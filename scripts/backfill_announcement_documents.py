#!/usr/bin/env python3
"""Build a resumable PIT announcement document and structured-event library.

The main StockWatch SQLite database remains the immutable metadata source.  This
script writes a separate research database and document tree under the history
directory so a partial PDF backfill can never poison the production store.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from config import get_config


EXTRACTOR_VERSION = "cninfo_rules_v1"
DETAIL_API = "https://www.cninfo.com.cn/new/announcement/bulletin_detail"
STATIC_ROOT = "https://static.cninfo.com.cn"
CATEGORY_TERMS = {
    "earnings": ("业绩预告", "业绩快报"),
    "buyback": ("回购",),
    "holding_change": ("增持", "减持", "持股变动"),
    "major_contract": ("重大合同", "中标", "签订合同", "项目合同"),
    "inquiry_penalty": ("问询函", "监管函", "处罚", "警示函", "立案"),
    "capital_action": ("分红", "权益分派", "利润分配", "股权激励"),
}
CATEGORY_PRIORITY = tuple(CATEGORY_TERMS)
PERCENT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
MONEY_RE = re.compile(
    r"(?:人民币)?\s*([0-9]+(?:\.[0-9]+)?)\s*(万亿元|亿元|万元|元|亿|万)"
)
SPACE_RE = re.compile(r"\s+")


def _parse_args() -> argparse.Namespace:
    history = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Backfill six high-information CNINFO document classes.")
    parser.add_argument("--source-db", default=str(Path(get_config().db_path).expanduser()))
    parser.add_argument("--library-db", default=str(history / "announcement_event_library.sqlite"))
    parser.add_argument("--document-dir", default=str(history / "announcement_documents"))
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument("--categories", default=",".join(CATEGORY_PRIORITY))
    parser.add_argument("--limit-per-category-year", type=int, default=50,
                        help="0 means every matching metadata row; positive values create a balanced batch.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-document-mb", type=float, default=20.0)
    parser.add_argument("--keep-pdf", action="store_true",
                        help="Keep PDFs after text extraction. Hashes and gzipped text are always retained.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-text-empty", action="store_true",
                        help="Retry scanned/empty PDFs with the OCR fallback.")
    return parser.parse_args()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            source TEXT NOT NULL,
            announcement_id TEXT NOT NULL,
            code TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            published_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            detail_url TEXT,
            document_url TEXT,
            document_type TEXT,
            local_path TEXT,
            text_path TEXT,
            byte_size INTEGER,
            document_sha256 TEXT,
            text_sha256 TEXT,
            text_char_count INTEGER,
            text_extraction_method TEXT,
            status TEXT NOT NULL,
            error TEXT,
            fetched_at TEXT,
            extracted_at TEXT,
            extractor_version TEXT NOT NULL,
            PRIMARY KEY (source, announcement_id)
        );
        CREATE INDEX IF NOT EXISTS idx_documents_category_time
            ON documents(category, published_at);
        CREATE INDEX IF NOT EXISTS idx_documents_code_available
            ON documents(code, available_at);

        CREATE TABLE IF NOT EXISTS events (
            source TEXT NOT NULL,
            announcement_id TEXT NOT NULL,
            event_index INTEGER NOT NULL,
            code TEXT NOT NULL,
            category TEXT NOT NULL,
            event_type TEXT NOT NULL,
            direction INTEGER NOT NULL,
            event_status TEXT NOT NULL,
            magnitude_value REAL,
            magnitude_unit TEXT,
            magnitude_percent REAL,
            signed_score REAL NOT NULL,
            novelty REAL NOT NULL,
            confidence REAL NOT NULL,
            evidence TEXT NOT NULL,
            title_fingerprint TEXT NOT NULL,
            published_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            document_sha256 TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            extracted_at TEXT NOT NULL,
            PRIMARY KEY (source, announcement_id, event_index),
            FOREIGN KEY (source, announcement_id)
                REFERENCES documents(source, announcement_id)
        );
        CREATE INDEX IF NOT EXISTS idx_events_code_available
            ON events(code, available_at);
        CREATE INDEX IF NOT EXISTS idx_events_category_available
            ON events(category, available_at);
        """
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
    if "text_extraction_method" not in columns:
        conn.execute("ALTER TABLE documents ADD COLUMN text_extraction_method TEXT")
        conn.commit()
    return conn


def _available_at(value: str) -> str:
    ts = pd.to_datetime(value, errors="raise")
    if ts.tzinfo is not None:
        ts = ts.tz_convert("Asia/Shanghai").tz_localize(None)
    if ts.time().hour == 0 and ts.time().minute == 0 and ts.time().second == 0:
        ts = ts.normalize() + pd.Timedelta(hours=15, seconds=1)
    return ts.isoformat(sep=" ", timespec="seconds")


def _normalize_title(title: str) -> str:
    text = SPACE_RE.sub("", str(title or "")).lower()
    text = re.sub(r"(?:关于|公告|的|公司|股份有限公司)", "", text)
    return text[:300]


def _classify(title: str, allowed: set[str] | None = None) -> str | None:
    allowed = allowed or set(CATEGORY_PRIORITY)
    text = str(title or "")
    for category in CATEGORY_PRIORITY:
        if category in allowed and any(term in text for term in CATEGORY_TERMS[category]):
            return category
    return None


def _candidate_rows(source_db: Path, categories: list[str], start: str, end: str,
                    limit_per_category_year: int) -> list[dict]:
    rows: list[dict] = []
    with sqlite3.connect(source_db) as conn:
        conn.row_factory = sqlite3.Row
        for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            year_start = max(pd.Timestamp(start), pd.Timestamp(f"{year}-01-01")).isoformat(sep=" ")
            year_end = min(pd.Timestamp(end) + pd.Timedelta(days=1), pd.Timestamp(f"{year + 1}-01-01")).isoformat(sep=" ")
            for category in categories:
                terms = CATEGORY_TERMS[category]
                where = " OR ".join("title LIKE ?" for _ in terms)
                sql = f"""
                    SELECT source, announcement_id, code, title, published_at, url
                    FROM announcements
                    WHERE published_at >= ? AND published_at < ? AND ({where})
                    ORDER BY published_at, announcement_id
                """
                params: list[object] = [year_start, year_end, *[f"%{term}%" for term in terms]]
                eligible = []
                for raw in conn.execute(sql, params):
                    row = dict(raw)
                    if _classify(row["title"], {category}) != category:
                        continue
                    row["category"] = category
                    eligible.append(row)
                if limit_per_category_year <= 0 or len(eligible) <= limit_per_category_year:
                    rows.extend(eligible)
                elif limit_per_category_year == 1:
                    rows.append(eligible[len(eligible) // 2])
                else:
                    # Deterministic time-stratified coverage; taking the latest N would
                    # silently concentrate every research batch in late December.
                    indexes = [
                        round(index * (len(eligible) - 1) / (limit_per_category_year - 1))
                        for index in range(limit_per_category_year)
                    ]
                    rows.extend(eligible[index] for index in indexes)
    deduped = {}
    for row in rows:
        key = (row["source"], row["announcement_id"])
        deduped.setdefault(key, row)
    return list(deduped.values())


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) StockWatchResearch/1.0",
        "Referer": "https://www.cninfo.com.cn/",
        "Origin": "https://www.cninfo.com.cn",
        "Accept": "application/json, text/plain, */*",
    })
    return session


def _request(session: requests.Session, method: str, url: str, *, timeout: float,
             retries: int, **kwargs) -> requests.Response:
    error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except Exception as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    assert error is not None
    raise error


def _resolve_document(session: requests.Session, announcement_id: str,
                      timeout: float, retries: int) -> tuple[str, str]:
    response = _request(
        session, "POST", DETAIL_API, timeout=timeout, retries=retries,
        params={"announceId": announcement_id, "flag": "false", "announceTime": ""},
    )
    payload = response.json()
    announcement = payload.get("announcement") or {}
    adjunct = str(announcement.get("adjunctUrl") or "").lstrip("/")
    if not adjunct:
        raise RuntimeError("CNINFO detail API returned no adjunctUrl")
    document_type = str(
        announcement.get("adjunctType") or Path(adjunct).suffix.lstrip(".") or "PDF"
    ).upper().lstrip(".")
    return f"{STATIC_ROOT}/{adjunct}", document_type


def _ocr_pdf_text(pdf_path: Path, max_pages: int = 40) -> str:
    with tempfile.TemporaryDirectory(prefix="stockwatch-ocr-") as temp_dir:
        prefix = Path(temp_dir) / "page"
        render = subprocess.run(
            ["pdftoppm", "-f", "1", "-l", str(max_pages), "-r", "150", "-jpeg", str(pdf_path), str(prefix)],
            check=False, capture_output=True,
        )
        if render.returncode != 0:
            error = render.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"pdftoppm OCR render failed: {error[:300]}")
        parts = []
        for image_path in sorted(Path(temp_dir).glob("page-*.jpg")):
            result = subprocess.run(
                ["tesseract", str(image_path), "stdout", "-l", "chi_sim+eng", "--psm", "6"],
                check=False, capture_output=True,
            )
            if result.returncode == 0:
                parts.append(result.stdout.decode("utf-8", errors="replace"))
        return "\n".join(parts).replace("\x00", "").strip()


def _extract_pdf_text(pdf_path: Path) -> tuple[str, str]:
    result = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        check=False, capture_output=True,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"pdftotext failed: {error[:300]}")
    text = result.stdout.decode("utf-8", errors="replace").replace("\x00", "").strip()
    if text:
        return text, "pdftotext"
    ocr_text = _ocr_pdf_text(pdf_path)
    return ocr_text, "tesseract_ocr" if ocr_text else "empty"


def _money_value(text: str) -> tuple[float | None, str | None]:
    matches = MONEY_RE.findall(text)
    if not matches:
        return None, None
    scale = {"元": 1.0, "万": 1e4, "万元": 1e4, "亿": 1e8, "亿元": 1e8, "万亿元": 1e12}
    values = [(float(value) * scale[unit], unit) for value, unit in matches]
    value, unit = max(values, key=lambda item: abs(item[0]))
    return value, unit


def _direction(category: str, title: str, text: str) -> int:
    """Prefer the event-bearing title over incidental words in long legal text."""
    title = str(title or "")
    if category == "inquiry_penalty":
        return -1
    if category == "buyback":
        return -1 if any(term in title for term in ("终止", "取消")) else 1
    if category == "holding_change":
        if any(term in title for term in ("终止减持", "取消减持")):
            return 1
        if "减持" in title:
            return -1
        if "增持" in title:
            return 1
    if category == "major_contract":
        return -1 if any(term in title for term in ("终止", "取消", "未中标")) else 1
    if category == "capital_action":
        return -1 if any(term in title for term in ("终止", "取消", "不分配")) else 1
    if category == "earnings":
        negative = ("预减", "下降", "亏损", "首亏", "续亏", "由盈转亏", "转亏")
        positive = ("预增", "增长", "扭亏", "减亏")
        evidence = f"{title} {text[:6000]}"
        if any(term in evidence for term in negative):
            return -1
        if any(term in evidence for term in positive):
            return 1
    return 0


def _event_status(category: str, title: str, text: str = "") -> str:
    title = str(title or "")
    if any(term in title for term in ("终止", "取消")):
        return "cancelled"
    if any(term in title for term in ("完成", "实施完毕", "结果", "已履行")):
        return "completed"
    if any(term in title for term in ("进展", "实施", "累计")):
        return "executing"
    if any(term in title for term in ("预案", "计划", "拟", "提示性")):
        return "planned"
    if category == "earnings":
        return "forecast" if "预告" in text else "reported"
    return "reported"


def _relevant_text(row: dict, text: str) -> str:
    terms = CATEGORY_TERMS[row["category"]]
    lines = []
    for raw_line in text.splitlines():
        line = SPACE_RE.sub(" ", raw_line).strip()
        if not line or len(line) > 600:
            continue
        has_event_term = any(term in line for term in terms)
        has_magnitude = bool(PERCENT_RE.search(line) or MONEY_RE.search(line))
        if has_event_term or has_magnitude:
            lines.append(line)
        if len(lines) >= 120:
            break
    return SPACE_RE.sub(" ", f"{row['title']} {' '.join(lines)}").strip()


def _extract_event(row: dict, text: str, document_sha256: str, novelty: float) -> dict:
    evidence_text = _relevant_text(row, text)
    direction = _direction(row["category"], row["title"], evidence_text)
    percentages = [abs(float(value)) for value in PERCENT_RE.findall(evidence_text)]
    magnitude_percent = min(max(percentages), 500.0) if percentages else None
    magnitude_value, magnitude_unit = _money_value(evidence_text)
    magnitude_component = 0.0
    if magnitude_percent is not None:
        magnitude_component = min(magnitude_percent / 100.0, 5.0)
    elif magnitude_value is not None:
        magnitude_component = min(max(0.0, __import__("math").log10(max(1.0, magnitude_value)) - 4.0) / 4.0, 3.0)
    signed_score = float(direction) * (1.0 + magnitude_component) * float(novelty)
    confidence = 0.9 if direction and (magnitude_percent is not None or magnitude_value is not None) else (0.7 if direction else 0.4)
    title_fp = hashlib.sha256(_normalize_title(row["title"]).encode("utf-8")).hexdigest()
    return {
        "source": row["source"],
        "announcement_id": row["announcement_id"],
        "event_index": 0,
        "code": str(row["code"]).zfill(6),
        "category": row["category"],
        "event_type": row["category"],
        "direction": direction,
        "event_status": _event_status(row["category"], row["title"], evidence_text),
        "magnitude_value": magnitude_value,
        "magnitude_unit": magnitude_unit,
        "magnitude_percent": magnitude_percent,
        "signed_score": signed_score,
        "novelty": novelty,
        "confidence": confidence,
        "evidence": evidence_text[:800],
        "title_fingerprint": title_fp,
        "published_at": row["published_at"],
        "available_at": _available_at(row["published_at"]),
        "document_sha256": document_sha256,
        "extractor_version": EXTRACTOR_VERSION,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def _download_one(row: dict, document_dir: Path, timeout: float, retries: int,
                  max_bytes: int, keep_pdf: bool) -> dict:
    session = _session()
    now = datetime.now().isoformat(timespec="seconds")
    try:
        document_url, document_type = _resolve_document(session, row["announcement_id"], timeout, retries)
        response = _request(session, "GET", document_url, timeout=timeout, retries=retries, stream=True)
        length = int(response.headers.get("content-length") or 0)
        if length and length > max_bytes:
            raise RuntimeError(f"document exceeds max size: {length} > {max_bytes}")
        year = pd.Timestamp(row["published_at"]).year
        base = document_dir / str(year) / row["category"]
        base.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if document_type == "PDF" else f".{document_type.lower()}"
        final_path = base / f"{row['announcement_id']}{suffix}"
        with tempfile.NamedTemporaryFile(dir=base, suffix=suffix, delete=False) as handle:
            tmp_path = Path(handle.name)
            digest = hashlib.sha256()
            total = 0
            for chunk in response.iter_content(1024 * 128):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise RuntimeError(f"document exceeds max size: > {max_bytes}")
                handle.write(chunk)
                digest.update(chunk)
        document_sha = digest.hexdigest()
        if document_type != "PDF":
            raise RuntimeError(f"unsupported document type: {document_type}")
        text, text_extraction_method = _extract_pdf_text(tmp_path)
        text_bytes = text.encode("utf-8")
        text_sha = hashlib.sha256(text_bytes).hexdigest()
        text_path = base / f"{row['announcement_id']}.txt.gz"
        with gzip.open(text_path, "wb", compresslevel=6) as handle:
            handle.write(text_bytes)
        if keep_pdf:
            tmp_path.replace(final_path)
            local_path = str(final_path)
        else:
            tmp_path.unlink(missing_ok=True)
            local_path = None
        return {
            **row,
            "document_url": document_url,
            "document_type": document_type,
            "local_path": local_path,
            "text_path": str(text_path),
            "byte_size": total,
            "document_sha256": document_sha,
            "text_sha256": text_sha,
            "text_char_count": len(text),
            "text_extraction_method": text_extraction_method,
            "text": text,
            "status": "done" if text else "text_empty",
            "error": None if text else "pdftotext produced empty text; PDF may be scanned",
            "fetched_at": now,
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        try:
            if "tmp_path" in locals():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {
            **row,
            "document_url": locals().get("document_url"),
            "document_type": locals().get("document_type"),
            "local_path": None,
            "text_path": None,
            "byte_size": None,
            "document_sha256": None,
            "text_sha256": None,
            "text_char_count": 0,
            "text_extraction_method": None,
            "text": "",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "fetched_at": now,
            "extracted_at": None,
        }


def _novelty(conn: sqlite3.Connection, row: dict) -> float:
    fingerprint = hashlib.sha256(_normalize_title(row["title"]).encode("utf-8")).hexdigest()
    prior = conn.execute(
        """SELECT 1 FROM events
           WHERE code=? AND title_fingerprint=? AND published_at < ? LIMIT 1""",
        (str(row["code"]).zfill(6), fingerprint, row["published_at"]),
    ).fetchone()
    return 0.35 if prior else 1.0


def _recompute_novelty(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """SELECT source, announcement_id, event_index, code, title_fingerprint,
                  published_at, novelty, signed_score
           FROM events ORDER BY code, title_fingerprint, published_at, announcement_id"""
    ).fetchall()
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["code"], row["title_fingerprint"])
        novelty = 0.35 if key in seen else 1.0
        seen.add(key)
        old_novelty = float(row["novelty"] or 1.0)
        score = float(row["signed_score"] or 0.0) / old_novelty * novelty
        conn.execute(
            """UPDATE events SET novelty=?, signed_score=?
               WHERE source=? AND announcement_id=? AND event_index=?""",
            (novelty, score, row["source"], row["announcement_id"], row["event_index"]),
        )
    conn.commit()


def _write_result(conn: sqlite3.Connection, result: dict) -> None:
    document_columns = [
        "source", "announcement_id", "code", "category", "title", "published_at", "available_at",
        "detail_url", "document_url", "document_type", "local_path", "text_path", "byte_size",
        "document_sha256", "text_sha256", "text_char_count", "status", "error", "fetched_at",
        "extracted_at", "extractor_version", "text_extraction_method",
    ]
    doc = {
        **result,
        "available_at": _available_at(result["published_at"]),
        "detail_url": result.get("url"),
        "extractor_version": EXTRACTOR_VERSION,
    }
    placeholders = ",".join("?" for _ in document_columns)
    updates = ",".join(f"{name}=excluded.{name}" for name in document_columns[2:])
    conn.execute(
        f"INSERT INTO documents ({','.join(document_columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT(source,announcement_id) DO UPDATE SET {updates}",
        [doc.get(name) for name in document_columns],
    )
    conn.execute("DELETE FROM events WHERE source=? AND announcement_id=?", (result["source"], result["announcement_id"]))
    if result["status"] == "done":
        event = _extract_event(result, result["text"], result["document_sha256"], _novelty(conn, result))
        event_columns = list(event)
        conn.execute(
            f"INSERT INTO events ({','.join(event_columns)}) VALUES ({','.join('?' for _ in event_columns)})",
            [event[name] for name in event_columns],
        )
    conn.commit()


def _summary(conn: sqlite3.Connection, requested: int) -> dict:
    rows = conn.execute(
        """SELECT category, status, COUNT(*) count, COALESCE(SUM(byte_size),0) bytes,
                  COALESCE(SUM(text_char_count),0) chars
           FROM documents GROUP BY category, status ORDER BY category, status"""
    ).fetchall()
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_this_run": requested,
        "documents": [dict(row) for row in rows],
        "event_count": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        "extractor_version": EXTRACTOR_VERSION,
    }


def main() -> None:
    args = _parse_args()
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    unknown = sorted(set(categories) - set(CATEGORY_PRIORITY))
    if unknown:
        raise RuntimeError(f"unknown categories: {unknown}")
    source_db = Path(args.source_db).expanduser()
    library_db = Path(args.library_db).expanduser()
    document_dir = Path(args.document_dir).expanduser()
    if not source_db.exists():
        raise RuntimeError(f"source DB missing: {source_db}")
    candidates = _candidate_rows(
        source_db, categories, args.start, args.end, args.limit_per_category_year,
    )
    conn = _connect(library_db)
    if not args.force:
        terminal_statuses = ("done",) if args.retry_text_empty else ("done", "text_empty")
        status_placeholders = ",".join("?" for _ in terminal_statuses)
        done = {
            tuple(row) for row in conn.execute(
                f"SELECT source, announcement_id FROM documents WHERE status IN ({status_placeholders})",
                terminal_statuses,
            )
        }
        candidates = [row for row in candidates if (row["source"], row["announcement_id"]) not in done]
    print(f"announcement document candidates={len(candidates)} categories={categories}", flush=True)
    max_bytes = int(max(1.0, args.max_document_mb) * 1024 * 1024)
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                _download_one, row, document_dir, args.timeout, args.retries,
                max_bytes, args.keep_pdf,
            )
            for row in candidates
        ]
        for future in as_completed(futures):
            result = future.result()
            _write_result(conn, result)
            completed += 1
            if completed % 25 == 0 or completed == len(candidates):
                print(f"documents processed={completed}/{len(candidates)} last={result['status']}", flush=True)
    _recompute_novelty(conn)
    summary = _summary(conn, len(candidates))
    report = library_db.with_suffix(".report.json")
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"announcement event library: {library_db}")
    print(f"report: {report}")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
