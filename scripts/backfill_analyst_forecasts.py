#!/usr/bin/env python3
"""Resumably capture publication-dated analyst EPS forecasts and source PDFs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


EXTRACTOR_VERSION = "eastmoney_analyst_forecast_v1"
EPS_COLUMN_RE = re.compile(r"^(?P<year>\d{4})-盈利预测-收益$")
REPORT_ID_RE = re.compile(r"H3_(?P<id>.+?)_1\.pdf$")


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Backfill analyst forecast reports.")
    parser.add_argument("--output", default=str(root / "analyst_forecasts.parquet"))
    parser.add_argument("--progress", default=str(root / "analyst_forecast_progress.json"))
    parser.add_argument("--documents", default=str(root / "analyst_documents"))
    parser.add_argument("--codes", default="", help="Comma-separated stock codes.")
    parser.add_argument("--max-codes", type=int, default=0)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--download-pdfs", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _canonical_hash(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _report_id(url: str) -> str:
    match = REPORT_ID_RE.search(str(url or ""))
    if match:
        return match.group("id")
    return hashlib.sha256(str(url or "").encode("utf-8")).hexdigest()[:24]


def _normalize_report_frame(frame: pd.DataFrame, fetched_at: str) -> pd.DataFrame:
    columns = [column for column in frame.columns if EPS_COLUMN_RE.match(str(column))]
    rows = []
    for source in frame.to_dict("records"):
        published = pd.to_datetime(source.get("日期"), errors="coerce")
        if pd.isna(published):
            continue
        published = published.normalize()
        available = published + pd.Timedelta(hours=15, seconds=1)
        pdf_url = str(source.get("报告PDF链接") or "")
        base = {
            "code": str(source.get("股票代码") or "").zfill(6),
            "name": source.get("股票简称"),
            "title": source.get("报告名称"),
            "rating": source.get("东财评级"),
            "institution": source.get("机构"),
            "industry": source.get("行业"),
            "published_at": published,
            "available_at": available,
            "report_pdf_url": pdf_url,
            "report_id": _report_id(pdf_url),
            "source_row_sha256": _canonical_hash(source),
            "fetched_at": fetched_at,
            "extractor_version": EXTRACTOR_VERSION,
            "document_sha256": None,
            "document_path": None,
            "source_document_verified": False,
            "pit_verified": False,
            "pit_note": (
                "publication date comes from the current Eastmoney report index; "
                "the original historical vendor vintage has not been independently archived"
            ),
        }
        for column in columns:
            eps = pd.to_numeric(source.get(column), errors="coerce")
            if pd.isna(eps):
                continue
            target_period = EPS_COLUMN_RE.match(str(column)).group("year")
            pe_column = f"{target_period}-盈利预测-市盈率"
            pe = pd.to_numeric(source.get(pe_column), errors="coerce")
            rows.append({
                **base,
                "target_period": target_period,
                "forecast_eps": float(eps),
                "forecast_pe": None if pd.isna(pe) else float(pe),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(
        ["code", "report_id", "target_period"], keep="last"
    )


def _load_codes(root: Path, raw: str) -> list[str]:
    if raw.strip():
        return sorted({item.strip().zfill(6) for item in raw.split(",") if item.strip()})
    manifest = root / "history_manifest.json"
    if manifest.exists():
        payload = json.loads(manifest.read_text())
        codes = [str(code).zfill(6) for code in payload.get("codes") or []]
        if codes:
            return sorted(set(codes))
    return sorted(path.stem for path in (root / "stocks").glob("*.parquet"))


def _load_progress(path: Path, force: bool) -> dict:
    if force or not path.exists():
        return {"completed_codes": [], "failed": {}}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"completed_codes": [], "failed": {}}


def _write_progress(path: Path, progress: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2))


def _download_pdf(row: dict, documents: Path) -> dict:
    import requests

    url = str(row.get("report_pdf_url") or "")
    if not url:
        return row
    target = documents / row["code"] / f"{row['report_id']}.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 100:
        content = target.read_bytes()
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise RuntimeError("analyst document response is not a PDF")
        target.write_bytes(content)
    item = dict(row)
    item["document_sha256"] = hashlib.sha256(content).hexdigest()
    item["document_path"] = str(target)
    item["source_document_verified"] = True
    # A PDF hash proves source-document integrity, not that today's API row is
    # an archived historical vintage. Keep the stronger PIT claim false.
    item["pit_verified"] = False
    return item


def _merge_output(path: Path, additions: list[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    if path.exists():
        frames.append(pd.read_parquet(path))
    frames.extend(frame for frame in additions if not frame.empty)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["code", "report_id", "target_period"], keep="last"
    )
    result["source_document_verified"] = result.get(
        "document_sha256", pd.Series(index=result.index, dtype="object")
    ).notna()
    # Version 1 captures the report index as it exists today. Even with the
    # original PDF hashed, that is not an archived vendor vintage.
    result["pit_verified"] = False
    result["pit_note"] = (
        "publication date comes from the current Eastmoney report index; "
        "the original historical vendor vintage has not been independently archived"
    )
    result = result.sort_values(["available_at", "code", "target_period"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    return result


def main() -> None:
    import akshare as ak

    args = _parse_args()
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    output = Path(args.output).expanduser()
    progress_path = Path(args.progress).expanduser()
    documents = Path(args.documents).expanduser()
    progress = _load_progress(progress_path, args.force)
    completed = set(progress.get("completed_codes") or [])
    codes = [code for code in _load_codes(root, args.codes) if args.force or code not in completed]
    if args.max_codes:
        codes = codes[:args.max_codes]
    cutoff = pd.to_datetime(args.start_date)
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")
    additions = []

    for index, code in enumerate(codes, 1):
        try:
            raw = ak.stock_research_report_em(symbol=code)
            frame = _normalize_report_frame(raw, fetched_at)
            if not frame.empty:
                frame = frame[frame["published_at"] >= cutoff].copy()
            if args.download_pdfs and not frame.empty:
                unique_documents = {}
                for row in frame.to_dict("records"):
                    report_id = row["report_id"]
                    if report_id not in unique_documents:
                        unique_documents[report_id] = _download_pdf(row, documents)
                    document = unique_documents[report_id]
                    row["document_sha256"] = document["document_sha256"]
                    row["document_path"] = document["document_path"]
                    row["source_document_verified"] = document["source_document_verified"]
                    row["pit_verified"] = document["pit_verified"]
                    unique_documents[(report_id, row["target_period"])] = row
                frame = pd.DataFrame([
                    value for key, value in unique_documents.items() if isinstance(key, tuple)
                ])
            additions.append(frame)
            completed.add(code)
            progress.setdefault("failed", {}).pop(code, None)
        except KeyError as exc:
            if str(exc) != "'infoCode'":
                progress.setdefault("failed", {})[code] = str(exc)[:300]
            else:
                # AKShare raises this when Eastmoney returns no report rows and
                # therefore no infoCode column. Treat it as audited no-data,
                # not a retryable network failure.
                completed.add(code)
                progress.setdefault("no_data_codes", []).append(code)
                progress["no_data_codes"] = sorted(set(progress["no_data_codes"]))
                progress.setdefault("failed", {}).pop(code, None)
        except Exception as exc:
            progress.setdefault("failed", {})[code] = str(exc)[:300]
        progress["completed_codes"] = sorted(completed)
        _write_progress(progress_path, progress)
        if index % 20 == 0:
            _merge_output(output, additions)
            additions = []
        print(f"analyst forecasts {index}/{len(codes)} code={code}", flush=True)
        time.sleep(max(args.sleep, 0.0))

    result = _merge_output(output, additions)
    progress.update({
        "output": str(output),
        "rows": int(len(result)),
        "codes_with_forecasts": int(result["code"].nunique()) if len(result) else 0,
        "pit_verified_rows": int(result["pit_verified"].sum()) if len(result) else 0,
        "source_document_verified_rows": (
            int(result["source_document_verified"].sum()) if len(result) else 0
        ),
        "extractor_version": EXTRACTOR_VERSION,
        "warning": (
            "The endpoint exposes current target-year fields; older reports with empty forecast values "
            "cannot reconstruct historical consensus and remain absent."
        ),
    })
    _write_progress(progress_path, progress)
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
