#!/usr/bin/env python3
"""Capture official CSI 500 membership sources without normalizing them.

The public CSI endpoints expose the latest full constituent/weight snapshots and
historical rebalance announcements.  Raw bytes are retained so normalization is
repeatable and every derived row can carry a source hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


EXTRACTOR_VERSION = "csi500_official_capture_v1"
INDEX_CODE = "000905"
CSI_HOME = "https://www.csindex.com.cn/csindex-home"
SEARCH_URL = f"{CSI_HOME}/announcement/queryAnnouncementByVo"
DETAIL_URL = f"{CSI_HOME}/announcement/queryAnnouncementById"
CONSTITUENTS_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/"
    f"autofile/cons/{INDEX_CODE}cons.xls"
)
CURRENT_WEIGHT_URL = (
    "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/"
    f"autofile/closeweight/{INDEX_CODE}closeweight.xls"
)
HEADERS = {
    "User-Agent": "StockWatch CSI500 PIT research/1.0 (+raw archival)",
    "Accept": "application/json, text/plain, */*",
}


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Capture official CSI500 PIT sources.")
    parser.add_argument("--raw-dir", default=str(root / "csi500_official_raw"))
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.now().date().isoformat())
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--pace", type=float, default=0.35)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_bytes(data)
    temp.replace(path)


def _request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    timeout: float,
    retries: int,
    **kwargs,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            if "text/html" in response.headers.get("content-type", "").lower():
                prefix = response.content[:200].lower()
                if b"<title>405</title>" in prefix or b"parameter errors" in prefix:
                    raise RuntimeError("CSI endpoint returned an anti-bot/error page")
            return response
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch official source {url}: {last_error}")


def _conservative_available_at(publish_date: str) -> str:
    """Date-only announcements are not assumed tradable on publication day."""

    published = datetime.fromisoformat(publish_date).date()
    next_day = datetime.combine(published + timedelta(days=1), datetime.min.time())
    return next_day.replace(tzinfo=timezone(timedelta(hours=8))).isoformat()


def _source_entry(
    *,
    kind: str,
    source_url: str,
    local_path: Path,
    raw_root: Path,
    data: bytes,
    fetched_at: str,
    published_at: str | None = None,
    available_at: str | None = None,
    notice_id: int | None = None,
) -> dict:
    return {
        "kind": kind,
        "notice_id": notice_id,
        "published_at": published_at,
        "available_at": available_at,
        "source_url": source_url,
        "source_hash": _sha256(data),
        "byte_size": len(data),
        "local_path": str(local_path.relative_to(raw_root)),
        "fetched_at": fetched_at,
        "extractor_version": EXTRACTOR_VERSION,
    }


def _capture_url(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    timeout: float,
    retries: int,
    force: bool,
) -> bytes:
    if path.exists() and not force:
        return path.read_bytes()
    response = _request(session, "GET", url, timeout=timeout, retries=retries)
    _write_bytes(path, response.content)
    return response.content


def _attachment_name(notice_id: int, index: int, file_name: str, file_url: str) -> str:
    suffix = Path(unquote(urlparse(file_url).path)).suffix or Path(file_name).suffix or ".bin"
    return f"{notice_id}_{index:02d}{suffix.lower()}"


def capture(args: argparse.Namespace) -> dict:
    raw_root = Path(args.raw_dir).expanduser()
    raw_root.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    entries: list[dict] = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for kind, url, relative in (
        ("current_constituents", CONSTITUENTS_URL, Path("snapshots/000905cons.xls")),
        ("current_weight_snapshot", CURRENT_WEIGHT_URL, Path("snapshots/000905closeweight.xls")),
    ):
        path = raw_root / relative
        data = _capture_url(
            session, url, path, timeout=args.timeout, retries=args.retries, force=args.force
        )
        entries.append(
            _source_entry(
                kind=kind,
                source_url=url,
                local_path=path,
                raw_root=raw_root,
                data=data,
                fetched_at=fetched_at,
                available_at=fetched_at,
            )
        )

    payload = {
        "searchInput": "中证500",
        "startDate": args.start,
        "endDate": args.end,
        "page": {"desc": "", "key": "", "page": 1, "rows": 100, "sortBy": ""},
    }
    search_response = _request(
        session,
        "POST",
        SEARCH_URL,
        timeout=args.timeout,
        retries=args.retries,
        json=payload,
    )
    search_obj = search_response.json()
    if str(search_obj.get("code")) != "200":
        raise RuntimeError(f"CSI announcement search failed: {search_obj}")
    if int(search_obj.get("total") or 0) > 100:
        raise RuntimeError("CSI announcement result exceeded one page; refusing partial capture")
    search_path = raw_root / "announcement_search.json"
    search_data = json.dumps(search_obj, ensure_ascii=False, indent=2).encode("utf-8")
    _write_bytes(search_path, search_data)
    entries.append(
        _source_entry(
            kind="announcement_search",
            source_url=SEARCH_URL,
            local_path=search_path,
            raw_root=raw_root,
            data=search_data,
            fetched_at=fetched_at,
        )
    )

    rows = [
        row
        for row in search_obj.get("data", [])
        if row.get("theme") == "指数调样" and row.get("noticeType") == "announcement"
    ]
    for row in sorted(rows, key=lambda item: (item["publishDate"], int(item["id"]))):
        notice_id = int(row["id"])
        detail_path = raw_root / "announcements" / f"{notice_id}.json"
        if detail_path.exists() and not args.force:
            detail_data = detail_path.read_bytes()
            detail_obj = json.loads(detail_data)
        else:
            time.sleep(max(args.pace, 0.0))
            response = _request(
                session,
                "GET",
                DETAIL_URL,
                timeout=args.timeout,
                retries=args.retries,
                params={"id": notice_id},
            )
            api_obj = response.json()
            if str(api_obj.get("code")) != "200" or not api_obj.get("data"):
                raise RuntimeError(f"CSI announcement detail failed for {notice_id}: {api_obj}")
            detail_obj = api_obj["data"]
            detail_data = json.dumps(detail_obj, ensure_ascii=False, indent=2).encode("utf-8")
            _write_bytes(detail_path, detail_data)

        content = f"{detail_obj.get('title', '')} {detail_obj.get('content', '')}"
        if "中证500" not in content.replace(" ", ""):
            continue
        published_at = str(detail_obj.get("publishDate") or row["publishDate"])
        available_at = _conservative_available_at(published_at)
        entries.append(
            _source_entry(
                kind="rebalance_announcement",
                source_url=f"{DETAIL_URL}?id={notice_id}",
                local_path=detail_path,
                raw_root=raw_root,
                data=detail_data,
                fetched_at=fetched_at,
                published_at=published_at,
                available_at=available_at,
                notice_id=notice_id,
            )
        )

        for index, enclosure in enumerate(detail_obj.get("enclosureList") or []):
            file_url = enclosure.get("fileUrl")
            if not file_url:
                continue
            file_name = _attachment_name(
                notice_id, index, str(enclosure.get("fileName") or ""), file_url
            )
            attachment_path = raw_root / "attachments" / file_name
            attachment_data = _capture_url(
                session,
                file_url,
                attachment_path,
                timeout=args.timeout,
                retries=args.retries,
                force=args.force,
            )
            entries.append(
                _source_entry(
                    kind="rebalance_attachment",
                    source_url=file_url,
                    local_path=attachment_path,
                    raw_root=raw_root,
                    data=attachment_data,
                    fetched_at=fetched_at,
                    published_at=published_at,
                    available_at=available_at,
                    notice_id=notice_id,
                )
            )

    manifest = {
        "index_code": INDEX_CODE,
        "capture_started_at": fetched_at,
        "query_start": args.start,
        "query_end": args.end,
        "extractor_version": EXTRACTOR_VERSION,
        "sources": entries,
    }
    manifest_path = raw_root / "manifest.json"
    _write_bytes(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    )
    return manifest


def main() -> None:
    args = _parse_args()
    manifest = capture(args)
    counts: dict[str, int] = {}
    for source in manifest["sources"]:
        counts[source["kind"]] = counts.get(source["kind"], 0) + 1
    print(json.dumps({"raw_dir": args.raw_dir, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
