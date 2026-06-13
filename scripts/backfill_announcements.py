#!/usr/bin/env python3
"""Resumable CNINFO announcement backfill into the local raw store."""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import requests

from utils.storage import Storage


CNINFO_PAGE_SIZE = 30
MARKET_PROGRESS_CODE = "__MARKET__"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill raw CNINFO announcements with resumable progress.")
    parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD. Defaults to training_set min date.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD. Defaults to training_set max date.")
    parser.add_argument("--codes", default="", help="Comma-separated stock codes for by-code mode.")
    parser.add_argument("--max-codes", type=int, default=0, help="Limit codes for smoke tests.")
    parser.add_argument("--mode", choices=["by-code", "market"], default="by-code",
                        help="by-code tracks code+quarter chunks; market fetches all-market date chunks.")
    parser.add_argument("--chunk-months", type=int, default=3, help="Months per by-code chunk.")
    parser.add_argument("--chunk-days", type=int, default=7, help="Days per market chunk.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent chunk workers.")
    parser.add_argument("--delay", type=float, default=0.2, help="Polite delay between CNINFO page requests.")
    parser.add_argument("--timeout", type=float, default=15.0, help="Per CNINFO request read timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Chunk retry count.")
    parser.add_argument("--force", action="store_true", help="Refetch chunks already marked done.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _load_universe(root: Path, args: argparse.Namespace) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")

    raw = pd.read_parquet(data_path, columns=["trade_date", "code"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    start = pd.Timestamp(args.start).normalize() if args.start else raw["trade_date"].min()
    end = pd.Timestamp(args.end).normalize() if args.end else raw["trade_date"].max()

    if args.codes:
        codes = [code.strip().zfill(6) for code in args.codes.split(",") if code.strip()]
    else:
        codes = sorted(raw["code"].dropna().astype(str).str.zfill(6).unique())
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    if not codes and args.mode == "by-code":
        raise RuntimeError("股票代码列表为空")
    if start > end:
        raise RuntimeError("start 不能晚于 end")
    return codes, start, end


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    })
    return session


def _request_json(session: requests.Session, url: str, timeout: float, **kwargs) -> dict:
    response = session.post(url, timeout=(5, timeout), **kwargs)
    response.raise_for_status()
    return response.json()


def _get_json(session: requests.Session, url: str, timeout: float) -> dict:
    response = session.get(url, timeout=(5, timeout))
    response.raise_for_status()
    return response.json()


def _cninfo_stock_org_ids(timeout: float, retries: int, delay: float) -> dict[str, str]:
    url = "http://www.cninfo.com.cn/new/data/szse_stock.json"
    last_exc = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            items = _get_json(_session(), url, timeout).get("stockList", [])
            return {
                str(item.get("code", "")).zfill(6): str(item.get("orgId", ""))
                for item in items
                if item.get("code") and item.get("orgId")
            }
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                _sleep_backoff(delay, attempt)
    raise last_exc


def _sleep_backoff(delay: float, attempt: int) -> None:
    base = max(delay, 0.1) * (2 ** max(0, attempt - 1))
    time.sleep(min(base, 8.0) + random.uniform(0, max(delay, 0.1)))


def _clean_title(title: str) -> str:
    text = re.sub(r"</?em>", "", str(title or ""))
    return " ".join(text.split())[:200]


def _parse_announcement_time(value) -> str | None:
    published_at = pd.to_datetime(value, unit="ms", utc=True, errors="coerce")
    if pd.isna(published_at):
        return None
    published_at = published_at.tz_convert("Asia/Shanghai").tz_localize(None)
    return published_at.isoformat(sep=" ", timespec="seconds")


def _announcement_row(item: dict, fallback_code: str = "", fallback_org_id: str = "") -> dict | None:
    raw_code = str(item.get("secCode") or fallback_code or "").strip()
    announcement_id = item.get("announcementId")
    published_at = _parse_announcement_time(item.get("announcementTime"))
    if not raw_code or not announcement_id or not published_at:
        return None
    code = raw_code.zfill(6)
    org_id = str(item.get("orgId") or fallback_org_id or "")
    url = (
        "http://www.cninfo.com.cn/new/disclosure/detail?"
        f"stockCode={code}&announcementId={announcement_id}&orgId={org_id}"
    )
    return {
        "source": "cninfo",
        "announcement_id": str(announcement_id),
        "code": code,
        "name": str(item.get("secName") or ""),
        "title": _clean_title(item.get("announcementTitle", "")),
        "published_at": published_at,
        "url": url,
        "org_id": org_id,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def _payload(start: str, end: str, stock: str) -> dict:
    return {
        "pageNum": "1",
        "pageSize": str(CNINFO_PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": stock,
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{start}~{end}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }


def _fetch_pages(stock: str, start: str, end: str, timeout: float, delay: float,
                 fallback_code: str = "", fallback_org_id: str = "") -> list[dict]:
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    session = _session()
    payload = _payload(start, end, stock)
    first_page = _request_json(session, url, timeout, params=payload)
    total = int(first_page.get("totalAnnouncement") or 0)
    page_count = max(1, math.ceil(total / CNINFO_PAGE_SIZE))
    pages = [first_page]
    for page in range(2, page_count + 1):
        payload["pageNum"] = str(page)
        if delay > 0:
            time.sleep(delay + random.uniform(0, delay))
        pages.append(_request_json(session, url, timeout, data=payload))

    rows = []
    seen = set()
    for page in pages:
        for item in page.get("announcements") or []:
            row = _announcement_row(item, fallback_code=fallback_code, fallback_org_id=fallback_org_id)
            if not row:
                continue
            key = (row["source"], row["announcement_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def _fetch_code_chunk(code: str, org_id: str, start: str, end: str,
                      timeout: float, delay: float) -> list[dict]:
    return _fetch_pages(
        stock=f"{code},{org_id}",
        start=start,
        end=end,
        timeout=timeout,
        delay=delay,
        fallback_code=code,
        fallback_org_id=org_id,
    )


def _fetch_market_chunk(start: str, end: str, timeout: float, delay: float) -> list[dict]:
    return _fetch_pages(stock="", start=start, end=end, timeout=timeout, delay=delay)


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    months = max(1, months)
    current = start.normalize()
    while current <= end:
        chunk_end = min(current + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def _day_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    days = max(1, days)
    current = start.normalize()
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=days - 1), end)
        yield current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        current = chunk_end + pd.Timedelta(days=1)


def _build_tasks(args: argparse.Namespace, codes: list[str], start: pd.Timestamp,
                 end: pd.Timestamp, org_ids: dict[str, str]) -> list[dict]:
    if args.mode == "market":
        return [
            {"mode": "market", "code": MARKET_PROGRESS_CODE, "chunk_start": chunk_start, "chunk_end": chunk_end}
            for chunk_start, chunk_end in _day_chunks(start, end, args.chunk_days)
        ]

    tasks = []
    for code in codes:
        org_id = org_ids.get(code)
        if not org_id:
            print(f"skip {code}: missing cninfo orgId", flush=True)
            continue
        for chunk_start, chunk_end in _month_chunks(start, end, args.chunk_months):
            tasks.append({
                "mode": "by-code",
                "code": code,
                "org_id": org_id,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
            })
    return tasks


def _skip_done(tasks: list[dict], storage: Storage, force: bool) -> list[dict]:
    if force:
        return tasks
    done = {
        (row["code"], row["chunk_start"], row["chunk_end"])
        for row in storage.get_announcement_progress()
        if row.get("status") == "done"
    }
    return [
        task for task in tasks
        if (task["code"], task["chunk_start"], task["chunk_end"]) not in done
    ]


def _run_task(task: dict, args: argparse.Namespace, storage: Storage,
              db_lock: threading.Lock) -> dict:
    code = task["code"]
    chunk_start = task["chunk_start"]
    chunk_end = task["chunk_end"]
    with db_lock:
        storage.mark_announcement_chunk(code, chunk_start, chunk_end, "pending", 0)

    last_error = ""
    for attempt in range(1, max(1, args.retries) + 1):
        try:
            if task["mode"] == "market":
                rows = _fetch_market_chunk(chunk_start, chunk_end, args.timeout, args.delay)
            else:
                rows = _fetch_code_chunk(
                    task["code"], task["org_id"], chunk_start, chunk_end, args.timeout, args.delay,
                )
            with db_lock:
                storage.upsert_announcements(rows)
                storage.mark_announcement_chunk(
                    code, chunk_start, chunk_end, "done", attempt, len(rows),
                )
            return {"status": "done", "rows": len(rows), "task": task}
        except Exception as exc:
            last_error = str(exc)
            if attempt < args.retries:
                _sleep_backoff(args.delay, attempt)

    with db_lock:
        storage.mark_announcement_chunk(
            code, chunk_start, chunk_end, "failed", max(1, args.retries), 0, last_error[:500],
        )
    return {"status": "failed", "rows": 0, "error": last_error, "task": task}


def main() -> None:
    args = _parse_args()
    root = _root()
    codes, start, end = _load_universe(root, args)
    storage = Storage()

    org_ids = {}
    if args.mode == "by-code":
        org_ids = _cninfo_stock_org_ids(args.timeout, args.retries, args.delay)
    tasks = _build_tasks(args, codes, start, end, org_ids)
    tasks = _skip_done(tasks, storage, args.force)
    if not tasks:
        print("announcement backfill: no pending chunks")
        return

    print(
        f"announcement backfill mode={args.mode}, chunks={len(tasks)}, "
        f"start={start.date()}, end={end.date()}, workers={args.workers}",
        flush=True,
    )

    done = 0
    failed = 0
    rows = 0
    db_lock = threading.Lock()
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_task, task, args, storage, db_lock) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows += result["rows"]
            if result["status"] == "done":
                done += 1
            else:
                failed += 1
            task = result["task"]
            print(
                f"{idx}/{len(tasks)} {result['status']} "
                f"{task['code']} {task['chunk_start']}~{task['chunk_end']} "
                f"rows={result['rows']} done={done} failed={failed}",
                flush=True,
            )

    print(f"announcement backfill finished: done={done}, failed={failed}, rows_upserted={rows}")
    if failed:
        print("failed chunks are recorded in announcement_fetch_progress and will be retried on the next run")


if __name__ == "__main__":
    main()
