#!/usr/bin/env python3
"""Build point-in-time daily news and announcement features."""
from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import sys
from datetime import time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import get_config
from utils.storage import Storage


CUTOFF_TIME = time(15, 0)
CNINFO_PAGE_SIZE = 30
NEWS_WINDOWS = (1, 3, 7)
ANN_WINDOWS = (7, 20)
ANN_TYPES = {
    "earnings": ("业绩", "预告", "快报", "年度报告", "季度报告", "半年报"),
    "holding": ("增持", "减持", "回购"),
    "inquiry": ("问询", "关注函", "监管函"),
    "risk": ("处罚", "立案", "诉讼", "仲裁", "违规"),
    "capital_action": ("重组", "并购", "收购", "定增", "非公开发行"),
    "dividend": ("分红", "权益分派", "利润分配"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PIT daily sentiment/event features.")
    parser.add_argument("--output", default=None, help="Output parquet path.")
    parser.add_argument("--start", default=None, help="Feature start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Feature end date, YYYY-MM-DD.")
    parser.add_argument("--codes", default="", help="Comma-separated stock codes.")
    parser.add_argument("--max-codes", type=int, default=0, help="Limit codes for smoke tests.")
    parser.add_argument("--announcement-timeout", type=float, default=15.0, help="Per cninfo request timeout in seconds.")
    parser.add_argument("--announcement-retries", type=int, default=3, help="Per cninfo request retry count.")
    parser.add_argument("--announcement-chunk-months", type=int, default=3, help="Months per cninfo date chunk.")
    parser.add_argument("--no-news", action="store_true", help="Skip forward-collected news features.")
    parser.add_argument("--no-announcements", action="store_true", help="Skip cninfo announcement features.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _load_grid(root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Index, list[str], pd.Timestamp, pd.Timestamp]:
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")

    raw = pd.read_parquet(data_path, columns=["trade_date", "code"]).drop_duplicates()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.normalize()
    calendar_dates = pd.Index(sorted(raw["trade_date"].drop_duplicates()))
    output_start = pd.Timestamp(args.start).normalize() if args.start else calendar_dates.min()
    output_end = pd.Timestamp(args.end).normalize() if args.end else calendar_dates.max()

    output_grid = raw[(raw["trade_date"] >= output_start) & (raw["trade_date"] <= output_end)]
    if args.codes:
        codes = [code.strip().zfill(6) for code in args.codes.split(",") if code.strip()]
        output_grid = output_grid[output_grid["code"].isin(codes)]
    codes = sorted(output_grid["code"].drop_duplicates())
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
        output_grid = output_grid[output_grid["code"].isin(codes)]
    if output_grid.empty:
        raise RuntimeError("特征网格为空，请检查日期或股票代码参数")

    warmup = max(max(NEWS_WINDOWS), max(ANN_WINDOWS))
    start_idx = int(calendar_dates.searchsorted(output_start, side="left"))
    calc_start = calendar_dates[max(0, start_idx - warmup)]
    grid = raw[
        raw["code"].isin(codes)
        & (raw["trade_date"] >= calc_start)
        & (raw["trade_date"] <= output_end)
    ]
    if args.start:
        grid = grid[grid["trade_date"] >= calc_start]
    if args.end:
        grid = grid[grid["trade_date"] <= output_end]
    grid = grid.sort_values(["code", "trade_date"]).reset_index(drop=True)
    return grid, calendar_dates, codes, output_start, output_end


def _to_feature_date(value, trade_dates: pd.Index) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    day = ts.normalize()
    if day < trade_dates.min():
        return None
    idx = int(trade_dates.searchsorted(day, side="left"))
    if idx >= len(trade_dates):
        return None
    if trade_dates[idx] == day and ts.time() <= CUTOFF_TIME:
        return trade_dates[idx]
    idx += 1 if trade_dates[idx] == day else 0
    if idx >= len(trade_dates):
        return None
    return trade_dates[idx]


def _empty_events(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _load_news_events(codes: list[str], trade_dates: pd.Index) -> pd.DataFrame:
    Storage()  # run migrations before reading the table directly
    db_path = get_config().db_path
    if not Path(db_path).expanduser().exists():
        return _empty_events(["code", "feature_date", "score", "title"])
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT code, title, source, ts, sentiment_score, available_at,
               fetched_at, sentiment_model_version
        FROM news
        WHERE code IN ({placeholders})
          AND COALESCE(available_at, fetched_at) IS NOT NULL
    """
    with sqlite3.connect(str(db_path)) as conn:
        rows = pd.read_sql_query(query, conn, params=codes)
    if rows.empty:
        return _empty_events(["code", "feature_date", "score", "title"])

    rows["feature_date"] = rows["available_at"].where(
        rows["available_at"].notna(), rows["fetched_at"],
    ).map(lambda value: _to_feature_date(value, trade_dates))
    rows = rows.dropna(subset=["feature_date"]).copy()
    rows["feature_date"] = pd.to_datetime(rows["feature_date"]).dt.normalize()
    rows["score"] = pd.to_numeric(rows["sentiment_score"], errors="coerce")
    return rows[["code", "feature_date", "score", "title"]]


def _classify_announcement(title: str) -> str:
    text = str(title or "")
    for name, keywords in ANN_TYPES.items():
        if any(keyword in text for keyword in keywords):
            return name
    return "other"


def _request_json(session, method: str, url: str, timeout: float, retries: int, **kwargs) -> dict:
    last_exc = None
    for _attempt in range(max(1, retries)):
        try:
            response = session.request(method, url, timeout=(5, timeout), **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_exc = exc
    raise last_exc


def _cninfo_stock_org_ids(timeout: float, retries: int) -> dict[str, str]:
    import requests

    url = "http://www.cninfo.com.cn/new/data/szse_stock.json"
    session = requests.Session()
    items = _request_json(session, "GET", url, timeout, retries).get("stockList", [])
    return {
        str(item.get("code", "")).zfill(6): str(item.get("orgId", ""))
        for item in items
        if item.get("code") and item.get("orgId")
    }


def _clean_title(title: str) -> str:
    text = re.sub(r"</?em>", "", str(title or ""))
    return " ".join(text.split())[:200]


def _fetch_cninfo_announcements(code: str, org_id: str, start: str, end: str,
                                timeout: float, retries: int) -> list[dict]:
    import requests

    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
    })
    payload = {
        "pageNum": "1",
        "pageSize": str(CNINFO_PAGE_SIZE),
        "column": "szse",
        "tabName": "fulltext",
        "plate": "",
        "stock": f"{code},{org_id}",
        "searchkey": "",
        "secid": "",
        "category": "",
        "trade": "",
        "seDate": f"{start[:4]}-{start[4:6]}-{start[6:]}~{end[:4]}-{end[4:6]}-{end[6:]}",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }

    first_page = _request_json(session, "POST", url, timeout, retries, params=payload)
    total = int(first_page.get("totalAnnouncement") or 0)
    page_count = max(1, math.ceil(total / CNINFO_PAGE_SIZE))
    pages = [first_page]
    for page in range(2, page_count + 1):
        payload["pageNum"] = str(page)
        pages.append(_request_json(session, "POST", url, timeout, retries, data=payload))

    rows = []
    for page in pages:
        for item in page.get("announcements") or []:
            published_at = pd.to_datetime(
                item.get("announcementTime"), unit="ms", utc=True, errors="coerce",
            )
            if pd.isna(published_at):
                continue
            published_at = published_at.tz_convert("Asia/Shanghai").tz_localize(None)
            announcement_id = item.get("announcementId", "")
            item_org_id = item.get("orgId") or org_id
            link = (
                "http://www.cninfo.com.cn/new/disclosure/detail?"
                f"stockCode={code}&announcementId={announcement_id}"
                f"&orgId={item_org_id}&announcementTime={published_at}"
            )
            rows.append({
                "代码": str(item.get("secCode") or code).zfill(6),
                "简称": str(item.get("secName") or ""),
                "公告标题": _clean_title(item.get("announcementTitle", "")),
                "公告时间": published_at,
                "公告链接": link,
            })
    return rows


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, months: int):
    months = max(1, months)
    current = start.normalize()
    end = end.normalize()
    while current <= end:
        chunk_end = min(current + pd.DateOffset(months=months) - pd.Timedelta(days=1), end)
        yield current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")
        current = chunk_end + pd.Timedelta(days=1)


def _load_announcement_events(codes: list[str], trade_dates: pd.Index,
                              start_date: pd.Timestamp, end_date: pd.Timestamp,
                              timeout: float, retries: int, chunk_months: int) -> pd.DataFrame:
    start = start_date - timedelta(days=45)
    end = end_date
    rows = []
    org_ids = _cninfo_stock_org_ids(timeout, retries)
    for idx, code in enumerate(codes, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(codes):
            print(f"announcements: {idx}/{len(codes)} {code}", flush=True)
        org_id = org_ids.get(code)
        if not org_id:
            print(f"announcement fetch skipped {code}: missing cninfo orgId", flush=True)
            continue
        items = []
        seen_items = set()
        for chunk_start, chunk_end in _date_chunks(pd.Timestamp(start), pd.Timestamp(end), chunk_months):
            try:
                chunk_items = _fetch_cninfo_announcements(code, org_id, chunk_start, chunk_end, timeout, retries)
            except Exception as exc:
                print(f"announcement fetch skipped {code} {chunk_start}-{chunk_end}: {exc}", flush=True)
                continue
            for item in chunk_items:
                key = (item.get("代码"), str(item.get("公告时间")), item.get("公告标题"))
                if key in seen_items:
                    continue
                seen_items.add(key)
                items.append(item)
        for row in items:
            published_at = pd.to_datetime(row.get("公告时间"), errors="coerce")
            if pd.isna(published_at):
                continue
            available_at = published_at
            feature_date = _to_feature_date(available_at, trade_dates)
            if feature_date is None:
                continue
            title = str(row.get("公告标题", ""))[:200]
            rows.append({
                "code": str(row.get("代码", code)).zfill(6),
                "feature_date": feature_date,
                "title": title,
                "ann_type": _classify_announcement(title),
                "source": "cninfo",
            })
    if not rows:
        return _empty_events(["code", "feature_date", "title", "ann_type", "source"])
    out = pd.DataFrame(rows)
    out["feature_date"] = pd.to_datetime(out["feature_date"]).dt.normalize()
    return out


def _base_frame(grid: pd.DataFrame, trade_dates: pd.Index) -> pd.DataFrame:
    index_map = {date: i for i, date in enumerate(trade_dates)}
    out = grid.copy()
    out["trade_idx"] = out["trade_date"].map(index_map).astype(int)
    return out


def _rolling_sum(frame: pd.DataFrame, col: str, window: int) -> pd.Series:
    return frame.groupby("code", sort=False)[col].transform(
        lambda values: values.rolling(window, min_periods=1).sum()
    )


def _rolling_max(frame: pd.DataFrame, col: str, window: int) -> pd.Series:
    return frame.groupby("code", sort=False)[col].transform(
        lambda values: values.rolling(window, min_periods=1).max()
    )


def _add_news_features(frame: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        for window in NEWS_WINDOWS:
            frame[f"news_score_{window}d"] = float("nan")
        frame["news_count_7d"] = 0
        frame["news_pos_count_7d"] = 0
        frame["news_neg_count_7d"] = 0
        frame["news_absmax_7d"] = float("nan")
        frame["news_latest_age_days"] = float("nan")
        frame["has_news_7d"] = 0
        return frame

    news = news.copy()
    news["is_scored"] = news["score"].notna().astype(int)
    news["score_sum"] = news["score"].fillna(0.0)
    news["pos"] = (news["score"] > 0.08).astype(int)
    news["neg"] = (news["score"] < -0.08).astype(int)
    daily = news.groupby(["code", "feature_date"]).agg(
        news_event_count=("title", "size"),
        news_scored_count=("is_scored", "sum"),
        news_score_sum=("score_sum", "sum"),
        news_pos_daily=("pos", "sum"),
        news_neg_daily=("neg", "sum"),
        news_absmax_daily=("score", lambda values: values.abs().max()),
    ).reset_index().rename(columns={"feature_date": "trade_date"})

    out = frame.merge(daily, on=["code", "trade_date"], how="left")
    fill_cols = ["news_event_count", "news_scored_count", "news_score_sum", "news_pos_daily", "news_neg_daily"]
    out[fill_cols] = out[fill_cols].fillna(0)
    for window in NEWS_WINDOWS:
        score_sum = _rolling_sum(out, "news_score_sum", window)
        scored_count = _rolling_sum(out, "news_scored_count", window)
        out[f"news_score_{window}d"] = (score_sum / scored_count).where(scored_count > 0)

    out["news_count_7d"] = _rolling_sum(out, "news_event_count", 7).astype(int)
    out["news_pos_count_7d"] = _rolling_sum(out, "news_pos_daily", 7).astype(int)
    out["news_neg_count_7d"] = _rolling_sum(out, "news_neg_daily", 7).astype(int)
    out["news_absmax_7d"] = _rolling_max(out, "news_absmax_daily", 7)
    out["has_news_7d"] = (out["news_count_7d"] > 0).astype(int)

    event_idx = out["trade_idx"].where(out["news_event_count"] > 0)
    last_idx = event_idx.groupby(out["code"], sort=False).ffill()
    age = out["trade_idx"] - last_idx
    out["news_latest_age_days"] = age.where(age <= 7)
    return out.drop(columns=[
        "news_event_count", "news_scored_count", "news_score_sum",
        "news_pos_daily", "news_neg_daily", "news_absmax_daily",
    ])


def _add_announcement_features(frame: pd.DataFrame, anns: pd.DataFrame) -> pd.DataFrame:
    if anns.empty:
        frame["ann_count_7d"] = 0
        frame["ann_count_20d"] = 0
        frame["ann_latest_age_days"] = float("nan")
        frame["has_ann_7d"] = 0
        for ann_type in [*ANN_TYPES.keys(), "other"]:
            frame[f"ann_{ann_type}_count_20d"] = 0
        return frame

    anns = anns.copy()
    type_dummies = pd.get_dummies(anns["ann_type"], prefix="ann")
    anns = pd.concat([anns, type_dummies], axis=1)
    agg = {"title": "size"}
    for col in type_dummies.columns:
        agg[col] = "sum"
    daily = anns.groupby(["code", "feature_date"]).agg(agg).reset_index()
    daily = daily.rename(columns={"feature_date": "trade_date", "title": "ann_event_count"})

    out = frame.merge(daily, on=["code", "trade_date"], how="left")
    ann_cols = ["ann_event_count", *type_dummies.columns]
    out[ann_cols] = out[ann_cols].fillna(0)
    out["ann_count_7d"] = _rolling_sum(out, "ann_event_count", 7).astype(int)
    out["ann_count_20d"] = _rolling_sum(out, "ann_event_count", 20).astype(int)
    out["has_ann_7d"] = (out["ann_count_7d"] > 0).astype(int)
    all_type_cols = [f"ann_{ann_type}" for ann_type in [*ANN_TYPES.keys(), "other"]]
    for ann_type in [*ANN_TYPES.keys(), "other"]:
        col = f"ann_{ann_type}"
        if col not in out.columns:
            out[col] = 0
        out[f"{col}_count_20d"] = _rolling_sum(out, col, 20).astype(int)

    event_idx = out["trade_idx"].where(out["ann_event_count"] > 0)
    last_idx = event_idx.groupby(out["code"], sort=False).ffill()
    age = out["trade_idx"] - last_idx
    out["ann_latest_age_days"] = age.where(age <= 20)
    return out.drop(columns=["ann_event_count", *all_type_cols])


def main() -> None:
    args = _parse_args()
    root = _root()
    grid, trade_dates, codes, output_start, output_end = _load_grid(root, args)
    frame = _base_frame(grid, trade_dates)

    if not args.no_news:
        news = _load_news_events(codes, trade_dates)
        frame = _add_news_features(frame, news)
        print(f"news events used: {len(news)}")
    if not args.no_announcements:
        anns = _load_announcement_events(
            codes,
            trade_dates,
            pd.to_datetime(grid["trade_date"]).min(),
            output_end,
            args.announcement_timeout,
            args.announcement_retries,
            args.announcement_chunk_months,
        )
        frame = _add_announcement_features(frame, anns)
        print(f"announcement events used: {len(anns)}")

    frame = frame[(frame["trade_date"] >= output_start) & (frame["trade_date"] <= output_end)]
    frame = frame.drop(columns=["trade_idx"]).sort_values(["trade_date", "code"])
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    output = Path(args.output).expanduser() if args.output else root / "sentiment_features.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    print(f"sentiment features saved: {output}, rows={len(frame)}, codes={len(codes)}")


if __name__ == "__main__":
    main()
