#!/usr/bin/env python3
"""Build point-in-time fundamental features from quarterly earnings reports.

Source: AKShare stock_yjbb_em(date=<report period>) — one row per stock per period
with ROE / gross margin / growth / EPS / operating cash flow per share AND the
announcement date (最新公告日期). We key each row by its announcement date so the
offline training-set merge can do a strict as-of join (a feature for trade date D
only uses reports announced BEFORE D — no look-ahead).

The output preserves the wider PIT information set (growth, profitability,
cash-flow quality and margins).  The legacy model may continue selecting only
`ocf_to_eps`; new baselines must choose features explicitly rather than silently
turning unavailable fields into zeros.

Output: ~/.stockwatch/history/fundamental_features.parquet
Date-only announcement timestamps are conservatively made available after the
close.  Because the upstream endpoint is a current snapshot, each row is hashed
and marked ``vintage_verified=False`` rather than presented as a true historical
vendor vintage.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REN = {
    "股票代码": "code", "净资产收益率": "roe", "销售毛利率": "gross_margin",
    "每股收益": "eps", "每股经营现金流量": "ocfps", "最新公告日期": "available_at",
    "营业总收入-营业总收入": "revenue",
    "营业总收入-同比增长": "revenue_yoy",
    "营业总收入-季度环比增长": "revenue_qoq",
    "净利润-净利润": "net_profit",
    "净利润-同比增长": "net_profit_yoy",
    "净利润-季度环比增长": "net_profit_qoq",
    "每股净资产": "bvps",
    "所处行业": "industry",
}

PIT_FUNDAMENTAL_FEATURES = [
    "ocf_to_eps", "roe", "gross_margin", "revenue_yoy", "revenue_qoq",
    "net_profit_yoy", "net_profit_qoq", "net_margin", "eps", "ocfps", "bvps",
]
EXTRACTION_VERSION = "pit_fundamental_v2"


def _quarter_ends(start_year: int) -> list[str]:
    out = []
    today = date.today()
    for year in range(start_year, today.year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            d = date(year, int(md[:2]), int(md[2:]))
            if d <= today:
                out.append(f"{year}{md}")
    return out


def _row_hash(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prepare_fundamental_frame(frame, report_period: str, fetched_at: str):
    import numpy as np
    import pandas as pd

    data = frame.rename(columns=REN).copy()
    if "code" not in data or "available_at" not in data:
        raise ValueError("fundamental source is missing code or available_at")
    data["source_row_sha256"] = [
        _row_hash(row) for row in frame.to_dict("records")
    ]
    data["code"] = data["code"].astype(str).str.zfill(6)
    announced = pd.to_datetime(data["available_at"], errors="coerce")
    midnight = announced.notna() & announced.dt.time.eq(datetime.min.time())
    data["available_at"] = announced.where(
        ~midnight,
        announced.dt.normalize() + pd.Timedelta(hours=15, seconds=1),
    )
    numeric = [
        "roe", "gross_margin", "eps", "ocfps", "revenue", "revenue_yoy",
        "revenue_qoq", "net_profit", "net_profit_yoy", "net_profit_qoq", "bvps",
    ]
    for column in numeric:
        if column not in data:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["ocf_to_eps"] = data["ocfps"] / data["eps"].replace(0, np.nan)
    data["ocf_to_eps"] = data["ocf_to_eps"].clip(-10, 10)
    data["net_margin"] = data["net_profit"] / data["revenue"].replace(0, np.nan)
    data["report_period"] = str(report_period)
    data["snapshot_fetched_at"] = fetched_at
    data["extraction_version"] = EXTRACTION_VERSION
    data["vintage_verified"] = False
    data["vintage_note"] = (
        "AKShare/Eastmoney current snapshot; original historical vendor vintage not verified"
    )
    if "industry" not in data:
        data["industry"] = None
    keep = [
        "code", "available_at", "report_period", *PIT_FUNDAMENTAL_FEATURES,
        "revenue", "net_profit", "industry", "source_row_sha256",
        "snapshot_fetched_at", "extraction_version", "vintage_verified", "vintage_note",
    ]
    return data[keep].dropna(subset=["available_at"])


def main():
    import akshare as ak
    import pandas as pd

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    start_year = int(os.getenv("STOCKWATCH_FUNDAMENTAL_START_YEAR", "2021"))
    periods = _quarter_ends(start_year)
    fetched_at = datetime.now().astimezone().isoformat(timespec="seconds")

    frames = []
    for p in periods:
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_yjbb_em(date=p)
                break
            except Exception as e:
                print(f"{p} retry {attempt}: {str(e)[:80]}")
                time.sleep(2)
        if df is None or df.empty:
            print(f"{p}: 无数据，跳过")
            continue
        try:
            prepared = _prepare_fundamental_frame(df, p, fetched_at)
        except ValueError:
            print(f"{p}: 字段缺失，跳过 ({list(df.columns)[:6]})")
            continue
        frames.append(prepared)
        print(f"{p}: {len(prepared)} 行")

    if not frames:
        raise RuntimeError("未取到任何业绩报表数据")
    data = pd.concat(frames, ignore_index=True)
    out = data.sort_values(["available_at", "code", "report_period"])
    out_path = root / "fundamental_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"fundamental features saved: {out_path}, rows={len(out)}, "
          f"codes={out['code'].nunique()}, "
          f"{out['available_at'].min().date()}..{out['available_at'].max().date()}")


if __name__ == "__main__":
    main()
