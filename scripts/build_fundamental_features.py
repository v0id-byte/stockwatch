#!/usr/bin/env python3
"""Build point-in-time fundamental features from quarterly earnings reports.

Source: AKShare stock_yjbb_em(date=<report period>) — one row per stock per period
with ROE / gross margin / growth / EPS / operating cash flow per share AND the
announcement date (最新公告日期). We key each row by its announcement date so the
offline training-set merge can do a strict as-of join (a feature for trade date D
only uses reports announced BEFORE D — no look-ahead).

Primary output factor is earnings quality `ocf_to_eps` (operating cash flow per
share / EPS): it is the one fundamental that showed a small but stable positive IC
on A-shares in risk-off regimes, used as a diversifying input to the bear model.

Output: ~/.stockwatch/history/fundamental_features.parquet
Columns: code, available_at, report_period, ocf_to_eps, roe, gross_margin
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REN = {
    "股票代码": "code", "净资产收益率": "roe", "销售毛利率": "gross_margin",
    "每股收益": "eps", "每股经营现金流量": "ocfps", "最新公告日期": "available_at",
}


def _quarter_ends(start_year: int) -> list[str]:
    out = []
    today = date.today()
    for year in range(start_year, today.year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            d = date(year, int(md[:2]), int(md[2:]))
            if d <= today:
                out.append(f"{year}{md}")
    return out


def main():
    import akshare as ak
    import numpy as np
    import pandas as pd

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    start_year = int(os.getenv("STOCKWATCH_FUNDAMENTAL_START_YEAR", "2021"))
    periods = _quarter_ends(start_year)

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
        df = df.rename(columns=REN)
        if "code" not in df.columns or "available_at" not in df.columns:
            print(f"{p}: 字段缺失，跳过 ({list(df.columns)[:6]})")
            continue
        df["report_period"] = p
        keep = ["code", "available_at", "report_period", "roe", "gross_margin", "eps", "ocfps"]
        frames.append(df[[c for c in keep if c in df.columns]])
        print(f"{p}: {len(df)} 行")

    if not frames:
        raise RuntimeError("未取到任何业绩报表数据")
    data = pd.concat(frames, ignore_index=True)
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["available_at"] = pd.to_datetime(data["available_at"], errors="coerce")
    for c in ["roe", "gross_margin", "eps", "ocfps"]:
        data[c] = pd.to_numeric(data.get(c), errors="coerce")
    data["ocf_to_eps"] = data["ocfps"] / data["eps"].replace(0, np.nan)
    data = data.dropna(subset=["available_at"]).sort_values("available_at")
    # clip the earnings-quality ratio to a sane band (one-off items create huge outliers)
    data["ocf_to_eps"] = data["ocf_to_eps"].clip(-10, 10)
    out = data[["code", "available_at", "report_period", "ocf_to_eps", "roe", "gross_margin"]]
    out_path = root / "fundamental_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"fundamental features saved: {out_path}, rows={len(out)}, "
          f"codes={out['code'].nunique()}, "
          f"{out['available_at'].min().date()}..{out['available_at'].max().date()}")


if __name__ == "__main__":
    main()
