#!/usr/bin/env python3
"""Download HS300 history for offline LightGBM training."""
from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market import MarketData


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _columns(df):
    return {
        "trade_date": "日期",
        "open": "开盘",
        "high": "最高",
        "low": "最低",
        "close": "收盘",
        "volume": "成交量",
        "amount": "成交额",
    }


def main():
    import akshare as ak
    import pandas as pd

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    stock_dir = root / "stocks"
    stock_dir.mkdir(parents=True, exist_ok=True)
    start = os.getenv("STOCKWATCH_HISTORY_START", (date.today() - timedelta(days=365 * 5 + 30)).strftime("%Y%m%d"))
    end = os.getenv("STOCKWATCH_HISTORY_END", date.today().strftime("%Y%m%d"))

    cons = ak.index_stock_cons(symbol="000300")
    code_col = next((col for col in cons.columns if "代码" in str(col)), "")
    if not code_col:
        raise RuntimeError(f"沪深300成分列识别失败: {list(cons.columns)}")
    codes = [_normalize_code(code) for code in cons[code_col].tolist()]
    codes = [code for code in codes if code]
    print(f"download HS300 stocks: {len(codes)}, {start}..{end}")

    for code in tqdm(codes, desc="stocks"):
        path = stock_dir / f"{code}.parquet"
        if path.exists():
            continue
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="qfq")
            mapping = _columns(df)
            out = pd.DataFrame({
                key: df[value] if value in df.columns else 0
                for key, value in mapping.items()
            })
            out.to_parquet(path, index=False)
        except Exception as e:
            print(f"download failed {code}: {e}")

    market = MarketData().get_index_kline("sh000300", limit=1500)
    if market:
        pd.DataFrame(market).to_parquet(root / "market_sh000300.parquet", index=False)
    print(f"history saved to {root}")


if __name__ == "__main__":
    main()
