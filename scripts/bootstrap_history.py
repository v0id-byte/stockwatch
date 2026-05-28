#!/usr/bin/env python3
"""Download HS300 history for offline LightGBM training."""
from __future__ import annotations

import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market import MarketData

PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


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


def _download_one(ak, pd, code, start, end, mapping_fn, max_retries=3):
    """下载单只股票历史行情，带重试和基础行数校验。"""
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
            if df is None or len(df) < 200:
                raise ValueError(f"行数异常({0 if df is None else len(df)})")
            mapping = mapping_fn(df)
            return pd.DataFrame({
                key: df[value] if value in df.columns else 0
                for key, value in mapping.items()
            })
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"download failed {code}: {e}")
                return None
    return None


def _disable_proxy_env():
    removed = [key for key in PROXY_ENV_KEYS if os.environ.pop(key, None)]
    if removed:
        print(f"已忽略代理环境变量: {', '.join(removed)}")


def main():
    _disable_proxy_env()

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

    success, skipped, failed = 0, 0, 0
    failed_codes = []
    for code in tqdm(codes, desc="stocks"):
        path = stock_dir / f"{code}.parquet"
        if path.exists():
            skipped += 1
            continue
        out = _download_one(ak, pd, code, start, end, _columns)
        if out is not None:
            out.to_parquet(path, index=False)
            success += 1
        else:
            failed += 1
            failed_codes.append(code)
        time.sleep(0.4)

    if failed_codes:
        (root / "failed_codes.txt").write_text("\n".join(failed_codes))

    print(f"下载完成：成功 {success} / 跳过 {skipped} / 失败 {failed} / 总 {len(codes)}")
    if failed_codes:
        print(f"失败代码已写入 {root / 'failed_codes.txt'}，可重跑脚本补下载")

    market = MarketData().get_index_kline("sh000300", limit=1500)
    if market:
        pd.DataFrame(market).to_parquet(root / "market_sh000300.parquet", index=False)
    print(f"history saved to {root}")


if __name__ == "__main__":
    main()
