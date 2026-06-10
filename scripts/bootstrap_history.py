#!/usr/bin/env python3
"""Download index-constituent history for offline LightGBM training."""
from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market import MarketData

DEFAULT_INDEX_SYMBOLS = ("000300", "000905")  # HS300 + CSI500
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


def _env_list(name: str, default: tuple[str, ...] = ()) -> list[str]:
    raw = os.getenv(name, ",".join(default))
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _load_existing_failures(root: Path) -> list[str]:
    path = root / "failed_codes.txt"
    if not path.exists():
        return []
    return [_normalize_code(line) for line in path.read_text().splitlines() if _normalize_code(line)]


def _read_last_date(pd, path: Path) -> str:
    try:
        df = pd.read_parquet(path, columns=["trade_date"])
        if df.empty:
            return ""
        return str(df["trade_date"].max()).replace("-", "")
    except Exception:
        return ""


def _filter_dates(df, start: str, end: str):
    trade_date = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    return df[(trade_date >= start) & (trade_date <= end)].copy()


def _download_tencent(pd, code, start, end):
    records = MarketData.get_daily_kline(code, limit=800)
    if not records:
        return None
    df = pd.DataFrame(records)
    df = _filter_dates(df, start, end)
    if len(df) < 200:
        return None
    return df


def _download_one(ak, pd, code, start, end, mapping_fn, max_retries=1):
    """下载单只股票历史行情，带重试和基础行数校验。"""
    last_error = None
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
            }), "akshare"
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    fallback = _download_tencent(pd, code, start, end)
    if fallback is not None:
        return fallback, "tencent"
    print(f"download failed {code}: {last_error}")
    return None, ""


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
    index_symbols = _env_list("STOCKWATCH_INDEX_SYMBOLS", DEFAULT_INDEX_SYMBOLS)
    extra_codes = _env_list("STOCKWATCH_EXTRA_CODES")
    retry_failed = _env_bool("STOCKWATCH_RETRY_FAILED", True)
    refresh_existing = _env_bool("STOCKWATCH_REFRESH_EXISTING", False)
    workers = max(1, _env_int("STOCKWATCH_DOWNLOAD_WORKERS", 4))

    index_counts = {}
    codes = []
    for symbol in index_symbols:
        cons = ak.index_stock_cons(symbol=symbol)
        code_col = next((col for col in cons.columns if "代码" in str(col)), "")
        if not code_col:
            raise RuntimeError(f"{symbol} 成分列识别失败: {list(cons.columns)}")
        symbol_codes = [_normalize_code(code) for code in cons[code_col].tolist()]
        symbol_codes = [code for code in symbol_codes if code]
        index_counts[symbol] = {
            "rows": len(symbol_codes),
            "unique": len(set(symbol_codes)),
        }
        codes.extend(symbol_codes)

    if retry_failed:
        extra_codes.extend(_load_existing_failures(root))
    codes.extend(extra_codes)
    codes = sorted(dict.fromkeys(_normalize_code(code) for code in codes if _normalize_code(code)))
    print(f"download index stocks: {len(codes)} unique, indices={index_symbols}, {start}..{end}")
    print(f"index counts: {index_counts}")

    def download_task(code: str) -> tuple[str, str]:
        path = stock_dir / f"{code}.parquet"
        if path.exists() and not refresh_existing:
            last_date = _read_last_date(pd, path)
            if last_date and last_date >= end:
                return code, "skipped"
            return code, "skipped"
        out, source = _download_one(ak, pd, code, start, end, _columns)
        if out is not None:
            out.to_parquet(path, index=False)
            return code, f"success_{source}"
        return code, "failed"

    success, skipped, failed = 0, 0, 0
    source_counts = {"akshare": 0, "tencent": 0}
    failed_codes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_task, code) for code in codes]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"stocks x{workers}"):
            code, status = future.result()
            if status.startswith("success_"):
                success += 1
                source = status.removeprefix("success_")
                source_counts[source] = source_counts.get(source, 0) + 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                failed_codes.append(code)

    if failed_codes:
        (root / "failed_codes.txt").write_text("\n".join(failed_codes))
    elif (root / "failed_codes.txt").exists():
        (root / "failed_codes.txt").unlink()

    print(f"下载完成：成功 {success} / 跳过 {skipped} / 失败 {failed} / 总 {len(codes)}")
    if failed_codes:
        print(f"失败代码已写入 {root / 'failed_codes.txt'}，可重跑脚本补下载")

    market = MarketData().get_index_kline("sh000300", limit=1500)
    if market:
        pd.DataFrame(market).to_parquet(root / "market_sh000300.parquet", index=False)
    manifest = {
        "index_symbols": index_symbols,
        "index_counts": index_counts,
        "unique_codes": len(codes),
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "source_counts": source_counts,
        "start": start,
        "end": end,
        "refresh_existing": refresh_existing,
    }
    (root / "history_manifest.json").write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2))
    print(f"history saved to {root}")


if __name__ == "__main__":
    main()
