#!/usr/bin/env python3
"""Download auditable raw-price history for offline model research."""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# HS300 + CSI500 + CSI1000 (~1800 investable names). Broader training breadth
# modestly improves cross-sectional IC. For max breadth add CSI2000 (932000) via
# STOCKWATCH_INDEX_SYMBOLS=000300,000905,000852,932000 — but the extra micro-caps
# mostly inflate backtested return via illiquidity, they do not cut drawdown.
DEFAULT_INDEX_SYMBOLS = ("000300", "000905", "000852")
BENCHMARK_INDEX_CODES = ("sh000300", "sh000905")
BENCHMARK_AKSHARE_SYMBOLS = {
    "sh000300": "sh000300",
    "sh000905": "csi000905",
}
EXPECTED_INDEX_MEMBER_COUNTS = {
    "000300": 300,
    "000905": 500,
    "000852": 1000,
    "932000": 2000,
}
PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)
SINA_DOWNLOAD_LOCK = Lock()
MARKET_DATA_SCHEMA_VERSION = 2
HISTORY_REQUIRED_COLUMNS = {
    "trade_date", "open", "high", "low", "close",
    "raw_open", "raw_high", "raw_low", "raw_close",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor",
    "volume", "volume_shares", "volume_lots", "amount", "turnover", "vwap", "adj_vwap", "amihud_1d",
    "data_source", "market_data_schema_version", "return_adjustment",
}
PIT_UNIVERSE_REQUIRED_COLUMNS = {
    "trade_date", "code", "index_code", "is_member", "is_listed", "is_st",
    "is_suspended", "is_limit_up", "is_limit_down",
}


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _normalize_history(pd, raw, adjusted, source: str):
    """Normalize source-specific units without mixing price conventions.

    ``stock_zh_a_hist`` reports volume in lots and turnover in percent, while
    ``stock_zh_a_daily`` reports volume in shares and turnover as a decimal.
    Raw OHLC is retained for price/capitalization calculations.  HFQ OHLC is a
    separate return basis; it must never be used as the observed market price.
    """
    if source == "eastmoney":
        raw_columns = {
            "日期": "trade_date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume_lots",
            "成交额": "amount", "换手率": "turnover_pct",
        }
        adjusted_columns = {
            "日期": "trade_date", "开盘": "adj_open", "最高": "adj_high",
            "最低": "adj_low", "收盘": "adj_close",
        }
        missing = set(raw_columns) - set(raw.columns)
        adjusted_missing = set(adjusted_columns) - set(adjusted.columns)
        if missing or adjusted_missing:
            raise ValueError(
                f"EastMoney history schema mismatch: raw_missing={sorted(missing)}, "
                f"hfq_missing={sorted(adjusted_missing)}"
            )
        out = raw[list(raw_columns)].rename(columns=raw_columns).copy()
        adj = adjusted[list(adjusted_columns)].rename(columns=adjusted_columns).copy()
        out["volume_lots"] = pd.to_numeric(out["volume_lots"], errors="coerce")
        out["volume"] = out["volume_lots"] * 100.0
        out["turnover_pct"] = pd.to_numeric(out["turnover_pct"], errors="coerce")
        out["turnover"] = out["turnover_pct"] / 100.0
    elif source == "baostock":
        # Baostock is the only free source that keeps delisted stocks' full
        # history; volume is in shares and turn is a percentage.
        raw_columns = {
            "date": "trade_date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
            "amount": "amount", "turn": "turnover_pct",
        }
        adjusted_columns = {
            "date": "trade_date", "open": "adj_open", "high": "adj_high",
            "low": "adj_low", "close": "adj_close",
        }
        missing = set(raw_columns) - set(raw.columns)
        adjusted_missing = set(adjusted_columns) - set(adjusted.columns)
        if missing or adjusted_missing:
            raise ValueError(
                f"Baostock history schema mismatch: raw_missing={sorted(missing)}, "
                f"hfq_missing={sorted(adjusted_missing)}"
            )
        out = raw[list(raw_columns)].rename(columns=raw_columns).copy()
        adj = adjusted[list(adjusted_columns)].rename(columns=adjusted_columns).copy()
        for column in ("open", "high", "low", "close", "volume", "amount", "turnover_pct"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
        # Baostock emits placeholder rows on suspension days (zero/empty price
        # or volume).  Other sources emit no bar at all; align on that so the
        # PIT layer sees a gap and resolves it through suspension evidence.
        out = out[(out["close"] > 0) & (out["volume"] > 0) & (out["amount"] > 0)]
        out["turnover_pct"] = out["turnover_pct"].fillna(0.0)
        out["turnover"] = out["turnover_pct"] / 100.0
        out["volume_lots"] = out["volume"] / 100.0
        for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
            adj[column] = pd.to_numeric(adj[column], errors="coerce")
        adj = adj[adj["adj_close"] > 0]
    elif source == "sina":
        raw_columns = {
            "date": "trade_date", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume",
            "amount": "amount", "outstanding_share": "float_a_share",
            "turnover": "turnover",
        }
        adjusted_columns = {
            "date": "trade_date", "open": "adj_open", "high": "adj_high",
            "low": "adj_low", "close": "adj_close",
        }
        missing = set(raw_columns) - set(raw.columns)
        adjusted_missing = set(adjusted_columns) - set(adjusted.columns)
        if missing or adjusted_missing:
            raise ValueError(
                f"Sina history schema mismatch: raw_missing={sorted(missing)}, "
                f"hfq_missing={sorted(adjusted_missing)}"
            )
        out = raw[list(raw_columns)].rename(columns=raw_columns).copy()
        adj = adjusted[list(adjusted_columns)].rename(columns=adjusted_columns).copy()
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
        out["volume_lots"] = out["volume"] / 100.0
        out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
        out["turnover_pct"] = out["turnover"] * 100.0
    else:
        raise ValueError(f"unsupported history source: {source}")

    numeric = [
        "open", "high", "low", "close", "volume", "volume_lots", "amount",
        "turnover", "turnover_pct",
    ]
    if "float_a_share" in out.columns:
        numeric.append("float_a_share")
    for column in numeric:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
        adj[column] = pd.to_numeric(adj[column], errors="coerce")

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    adj["trade_date"] = pd.to_datetime(adj["trade_date"], errors="coerce")
    out = out.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    adj = adj.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last")
    out = out.merge(adj, on="trade_date", how="left", validate="one_to_one")
    out = out.sort_values("trade_date").reset_index(drop=True)
    # Some sources (Sina, Baostock) emit zero-volume placeholder rows on
    # suspension days.  A bar means actual trading; gaps are resolved by the
    # PIT layer's suspension evidence, uniformly across sources.
    out = out[(out["volume"] > 0) & (out["amount"] > 0)].reset_index(drop=True)
    if out[["adj_open", "adj_high", "adj_low", "adj_close"]].isna().any().any():
        raise ValueError("HFQ history does not cover every raw-price trade date")
    price_columns = [
        "open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close",
    ]
    if out[price_columns].le(0).any().any():
        raise ValueError("raw/HFQ OHLC contains zero or negative prices")
    if out[["volume", "amount", "turnover"]].lt(0).any().any():
        raise ValueError("volume/amount/turnover contains negative values")

    valid_price = out["close"] > 0
    valid_volume = out["volume"] > 0
    valid_amount = out["amount"] > 0
    out["adj_factor"] = (out["adj_close"] / out["close"]).where(valid_price)
    out["vwap"] = (out["amount"] / out["volume"]).where(valid_volume & valid_amount)
    out["adj_vwap"] = out["vwap"] * out["adj_factor"]
    out["amihud_1d"] = out["adj_close"].pct_change().abs().div(out["amount"]).where(valid_amount)
    if "float_a_share" not in out.columns:
        out["float_a_share"] = (out["volume"] / out["turnover"]).where(out["turnover"] > 0)
    out["float_market_cap"] = out["close"] * out["float_a_share"]
    for column in ("open", "high", "low", "close"):
        out[f"raw_{column}"] = out[column]
    out["volume_shares"] = out["volume"]
    out["data_source"] = source
    out["market_data_schema_version"] = MARKET_DATA_SCHEMA_VERSION
    out["return_adjustment"] = "hfq"
    out["trade_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    return out


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


def _history_file_is_current(pd, path: Path, start: str, end: str) -> bool:
    """Old qfq-only parquet files must not be silently reused as schema v2."""
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    if frame.empty or not HISTORY_REQUIRED_COLUMNS.issubset(frame.columns):
        return False
    schema_values = pd.to_numeric(
        frame["market_data_schema_version"], errors="coerce"
    ).dropna().unique()
    if len(schema_values) != 1 or int(schema_values[0]) != MARKET_DATA_SCHEMA_VERSION:
        return False
    dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        return False
    if not frame["return_adjustment"].eq("hfq").all():
        return False
    trade_date = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
    if not len(trade_date) or trade_date.max() < end:
        return False
    # Fail safe when the requested research horizon is extended. A newly listed
    # stock may be fetched again on a later run, which is preferable to silently
    # claiming a 20-year manifest over a 5-year file.
    return bool(trade_date.min() <= start)


def _filter_dates(df, start: str, end: str):
    trade_date = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    return df[(trade_date >= start) & (trade_date <= end)].copy()


def _download_sina(ak, pd, code, start, end):
    """Download paired raw/HFQ Sina history when EastMoney is unavailable."""
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    try:
        # AKShare's Sina path initializes a MiniRacer runtime that is unsafe to
        # initialize concurrently in the same process.
        with SINA_DOWNLOAD_LOCK:
            raw = ak.stock_zh_a_daily(
                symbol=f"{prefix}{code}",
                start_date=start,
                end_date=end,
                adjust="",
            )
            adjusted = ak.stock_zh_a_daily(
                symbol=f"{prefix}{code}",
                start_date=start,
                end_date=end,
                adjust="hfq",
            )
    except Exception:
        return None
    if raw is None or raw.empty or adjusted is None or adjusted.empty:
        return None
    try:
        out = _normalize_history(pd, raw, adjusted, "sina")
    except (KeyError, ValueError):
        return None
    out = _filter_dates(out, start, end)
    return None if out.empty else out


BAOSTOCK_LOCK = Lock()
_BAOSTOCK_STATE = {"ready": False}


def _download_baostock(pd, code, start, end):
    """Last-resort source: the only free feed that serves delisted stocks."""
    try:
        import baostock as bs
    except ImportError:
        return None
    prefix = "sh" if code.startswith(("5", "6", "9")) else "sz"
    symbol = f"{prefix}.{code}"
    start_iso = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_iso = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    fields = "date,open,high,low,close,volume,amount,turn"

    with BAOSTOCK_LOCK:
        if not _BAOSTOCK_STATE["ready"]:
            login = bs.login()
            if getattr(login, "error_code", "1") != "0":
                return None
            _BAOSTOCK_STATE["ready"] = True

        def _query(adjust_flag):
            rs = bs.query_history_k_data_plus(
                symbol, fields, start_date=start_iso, end_date=end_iso,
                frequency="d", adjustflag=adjust_flag,
            )
            if getattr(rs, "error_code", "1") != "0":
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            return pd.DataFrame(rows, columns=rs.fields) if rows else None

        raw = _query("3")
        adjusted = _query("1")
    if raw is None or raw.empty or adjusted is None or adjusted.empty:
        return None
    try:
        out = _normalize_history(pd, raw, adjusted, "baostock")
    except (KeyError, ValueError):
        return None
    out = _filter_dates(out, start, end)
    return None if out.empty else out


def _download_one(ak, pd, code, start, end, mapping_fn=None, max_retries=1):
    """下载单只股票历史行情，带重试和基础行数校验。"""
    last_error = None
    for attempt in range(max_retries):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="",
            )
            adjusted = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="hfq",
            )
            if raw is None or raw.empty or adjusted is None or adjusted.empty:
                raise ValueError("行情为空")
            return _normalize_history(pd, raw, adjusted, "eastmoney"), "akshare"
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    fallback = _download_sina(ak, pd, code, start, end)
    if fallback is not None:
        return fallback, "sina"
    fallback = _download_baostock(pd, code, start, end)
    if fallback is not None:
        return fallback, "baostock"
    # Tencent's endpoint used here exposes qfq OHLC without turnover/amount.
    # Accepting it would silently reintroduce mixed units, so history fails
    # closed after the paired raw/HFQ sources are exhausted.
    print(f"download failed {code}: {last_error}")
    return None, ""


def _disable_proxy_env():
    removed = [key for key in PROXY_ENV_KEYS if os.environ.pop(key, None)]
    if removed:
        print(f"已忽略代理环境变量: {', '.join(removed)}")


def _index_constituents(ak, symbol: str) -> tuple[list[str], str]:
    """Prefer the official CSI constituent directory over Sina's lossy table."""
    try:
        frame = ak.index_stock_cons_csindex(symbol=symbol)
        source = "csindex"
    except Exception as exc:
        print(f"中证成分接口失败 {symbol}，回退新浪: {exc}")
        frame = ak.index_stock_cons(symbol=symbol)
        source = "sina_fallback"
    code_col = next((col for col in frame.columns if "成分券代码" in str(col)), "")
    if not code_col:
        code_col = next((col for col in frame.columns if "代码" in str(col)), "")
    if not code_col:
        raise RuntimeError(f"{symbol} 成分列识别失败: {list(frame.columns)}")
    codes = [_normalize_code(code) for code in frame[code_col].tolist()]
    return [code for code in codes if code], source


def _pit_universe_scope(
    pd,
    path: Path,
    index_symbols: list[str],
    expected_counts: dict[str, int] | None = None,
) -> tuple[list[str], dict]:
    """Load the code download scope from an externally captured PIT universe."""
    frame = pd.read_parquet(path)
    missing = PIT_UNIVERSE_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"PIT universe missing required columns: {sorted(missing)}")
    frame = frame.copy()
    status_columns = sorted(PIT_UNIVERSE_REQUIRED_COLUMNS - {"trade_date", "code", "index_code"})
    for column in status_columns:
        if not pd.api.types.is_bool_dtype(frame[column].dtype) or frame[column].isna().any():
            raise RuntimeError(f"PIT universe status column must be non-null boolean: {column}")
    frame["code"] = frame["code"].astype(str).map(_normalize_code)
    frame["index_code"] = frame["index_code"].astype(str).map(_normalize_code)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    if frame["trade_date"].isna().any() or frame["code"].eq("").any() or frame["index_code"].eq("").any():
        raise RuntimeError("PIT universe contains invalid date, code, or index_code values")
    duplicates = frame.duplicated(["trade_date", "code", "index_code"]).sum()
    if duplicates:
        raise RuntimeError(f"PIT universe has {int(duplicates)} duplicate date/code/index rows")
    stock_status_columns = [column for column in status_columns if column != "is_member"]
    conflicts = frame.groupby(["trade_date", "code"])[stock_status_columns].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise RuntimeError(
            f"PIT universe has {int(conflicts.sum())} conflicting date/code stock-status rows"
        )
    frame = frame[(frame["code"] != "") & frame["index_code"].isin(index_symbols)]
    frame = frame[frame["is_member"].eq(True) & frame["is_listed"].eq(True)]
    if frame.empty:
        raise RuntimeError("PIT universe has no listed member rows for requested indices")
    expected_counts = EXPECTED_INDEX_MEMBER_COUNTS if expected_counts is None else expected_counts
    count_checks = {}
    for symbol in index_symbols:
        expected = expected_counts.get(symbol)
        if expected is None:
            continue
        daily_counts = frame.loc[frame["index_code"].eq(symbol)].groupby("trade_date")["code"].nunique()
        if daily_counts.empty:
            raise RuntimeError(f"PIT universe has no rows for requested index {symbol}")
        bad = daily_counts[daily_counts.ne(expected)]
        if not bad.empty:
            sample = {str(day.date()): int(value) for day, value in bad.head(5).items()}
            raise RuntimeError(
                f"PIT universe member-count gate failed for {symbol}: expected {expected}, sample={sample}"
            )
        count_checks[symbol] = {
            "expected": expected,
            "dates": int(len(daily_counts)),
            "min": int(daily_counts.min()),
            "max": int(daily_counts.max()),
        }
    codes = sorted(frame["code"].unique().tolist())
    meta = {
        "path": str(path),
        "rows": int(len(frame)),
        "codes": len(codes),
        "date_start": str(frame["trade_date"].min().date()),
        "date_end": str(frame["trade_date"].max().date()),
        "required_status_columns": sorted(PIT_UNIVERSE_REQUIRED_COLUMNS),
        "member_count_checks": count_checks,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "PASS",
        "training_gate": "PASS",
    }
    return codes, meta


def _download_index_baostock(pd, index_code: str, start: str, end: str):
    """Benchmark fallback when the EastMoney index endpoint throttles this IP."""
    try:
        import baostock as bs
    except ImportError:
        return None
    symbol = f"{index_code[:2]}.{index_code[2:]}"
    with BAOSTOCK_LOCK:
        if not _BAOSTOCK_STATE["ready"]:
            login = bs.login()
            if getattr(login, "error_code", "1") != "0":
                return None
            _BAOSTOCK_STATE["ready"] = True
        rs = bs.query_history_k_data_plus(
            symbol, "date,open,high,low,close,volume,amount",
            start_date=f"{start[:4]}-{start[4:6]}-{start[6:]}",
            end_date=f"{end[:4]}-{end[4:6]}-{end[6:]}",
            frequency="d",
        )
        if getattr(rs, "error_code", "1") != "0":
            return None
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=rs.fields)
    for column in ("open", "high", "low", "close", "volume", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.rename(columns={"date": "trade_date"})


def _save_benchmarks(ak, pd, root: Path, start: str, end: str) -> dict[str, int]:
    counts = {}
    for index_code in BENCHMARK_INDEX_CODES:
        symbol = BENCHMARK_AKSHARE_SYMBOLS[index_code]
        try:
            raw = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)
        except Exception as exc:
            print(f"基准指数 EastMoney 失败 {index_code}，回退 baostock: {exc}")
            raw = _download_index_baostock(pd, index_code, start, end)
        if raw is None or raw.empty:
            raise RuntimeError(f"基准指数下载失败: {index_code}")
        frame = raw.rename(columns={
            "日期": "trade_date", "date": "trade_date",
            "开盘": "open", "open": "open",
            "收盘": "close", "close": "close",
            "最高": "high", "high": "high",
            "最低": "low", "low": "low",
            "成交量": "volume", "volume": "volume",
            "成交额": "amount", "amount": "amount",
        }).copy()
        required = {"trade_date", "open", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"基准指数 {index_code} schema mismatch: {sorted(missing)}")
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.dropna(subset=["trade_date"]).sort_values("trade_date")
        frame = frame.drop_duplicates("trade_date", keep="last")
        requested_start = pd.Timestamp(start)
        requested_end = pd.Timestamp(end)
        if (
            frame.empty
            or frame["trade_date"].min() > requested_start + pd.Timedelta(days=7)
            or frame["trade_date"].max() < requested_end - pd.Timedelta(days=7)
        ):
            raise RuntimeError(
                f"基准指数 {index_code} 未覆盖请求区间 {start}..{end}: "
                f"{frame['trade_date'].min()}..{frame['trade_date'].max()}"
            )
        frame["raw_open"] = frame["open"]
        frame["raw_close"] = frame["close"]
        frame.to_parquet(root / f"market_{index_code}.parquet", index=False)
        counts[index_code] = len(frame)
    return counts


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
    if _env_bool("STOCKWATCH_BENCHMARK_ONLY", False):
        counts = _save_benchmarks(ak, pd, root, start, end)
        print(f"benchmark history saved to {root}: {counts}")
        return
    index_symbols = _env_list("STOCKWATCH_INDEX_SYMBOLS", DEFAULT_INDEX_SYMBOLS)
    extra_codes = _env_list("STOCKWATCH_EXTRA_CODES")
    retry_failed = _env_bool("STOCKWATCH_RETRY_FAILED", True)
    refresh_existing = _env_bool("STOCKWATCH_REFRESH_EXISTING", False)
    workers = max(1, _env_int("STOCKWATCH_DOWNLOAD_WORKERS", 4))

    index_counts = {}
    pit_path = root / "pit_universe_daily.parquet"
    pit_meta = None
    if pit_path.exists():
        codes, pit_meta = _pit_universe_scope(pd, pit_path, index_symbols)
        constituent_membership_kind = "point_in_time_daily"
        for symbol in index_symbols:
            index_counts[symbol] = {"source": "pit_universe_daily.parquet"}
    else:
        codes = []
        constituent_membership_kind = "current_snapshot_not_point_in_time"
        print(
            "WARNING: pit_universe_daily.parquet is missing; current constituents are "
            "download scope only and build_training_set.py will fail closed."
        )
        for symbol in index_symbols:
            symbol_codes, constituent_source = _index_constituents(ak, symbol)
            index_counts[symbol] = {
                "rows": len(symbol_codes),
                "unique": len(set(symbol_codes)),
                "source": constituent_source,
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
            if _history_file_is_current(pd, path, start, end):
                return code, "skipped"
        out, source = _download_one(ak, pd, code, start, end)
        if out is not None:
            out.to_parquet(path, index=False)
            return code, f"success_{source}"
        return code, "failed"

    success, skipped, failed = 0, 0, 0
    completed_codes = []
    source_counts = {"akshare": 0, "sina": 0}
    failed_codes = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_task, code) for code in codes]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"stocks x{workers}"):
            code, status = future.result()
            if status.startswith("success_"):
                success += 1
                completed_codes.append(code)
                source = status.removeprefix("success_")
                source_counts[source] = source_counts.get(source, 0) + 1
            elif status == "skipped":
                skipped += 1
                completed_codes.append(code)
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

    benchmark_counts = _save_benchmarks(ak, pd, root, start, end)
    manifest = {
        "index_symbols": index_symbols,
        "index_counts": index_counts,
        "unique_codes": len(codes),
        "requested_codes": codes,
        "codes": sorted(completed_codes),
        "excluded_failed_codes": sorted(failed_codes),
        "constituents_fetched_at": datetime.now().isoformat(timespec="seconds"),
        "constituent_membership_kind": constituent_membership_kind,
        "pit_universe": pit_meta or {
            "path": str(pit_path),
            "status": "MISSING",
            "training_gate": "FAIL",
            "note": "Current constituents are download scope only and are not valid historical membership.",
        },
        "success": success,
        "skipped": skipped,
        "failed": failed,
        "source_counts": source_counts,
        "start": start,
        "end": end,
        "refresh_existing": refresh_existing,
        "benchmark_counts": benchmark_counts,
        "market_data_schema": {
            "version": MARKET_DATA_SCHEMA_VERSION,
            "raw_ohlc": "raw_open/raw_high/raw_low/raw_close (plus compatibility aliases open/high/low/close); unadjusted CNY per share",
            "return_ohlc": "adj_open/adj_high/adj_low/adj_close; AKShare hfq",
            "adjustment_factor": "adj_factor = adj_close / raw close",
            "volume": "volume_shares (plus compatibility alias volume); shares",
            "volume_lots": "lots; 1 lot = 100 shares",
            "amount": "CNY",
            "turnover": "decimal fraction of float shares",
            "vwap": "amount_CNY / volume_shares; CNY per share",
            "adj_vwap": "raw vwap * adj_factor; same HFQ basis as adj_ohlc",
            "amihud_1d": "abs(hfq close-to-close return) / amount_CNY; unscaled",
            "float_market_cap": "raw close * point-in-time/turnover-implied float shares",
            "revision_risk": (
                "HFQ prices/factors are vendor-derived and may be revised after corporate-action "
                "corrections; persist this manifest and raw parquet snapshots for reproducibility."
            ),
        },
    }
    (root / "history_manifest.json").write_text(__import__("json").dumps(manifest, ensure_ascii=False, indent=2))
    print(f"history saved to {root}")


if __name__ == "__main__":
    main()
