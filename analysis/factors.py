"""Alpha158 factor calculation implemented with pandas."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from loguru import logger

warnings.filterwarnings("ignore", category=PerformanceWarning)


WINDOWS = [5, 10, 20, 30, 60]
ROLLING_FEATURES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP",
    "VSUMN", "VSUMD",
]
BASE_FEATURES = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
KBAR_FEATURES = ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]
ALPHA158_FEATURES = (
    BASE_FEATURES
    + KBAR_FEATURES
    + [f"{name}{window}" for window in WINDOWS for name in ROLLING_FEATURES]
)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out = out.sort_values("trade_date")
        out.index = pd.to_datetime(out["trade_date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _market_return(market_df: pd.DataFrame, index: pd.Index) -> pd.Series:
    if market_df is None or market_df.empty:
        return pd.Series(0.0, index=index)
    market = _clean_frame(market_df)
    ret = market["close"].pct_change()
    ret = ret.reindex(index).ffill().fillna(0.0)
    return ret


def _rolling_rank_last(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(
        lambda values: pd.Series(values).rank(pct=True).iloc[-1],
        raw=False,
    )


def _rolling_argmax(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: (np.argmax(values) + 1) / window, raw=True)


def _rolling_argmin(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: (np.argmin(values) + 1) / window, raw=True)


def compute_alpha158_frame(kline_df: pd.DataFrame, market_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return an all-date Alpha158 frame; latest row is used online."""
    df = _clean_frame(kline_df)
    if df.empty:
        return pd.DataFrame(columns=ALPHA158_FEATURES)

    eps = 1e-12
    close = df["close"].replace(0, np.nan)
    open_ = df["open"].replace(0, np.nan)
    high = df["high"]
    low = df["low"]
    volume = df["volume"].replace(0, np.nan)
    amount = df["amount"]
    k_range = (high - low).replace(0, np.nan)
    ret = close.pct_change()
    market_ret = _market_return(market_df, df.index)
    log_volume = np.log(df["volume"].replace(0, np.nan))

    factors = pd.DataFrame(index=df.index)
    vwap = (amount / df["volume"].replace(0, np.nan)).where(amount > 0, (high + low + close) / 3)
    factors["OPEN0"] = df["open"] / close
    factors["HIGH0"] = high / close
    factors["LOW0"] = low / close
    factors["VWAP0"] = vwap / close

    factors["KMID"] = (close - open_) / open_
    factors["KLEN"] = (high - low) / open_
    factors["KMID2"] = (close - open_) / (k_range + eps)
    factors["KUP"] = (high - np.maximum(open_, close)) / open_
    factors["KUP2"] = (high - np.maximum(open_, close)) / (k_range + eps)
    factors["KLOW"] = (np.minimum(open_, close) - low) / open_
    factors["KLOW2"] = (np.minimum(open_, close) - low) / (k_range + eps)
    factors["KSFT"] = (2 * close - high - low) / open_
    factors["KSFT2"] = (2 * close - high - low) / (k_range + eps)

    pos_ret = ret.clip(lower=0)
    neg_ret = (-ret.clip(upper=0))
    vol_delta = df["volume"].diff()
    vol_pos = vol_delta.clip(lower=0)
    vol_neg = (-vol_delta.clip(upper=0))

    for window in WINDOWS:
        beta = ret.rolling(window).cov(market_ret) / (market_ret.rolling(window).var() + eps)
        corr = ret.rolling(window).corr(market_ret)
        min_close = close.rolling(window).min()
        max_close = close.rolling(window).max()
        ret_denom = pos_ret.rolling(window).sum() + neg_ret.rolling(window).sum() + eps
        vol_denom = vol_pos.rolling(window).sum() + vol_neg.rolling(window).sum() + eps
        value = close * df["volume"]

        factors[f"ROC{window}"] = close.shift(window) / close
        factors[f"MA{window}"] = close.rolling(window).mean() / close
        factors[f"STD{window}"] = close.rolling(window).std() / close
        factors[f"BETA{window}"] = beta
        factors[f"RSQR{window}"] = corr.pow(2)
        factors[f"RESI{window}"] = (ret - beta * market_ret).rolling(window).std()
        factors[f"MAX{window}"] = high.rolling(window).max() / close
        factors[f"MIN{window}"] = low.rolling(window).min() / close
        factors[f"QTLU{window}"] = close.rolling(window).quantile(0.8) / close
        factors[f"QTLD{window}"] = close.rolling(window).quantile(0.2) / close
        factors[f"RANK{window}"] = _rolling_rank_last(close, window)
        factors[f"RSV{window}"] = (close - min_close) / (max_close - min_close + eps)
        factors[f"IMAX{window}"] = _rolling_argmax(high, window)
        factors[f"IMIN{window}"] = _rolling_argmin(low, window)
        factors[f"IMXD{window}"] = factors[f"IMAX{window}"] - factors[f"IMIN{window}"]
        factors[f"CORR{window}"] = close.rolling(window).corr(log_volume)
        factors[f"CORD{window}"] = close.diff().rolling(window).corr(log_volume.diff())
        factors[f"CNTP{window}"] = (ret > 0).rolling(window).mean()
        factors[f"CNTN{window}"] = (ret < 0).rolling(window).mean()
        factors[f"CNTD{window}"] = factors[f"CNTP{window}"] - factors[f"CNTN{window}"]
        factors[f"SUMP{window}"] = pos_ret.rolling(window).sum() / ret_denom
        factors[f"SUMN{window}"] = neg_ret.rolling(window).sum() / ret_denom
        factors[f"SUMD{window}"] = factors[f"SUMP{window}"] - factors[f"SUMN{window}"]
        factors[f"VMA{window}"] = df["volume"].rolling(window).mean() / volume
        factors[f"VSTD{window}"] = df["volume"].rolling(window).std() / volume
        factors[f"WVMA{window}"] = value.rolling(window).std() / (value.rolling(window).mean() + eps)
        factors[f"VSUMP{window}"] = vol_pos.rolling(window).sum() / vol_denom
        factors[f"VSUMN{window}"] = vol_neg.rolling(window).sum() / vol_denom
        factors[f"VSUMD{window}"] = factors[f"VSUMP{window}"] - factors[f"VSUMN{window}"]

    factors = factors.reindex(columns=ALPHA158_FEATURES)
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "trade_date" in df.columns:
        factors.insert(0, "trade_date", df["trade_date"].values)
    return factors


def compute_alpha158(kline_df: pd.DataFrame, market_df: pd.DataFrame | None = None) -> dict:
    """
    Compute latest Alpha158 values.

    kline_df columns: open/high/low/close/volume/amount; optional trade_date.
    market_df is aligned by trade_date when present and is used for BETA/RSQR/RESI.
    """
    factors = compute_alpha158_frame(kline_df, market_df)
    if factors.empty:
        return {name: 0.0 for name in ALPHA158_FEATURES}
    latest = factors.iloc[-1]
    return {name: float(latest.get(name, 0.0)) for name in ALPHA158_FEATURES}


def summarize_alpha158_cross_section(factors_by_code: dict[str, dict]) -> dict[str, str]:
    """Format Top 5 positive/negative z-score signals for each code."""
    if not factors_by_code:
        return {}
    frame = pd.DataFrame.from_dict(factors_by_code, orient="index")
    frame = frame.reindex(columns=ALPHA158_FEATURES).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    std = frame.std(axis=0).replace(0, np.nan)
    zscores = ((frame - frame.mean(axis=0)) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    summaries = {}
    for code, row in zscores.iterrows():
        top = row.sort_values(ascending=False).head(5)
        bottom = row.sort_values(ascending=True).head(5)
        pos = ", ".join(f"{name}(+{value:.1f}sigma)" for name, value in top.items())
        neg = ", ".join(f"{name}({value:.1f}sigma)" for name, value in bottom.items())
        summaries[code] = (
            "Alpha158 摘要:\n"
            f"  最显著的正面信号: {pos}\n"
            f"  最显著的负面信号: {neg}"
        )
    logger.info(f"Alpha158 因子摘要生成: {len(summaries)} 只, {len(ALPHA158_FEATURES)} 个因子")
    return summaries
