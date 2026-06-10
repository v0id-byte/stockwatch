"""Alpha158 factor calculation implemented with pandas."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from loguru import logger

warnings.filterwarnings("ignore", category=PerformanceWarning)


WINDOWS = [5, 10, 20, 30, 60, 120, 250]
ROLLING_FEATURES = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP",
    "VSUMN", "VSUMD", "RET", "MOM", "DD", "UPR", "DNR", "SHARPE",
    "VOLZ", "TURN", "AMTMA", "ILLIQ", "PVCHG", "RELV",
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
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std()
        up_days = (ret > 0).rolling(window).sum()
        down_days = (ret < 0).rolling(window).sum()
        volume_mean = df["volume"].rolling(window).mean()

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
        factors[f"RET{window}"] = close / close.shift(window) - 1
        factors[f"MOM{window}"] = ret_mean / (ret_std + eps)
        factors[f"DD{window}"] = close / (max_close + eps) - 1
        factors[f"UPR{window}"] = up_days / (down_days + eps)
        factors[f"DNR{window}"] = down_days / window
        factors[f"SHARPE{window}"] = ret_mean / (ret_std + eps) * np.sqrt(252)
        factors[f"VOLZ{window}"] = (df["volume"] - volume_mean) / (df["volume"].rolling(window).std() + eps)
        factors[f"TURN{window}"] = df["volume"] / (volume_mean + eps)
        factors[f"AMTMA{window}"] = value.rolling(window).mean() / (value + eps)
        factors[f"ILLIQ{window}"] = ret.abs().rolling(window).mean() / (value.rolling(window).mean() + eps)
        factors[f"PVCHG{window}"] = ret.rolling(window).corr(df["volume"].pct_change())
        factors[f"RELV{window}"] = (ret - market_ret).rolling(window).sum()

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


def _factor_value(factors: dict, name: str) -> float:
    try:
        value = float(factors.get(name, 0.0) or 0.0)
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _summarize_single_alpha158(factors: dict) -> str:
    signals = []

    roc5 = _factor_value(factors, "ROC5")
    if 0 < roc5 <= 0.98:
        signals.append("5日动量偏强")
    elif roc5 >= 1.02:
        signals.append("5日动量偏弱")

    roc20 = _factor_value(factors, "ROC20")
    if 0 < roc20 <= 0.95:
        signals.append("20日动量偏强")
    elif roc20 >= 1.05:
        signals.append("20日动量偏弱")

    roc120 = _factor_value(factors, "ROC120")
    if 0 < roc120 <= 0.85:
        signals.append("120日中线动量偏强")
    elif roc120 >= 1.15:
        signals.append("120日中线动量偏弱")

    roc250 = _factor_value(factors, "ROC250")
    if 0 < roc250 <= 0.8:
        signals.append("250日长线动量偏强")
    elif roc250 >= 1.2:
        signals.append("250日长线动量偏弱")

    ma20 = _factor_value(factors, "MA20")
    if 0 < ma20 <= 0.98:
        signals.append("站上20日均线")
    elif ma20 >= 1.02:
        signals.append("低于20日均线")

    ma120 = _factor_value(factors, "MA120")
    if 0 < ma120 <= 0.95:
        signals.append("站上半年线")
    elif ma120 >= 1.05:
        signals.append("低于半年线")

    ma250 = _factor_value(factors, "MA250")
    if 0 < ma250 <= 0.95:
        signals.append("站上年线")
    elif ma250 >= 1.05:
        signals.append("低于年线")

    rsv20 = _factor_value(factors, "RSV20")
    if rsv20 >= 0.8:
        signals.append("20日价格位置偏高")
    elif 0 < rsv20 <= 0.2:
        signals.append("20日价格位置偏低")

    cntd20 = _factor_value(factors, "CNTD20")
    if cntd20 >= 0.2:
        signals.append("近20日上涨天数占优")
    elif cntd20 <= -0.2:
        signals.append("近20日下跌天数占优")

    sumd20 = _factor_value(factors, "SUMD20")
    if sumd20 >= 0.2:
        signals.append("近20日正收益强于负收益")
    elif sumd20 <= -0.2:
        signals.append("近20日负收益强于正收益")

    sumd120 = _factor_value(factors, "SUMD120")
    if sumd120 >= 0.2:
        signals.append("近120日正收益占优")
    elif sumd120 <= -0.2:
        signals.append("近120日负收益占优")

    dd120 = _factor_value(factors, "DD120")
    if dd120 <= -0.25:
        signals.append("距120日高点回撤较深")
    elif -0.08 <= dd120 < 0:
        signals.append("接近120日阶段高位")

    relv60 = _factor_value(factors, "RELV60")
    if relv60 >= 0.08:
        signals.append("近60日跑赢大盘")
    elif relv60 <= -0.08:
        signals.append("近60日跑输大盘")

    sharpe120 = _factor_value(factors, "SHARPE120")
    if sharpe120 >= 1.0:
        signals.append("120日收益/波动比偏好")
    elif sharpe120 <= -1.0:
        signals.append("120日收益/波动比偏弱")

    vsumd20 = _factor_value(factors, "VSUMD20")
    if vsumd20 >= 0.2:
        signals.append("量能扩张占优")
    elif vsumd20 <= -0.2:
        signals.append("量能收缩占优")

    turn20 = _factor_value(factors, "TURN20")
    if turn20 >= 1.8:
        signals.append("短期成交明显放大")
    elif 0 < turn20 <= 0.6:
        signals.append("短期成交明显萎缩")

    if not signals:
        signals.append("动量、位置、量能因子整体接近中性")

    valid_count = sum(1 for name in ALPHA158_FEATURES if name in factors)
    return (
        "Alpha158 摘要:\n"
        f"  已计算因子: {valid_count}/{len(ALPHA158_FEATURES)}\n"
        f"  重点信号: {'；'.join(signals[:5])}"
    )


def summarize_alpha158_cross_section(factors_by_code: dict[str, dict]) -> dict[str, str]:
    """Format Top 5 positive/negative z-score signals for each code."""
    if not factors_by_code:
        return {}
    frame = pd.DataFrame.from_dict(factors_by_code, orient="index")
    frame = frame.reindex(columns=ALPHA158_FEATURES).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(frame) < 2:
        summaries = {
            code: _summarize_single_alpha158(row.to_dict())
            for code, row in frame.iterrows()
        }
        logger.info(f"Alpha158 单票摘要生成: {len(summaries)} 只, {len(ALPHA158_FEATURES)} 个因子")
        return summaries

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
