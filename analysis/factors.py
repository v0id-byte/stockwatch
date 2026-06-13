"""Alpha158 factor calculation implemented with pandas.

References:
    Qlib: A AI-oriented Quantitative Investment Platform
        Liu et al. (2021) — https://arxiv.org/abs/2009.11189
    101 Formulaic Alphas (WorldQuant style)
        Kakushadze (2016) — https://arxiv.org/abs/1601.00991
    This implementation follows Qlib's Alpha158/Alpha360 feature definitions.
"""
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

# Sign-stable cross-sectional alpha factors, selected by their forward-return IC
# being the SAME sign and non-trivial magnitude in BOTH the 2022-2024 bear and
# the 2025-2026 bull regimes (see scripts/backtest_strategy.py). Three diversified
# families that survive regime changes in A-shares:
#   - short/medium reversal:        RET/RELV/ROC/RESI/CORD
#   - oversold price position:      QTLD/QTLU/MA/IMIN
#   - low turnover / low attention: TURN/VOLZ/AMTMA/VMA/VSUMN/WVMA
# Pure illiquidity (ILLIQ) and long-horizon volatility/beta/R2 (STD/BETA/RSQR) are
# deliberately EXCLUDED: their IC sign flips across regimes, which is what made the
# previous "stable" model post a negative out-of-sample IC.
ROBUST_FEATURES = [
    "RET20", "RET30", "RET60", "RELV20", "RELV30", "RELV60", "ROC20", "ROC30", "ROC60",
    "QTLD20", "QTLD30", "QTLD60", "QTLU60", "QTLU120", "MA20", "MA30", "MA60", "IMIN30",
    "TURN120", "TURN250", "VOLZ120", "VOLZ250", "AMTMA60", "AMTMA120", "AMTMA250",
    "VMA120", "VMA250", "VSUMN60", "VSUMN120", "CORD60", "RESI10", "WVMA20",
]

# Regime-specialized sets (used by MARKET_REGIME / STOCKWATCH_LGBM_REGIME). On A-share
# history (2022-2026) cross-sectional factor IC is ~2x stronger in bear/sideways than in
# strong bull markets, and a few factors (illiquidity, low-vol) are predictive in risk-off
# but flat in bull. So the two regimes are handled asymmetrically:
#   BEAR — high-conviction defensive stock-picking: robust set PLUS illiquidity, deeper
#          oversold (drawdown / distance-from-high) factors, which measurably lift bear IC.
#   BULL — light touch: cross-sectional alpha is thin in strong uptrends, so drop the
#          short-term reversal factors (don't fight winners) and keep only the position /
#          liquidity / low-turnover factors that stay weakly positive in bull.
BEAR_FEATURES = ROBUST_FEATURES + [
    "ILLIQ20", "ILLIQ60", "ILLIQ120", "DD60", "DD120", "MAX60", "IMIN60", "QTLD120",
]
# NOTE on fundamentals: the earnings-quality factor `ocf_to_eps` (operating cash
# flow / EPS) was built and tested as a bear-model diversifier. A clean out-of-sample
# check (train bear-days <2024, test >=2024) showed it does NOT help — IC +0.1070
# WITHOUT vs +0.1039 WITH — despite a high in-sample split gain (overfitting). So it
# is intentionally NOT in BEAR_FEATURES. The PIT pipeline (scripts/
# build_fundamental_features.py + the training-set merge + analysis/fundamental.py)
# is kept as opt-in research infra: append "ocf_to_eps" (or another fundamental)
# here to re-enable, and the online path will inject it automatically.
BULL_FEATURES = [
    "MA20", "MA30", "MA60", "QTLD20", "QTLD30", "QTLD60", "QTLU60", "QTLU120", "IMIN30",
    "ILLIQ20", "ILLIQ60", "TURN120", "TURN250", "AMTMA120", "AMTMA250", "VMA120",
    "VMA250", "VSUMN60",
]


def cross_sectional_rank_normalize(frame: "pd.DataFrame", columns: list[str]) -> "pd.DataFrame":
    """Per-column cross-sectional percentile rank centered to [-0.5, 0.5].

    Used to match training (features are rank-normalized within each trade date)
    when scoring a batch of stocks online. A single-row batch maps to all-zeros,
    which the model treats as a neutral (un-rankable) input.
    """
    out = frame.copy()
    n = len(out)
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
            continue
        if n <= 1:
            out[col] = 0.0
        else:
            out[col] = out[col].rank(pct=True) - 0.5
    return out


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

    vwap = (amount / df["volume"].replace(0, np.nan)).where(amount > 0, (high + low + close) / 3)
    factor_data = {
        "OPEN0": df["open"] / close,
        "HIGH0": high / close,
        "LOW0": low / close,
        "VWAP0": vwap / close,
        "KMID": (close - open_) / open_,
        "KLEN": (high - low) / open_,
        "KMID2": (close - open_) / (k_range + eps),
        "KUP": (high - np.maximum(open_, close)) / open_,
        "KUP2": (high - np.maximum(open_, close)) / (k_range + eps),
        "KLOW": (np.minimum(open_, close) - low) / open_,
        "KLOW2": (np.minimum(open_, close) - low) / (k_range + eps),
        "KSFT": (2 * close - high - low) / open_,
        "KSFT2": (2 * close - high - low) / (k_range + eps),
    }

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

        imax = _rolling_argmax(high, window)
        imin = _rolling_argmin(low, window)
        cntp = (ret > 0).rolling(window).mean()
        cntn = (ret < 0).rolling(window).mean()
        sump = pos_ret.rolling(window).sum() / ret_denom
        sumn = neg_ret.rolling(window).sum() / ret_denom
        vsump = vol_pos.rolling(window).sum() / vol_denom
        vsumn = vol_neg.rolling(window).sum() / vol_denom
        factor_data.update({
            f"ROC{window}": close.shift(window) / close,
            f"MA{window}": close.rolling(window).mean() / close,
            f"STD{window}": close.rolling(window).std() / close,
            f"BETA{window}": beta,
            f"RSQR{window}": corr.pow(2),
            f"RESI{window}": (ret - beta * market_ret).rolling(window).std(),
            f"MAX{window}": high.rolling(window).max() / close,
            f"MIN{window}": low.rolling(window).min() / close,
            f"QTLU{window}": close.rolling(window).quantile(0.8) / close,
            f"QTLD{window}": close.rolling(window).quantile(0.2) / close,
            f"RANK{window}": _rolling_rank_last(close, window),
            f"RSV{window}": (close - min_close) / (max_close - min_close + eps),
            f"IMAX{window}": imax,
            f"IMIN{window}": imin,
            f"IMXD{window}": imax - imin,
            f"CORR{window}": close.rolling(window).corr(log_volume),
            f"CORD{window}": close.diff().rolling(window).corr(log_volume.diff()),
            f"CNTP{window}": cntp,
            f"CNTN{window}": cntn,
            f"CNTD{window}": cntp - cntn,
            f"SUMP{window}": sump,
            f"SUMN{window}": sumn,
            f"SUMD{window}": sump - sumn,
            f"VMA{window}": df["volume"].rolling(window).mean() / volume,
            f"VSTD{window}": df["volume"].rolling(window).std() / volume,
            f"WVMA{window}": value.rolling(window).std() / (value.rolling(window).mean() + eps),
            f"VSUMP{window}": vsump,
            f"VSUMN{window}": vsumn,
            f"VSUMD{window}": vsump - vsumn,
            f"RET{window}": close / close.shift(window) - 1,
            f"MOM{window}": ret_mean / (ret_std + eps),
            f"DD{window}": close / (max_close + eps) - 1,
            f"UPR{window}": up_days / (down_days + eps),
            f"DNR{window}": down_days / window,
            f"SHARPE{window}": ret_mean / (ret_std + eps) * np.sqrt(252),
            f"VOLZ{window}": (df["volume"] - volume_mean) / (df["volume"].rolling(window).std() + eps),
            f"TURN{window}": df["volume"] / (volume_mean + eps),
            f"AMTMA{window}": value.rolling(window).mean() / (value + eps),
            f"ILLIQ{window}": ret.abs().rolling(window).mean() / (value.rolling(window).mean() + eps),
            f"PVCHG{window}": ret.rolling(window).corr(df["volume"].pct_change()),
            f"RELV{window}": (ret - market_ret).rolling(window).sum(),
        })

    factors = pd.DataFrame(factor_data, index=df.index).reindex(columns=ALPHA158_FEATURES)
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
