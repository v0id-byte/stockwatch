"""Exact pandas implementation of Microsoft Qlib's default Alpha158 fields.

The feature names, order, windows, and formulas mirror Qlib commit
``d5379c520f66a39953bad76234a7019a72796fd0`` in ``Alpha158DL`` and its
``ops.py``/``rolling.pyx`` operator implementations:
https://github.com/microsoft/qlib/blob/d5379c520f66a39953bad76234a7019a72796fd0/qlib/contrib/data/loader.py

Unlike the legacy feature family in :mod:`analysis.factors`, this module has no
market-index inputs or StockWatch-only operators. Raw NaNs and infinities are
preserved just as they are by Qlib's expression layer; normalization/imputation
belongs to a later dataset processor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


QLIB_ALPHA158_WINDOWS = (5, 10, 20, 30, 60)
QLIB_ALPHA158_KBAR_FEATURES = (
    "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
)
QLIB_ALPHA158_PRICE_FEATURES = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
QLIB_ALPHA158_ROLLING_FEATURES = (
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD",
    "RANK", "RSV", "IMAX", "IMIN", "IMXD", "CORR", "CORD", "CNTP", "CNTN",
    "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN",
    "VSUMD",
)
QLIB_ALPHA158_FEATURES = (
    *QLIB_ALPHA158_KBAR_FEATURES,
    *QLIB_ALPHA158_PRICE_FEATURES,
    *(
        f"{name}{window}"
        for name in QLIB_ALPHA158_ROLLING_FEATURES
        for window in QLIB_ALPHA158_WINDOWS
    ),
)

_REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume", "vwap")
_EPS = 1e-12
_QLIB_ZERO_STD_ATOL = 2e-5


def _clean_qlib_input(kline_df: pd.DataFrame) -> pd.DataFrame:
    missing = [name for name in _REQUIRED_COLUMNS if name not in kline_df.columns]
    if missing:
        raise ValueError(
            "exact Qlib Alpha158 requires input columns "
            f"{', '.join(_REQUIRED_COLUMNS)}; missing: {', '.join(missing)}"
        )
    frame = kline_df.copy()
    if "trade_date" in frame.columns:
        frame = frame.sort_values("trade_date", kind="stable")
        frame.index = pd.to_datetime(frame["trade_date"])
    for name in _REQUIRED_COLUMNS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def _rolling_trend(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Qlib Slope/Rsquare/Resi against the observation position in each window."""
    x = pd.Series(np.arange(1.0, len(series) + 1.0), index=series.index)
    valid = series.notna()
    rolling = lambda values: values.rolling(window, min_periods=1).sum()

    count = rolling(valid.astype(float))
    sum_x = rolling(x.where(valid))
    sum_x2 = rolling(x.pow(2).where(valid))
    sum_y = rolling(series)
    sum_y2 = rolling(series.pow(2))
    sum_xy = rolling((x * series).where(valid))

    covariance_numerator = count * sum_xy - sum_x * sum_y
    x_variance_numerator = count * sum_x2 - sum_x.pow(2)
    y_variance_numerator = count * sum_y2 - sum_y.pow(2)
    slope = covariance_numerator / x_variance_numerator
    intercept = (sum_y - slope * sum_x) / count
    residual = series - (slope * x + intercept)
    rsquare = covariance_numerator.pow(2) / (
        x_variance_numerator * y_variance_numerator
    )

    # Qlib explicitly nulls Rsquare when rolling price std is effectively zero.
    near_constant = np.isclose(
        series.rolling(window, min_periods=1).std(),
        0,
        atol=_QLIB_ZERO_STD_ATOL,
    )
    return slope, rsquare.mask(near_constant), residual


def _qlib_corr(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    """Qlib Corr, including its near-constant-series NaN rule."""
    result = left.rolling(window, min_periods=1).corr(right)
    left_constant = np.isclose(
        left.rolling(window, min_periods=1).std(),
        0,
        atol=_QLIB_ZERO_STD_ATOL,
    )
    right_constant = np.isclose(
        right.rolling(window, min_periods=1).std(),
        0,
        atol=_QLIB_ZERO_STD_ATOL,
    )
    return result.mask(left_constant | right_constant)


def _rolling_index(series: pd.Series, window: int, *, maximum: bool) -> pd.Series:
    selector = np.argmax if maximum else np.argmin
    return series.rolling(window, min_periods=1).apply(
        lambda values: selector(values) + 1,
        raw=True,
    ) / window


def compute_qlib_alpha158_frame(kline_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Qlib's default 158 raw features for every input row.

    Required OHLC and VWAP columns must share one explicit price-adjustment
    basis; ``volume`` must use one explicit volume basis. This function does not
    construct or guess either basis. ``trade_date`` is optional and is copied to
    the output. Raw NaNs and infinities are deliberately preserved.
    """
    frame = _clean_qlib_input(kline_df)
    if frame.empty:
        result = pd.DataFrame(index=frame.index, columns=QLIB_ALPHA158_FEATURES, dtype=float)
        if "trade_date" in frame.columns:
            result.insert(0, "trade_date", frame["trade_date"].to_numpy())
        return result

    open_ = frame["open"]
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    volume = frame["volume"]
    vwap = frame["vwap"]
    price_range = high - low

    data: dict[str, pd.Series] = {
        "KMID": (close - open_) / open_,
        "KLEN": price_range / open_,
        "KMID2": (close - open_) / (price_range + _EPS),
        "KUP": (high - np.maximum(open_, close)) / open_,
        "KUP2": (high - np.maximum(open_, close)) / (price_range + _EPS),
        "KLOW": (np.minimum(open_, close) - low) / open_,
        "KLOW2": (np.minimum(open_, close) - low) / (price_range + _EPS),
        "KSFT": (2 * close - high - low) / open_,
        "KSFT2": (2 * close - high - low) / (price_range + _EPS),
        "OPEN0": open_ / close,
        "HIGH0": high / close,
        "LOW0": low / close,
        "VWAP0": vwap / close,
    }

    previous_close = close.shift(1)
    close_change = close - previous_close
    positive_close_change = pd.Series(np.maximum(close_change, 0), index=frame.index)
    negative_close_change = pd.Series(np.maximum(-close_change, 0), index=frame.index)
    absolute_close_change = close_change.abs()
    close_ratio = close / previous_close

    previous_volume = volume.shift(1)
    volume_change = volume - previous_volume
    positive_volume_change = pd.Series(np.maximum(volume_change, 0), index=frame.index)
    negative_volume_change = pd.Series(np.maximum(-volume_change, 0), index=frame.index)
    absolute_volume_change = volume_change.abs()
    log_volume = np.log(volume + 1)
    log_volume_ratio = np.log(volume / previous_volume + 1)

    for window in QLIB_ALPHA158_WINDOWS:
        rolling_close = close.rolling(window, min_periods=1)
        rolling_high = high.rolling(window, min_periods=1)
        rolling_low = low.rolling(window, min_periods=1)
        rolling_volume = volume.rolling(window, min_periods=1)
        slope, rsquare, residual = _rolling_trend(close, window)
        index_max = _rolling_index(high, window, maximum=True)
        index_min = _rolling_index(low, window, maximum=False)

        gain_sum = positive_close_change.rolling(window, min_periods=1).sum()
        loss_sum = negative_close_change.rolling(window, min_periods=1).sum()
        absolute_change_sum = absolute_close_change.rolling(window, min_periods=1).sum()
        volume_gain_sum = positive_volume_change.rolling(window, min_periods=1).sum()
        volume_loss_sum = negative_volume_change.rolling(window, min_periods=1).sum()
        absolute_volume_change_sum = absolute_volume_change.rolling(window, min_periods=1).sum()
        volume_weighted_move = (close_ratio - 1).abs() * volume

        data[f"ROC{window}"] = close.shift(window) / close
        data[f"MA{window}"] = rolling_close.mean() / close
        data[f"STD{window}"] = rolling_close.std() / close
        data[f"BETA{window}"] = slope / close
        data[f"RSQR{window}"] = rsquare
        data[f"RESI{window}"] = residual / close
        data[f"MAX{window}"] = rolling_high.max() / close
        data[f"MIN{window}"] = rolling_low.min() / close
        data[f"QTLU{window}"] = rolling_close.quantile(0.8) / close
        data[f"QTLD{window}"] = rolling_close.quantile(0.2) / close
        data[f"RANK{window}"] = rolling_close.rank(pct=True)
        data[f"RSV{window}"] = (
            (close - rolling_low.min())
            / (rolling_high.max() - rolling_low.min() + _EPS)
        )
        data[f"IMAX{window}"] = index_max
        data[f"IMIN{window}"] = index_min
        data[f"IMXD{window}"] = index_max - index_min
        data[f"CORR{window}"] = _qlib_corr(close, log_volume, window)
        data[f"CORD{window}"] = _qlib_corr(close_ratio, log_volume_ratio, window)
        data[f"CNTP{window}"] = (close > previous_close).rolling(window, min_periods=1).mean()
        data[f"CNTN{window}"] = (close < previous_close).rolling(window, min_periods=1).mean()
        data[f"CNTD{window}"] = data[f"CNTP{window}"] - data[f"CNTN{window}"]
        data[f"SUMP{window}"] = gain_sum / (absolute_change_sum + _EPS)
        data[f"SUMN{window}"] = loss_sum / (absolute_change_sum + _EPS)
        data[f"SUMD{window}"] = (gain_sum - loss_sum) / (absolute_change_sum + _EPS)
        data[f"VMA{window}"] = rolling_volume.mean() / (volume + _EPS)
        data[f"VSTD{window}"] = rolling_volume.std() / (volume + _EPS)
        data[f"WVMA{window}"] = (
            volume_weighted_move.rolling(window, min_periods=1).std()
            / (volume_weighted_move.rolling(window, min_periods=1).mean() + _EPS)
        )
        data[f"VSUMP{window}"] = volume_gain_sum / (absolute_volume_change_sum + _EPS)
        data[f"VSUMN{window}"] = volume_loss_sum / (absolute_volume_change_sum + _EPS)
        data[f"VSUMD{window}"] = (
            (volume_gain_sum - volume_loss_sum) / (absolute_volume_change_sum + _EPS)
        )

    result = pd.DataFrame(data, index=frame.index).reindex(columns=QLIB_ALPHA158_FEATURES)
    if "trade_date" in frame.columns:
        result.insert(0, "trade_date", frame["trade_date"].to_numpy())
    return result


def compute_qlib_alpha158(kline_df: pd.DataFrame) -> dict[str, float]:
    """Return the latest row of exact Qlib Alpha158 raw features."""
    frame = compute_qlib_alpha158_frame(kline_df)
    if frame.empty:
        return {name: float("nan") for name in QLIB_ALPHA158_FEATURES}
    latest = frame.iloc[-1]
    return {name: float(latest[name]) for name in QLIB_ALPHA158_FEATURES}
