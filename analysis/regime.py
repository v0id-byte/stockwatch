"""Market volatility regime detection."""
from __future__ import annotations

import math

import pandas as pd
from loguru import logger

from data.market import MarketData
from utils.storage import Storage


DEFAULT_REGIME = {
    "regime": "normal",
    "vol_20d": 0.0,
    "percentile": 0.0,
    "confidence_floor": 0.6,
    "context": "大盘 regime: normal (未启用或数据不足)",
}


def _regime_for_percentile(percentile: float) -> tuple[str, float]:
    if percentile > 0.9:
        return "crisis", 0.8
    if percentile > 0.7:
        return "volatile", 0.7
    if percentile <= 0.3:
        return "calm", 0.6
    return "normal", 0.6


def get_market_regime(market: MarketData, storage: Storage) -> dict:
    """Return regime info and cache rolling volatility history."""
    rows = market.get_index_kline("sh000001", limit=900)
    if len(rows) < 60:
        logger.info("波动率 regime：指数K线不足，使用 normal")
        return DEFAULT_REGIME.copy()

    df = pd.DataFrame(rows).sort_values("trade_date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    ret = df["close"].pct_change()
    df["vol_20d"] = ret.rolling(20).std() * math.sqrt(252)
    vols = df.dropna(subset=["vol_20d"]).copy()
    if len(vols) < 500:
        logger.info(f"波动率 regime：历史样本不足（{len(vols)}/500），使用 normal")
        return DEFAULT_REGIME.copy()

    latest = float(vols["vol_20d"].iloc[-1])
    percentile = float((vols["vol_20d"] <= latest).mean())
    regime, floor = _regime_for_percentile(percentile)
    cache_rows = []
    for _, row in vols.iterrows():
        row_percentile = float((vols["vol_20d"] <= row["vol_20d"]).mean())
        row_regime, _ = _regime_for_percentile(row_percentile)
        cache_rows.append({
            "trade_date": str(row["trade_date"]),
            "vol_20d": float(row["vol_20d"]),
            "regime": row_regime,
            "percentile": row_percentile,
        })
    storage.upsert_market_regime_history(cache_rows)

    context = (
        f"大盘 regime: {regime} "
        f"(波动率 {percentile:.0%} 分位，本日 confidence 阈值 {floor:.1f})"
    )
    logger.info(context)
    return {
        "regime": regime,
        "vol_20d": latest,
        "percentile": percentile,
        "confidence_floor": floor,
        "context": context,
    }
