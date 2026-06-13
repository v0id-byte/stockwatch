"""Latest point-in-time fundamental values for online scoring.

Only used when the active model (e.g. the bear-regime model) lists a fundamental
feature. Earnings-quality `ocf_to_eps` (operating cash flow per share / EPS) is a
small diversifying signal in risk-off regimes. Fetches the two most recent earnings
periods from AKShare and keeps the latest report visible per code. Fails soft: any
error returns an empty map and the feature stays neutral (0) for that run.
"""
from __future__ import annotations

from datetime import date

from loguru import logger

_CACHE: dict[str, dict[str, float]] = {}


def _recent_quarter_ends(n: int) -> list[str]:
    today = date.today()
    ends = []
    for year in (today.year, today.year - 1):
        for md in ("1231", "0930", "0630", "0331"):
            d = date(year, int(md[:2]), int(md[2:]))
            if d <= today:
                ends.append(f"{year}{md}")
    return ends[:n]


def get_latest_ocf_to_eps(codes: list[str]) -> dict[str, float]:
    """Return {code: ocf_to_eps} for the latest report visible today. Cached per day."""
    cache_key = date.today().isoformat()
    if cache_key not in _CACHE:
        _CACHE.clear()
        _CACHE[cache_key] = _fetch()
    table = _CACHE.get(cache_key, {})
    return {c: table[c] for c in (str(x).zfill(6) for x in codes) if c in table}


def _fetch() -> dict[str, float]:
    try:
        import akshare as ak
        import pandas as pd
    except Exception:
        return {}
    best: dict[str, float] = {}
    for period in _recent_quarter_ends(2):
        try:
            df = ak.stock_yjbb_em(date=period)
        except Exception as e:
            logger.debug(f"业绩报表获取失败 {period}: {e}")
            continue
        if df is None or df.empty or "股票代码" not in df.columns:
            continue
        eps = pd.to_numeric(df.get("每股收益"), errors="coerce")
        ocfps = pd.to_numeric(df.get("每股经营现金流量"), errors="coerce")
        ratio = (ocfps / eps.replace(0, float("nan"))).clip(-10, 10)
        for code, value in zip(df["股票代码"].astype(str).str.zfill(6), ratio):
            if code not in best and pd.notna(value):  # first (most recent period) wins
                best[code] = float(value)
    if best:
        logger.info(f"基本面 ocf_to_eps 已加载: {len(best)} 只")
    return best
