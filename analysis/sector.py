"""Sector mapping and 5-day relative strength."""
from __future__ import annotations

import re

import pandas as pd
from loguru import logger

from data.market import MarketData
from utils.storage import Storage


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _pick_column(columns, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return ""


def _build_sector_map() -> dict[str, str]:
    import akshare as ak

    mapping = {}
    boards = ak.stock_board_industry_name_em()
    name_col = _pick_column(boards.columns, ["板块名称", "名称"])
    if not name_col:
        return mapping
    for sector in boards[name_col].dropna().astype(str).tolist():
        try:
            cons = ak.stock_board_industry_cons_em(symbol=sector)
            code_col = _pick_column(cons.columns, ["代码", "股票代码"])
            if not code_col:
                continue
            for code in cons[code_col].dropna().tolist():
                normalized = _normalize_code(code)
                if normalized:
                    mapping[normalized] = sector
        except Exception as e:
            logger.debug(f"板块成分获取失败 {sector}: {e}")
    return mapping


def _sector_return_5d(sector: str) -> tuple[str, float] | None:
    import akshare as ak

    df = ak.stock_board_industry_hist_em(symbol=sector, period="日k", adjust="")
    date_col = _pick_column(df.columns, ["日期", "date"])
    close_col = _pick_column(df.columns, ["收盘", "close"])
    if not date_col or not close_col or len(df) < 6:
        return None
    df = df.sort_values(date_col)
    close = pd.to_numeric(df[close_col], errors="coerce").dropna()
    if len(close) < 6:
        return None
    trade_date = str(df[date_col].iloc[-1])[:10]
    return trade_date, float(close.iloc[-1] / close.iloc[-6] - 1)


def _benchmark_return_5d(market: MarketData) -> float:
    rows = market.get_index_kline("sh000300", limit=20)
    if len(rows) < 6:
        return 0.0
    close = pd.Series([row["close"] for row in rows], dtype="float64")
    return float(close.iloc[-1] / close.iloc[-6] - 1)


def get_sector_contexts(codes: list[str], market: MarketData, storage: Storage) -> dict[str, str]:
    """Return LLM-ready sector context for each code."""
    cached = storage.get_cached_stock_sectors(codes, max_age_days=30)
    missing = [code for code in codes if code not in cached]
    if missing:
        try:
            fresh = _build_sector_map()
            storage.upsert_stock_sectors(fresh)
            cached.update({code: fresh[code] for code in missing if code in fresh})
            logger.info(f"板块映射刷新: {len(fresh)} 只股票")
        except Exception as e:
            logger.warning(f"板块映射刷新失败，使用缓存: {e}")

    benchmark_5d = _benchmark_return_5d(market)
    contexts = {}
    strength_cache: dict[str, dict] = {}
    for code in codes:
        sector = cached.get(code, "未知")
        if sector == "未知":
            contexts[code] = "所属板块: 未知\n板块5日相对收益: 0.0%（未知）"
            continue

        if sector not in strength_cache:
            strength = None
            try:
                result = _sector_return_5d(sector)
                if result:
                    trade_date, return_5d = result
                    strength = {
                        "sector": sector,
                        "trade_date": trade_date,
                        "return_5d": return_5d,
                        "excess_return_5d": return_5d - benchmark_5d,
                    }
                    storage.upsert_sector_strength([strength])
            except Exception as e:
                logger.debug(f"板块强弱获取失败 {sector}: {e}")
            if strength is None:
                strength = storage.get_sector_strength(sector) or {
                    "return_5d": 0.0,
                    "excess_return_5d": 0.0,
                }
            strength_cache[sector] = strength

        excess = float(strength_cache[sector].get("excess_return_5d") or 0.0)
        label = "强于大盘" if excess > 0 else "弱于大盘" if excess < 0 else "与大盘接近"
        contexts[code] = f"所属板块: {sector}\n板块5日相对收益: {excess:+.1%}（{label}）"

    logger.info(f"板块上下文生成: {len(contexts)} 只")
    return contexts
