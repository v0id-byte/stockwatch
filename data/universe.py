"""分析池构建：自选股 + 热门拓展池"""
import json
import re
from datetime import date, timedelta
from pathlib import Path

import akshare as ak
from loguru import logger

from config import get_config
from data.market import MarketData

# 节假日兜底表（仅在 AKShare 接口与本地缓存均不可用时使用）。
# 主源：ak.tool_trade_date_hist_sina() → 成功后写入 ~/.stockwatch/trade_dates_cache.json
# 次源：本地缓存（30天内有效）
# 末源：下方硬编码表
_HOLIDAYS_FALLBACK = {
    # 2026
    "2026-01-01", "2026-01-02", "2026-01-26", "2026-01-27", "2026-01-28",
    "2026-01-29", "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02",
    "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
    "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01",
    "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06",
    "2026-10-07", "2026-10-08",
    # 2027（估算，以交易所正式公告为准；AKShare 主源会覆盖本表）
    "2027-01-01", "2027-01-02", "2027-01-03",
    "2027-02-15", "2027-02-16", "2027-02-17", "2027-02-18",
    "2027-02-19", "2027-02-20", "2027-02-21",
    "2027-04-05", "2027-04-06",
    "2027-05-01", "2027-05-02", "2027-05-03", "2027-05-04", "2027-05-05",
    "2027-06-19", "2027-06-20", "2027-06-21",
    "2027-10-01", "2027-10-02", "2027-10-03", "2027-10-04",
    "2027-10-05", "2027-10-06", "2027-10-07",
}

_TRADE_DATES_CACHE = Path.home() / ".stockwatch" / "trade_dates_cache.json"
_CACHE_TTL_DAYS = 30


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _load_trade_dates_cache() -> set[str]:
    """从本地文件加载缓存的交易日列表。"""
    try:
        if _TRADE_DATES_CACHE.exists():
            data = json.loads(_TRADE_DATES_CACHE.read_text())
            cached_at_str = data.get("cached_at", "")
            if not cached_at_str:
                return set()
            cached_at = date.fromisoformat(cached_at_str)
            today = date.today()
            if (today - cached_at).days < _CACHE_TTL_DAYS:
                dates = set(data.get("dates", []))
                if dates and max(dates) >= today.isoformat():
                    return dates
    except Exception:
        pass
    return set()


def _save_trade_dates_cache(dates: set[str]):
    try:
        _TRADE_DATES_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _TRADE_DATES_CACHE.write_text(json.dumps({
            "cached_at": date.today().isoformat(),
            "dates": sorted(dates),
        }))
        logger.debug(f"交易日历已缓存 {len(dates)} 条 → {_TRADE_DATES_CACHE}")
    except Exception as e:
        logger.debug(f"交易日历缓存写入失败: {e}")


class Universe:
    def __init__(self, storage=None):
        self.cfg = get_config()
        self.storage = storage
        self.market = MarketData()

    def is_trading_day(self) -> bool:
        """判断今天是否交易日（AKShare 主源 → 本地缓存 → 节假日兜底表）。"""
        today = date.today()
        if today.weekday() >= 5:  # 周六周日
            return False
        today_str = today.isoformat()

        # 1. AKShare 主源
        try:
            df = ak.tool_trade_date_hist_sina()
            trade_dates = {str(v)[:10] for v in df["trade_date"].tolist()}
            if trade_dates and max(trade_dates) >= today_str:
                _save_trade_dates_cache(trade_dates)  # 成功则刷新缓存
                return today_str in trade_dates
        except Exception as e:
            logger.debug(f"AKShare 交易日历获取失败，尝试本地缓存: {e}")

        # 2. 本地缓存（30天内有效）
        cached = _load_trade_dates_cache()
        if cached:
            logger.debug("使用本地缓存交易日历")
            return today_str in cached

        # 3. 节假日兜底表
        logger.warning("交易日历主源与缓存均不可用，使用兜底节假日表（仅供应急）")
        return today_str not in _HOLIDAYS_FALLBACK

    def get_today_codes(self) -> list[str]:
        """返回今日分析池 code 列表（上限50只）。"""
        if not self.is_trading_day():
            logger.info("今日非交易日，跳过")
            return []

        watchlist = self.cfg.watchlist
        today_str = date.today().isoformat()
        cache_key = f"hot_pool_{today_str}"

        hot_codes = self._get_cached_pool(cache_key)
        if not hot_codes:
            hot_codes = self._build_hot_pool()
            self._cache_pool(cache_key, hot_codes)

        all_codes = watchlist.copy()
        for c in hot_codes:
            if c not in all_codes and len(all_codes) < self.cfg.max_stocks_per_run:
                all_codes.append(c)

        logger.info(f"分析池共 {len(all_codes)} 只（自选 {len(watchlist)} + 热门 {len(all_codes)-len(watchlist)}）")
        return all_codes

    def _build_hot_pool(self) -> list[str]:
        """动态生成热门拓展池（龙虎榜 + 板块涨幅龙头 + 北向净买入）。"""
        hot = []
        today_str = date.today().strftime("%Y%m%d")

        try:
            df = ak.stock_lhb_detail_em(start_date=today_str, end_date=today_str)
            for code in df["代码"].unique()[:15]:
                normalized = _normalize_code(code)
                if normalized:
                    hot.append(normalized)
            logger.info(f"龙虎榜: {len(df)} 只上榜")
        except Exception as e:
            logger.warning(f"龙虎榜获取失败: {e}")

        try:
            df2 = ak.stock_sector_fund_flow_rank(indicator="今日")
            df2 = df2.sort_values("今日涨跌幅", ascending=False)
            top_sectors = df2.head(3)["名称"].tolist()
            logger.info(f"今日强势板块: {top_sectors}")
            for sector in top_sectors:
                try:
                    cons = ak.stock_sector_fund_flow_summary(symbol=sector, indicator="今日")
                    cons = cons.sort_values("今日涨跌幅", ascending=False)
                    code = _normalize_code(cons.iloc[0].get("代码", "")) if len(cons) else ""
                    if code:
                        hot.append(code)
                        logger.info(f"板块龙头: {sector} -> {code}")
                except Exception as sector_error:
                    logger.debug(f"板块龙头获取失败 {sector}: {sector_error}")
        except Exception as e:
            logger.warning(f"强势板块龙头获取失败: {e}")

        try:
            north = self.market.get_north_money()
            for item in north:
                hot.append(item["code"])
            logger.info(f"北向资金净买入Top10: {len(north)} 只")
        except Exception as e:
            logger.debug(f"北向资金候选获取失败: {e}")

        seen = set()
        result = []
        for c in hot:
            c = _normalize_code(c)
            if c and c not in seen and len(result) < 30:
                seen.add(c)
                result.append(c)
        logger.info(f"热门拓展池: {len(result)} 只")
        return result

    def _get_cached_pool(self, cache_key: str) -> list[str]:
        try:
            path = self.cfg.home_dir / f"{cache_key}.json"
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("codes", [])
        except Exception:
            return []

    def _cache_pool(self, cache_key: str, codes: list[str]):
        try:
            path = self.cfg.home_dir / f"{cache_key}.json"
            with open(path, "w") as f:
                json.dump({"codes": codes}, f)
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")
