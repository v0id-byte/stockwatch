"""分析池构建：自选股 + 热门拓展池"""
from datetime import date, datetime
from loguru import logger

import akshare as ak

from config import get_config
from data.market import MarketData

# 简单节假日（周末已跳过，最简化处理）
_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-02", "2026-01-26", "2026-01-27", "2026-01-28",
    "2026-01-29", "2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02",
    "2026-04-03", "2026-04-04", "2026-04-05", "2026-04-06",
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    "2026-06-22", "2026-06-23", "2026-06-24", "2026-06-25",
    "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01",
    "2026-10-02", "2026-10-03", "2026-10-04", "2026-10-05", "2026-10-06",
    "2026-10-07", "2026-10-08",
}


class Universe:
    def __init__(self, storage=None):
        self.cfg = get_config()
        self.storage = storage
        self.market = MarketData()

    def is_trading_day(self) -> bool:
        """判断今天是否交易日（周末 + 节假日）"""
        today = date.today()
        if today.weekday() >= 5:  # 周六周日
            return False
        if today.isoformat() in _HOLIDAYS_2026:
            return False
        return True

    def get_today_codes(self) -> list[str]:
        """返回今日分析池 code 列表（上限50只）"""
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
        """动态生成热门拓展池（龙虎榜 + 涨幅榜龙头）"""
        hot = []
        today_str = date.today().strftime("%Y%m%d")

        # 龙虎榜（近1日）
        try:
            df = ak.stock_lhb_detail_em(start_date=today_str, end_date=today_str)
            for code in df["代码"].unique()[:15]:
                hot.append(str(code).zfill(6))
            logger.info(f"龙虎榜: {len(df)} 只上榜")
        except Exception as e:
            logger.warning(f"龙虎榜获取失败: {e}")

        # 涨幅前排龙头（用 stock_sector_fund_flow_rank 的涨跌幅）
        try:
            df2 = ak.stock_sector_fund_flow_rank(indicator="今日")
            df2 = df2.sort_values("今日涨跌幅", ascending=False)
            top_sectors = df2.head(5)["名称"].tolist()
            logger.info(f"今日强势板块: {top_sectors[:3]}")
            # 每个板块取龙头股（简化处理，直接用 hot_rank_em）
            df3 = ak.stock_hot_rank_em()
            df3 = df3.sort_values("涨跌幅", ascending=False)
            for _, row in df3.head(15).iterrows():
                raw = str(row["代码"])
                code = raw[2:]  # 去掉 SH/SZ 前缀
                hot.append(code.zfill(6))
        except Exception as e:
            logger.warning(f"热门榜获取失败: {e}")

        # 北向（用 stock_hsgt_fund_flow_summary_em 看大盘情绪）
        try:
            df4 = ak.stock_hsgt_fund_flow_summary_em()
            logger.info(f"北向资金今日: {df4.iloc[0] if len(df4)>0 else 'N/A'}")
        except Exception as e:
            logger.debug(f"北向资金获取失败: {e}")

        # 去重 + 限制
        seen = set()
        result = []
        for c in hot:
            c = c.zfill(6)
            if c not in seen and len(result) < 30:
                seen.add(c)
                result.append(c)
        logger.info(f"热门拓展池: {len(result)} 只")
        return result

    def _get_cached_pool(self, cache_key: str) -> list[str]:
        try:
            import json as _json
            path = self.cfg.home_dir / f"{cache_key}.json"
            with open(path, "r") as f:
                data = _json.load(f)
            return data.get("codes", [])
        except Exception:
            return []

    def _cache_pool(self, cache_key: str, codes: list[str]):
        try:
            import json as _json
            path = self.cfg.home_dir / f"{cache_key}.json"
            with open(path, "w") as f:
                _json.dump({"codes": codes}, f)
        except Exception as e:
            logger.warning(f"缓存写入失败: {e}")
