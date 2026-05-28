"""新闻 + 公告 + 财联社电报"""
import hashlib
from datetime import datetime, timedelta, date
from loguru import logger

import akshare as ak


class NewsData:
    @staticmethod
    def _hash_item(title: str, content: str, ts: str) -> str:
        return hashlib.md5(f"{title}|{content}|{ts}".encode()).hexdigest()

    @staticmethod
    def get_news(code: str, days: int = 7) -> list[dict]:
        """获取近 days 天个股新闻"""
        results = []
        try:
            df = ak.stock_news_em(symbol=code)
            for _, row in df.iterrows():
                try:
                    dt = datetime.strptime(str(row.get("发布时间", "")), "%Y-%m-%d %H:%M:%S")
                except Exception:
                    dt = datetime.now()
                results.append({
                    "title": str(row.get("新闻标题", ""))[:200],
                    "content": str(row.get("新闻内容", ""))[:1000],
                    "source": str(row.get("文章来源", "东方财富")),
                    "ts": dt.isoformat(),
                })
        except Exception as e:
            logger.warning(f"东财新闻获取失败 {code}: {e}")

        # 去重
        seen = set()
        unique = []
        for item in results:
            h = NewsData._hash_item(item["title"], item["content"], item["ts"])
            if h not in seen:
                seen.add(h)
                unique.append(item)
        return unique[:30]

    @staticmethod
    def get_telegraph() -> list[dict]:
        """获取财联社/央视电报（大盘情绪用）"""
        items = []
        try:
            df = ak.news_cctv()
            for _, row in df.iterrows():
                try:
                    dt = datetime.strptime(str(row.get("date", "")), "%Y-%m-%d")
                except Exception:
                    dt = datetime.now()
                items.append({
                    "title": str(row.get("title", ""))[:200],
                    "content": str(row.get("content", ""))[:500],
                    "source": "央视新闻",
                    "ts": dt.isoformat(),
                })
        except Exception as e:
            logger.warning(f"财联社/央视电报获取失败: {e}")
        return items[:20]
