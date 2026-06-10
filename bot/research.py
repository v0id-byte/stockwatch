"""Natural-language stock research answers for the Feishu bot."""
from __future__ import annotations

import contextlib
import io
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

from loguru import logger

import akshare as ak

from analysis.technical import compute_tech_score
from data.market import MarketData
from data.news import NewsData
from utils.llm import get_llm_client
from utils.storage import Storage


_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_REORG_WORDS = ("重组", "并购", "吸收合并", "资产注入", "资产出售", "停牌", "复牌", "定增")


@dataclass
class StockRef:
    code: str
    name: str


def _ak_call(fn, *args, **kwargs):
    """Silence tqdm/progress output from AKShare during bot replies."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def _clean_text(value) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).upper()
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[\s`~!@#$%^&*()_+\-=\[\]{};:'\",.<>/?\\|，。！？、：；（）【】《》]+", "", text)


def _display_text(value, limit: int = 120) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    text = " ".join(text.split())
    return text[:limit]


def _date_yyyymmdd(days_ago: int = 0) -> str:
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")


@lru_cache(maxsize=1)
def _stock_name_table() -> list[StockRef]:
    df = _ak_call(ak.stock_info_a_code_name)
    rows = []
    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        if re.fullmatch(r"\d{6}", code) and name:
            rows.append(StockRef(code=code, name=name))
    return rows


def _quote_name(code: str, market: MarketData) -> str:
    quote = market.get_realtime_quote([code]).get(code, {})
    return str(quote.get("name") or "").strip()


def resolve_stock(text: str, market: MarketData) -> StockRef | None:
    code_match = _CODE_RE.search(text or "")
    if code_match:
        code = code_match.group(1)
        return StockRef(code=code, name=_quote_name(code, market) or code)

    cleaned_query = _clean_text(text)
    if not cleaned_query:
        return None

    matches = []
    try:
        for item in _stock_name_table():
            cleaned_name = _clean_text(item.name)
            if len(cleaned_name) >= 2 and cleaned_name in cleaned_query:
                matches.append((len(cleaned_name), item))
    except Exception as e:
        logger.warning(f"股票名称表获取失败: {e}")
        return None

    if not matches:
        return None
    matches.sort(key=lambda row: row[0], reverse=True)
    stock = matches[0][1]
    return StockRef(code=stock.code, name=_quote_name(stock.code, market) or stock.name)


def _has_reorg_intent(question: str) -> bool:
    return any(word in question for word in _REORG_WORDS)


def _announcement_from_cninfo(row: dict) -> dict:
    return {
        "source": "巨潮资讯公告",
        "date": _display_text(row.get("公告时间", "")),
        "title": _display_text(row.get("公告标题", ""), 160),
        "url": str(row.get("公告链接", "")),
    }


def _announcement_from_eastmoney(row: dict) -> dict:
    return {
        "source": "东方财富公告大全",
        "date": _display_text(row.get("公告日期", "")),
        "title": _display_text(row.get("公告标题", ""), 160),
        "url": str(row.get("网址", "")),
    }


def _dedupe_items(items: list[dict], limit: int) -> list[dict]:
    seen = set()
    unique = []
    for item in items:
        key = (item.get("date", ""), item.get("title", ""))
        if not item.get("title") or key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= limit:
            break
    return unique


def get_announcements(code: str, question: str) -> list[dict]:
    recent_items = []
    reorg_items = []
    end = _date_yyyymmdd()
    recent_start = _date_yyyymmdd(45)

    try:
        df = _ak_call(
            ak.stock_zh_a_disclosure_report_cninfo,
            symbol=code,
            start_date=recent_start,
            end_date=end,
        )
        recent_items.extend(_announcement_from_cninfo(dict(row)) for _, row in df.head(6).iterrows())
    except Exception as e:
        logger.warning(f"近期公告获取失败 {code}: {e}")

    if _has_reorg_intent(question):
        for keyword in ("重组", "吸收合并", "资产出售"):
            try:
                df = _ak_call(
                    ak.stock_zh_a_disclosure_report_cninfo,
                    symbol=code,
                    keyword=keyword,
                    start_date="20200101",
                    end_date=end,
                )
                reorg_items.extend(_announcement_from_cninfo(dict(row)) for _, row in df.head(5).iterrows())
            except Exception as e:
                logger.debug(f"重组公告获取失败 {code} {keyword}: {e}")
        if not reorg_items:
            try:
                df = _ak_call(ak.stock_individual_notice_report, security=code, symbol="资产重组")
                reorg_items.extend(_announcement_from_eastmoney(dict(row)) for _, row in df.head(5).iterrows())
            except Exception as e:
                logger.debug(f"东方财富重组公告获取失败 {code}: {e}")

    items = reorg_items + recent_items
    return _dedupe_items(items, 10)


def get_recent_news(code: str, days: int = 14) -> list[dict]:
    items = []
    for row in NewsData.get_news(code, days=days)[:8]:
        items.append({
            "source": _display_text(row.get("source", "新闻"), 40),
            "date": _display_text(row.get("ts", "")),
            "title": _display_text(row.get("title", ""), 160),
        })
    return items


def _ensure_kline(code: str, market: MarketData, storage: Storage) -> list[dict]:
    if not storage.kline_cached_today(code):
        for row in market.get_daily_kline(code):
            storage.upsert_kline(code, row["trade_date"], row)
    return storage.get_kline(code, "2020-01-01", datetime.now().strftime("%Y-%m-%d"))


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def build_trend_summary(kline: list[dict], quote: dict) -> dict:
    if len(kline) < 6:
        return {"summary": "K线数据不足，无法计算最近一周走势。", "levels": {}, "tech": {}}

    recent = kline[-6:]
    first_close = float(recent[0]["close"])
    last_close = float(recent[-1]["close"])
    week_pct = (last_close / first_close - 1) * 100 if first_close > 0 else 0.0
    highs = [float(row["high"]) for row in recent]
    lows = [float(row["low"]) for row in recent]
    closes = [float(row["close"]) for row in recent]
    volumes = [float(row.get("volume") or 0) for row in recent]
    prev_volumes = [float(row.get("volume") or 0) for row in kline[-11:-6]]
    avg_recent_vol = sum(volumes[1:]) / max(len(volumes[1:]), 1)
    avg_prev_vol = sum(prev_volumes) / max(len(prev_volumes), 1) if prev_volumes else 0
    vol_ratio = avg_recent_vol / avg_prev_vol if avg_prev_vol > 0 else 0

    latest = float(quote.get("close") or last_close)
    latest_pct = float(quote.get("pct_change") or 0)
    tech = compute_tech_score(kline)
    details = tech.get("details", {})
    support = min(float(row["low"]) for row in kline[-10:])
    resistance = max(float(row["high"]) for row in kline[-10:])

    return {
        "summary": (
            f"最近5个交易日从 {first_close:.2f} 到 {last_close:.2f}，累计 {_pct(week_pct)}；"
            f"区间高点 {max(highs):.2f}，低点 {min(lows):.2f}；"
            f"当前参考价 {latest:.2f}，今日涨跌 {_pct(latest_pct)}；"
            f"近5日均量较前5日 {'放大' if vol_ratio >= 1.15 else '缩小' if vol_ratio <= 0.85 else '基本持平'}。"
        ),
        "levels": {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "last_close": round(last_close, 2),
            "week_pct": round(week_pct, 2),
        },
        "tech": {
            "score": tech.get("score", 0),
            "ma5": details.get("ma5"),
            "ma10": details.get("ma10"),
            "ma20": details.get("ma20"),
            "macd_bar": details.get("macd_bar"),
            "rsi14": details.get("rsi14"),
            "boll_position": details.get("boll_position"),
            "vol_ratio": round(vol_ratio, 2) if math.isfinite(vol_ratio) else 0,
        },
        "closes": [
            {
                "date": row["trade_date"],
                "close": float(row["close"]),
                "pct": round((float(row["close"]) / closes[i - 1] - 1) * 100, 2) if i > 0 and closes[i - 1] > 0 else 0,
            }
            for i, row in enumerate(recent)
        ],
    }


def _format_sources(items: list[dict], prefix: str) -> str:
    if not items:
        return f"{prefix}: 暂无"
    lines = []
    for i, item in enumerate(items[:8], 1):
        label = item.get("label") or f"{prefix}{i}"
        lines.append(f"- [{label}] {item.get('date', '')} {item.get('source', '')}: {item.get('title', '')}")
    return "\n".join(lines)


def _label_items(items: list[dict], prefix: str) -> list[dict]:
    labeled = []
    for i, item in enumerate(items, 1):
        copied = dict(item)
        copied["label"] = f"{prefix}{i}"
        labeled.append(copied)
    return labeled


def answer_stock_question(question: str, stock: StockRef, market: MarketData, storage: Storage) -> str:
    quote = market.get_realtime_quote([stock.code]).get(stock.code, {})
    name = str(quote.get("name") or stock.name or stock.code)
    kline = _ensure_kline(stock.code, market, storage)
    trend = build_trend_summary(kline, quote)
    announcements = _label_items(get_announcements(stock.code, question)[:8], "公告")
    news = _label_items(get_recent_news(stock.code)[:6], "新闻")

    data_pack = {
        "question": question,
        "stock": {"code": stock.code, "name": name},
        "trend": trend,
        "announcements": announcements,
        "news": news,
        "source_priority": "公告优先级: 巨潮资讯公告 > 东方财富公告大全 > 个股新闻。",
    }
    system_prompt = (
        "你是给非专业家人使用的 A 股研究助手。只根据用户问题和提供的数据回答，"
        "不要编造未提供的公告、新闻、价格或结论。可以用均线、MACD、RSI、布林位置等指标佐证，"
        "但必须翻译成普通人能理解的话。不要给确定性买卖指令，改用“偏向建议 + 风险提醒 + 观察价位”。"
    )
    user_prompt = f"""请用中文回答下面的股票问题。

固定结构：
1. 先给一句“结论”
2. “消息/公告”：优先引用公告来源，重组问题要说明最新状态和关键节点
3. “最近一周走势”：说明涨跌、区间高低、量能、技术状态
4. “偏向建议”：用谨慎措辞，给观察价位/支撑压力，不要承诺收益
5. “主要风险”：列 2-4 条
不要使用 Markdown 表格或分隔线，飞书卡片里只用短段落和项目符号。

引用要求：
- 公告和新闻必须使用数据里 label 字段对应的编号，比如 [公告1]、[新闻1]。
- 如果数据里没有对应来源，要明确说“未查到近期对应公告/新闻”。

数据：
{json.dumps(data_pack, ensure_ascii=False, indent=2)}
"""
    client = get_llm_client()
    answer = client.chat([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt[:9000]},
    ], temperature=0.2, max_tokens=1800)
    answer = client._strip_think(answer).strip()
    if not answer:
        raise RuntimeError("模型没有返回分析内容")
    source_notes = [
        "资料来源",
        _format_sources(announcements, "公告"),
    ]
    if news:
        source_notes.append(_format_sources(news[:3], "新闻"))
    return answer + "\n\n" + "\n".join(source_notes)
