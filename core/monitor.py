"""盘中轻量监控：盯关键价 + 重大新闻扫描。"""
import hashlib
from datetime import datetime

from loguru import logger

from config import get_config
from data.market import MarketData
from data.news import NewsData
from push.feishu import FeishuClient, render_text_card
from utils.storage import Storage
from analysis.sentiment import SENTIMENT_SCORE_MODEL_VERSION, score_news_items


def _is_intraday_monitor_time(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 25 <= minutes <= 11 * 60 + 30) or (13 * 60 <= minutes <= 15 * 60)


def _send_alert_card(feishu: FeishuClient | None, chat_id: str, card: dict) -> bool:
    if feishu is None:
        return True
    if chat_id:
        return feishu.send_card_to(chat_id, card, "chat_id")
    return feishu.send_message(card)


def _sell_pressure(quote: dict) -> tuple[bool, str]:
    buy_volume = float(quote.get("buy_volume_5") or 0)
    sell_volume = float(quote.get("sell_volume_5") or 0)
    outer_volume = float(quote.get("outer_volume") or 0)
    inner_volume = float(quote.get("inner_volume") or 0)
    pct = float(quote.get("pct_change") or 0)
    book_ratio = sell_volume / buy_volume if buy_volume > 0 else 0
    trade_ratio = inner_volume / outer_volume if outer_volume > 0 else 0
    heavy = (book_ratio >= 1.8 and sell_volume > 0) or (trade_ratio >= 1.5 and pct <= -0.5)

    parts = []
    if book_ratio > 0:
        parts.append(f"五档卖/买量 {book_ratio:.1f}倍")
    if trade_ratio > 0:
        parts.append(f"内/外盘 {trade_ratio:.1f}倍")
    parts.append(f"今日涨跌 {pct:+.2f}%")
    return heavy, "，".join(parts)


def _news_event_key(code: str, item: dict) -> str:
    raw = f"{code}|{item.get('ts', '')}|{item.get('title', '')}"
    return "news:" + hashlib.md5(raw.encode()).hexdigest()


def _major_news_by_title(title: str) -> bool:
    keywords = [
        "停牌", "复牌", "立案", "处罚", "减持", "增持", "回购", "重组",
        "并购", "收购", "重大合同", "中标", "业绩预告", "亏损", "暴雷",
        "退市", "风险警示", "澄清", "事故", "制裁",
    ]
    return any(word in str(title or "") for word in keywords)


def _monitor_price_alerts(storage: Storage, market: MarketData, feishu: FeishuClient | None):
    alerts = storage.get_active_price_alerts()
    if not alerts:
        return
    codes = sorted({alert["code"] for alert in alerts})
    quotes = market.get_realtime_quote(codes)
    today = datetime.now().date().isoformat()

    for alert in alerts:
        if str(alert.get("last_notified_at") or "").startswith(today):
            continue
        quote = quotes.get(alert["code"], {})
        current_price = float(quote.get("close") or 0)
        trigger_price = float(alert.get("trigger_price") or 0)
        direction = alert.get("direction") or "below"
        if current_price <= 0 or trigger_price <= 0:
            continue
        if direction == "above":
            if current_price < trigger_price:
                continue
            level = "warning"
            if not get_config().alert_level_enabled(level):
                logger.info(f"压力位提醒被 ALERT_LEVELS 过滤 {alert['code']} level={level}")
                continue
            title = "触达观察压力位"
            template = "orange"
            lines = [
                f"**{alert.get('name', alert['code'])}**({alert['code']}) 涨到 {current_price:.2f}元",
                f"观察压力位：{trigger_price:.2f}元",
                "提醒：已到你设置的压力观察位，请复核计划；这不是卖出指令。",
            ]
        else:
            if current_price > trigger_price:
                continue
            heavy_pressure, pressure_text = _sell_pressure(quote)
            level = "warning" if heavy_pressure else "info"
            if not get_config().alert_level_enabled(level):
                logger.info(f"风险价提醒被 ALERT_LEVELS 过滤 {alert['code']} level={level}")
                continue
            title = "触达风险价，卖压偏重" if heavy_pressure else "触达风险价"
            template = "orange" if heavy_pressure else "green"
            action_line = "卖压偏重，建议先复核仓位和风险计划。" if heavy_pressure else "卖压未明显放大，请按自己的计划复核。"
            lines = [
                f"**{alert.get('name', alert['code'])}**({alert['code']}) 跌到 {current_price:.2f}元",
                f"风险价/关键价：{trigger_price:.2f}元",
                f"盘口：{pressure_text}",
                f"提醒：{action_line}",
            ]
        if alert.get("quantity"):
            lines.insert(2, f"计划数量：{float(alert['quantity']):g}股")
        card = render_text_card(title, lines, template=template)
        if _send_alert_card(feishu, alert.get("chat_id", ""), card):
            storage.mark_price_alert_notified(alert["id"])


def _monitor_major_news(storage: Storage, market: MarketData, feishu: FeishuClient):
    cfg = get_config()
    alerts = storage.get_active_price_alerts()
    positions = storage.get_active_tracked_positions()
    codes = list(cfg.watchlist)
    for row in [*alerts, *positions]:
        codes.append(row["code"])
    codes = list(dict.fromkeys(codes))[:30]
    if not codes:
        return

    quotes = market.get_realtime_quote(codes)
    for code in codes:
        name = quotes.get(code, {}).get("name", code)
        try:
            news = NewsData.get_news(code, days=1)[:8]
        except Exception as e:
            logger.warning(f"重大新闻扫描失败 {code}: {e}")
            continue
        if not news:
            continue
        storage.upsert_news(code, news)
        fresh_news = [item for item in news if not storage.alert_event_exists(_news_event_key(code, item))]
        if not fresh_news:
            continue
        batch = fresh_news[:5]
        try:
            scores = score_news_items(batch)
        except Exception as e:
            logger.warning(f"重大新闻打分失败 {code}: {e}")
            continue
        if len(scores) < len(batch):
            scores.extend([0.0] * (len(batch) - len(scores)))

        for item, score in zip(batch, scores):
            key = _news_event_key(code, item)
            title = str(item.get("title", ""))
            storage.update_news_sentiment(
                code, title, item.get("ts", ""), score,
                model_version=SENTIMENT_SCORE_MODEL_VERSION,
            )
            is_major = abs(score) >= 0.8 or _major_news_by_title(title)
            if not is_major:
                storage.mark_alert_event(key, "news_seen", code, title)
                continue

            direction = "正面" if score > 0.08 else "负面" if score < -0.08 else "中性"
            template = "red" if score < -0.08 else "green" if score > 0.08 else "blue"
            level = "critical" if score < -0.08 else "info"
            if not cfg.alert_level_enabled(level):
                storage.mark_alert_event(key, "news_filtered", code, title)
                logger.info(f"重大新闻提醒被 ALERT_LEVELS 过滤 {code} level={level}")
                continue
            card = render_text_card("个股重大消息提醒", [
                f"**{name}**({code}) {direction}消息 {score:+.2f}",
                f"来源：{item.get('source', '未知')}",
                f"标题：{title}",
                f"时间：{item.get('ts', '')}",
            ], template=template)
            if feishu.send_message(card):
                storage.mark_alert_event(key, "major_news", code, title)


def monitor_once(check_news: bool = False):
    """盘中轻量监控：盯关键价 + 可选重大新闻扫描。"""
    cfg = get_config()
    storage = Storage()
    market = MarketData()
    feishu = FeishuClient() if cfg.notify_channel == "feishu" else None
    if feishu is None:
        logger.info("NOTIFY_CHANNEL=web，盘中轻量监控会记录触价状态并跳过飞书推送")
    _monitor_price_alerts(storage, market, feishu)
    if check_news and feishu:
        _monitor_major_news(storage, market, feishu)
