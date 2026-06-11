"""Main entry: test / once / daemon"""
import sys
import os
import time
import uuid
import hashlib
from datetime import datetime, timedelta

# 确保从 stockwatch 目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import get_config
from utils.storage import Storage
from utils.llm import get_llm_client, get_token_usage, reset_token_usage
from data.market import MarketData
from data.news import NewsData
from data.universe import Universe
from analysis.technical import compute_tech_score
from analysis.sentiment import batch_sentiment_details, score_news_items
from decision.engine import DecisionEngine
from push.feishu import FeishuClient, render_card, render_text_card


ALERT_LEVEL_LABELS = {
    "critical": "红色/必须看",
    "warning": "橙色/建议看",
    "info": "蓝色/仅记录",
}


def _setup_log():
    cfg = get_config()
    logger.remove()
    logger.add(
        cfg.log_dir / "stockwatch_{time}.log",
        rotation="00:00",
        retention="7 days",
        level=cfg.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )


def _format_north_context(items: list[dict]) -> str:
    if not items:
        return "未知"
    parts = [
        f"{item['name']}({item['code']}) {item['net_buy']:.2f}万"
        for item in items[:5]
    ]
    return "北向净买入Top5: " + "、".join(parts)


def _tracked_alert_reason(position: dict, decision: dict) -> str:
    if not position:
        return ""
    today = datetime.now().date().isoformat()
    last_notified = str(position.get("last_notified_at") or "")
    if last_notified.startswith(today):
        return ""
    current_price = float(decision.get("_current_price") or 0)
    stop_loss = float(position.get("stop_loss") or decision.get("stop_loss") or 0)
    target_price = float(position.get("target_price") or decision.get("target_price") or 0)
    if stop_loss > 0 and current_price > 0 and current_price <= stop_loss:
        return f"持仓跟踪：现价 {current_price:.2f} 跌破止损 {stop_loss:.2f}"
    if target_price > 0 and current_price > 0 and current_price >= target_price * 0.98:
        return f"持仓跟踪：现价 {current_price:.2f} 接近目标 {target_price:.2f}"
    if decision.get("action") == "SELL" and decision.get("confidence", 0) >= 0.6:
        return "持仓跟踪：模型转为 SELL，建议复核仓位"
    return ""


def _decision_alert_level(decision: dict) -> str:
    action = str(decision.get("action") or "HOLD").upper()
    confidence = float(decision.get("confidence") or 0)
    one_liner = str(decision.get("one_liner") or "")
    if "跌破止损" in one_liner or "模型转为 SELL" in one_liner:
        return "critical"
    if action == "SELL" and confidence >= 0.75:
        return "critical"
    if action in {"BUY", "SELL"}:
        return "warning"
    return "info"


def _is_after_close_summary_time(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    return now.hour * 60 + now.minute >= 15 * 60


def _after_close_summary_key(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return f"after_close_summary:{now.date().isoformat()}"


def _format_family_watch_line(decisions: list[dict]) -> str:
    if not decisions:
        return "给家人看的结论：今天没有必须盯着看的股票，收盘后看这条就够了。"
    hold_count = sum(1 for d in decisions if d.get("action") == "HOLD")
    return f"给家人看的结论：今天已检查 {len(decisions)} 只，其中 {hold_count} 只以观察为主，不需要一直盯盘。"


def _send_after_close_summary(storage: Storage, feishu: FeishuClient,
                              decisions: list[dict], pushed_alerts: list[dict],
                              tracked_positions: list[dict], price_alerts: list[dict]) -> bool:
    cfg = get_config()
    if not (cfg.enable_after_close_summary or cfg.enable_reassurance_mode):
        return False
    if pushed_alerts or not _is_after_close_summary_time():
        return False
    key = _after_close_summary_key()
    if storage.alert_event_exists(key):
        return False

    watch_lines = [_format_family_watch_line(decisions)] if cfg.enable_family_brief else [
        f"今日已检查 {len(decisions)} 只，未触发需要立刻处理的提醒。"
    ]
    observe = []
    for item in decisions[:3]:
        name = item.get("name") or item.get("code")
        line = item.get("one_liner") or item.get("tech_summary") or "继续观察"
        observe.append(f"{name}({item.get('code')})：{line}")
    if not observe:
        observe.append("暂无需要单独关注的标的。")

    card = render_text_card("今日不用盯盘", [
        *watch_lines,
        f"持仓跟踪：{len(tracked_positions)} 只；盯价提醒：{len(price_alerts)} 条。",
        "系统会继续盯价格、公告、重大消息和模型转弱信号。",
        "明日观察：",
        *observe,
        "*仅供研究参考，不构成投资建议*",
    ], template="green")
    ok = feishu.send_message(card)
    if ok:
        storage.mark_alert_event(key, "after_close_summary", "", "今日不用盯盘")
    return ok


def _is_intraday_monitor_time(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return (9 * 60 + 25 <= minutes <= 11 * 60 + 30) or (13 * 60 <= minutes <= 15 * 60)


def _send_alert_card(feishu: FeishuClient, chat_id: str, card: dict) -> bool:
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


def _monitor_price_alerts(storage: Storage, market: MarketData, feishu: FeishuClient):
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
        if current_price <= 0 or trigger_price <= 0 or current_price > trigger_price:
            continue

        heavy_pressure, pressure_text = _sell_pressure(quote)
        level = "warning" if heavy_pressure else "info"
        if not get_config().alert_level_enabled(level):
            logger.info(f"盯价提醒被 ALERT_LEVELS 过滤 {alert['code']} level={level}")
            continue
        title = "触发加仓价，但卖压偏重" if heavy_pressure else "触发加仓价"
        template = "orange" if heavy_pressure else "green"
        action_line = "建议先别急着加，已挂买单可考虑撤单观察。" if heavy_pressure else "卖压未明显放大，可按计划关注加仓。"
        lines = [
            f"**{alert.get('name', alert['code'])}**({alert['code']}) 跌到 {current_price:.2f}元",
            f"盯价：{trigger_price:.2f}元",
            f"盘口：{pressure_text}",
            f"建议：{action_line}",
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
    codes = []
    for code in cfg.watchlist:
        codes.append(code)
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
            storage.update_news_sentiment(code, title, item.get("ts", ""), score)
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
    """盘中轻量监控：盯加仓价 + 可选重大新闻扫描。"""
    storage = Storage()
    market = MarketData()
    feishu = FeishuClient()
    _monitor_price_alerts(storage, market, feishu)
    if check_news:
        _monitor_major_news(storage, market, feishu)


def demo(query: str = ""):
    """无需飞书推送的终端体验入口；MiniMax 配好时输出完整问答，否则降级为规则快照。"""
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    cfg = get_config()
    storage = Storage()
    market = MarketData()
    query = (query or "600519 最近一周走势如何").strip()

    from bot.research import (
        answer_market_question,
        answer_stock_question,
        build_market_snapshot,
        format_market_snapshot,
        format_stock_snapshot,
        resolve_stock,
    )

    stock = resolve_stock(query, market)
    try:
        if stock and cfg.minimax_api_key:
            answer = answer_stock_question(query, stock, market, storage)
        elif stock:
            answer = format_stock_snapshot(query, stock, market, storage, include_sources=False)
        elif cfg.minimax_api_key:
            answer = answer_market_question(query, market, storage)
        else:
            answer = format_market_snapshot(
                build_market_snapshot(
                    market, storage,
                    include_news=False, include_north=False, include_regime=False,
                )
            )
    except Exception as e:
        logger.warning(f"demo 完整问答失败，降级输出快照: {e}")
        if stock:
            answer = format_stock_snapshot(query, stock, market, storage, include_sources=False)
        else:
            answer = format_market_snapshot(
                build_market_snapshot(
                    market, storage,
                    include_news=False, include_north=False, include_regime=False,
                )
            )

    print("=" * 40)
    print(f"StockWatch demo: {query}")
    print("=" * 40)
    print(answer)


def test():
    """测试 AKShare / MiniMax / 飞书 连接"""
    print("=" * 40)
    print("StockWatch 自检")
    print("=" * 40)

    # 1. AKShare
    try:
        data = MarketData()
        quotes = data.get_realtime_quote(["600519"])
        name = quotes.get("600519", {}).get("name", "未知")
        print(f"✅ AKShare 连接成功 → {name}")
    except Exception as e:
        print(f"❌ AKShare 失败: {e}")

    # 2. MiniMax
    try:
        llm = get_llm_client()
        result = llm.chat([{"role": "user", "content": "Hello, reply OK"}])
        print(f"✅ MiniMax 连接成功")
    except Exception as e:
        print(f"❌ MiniMax 失败: {e}")

    # 3. 飞书
    try:
        feishu = FeishuClient()
        token = feishu._get_token()
        print(f"✅ 飞书连接成功")
    except Exception as e:
        print(f"❌ 飞书失败: {e}")

    # 4. v2 模块冷启动检查
    try:
        cfg = get_config()
        storage = Storage()
        model = storage.get_latest_calibration_model("BUY")
        print(f"✅ v2 calibration 表可读 → {'已有模型' if model else '暂无模型'}")
        from analysis.factors import compute_alpha158
        import pandas as pd
        dates = pd.date_range("2026-01-01", periods=70).strftime("%Y-%m-%d")
        demo = pd.DataFrame([
            {"trade_date": dates[i], "open": 10+i*0.1, "high": 10.5+i*0.1,
             "low": 9.8+i*0.1, "close": 10.2+i*0.1, "volume": 100000+i*1000, "amount": 0}
            for i in range(70)
        ])
        factors = compute_alpha158(demo, demo)
        print(f"✅ Alpha158 可计算 → {len(factors)} 个因子")
        from analysis.lgbm import LgbmRanker
        ranker = LgbmRanker(cfg.lgbm_model_path)
        print(f"✅ LGBM 检查完成 → {'已加载' if ranker.model else '未加载'}")
        if cfg.enable_regime:
            from analysis.regime import get_market_regime
            regime = get_market_regime(data, storage)
            print(f"✅ regime 可获取 → {regime['regime']}")
        else:
            print("✅ regime 检查完成 → 未启用")
        if cfg.enable_sector:
            with storage._conn() as conn:
                count = conn.execute("SELECT COUNT(*) FROM stock_sector_map").fetchone()[0]
            print(f"✅ sector 映射表可读 → {count} 条缓存")
        else:
            print("✅ sector 检查完成 → 未启用")
    except Exception as e:
        print(f"❌ v2 模块检查失败: {e}")

    print("=" * 40)


def once():
    """立即跑一次完整流程"""
    _setup_log()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    logger.info(f"===== 开始运行 {run_id} =====")

    cfg = get_config()
    storage = Storage()
    market = MarketData()
    universe = Universe(storage)
    reset_token_usage()

    # 检查交易日
    if not universe.is_trading_day():
        logger.info("今日非交易日，跳过")
        return

    # 1. 构建分析池
    codes = universe.get_today_codes()
    tracked_positions = storage.get_active_tracked_positions()
    price_alerts = storage.get_active_price_alerts()
    tracked_by_code = {pos["code"]: pos for pos in tracked_positions}
    for row in [*tracked_positions, *price_alerts]:
        if row["code"] not in codes:
            codes.append(row["code"])
    logger.info(f"分析池: {len(codes)} 只")
    if not codes:
        logger.info("分析池为空，结束")
        return

    # 2. 获取实时报价
    quotes = market.get_realtime_quote(codes)
    logger.info(f"实时报价获取: {len(quotes)} 只")
    north_context = _format_north_context(market.get_north_money())
    sht_pct = market.get_index_pct()
    logger.info(f"大盘上下文: {north_context}; 上证涨跌 {sht_pct:.2f}%")

    regime_info = {
        "regime": "normal",
        "vol_20d": 0.0,
        "percentile": 0.0,
        "confidence_floor": cfg.min_confidence_to_push,
        "context": "大盘 regime: normal (未启用)",
    }
    if cfg.enable_regime:
        try:
            from analysis.regime import get_market_regime
            regime_info = get_market_regime(market, storage)
        except Exception as e:
            logger.warning(f"波动率 regime 获取失败，使用默认值: {e}")
    confidence_floor = max(cfg.min_confidence_to_push, regime_info.get("confidence_floor", cfg.min_confidence_to_push))

    sector_contexts = {}
    if cfg.enable_sector:
        try:
            from analysis.sector import get_sector_contexts
            sector_contexts = get_sector_contexts(codes, market, storage)
        except Exception as e:
            logger.warning(f"板块强弱获取失败，跳过: {e}")

    factor_needed = cfg.enable_alpha158 or cfg.enable_lgbm
    market_df = None
    if factor_needed:
        try:
            import pandas as pd
            market_df = pd.DataFrame(market.get_index_kline("sh000001", limit=320))
        except Exception as e:
            logger.warning(f"指数K线获取失败，Alpha/LGBM 将使用空大盘上下文: {e}")

    # 3. 技术分析
    llm_calls = 0
    decisions = []

    for code in codes:
        name = quotes.get(code, {}).get("name", code)
        logger.info(f"分析: {code}({name})")

        # K 线缓存
        if not storage.kline_cached_today(code):
            kline = market.get_daily_kline(code)
            for row in kline:
                storage.upsert_kline(code, row["trade_date"], row)

        kline = storage.get_kline(code, "2020-01-01", datetime.now().strftime("%Y-%m-%d"))
        if len(kline) < 20:
            logger.warning(f"K线不足，跳过 {code}")
            continue

        # 技术分
        tech_result = compute_tech_score(kline)
        tech_score = tech_result["score"]

        # 保存技术分用于调试
        logger.debug(f"{code} 技术分: {tech_score} {tech_result.get('details', {})}")

        # 4. 情绪分（并发）
        # 这里先记录，等待 batch_sentiment_details
        decisions.append({
            "code": code, "name": name,
            "tech_score": tech_score,
            "tech_details": tech_result.get("details", {}),
            "kline": kline,
            "quote": quotes.get(code, {}),
        })
        if factor_needed:
            try:
                from analysis.factors import compute_alpha158
                decisions[-1]["factors"] = compute_alpha158(pd.DataFrame(kline), market_df)
            except Exception as e:
                logger.warning(f"Alpha158 计算失败 {code}: {e}")
                decisions[-1]["factors"] = {}

    alpha_contexts = {}
    lgbm_contexts = {}
    factor_map = {d["code"]: d.get("factors", {}) for d in decisions if d.get("factors")}
    if cfg.enable_alpha158 and factor_map:
        try:
            from analysis.factors import summarize_alpha158_cross_section
            alpha_contexts = summarize_alpha158_cross_section(factor_map)
        except Exception as e:
            logger.warning(f"Alpha158 摘要生成失败: {e}")
    if cfg.enable_lgbm and factor_map:
        try:
            from analysis.lgbm import LgbmRanker, format_lgbm_context
            ranker = LgbmRanker(cfg.lgbm_model_path)
            scores = {code: ranker.predict(factors) for code, factors in factor_map.items()}
            lgbm_contexts = format_lgbm_context(scores)
        except Exception as e:
            logger.warning(f"LightGBM 推理失败，跳过: {e}")

    # 并发情绪分析（最多30只，减少 token 消耗）
    batch = [(d["code"], d["name"]) for d in decisions[:30]]
    sentiment_details = batch_sentiment_details(batch, storage=storage)
    for d in decisions:
        detail = sentiment_details.get(d["code"], {})
        d["sentiment"] = detail.get("score", 0.0)
        d["sentiment_context"] = detail.get("context", "")

    # 5. 最终决策
    engine = DecisionEngine(storage)
    all_alert_results = []
    run_results = []
    final_decisions = []

    for d in decisions:
        decision = engine.decide_one(
            d["code"], d["name"],
            d["tech_score"], d["sentiment"],
            d["kline"],
            tech_details=d.get("tech_details"),
            sentiment_context=d.get("sentiment_context", ""),
            north_context=north_context,
            sht_pct=sht_pct,
            alpha_summary=alpha_contexts.get(d["code"], ""),
            lgbm_context=lgbm_contexts.get(d["code"], ""),
            regime_context=regime_info.get("context", "大盘 regime: normal"),
            sector_context=sector_contexts.get(d["code"], "所属板块: 未知"),
            confidence_floor=confidence_floor,
        )
        tracked_reason = _tracked_alert_reason(tracked_by_code.get(d["code"]), decision)
        if tracked_reason:
            decision["one_liner"] = tracked_reason[:50]
            decision["_will_push"] = True
            storage.mark_position_notified(tracked_by_code[d["code"]]["id"])
        if decision.get("_will_push"):
            level = _decision_alert_level(decision)
            decision["_alert_level"] = level
            all_alert_results.append(decision)
            if cfg.alert_level_enabled(level):
                run_results.append(decision)
            else:
                logger.info(
                    f"决策提醒被 ALERT_LEVELS 过滤 {decision['code']} "
                    f"level={level}({ALERT_LEVEL_LABELS.get(level, level)})"
                )
        final_decisions.append(decision)
        storage.insert_decision(run_id, datetime.now().isoformat(), decision)
        llm_calls += 1

    # 6. 推送
    feishu = FeishuClient()
    push_ok = False
    push_status = "无可推送信号"
    if run_results:
        card = render_card(run_id, run_results, regime_info=regime_info if cfg.enable_regime else None)
        if card:
            push_ok = feishu.send_message(card)
            push_status = "成功" if push_ok else "失败"
            for d in run_results:
                storage.mark_decision_pushed(run_id, d["code"], push_ok)
                if not push_ok:
                    break
        else:
            push_status = "无可展示信号"
    if not all_alert_results:
        summary_ok = _send_after_close_summary(
            storage, feishu, final_decisions, run_results,
            tracked_positions, price_alerts,
        )
        if summary_ok:
            push_ok = True
            push_status = "安心总结成功"

    # 7. 记录统计
    tokens_used = get_token_usage()
    storage.insert_run(run_id, {
        "stocks_analyzed": len(decisions),
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "pushed_count": len(run_results),
        "pushed_ok": 1 if push_ok else 0,
    })

    logger.info(
        f"===== 运行结束 {run_id} =====\n"
        f"  分析: {len(decisions)} 只 | LLM调用: {llm_calls} 次 | "
        f"推送: {push_status} ({len(run_results)} 条)"
    )

    print(f"\n完成！分析 {len(decisions)} 只，推送 {push_status}\n")


def daemon():
    """守护进程模式，按调度时间自动运行"""
    import sched

    _setup_log()
    logger.info("StockWatch 守护进程启动")

    scheduler = sched.scheduler(time.time, time.sleep)
    last_news_check = {"at": datetime.min}

    def _run_full():
        try:
            once()
        except Exception as e:
            logger.error(f"运行异常: {e}")
        _schedule_next_full(scheduler)

    def _run_monitor():
        try:
            now = datetime.now()
            if _is_intraday_monitor_time(now):
                check_news = (now - last_news_check["at"]).total_seconds() >= 1800
                monitor_once(check_news=check_news)
                if check_news:
                    last_news_check["at"] = now
        except Exception as e:
            logger.error(f"盘中监控异常: {e}")
        _schedule_next_monitor(scheduler)

    def _schedule_next_full(sched_obj):
        now = datetime.now()
        targets = [
            (9, 10),   # 早盘
            (12, 30),  # 午间
            (15, 15),  # 收盘
        ]
        candidates = []
        for target_h, target_m in targets:
            next_run = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
            if next_run <= now:
                next_run = next_run + timedelta(days=1)
            candidates.append(next_run)

        next_run = min(candidates)
        delay = (next_run - now).total_seconds()
        logger.info(f"下次完整分析: {next_run.strftime('%Y-%m-%d %H:%M')}, 约 {int(delay//60)} 分钟后")
        sched_obj.enter(delay, 1, _run_full, ())

    def _schedule_next_monitor(sched_obj):
        sched_obj.enter(300, 2, _run_monitor, ())

    _schedule_next_full(scheduler)
    scheduler.enter(10, 2, _run_monitor, ())
    scheduler.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [test|once|daemon|monitor|bot|demo|dashboard|report]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "test":
        test()
    elif mode == "once":
        once()
    elif mode == "daemon":
        daemon()
    elif mode == "monitor":
        _setup_log()
        monitor_once(check_news=True)
    elif mode == "bot":
        _setup_log()
        from bot.runner import run_bot
        run_bot()
    elif mode == "demo":
        demo(" ".join(sys.argv[2:]))
    elif mode == "dashboard":
        os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
        from dashboard import main as dashboard_main
        raise SystemExit(dashboard_main(sys.argv[2:]))
    elif mode == "report":
        os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
        from analysis.report import main as report_main
        raise SystemExit(report_main(sys.argv[2:]))
    else:
        print(f"Unknown mode: {mode}")
