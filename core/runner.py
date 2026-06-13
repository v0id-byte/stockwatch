"""完整分析流程（once）及其辅助函数。"""
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from loguru import logger

from config import get_config
from utils.storage import Storage
from utils.llm import get_token_usage, reset_token_usage
from data.market import MarketData
from data.universe import Universe
from analysis.technical import compute_tech_score
from analysis.sentiment import SENTIMENT_SCORE_MODEL_VERSION, batch_sentiment_details
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


def once():
    """立即跑一次完整流程。"""
    _setup_log()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    logger.info(f"===== 开始运行 {run_id} =====")

    cfg = get_config()
    storage = Storage()
    market = MarketData()
    universe = Universe(storage)
    reset_token_usage()

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

    # 2. 实时报价 + 大盘
    quotes = market.get_realtime_quote(codes)
    logger.info(f"实时报价获取: {len(quotes)} 只")
    if cfg.enable_propagation:
        try:
            from analysis.propagation import (
                detect_leaders_from_quotes,
                find_related_candidates_from_history,
            )
            leaders = detect_leaders_from_quotes(
                quotes,
                threshold=cfg.propagation_leader_return_threshold,
            )
            slots = max(0, cfg.max_stocks_per_run - len(codes))
            if leaders and slots:
                leader_returns = {
                    code: float(quotes.get(code, {}).get("pct_change") or 0) / 100.0
                    for code in leaders
                }
                candidates = find_related_candidates_from_history(
                    cfg.propagation_history_dir,
                    leaders,
                    leader_returns,
                    existing_codes=set(codes),
                    max_candidates=min(cfg.propagation_max_candidates, slots),
                    min_corr=cfg.propagation_min_corr,
                )
                added = []
                for row in candidates:
                    code = row["code"]
                    if code not in codes:
                        codes.append(code)
                        added.append(code)
                if added:
                    quotes.update(market.get_realtime_quote(added))
                    logger.info(f"关联补涨扩展池: +{len(added)} 只 {added}")
            elif leaders:
                logger.info("关联补涨识别到领涨股，但分析池已达上限，跳过扩展")
            else:
                logger.info("关联补涨未识别到达阈值的领涨股")
        except Exception as e:
            logger.warning(f"关联补涨扩展失败，跳过: {e}")
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

    # 3. 并行拉取 K 线（网络 I/O；写入串行保证 SQLite 安全）
    today_str = datetime.now().strftime("%Y-%m-%d")
    codes_to_fetch = [c for c in codes if not storage.kline_cached_today(c)]
    if codes_to_fetch:
        logger.info(f"并发拉取K线 {len(codes_to_fetch)} 只（8线程）")
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(market.get_daily_kline, c): c for c in codes_to_fetch}
            for fut in as_completed(futures):
                c = futures[fut]
                try:
                    rows = fut.result()
                    for row in rows:
                        storage.upsert_kline(c, row["trade_date"], row)
                except Exception as e:
                    logger.warning(f"K线并发获取失败 {c}: {e}")

    # 4. 技术分析 + Alpha158（串行）
    llm_calls = 0
    decisions = []

    for code in codes:
        name = quotes.get(code, {}).get("name", code)
        logger.info(f"分析: {code}({name})")

        kline = storage.get_kline(code, "2020-01-01", today_str)
        if len(kline) < 20:
            logger.warning(f"K线不足，跳过 {code}")
            continue

        tech_result = compute_tech_score(kline)
        tech_score = tech_result["score"]
        logger.debug(f"{code} 技术分: {tech_score} {tech_result.get('details', {})}")

        decisions.append({
            "code": code, "name": name,
            "tech_score": tech_score,
            "tech_details": tech_result.get("details", {}),
            "kline": kline,
            "quote": quotes.get(code, {}),
        })
        if factor_needed:
            try:
                import pandas as pd
                from analysis.factors import compute_alpha158
                decisions[-1]["factors"] = compute_alpha158(pd.DataFrame(kline), market_df)
            except Exception as e:
                logger.warning(f"Alpha158 计算失败 {code}: {e}")
                decisions[-1]["factors"] = {}

    alpha_contexts = {}
    lgbm_contexts = {}
    propagation_contexts = {}
    if cfg.enable_propagation and decisions:
        try:
            from analysis.propagation import compute_latest_propagation_features
            kline_map = {d["code"]: d["kline"] for d in decisions}
            propagation_features, propagation_contexts = compute_latest_propagation_features(
                [d["code"] for d in decisions],
                quotes,
                kline_map,
                leader_return_threshold=cfg.propagation_leader_return_threshold,
                min_corr=cfg.propagation_min_corr,
            )
            for d in decisions:
                d.setdefault("factors", {}).update(propagation_features.get(d["code"], {}))
        except Exception as e:
            logger.warning(f"关联补涨特征生成失败，跳过: {e}")
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
            from analysis.factors import WINDOWS
            ranker = LgbmRanker(cfg.lgbm_model_path)
            # Long-window factors need full history; stocks with too few klines would
            # feed fillna(0) features the model never saw in training, so skip them.
            min_hist = max(WINDOWS)
            kline_len = {d["code"]: len(d.get("kline", [])) for d in decisions}
            eligible = {c: f for c, f in factor_map.items() if kline_len.get(c, 0) >= min_hist}
            scores = ranker.predict_batch(eligible)
            for code in factor_map:
                scores.setdefault(code, None)
            lgbm_contexts = format_lgbm_context(scores)
        except Exception as e:
            logger.warning(f"LightGBM 推理失败，跳过: {e}")
    if propagation_contexts:
        for d in decisions:
            code = d["code"]
            line = propagation_contexts.get(code, "")
            if line:
                current = lgbm_contexts.get(code, "")
                lgbm_contexts[code] = "\n".join(part for part in [current, line] if part)

    # 5. 情绪分析（批量）
    batch = [(d["code"], d["name"]) for d in decisions[:30]]
    sentiment_details = batch_sentiment_details(batch, storage=storage)
    for d in decisions:
        detail = sentiment_details.get(d["code"], {})
        d["sentiment"] = detail.get("score", 0.0)
        d["sentiment_context"] = detail.get("context", "")

    # 6. 最终决策
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

    # 7. 推送
    feishu = FeishuClient() if cfg.notify_channel == "feishu" else None
    push_ok = False
    push_status = "无可推送信号"
    if run_results:
        card = render_card(run_id, run_results, regime_info=regime_info if cfg.enable_regime else None)
        if card:
            if feishu:
                push_ok = feishu.send_message(card)
                push_status = "成功" if push_ok else "失败"
                for d in run_results:
                    storage.mark_decision_pushed(run_id, d["code"], push_ok)
                    if not push_ok:
                        break
            else:
                push_ok = True
                push_status = "已写入 Web 控制台"
        else:
            push_status = "无可展示信号"
    if feishu and not run_results:
        summary_ok = _send_after_close_summary(
            storage, feishu, final_decisions, run_results,
            tracked_positions, price_alerts,
        )
        if summary_ok:
            push_ok = True
            push_status = "安心总结成功"

    # 8. 统计
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
