"""Main entry: test / once / daemon"""
import sys
import os
import time
import uuid
from datetime import datetime, timedelta

# 确保从 stockwatch 目录运行
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import get_config
from utils.storage import Storage
from utils.llm import get_llm_client, get_token_usage, reset_token_usage
from data.market import MarketData
from data.universe import Universe
from analysis.technical import compute_tech_score
from analysis.sentiment import batch_sentiment
from decision.engine import DecisionEngine
from push.feishu import FeishuClient, render_card


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
        demo = pd.DataFrame([
            {"trade_date": f"2026-01-{i+1:02d}", "open": 10+i*0.1, "high": 10.5+i*0.1,
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
        # 这里先记录，等待 batch_sentiment
        decisions.append({
            "code": code, "name": name,
            "tech_score": tech_score,
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
    sentiments = batch_sentiment(batch, storage=storage)
    for d in decisions:
        d["sentiment"] = sentiments.get(d["code"], 0.0)

    # 5. 最终决策
    engine = DecisionEngine(storage)
    run_results = []

    for d in decisions:
        decision = engine.decide_one(
            d["code"], d["name"],
            d["tech_score"], d["sentiment"],
            d["kline"],
            north_context=north_context,
            sht_pct=sht_pct,
            alpha_summary=alpha_contexts.get(d["code"], ""),
            lgbm_context=lgbm_contexts.get(d["code"], ""),
            regime_context=regime_info.get("context", "大盘 regime: normal"),
            sector_context=sector_contexts.get(d["code"], "所属板块: 未知"),
            confidence_floor=confidence_floor,
        )
        if decision.get("_will_push"):
            run_results.append(decision)
        storage.insert_decision(run_id, datetime.now().isoformat(), decision)
        llm_calls += 1

    # 6. 推送
    feishu = FeishuClient()
    push_ok = False
    if run_results:
        card = render_card(run_id, run_results, regime_info=regime_info if cfg.enable_regime else None)
        if card:
            push_ok = feishu.send_message(card)
            for d in run_results:
                storage.mark_decision_pushed(run_id, d["code"], push_ok)
                if not push_ok:
                    break

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
        f"推送: {'成功' if push_ok else '失败'} ({len(run_results)} 条)"
    )

    print(f"\n完成！分析 {len(decisions)} 只，推送 {'成功' if push_ok else '失败'}\n")


def daemon():
    """守护进程模式，按调度时间自动运行"""
    import sched

    _setup_log()
    logger.info("StockWatch 守护进程启动")

    scheduler = sched.scheduler(time.time, time.sleep)

    def _run():
        try:
            once()
        except Exception as e:
            logger.error(f"运行异常: {e}")
        # 调度下次
        _schedule_next(scheduler)

    def _schedule_next(sched_obj):
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
        logger.info(f"下次运行: {next_run.strftime('%Y-%m-%d %H:%M')}, 约 {int(delay//60)} 分钟后")
        sched_obj.enter(delay, 1, _run, ())

    _schedule_next(scheduler)
    scheduler.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [test|once|daemon]")
        sys.exit(1)

    mode = sys.argv[1]
    if mode == "test":
        test()
    elif mode == "once":
        once()
    elif mode == "daemon":
        daemon()
    else:
        print(f"Unknown mode: {mode}")
