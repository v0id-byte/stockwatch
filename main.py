"""CLI 入口：test / once / daemon / monitor / bot / demo / dashboard / report"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test():
    """自检：AKShare / LLM / 飞书 连接。"""
    print("=" * 40)
    print("StockWatch 自检")
    print("=" * 40)

    try:
        import time
        from datetime import date
        from data.market import MarketData
        from utils.storage import Storage
        data = MarketData()
        _t = time.perf_counter()
        quotes = data.get_realtime_quote(["600519"])
        latency = (time.perf_counter() - _t) * 1000
        quote = quotes.get("600519", {})
        name = quote.get("name", "未知")
        Storage().upsert_source_health(
            "行情", bool(quotes), latency,
            None if quotes else "未获取到报价", len(quotes),
        )
        if quotes:
            qt = str(quote.get("quote_time") or "")
            stale = "" if qt[:8] == date.today().strftime("%Y%m%d") else "（非当日时间戳，可能非交易时段或数据延迟）"
            print(f"✅ 行情接口成功 → {name}（{latency:.0f}ms）{stale}")
        else:
            print("❌ 行情接口失败：未获取到报价")
    except Exception as e:
        print(f"❌ 行情接口失败: {e}")

    try:
        from config import get_config
        from utils.llm import get_llm_client
        cfg = get_config()
        if not cfg.ai_enabled:
            print("✅ 运行模式：规则模式（未启用AI）→ 跳过LLM测试；止损/盯价/公告提醒照常运行")
        else:
            llm = get_llm_client()
            llm.chat([{"role": "user", "content": "Hello, reply OK"}])
            print(f"✅ LLM 连接成功 → {cfg.llm_provider}/{cfg.llm_model}")
    except Exception as e:
        print(f"❌ LLM 失败: {e}")

    try:
        from config import get_config
        cfg = get_config()
        if cfg.notify_channel == "feishu":
            from push.feishu import FeishuClient
            FeishuClient()._get_token()
            print("✅ 飞书连接成功")
        else:
            print("✅ 通知渠道：Web 控制台（跳过飞书连接）")
    except Exception as e:
        print(f"❌ 通知渠道失败: {e}")

    try:
        from utils.storage import Storage
        from analysis.factors import compute_alpha158
        import pandas as pd
        storage = Storage()
        model = storage.get_latest_calibration_model("BUY")
        print(f"✅ v2 calibration 表可读 → {'已有模型' if model else '暂无模型'}")
        dates = pd.date_range("2026-01-01", periods=70).strftime("%Y-%m-%d")
        demo_df = pd.DataFrame([
            {"trade_date": dates[i], "open": 10 + i * 0.1, "high": 10.5 + i * 0.1,
             "low": 9.8 + i * 0.1, "close": 10.2 + i * 0.1,
             "volume": 100000 + i * 1000, "amount": 0}
            for i in range(70)
        ])
        factors = compute_alpha158(demo_df, demo_df)
        print(f"✅ Alpha158 可计算 → {len(factors)} 个因子")
        from analysis.lgbm import LgbmRanker
        from config import get_config
        cfg = get_config()
        ranker = LgbmRanker(cfg.lgbm_model_path)
        print(f"✅ LGBM 检查完成 → {'已加载' if ranker.model else '未加载'}")
        if cfg.enable_regime:
            from analysis.regime import get_market_regime
            from data.market import MarketData
            regime = get_market_regime(MarketData(), storage)
            print(f"✅ regime 可获取 → {regime['regime']}")
        else:
            print("✅ regime 检查完成 → 未启用")
    except Exception as e:
        print(f"❌ v2 模块检查失败: {e}")

    try:
        from utils.storage import Storage
        rows = Storage().get_source_health()
        if rows:
            print("-" * 40)
            print("数据源健康（历史累计）")
            for r in rows:
                flag = "✅" if r.get("status") == "ok" else "❌"
                last = r.get("last_ok_at") or r.get("last_error_at") or "-"
                extra = f"，最近错误：{r['last_error']}" if r.get("status") != "ok" and r.get("last_error") else ""
                print(f"  {flag} {r['source']}：最近 {last}，成功 {r.get('ok_count', 0)} / 失败 {r.get('fail_count', 0)}{extra}")
    except Exception as e:
        print(f"  数据源健康读取失败: {e}")

    print("=" * 40)


def demo(query: str = ""):
    """无需飞书推送的终端体验入口。"""
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    cfg_mod = __import__("config")
    cfg = cfg_mod.get_config()
    from utils.storage import Storage
    from data.market import MarketData
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
        from loguru import logger
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py [test|once|daemon|monitor|bot|demo|dashboard|report|backtest]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "test":
        test()

    elif mode == "once":
        from core.runner import once
        once()

    elif mode == "daemon":
        from core.scheduler import daemon
        daemon()

    elif mode == "monitor":
        from core.runner import _setup_log
        from core.monitor import monitor_once
        _setup_log()
        monitor_once(check_news=True)

    elif mode == "bot":
        from core.runner import _setup_log
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

    elif mode == "backtest":
        os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
        from scripts.backtest_strategy import main as backtest_main
        raise SystemExit(backtest_main(sys.argv[2:]))

    else:
        print(f"Unknown mode: {mode}")
