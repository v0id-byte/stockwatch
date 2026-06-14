"""守护进程调度：按固定时间自动触发完整分析。"""
import sched
import time
from datetime import datetime, timedelta

from loguru import logger

from config import get_config
from core.runner import once, _setup_log
from core.monitor import monitor_once, _is_intraday_monitor_time


def _resolve_and_calibrate():
    """结算到期决策（写回 decisions.success），让信号复盘/置信度校准的数据真正闭环。
    幂等且廉价：规则模式下没有 BUY/SELL 决策时几乎零开销。结算不需要 sklearn；
    仅当开启 ENABLE_CALIBRATION 且本机装了训练依赖时才顺带重训校准模型。"""
    from utils.storage import Storage
    from analysis.calibration import resolve_decisions

    cfg = get_config()
    storage = Storage()
    resolved = resolve_decisions(storage, cfg.calibration_lookback_days)
    if resolved:
        logger.info(f"决策结算：本轮结算 {resolved} 条 BUY/SELL")
    if cfg.enable_calibration:
        try:
            from scripts.train_calibration import train_action
            train_action(storage, "BUY", cfg.calibration_min_samples)
            train_action(storage, "SELL", cfg.calibration_min_samples)
        except Exception as e:  # 缺 sklearn 等训练依赖时静默跳过，不影响盯盘
            logger.info(f"置信度校准重训跳过（通常是未装训练依赖）：{e}")


def daemon():
    """守护进程模式，按调度时间自动运行。"""
    _setup_log()
    logger.info("StockWatch 守护进程启动")

    scheduler = sched.scheduler(time.time, time.sleep)
    last_news_check = {"at": datetime.min}

    def _run_full():
        try:
            once()
        except Exception as e:
            logger.error(f"运行异常: {e}")
        try:
            _resolve_and_calibrate()
        except Exception as e:
            logger.warning(f"决策结算异常，跳过: {e}")
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
