"""Idempotent scheduler anchored to the mainland China market clock."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Callable

from loguru import logger

from config import get_config
from core.clock import MARKET_TZ_NAME, market_now
from core.monitor import _is_intraday_monitor_time, monitor_once
from core.runner import _setup_log, once
from utils.storage import Storage


FULL_RUN_TIMES = ((9, 10), (12, 30), (15, 15))
SCORE_TIME = (18, 30)
FULL_CATCH_UP = timedelta(hours=6)


def _resolve_and_calibrate():
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
        except Exception as exc:
            logger.info(f"置信度校准重训跳过（通常是未装训练依赖）：{exc}")


def _scheduled_at(now: datetime, hour: int, minute: int) -> datetime:
    now = market_now(now)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def latest_due_full_slot(now: datetime | None = None) -> datetime | None:
    now = market_now(now)
    due = [_scheduled_at(now, hour, minute) for hour, minute in FULL_RUN_TIMES]
    due = [slot for slot in due if slot <= now and now - slot <= FULL_CATCH_UP]
    return max(due, default=None)


def current_monitor_slot(now: datetime | None = None) -> datetime | None:
    now = market_now(now)
    if not _is_intraday_monitor_time(now):
        return None
    minute = now.minute - now.minute % 5
    return now.replace(minute=minute, second=0, microsecond=0)


class AgentScheduler:
    """Run each slot at most once and catch up only the latest full run."""

    def __init__(
        self,
        storage: Storage | None = None,
        full_runner: Callable[[], object] = once,
        monitor_runner: Callable[..., object] = monitor_once,
        score_runner: Callable[[], object] | None = None,
        trading_day_checker: Callable[[], bool] | None = None,
    ):
        self.storage = storage or Storage()
        self.full_runner = full_runner
        self.monitor_runner = monitor_runner
        self.score_runner = score_runner or self._score
        self.trading_day_checker = trading_day_checker
        self.last_news_check: datetime | None = None
        self._trading_day_cache: tuple[object, bool] | None = None

    def _is_trading_day(self, now: datetime) -> bool:
        if now.weekday() >= 5:
            return False
        if self._trading_day_cache and self._trading_day_cache[0] == now.date():
            return self._trading_day_cache[1]
        if self.trading_day_checker is None:
            from data.universe import Universe

            result = Universe(storage=self.storage).is_trading_day()
        else:
            result = self.trading_day_checker()
        self._trading_day_cache = (now.date(), bool(result))
        return bool(result)

    @staticmethod
    def _score() -> object:
        cfg = get_config()
        if not cfg.enable_risk_model:
            return {"status": "disabled"}
        from core.model_scoring import push_risk_alerts, run_nightly_scoring

        storage = Storage()
        result = run_nightly_scoring(storage, cfg)
        result["risk_alerts"] = push_risk_alerts(storage, cfg)
        return result

    def _run_claimed(self, job_type: str, slot: datetime, callback: Callable[[], object]) -> bool:
        key = f"{job_type}:{slot.strftime('%Y-%m-%dT%H:%M')}"
        if not self.storage.claim_scheduled_job(key, job_type, slot.isoformat()):
            return False
        error = None
        try:
            logger.info(f"执行计划任务 {key} ({MARKET_TZ_NAME})")
            callback()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception(f"计划任务失败 {key}: {exc}")
        finally:
            self.storage.finish_scheduled_job(key, error)
        return True

    def tick(self, now: datetime | None = None) -> list[str]:
        now = market_now(now)
        ran: list[str] = []
        if not self._is_trading_day(now):
            return ran

        full_slot = latest_due_full_slot(now)
        if full_slot is not None:
            def full_callback():
                self.full_runner()
                _resolve_and_calibrate()

            if self._run_claimed("full", full_slot, full_callback):
                ran.append("full")

        monitor_slot = current_monitor_slot(now)
        if monitor_slot is not None:
            check_news = self.last_news_check is None or now - self.last_news_check >= timedelta(minutes=30)

            def monitor_callback():
                self.monitor_runner(check_news=check_news)

            if self._run_claimed("monitor", monitor_slot, monitor_callback):
                ran.append("monitor")
                if check_news:
                    self.last_news_check = now

        cfg = get_config()
        score_slot = _scheduled_at(now, *SCORE_TIME)
        if cfg.enable_risk_model and score_slot <= now and now - score_slot <= FULL_CATCH_UP:
            if self._run_claimed("score", score_slot, self.score_runner):
                ran.append("score")
        return ran

    def run_forever(self, stop_event: threading.Event | None = None, interval: float = 20.0) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            self.tick()
            stop_event.wait(interval)


def daemon():
    """Backward-compatible scheduler-only service entry."""
    _setup_log()
    logger.info(f"StockWatch 守护进程启动，市场时区={MARKET_TZ_NAME}")
    AgentScheduler().run_forever()
