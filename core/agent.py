"""Single-process StockWatch background agent.

The agent owns scheduling and the localhost API.  The desktop UI is a client;
closing it never stops this process.
"""
from __future__ import annotations

import queue
import signal
import threading
from dataclasses import dataclass, field
from typing import Callable

from loguru import logger

from config import get_config
from core.clock import MARKET_TZ_NAME, market_now
from core.instance_lock import InstanceLock
from core.monitor import monitor_once
from core.runtime import install_bundled_models
from core.runner import _setup_log, once
from core.scheduler import AgentScheduler
from core.settings import settings_path
from utils.storage import Storage


@dataclass
class AgentController:
    storage: Storage
    actions: queue.Queue[str] = field(default_factory=queue.Queue)
    started_at: str = field(default_factory=lambda: market_now().isoformat(timespec="seconds"))
    current_action: str | None = None
    last_error: str | None = None

    def submit(self, action: str) -> bool:
        action = action.strip().lower()
        if action not in {"full", "monitor", "score"}:
            return False
        self.actions.put(action)
        return True

    def status(self) -> dict[str, object]:
        cfg = get_config()
        alerts = self.storage.get_recent_web_alerts(5)
        return {
            "ok": True,
            "mode": "agent",
            "started_at": self.started_at,
            "market_time": market_now().isoformat(timespec="seconds"),
            "market_timezone": MARKET_TZ_NAME,
            "configured": settings_path().exists(),
            "current_action": self.current_action,
            "queued_actions": self.actions.qsize(),
            "last_error": self.last_error,
            "last_successful_run": self.storage.get_last_successful_run_ts(),
            "scheduled_jobs": self.storage.get_recent_scheduled_jobs(10),
            "recent_alerts": alerts,
            "risk_model": {
                "enabled": cfg.enable_risk_model,
                "installed": cfg.risk_model_path.exists(),
            },
        }


class StockWatchAgent:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        install_bundled_models()
        self.host = host
        self.port = port
        self.storage = Storage()
        self.controller = AgentController(self.storage)
        self.scheduler = AgentScheduler(storage=self.storage)
        self.stop_event = threading.Event()
        self._server = None

    def stop(self, *_args) -> None:
        self.stop_event.set()
        if self._server is not None:
            threading.Thread(target=self._server.shutdown, daemon=True).start()

    def _run_dashboard(self) -> None:
        from dashboard import create_dashboard_server

        try:
            self._server = create_dashboard_server(self.host, self.port, agent_controller=self.controller)
            logger.info(f"Agent API/dashboard listening on http://{self.host}:{self.port}")
            self._server.serve_forever(poll_interval=0.5)
        except Exception as exc:
            self.controller.last_error = f"api: {type(exc).__name__}: {exc}"
            logger.exception(f"Agent API stopped: {exc}")
            self.stop_event.set()

    def _run_bot(self) -> None:
        try:
            from bot.runner import run_bot

            run_bot()
        except Exception as exc:
            self.controller.last_error = f"bot: {type(exc).__name__}: {exc}"
            logger.exception(f"Agent bot stopped: {exc}")

    def _run_action(self, action: str) -> None:
        callbacks: dict[str, Callable[[], object]] = {
            "full": once,
            "monitor": lambda: monitor_once(check_news=True),
            "score": self.scheduler.score_runner,
        }
        self.controller.current_action = action
        self.controller.last_error = None
        try:
            callbacks[action]()
        except Exception as exc:
            self.controller.last_error = f"{action}: {type(exc).__name__}: {exc}"
            logger.exception(f"Manual agent action failed: {exc}")
        finally:
            self.controller.current_action = None

    def run(self) -> None:
        _setup_log()
        cfg = get_config()
        with InstanceLock(cfg.home_dir / "agent.lock"):
            try:
                signal.signal(signal.SIGTERM, self.stop)
                signal.signal(signal.SIGINT, self.stop)
            except ValueError:
                pass
            threading.Thread(target=self._run_dashboard, name="stockwatch-dashboard", daemon=True).start()
            if cfg.agent_enable_bot:
                threading.Thread(target=self._run_bot, name="stockwatch-bot", daemon=True).start()
            logger.info(f"StockWatch Agent started; market timezone={MARKET_TZ_NAME}")
            while not self.stop_event.is_set():
                try:
                    action = self.controller.actions.get_nowait()
                except queue.Empty:
                    self.scheduler.tick()
                    self.stop_event.wait(20)
                else:
                    self._run_action(action)
            if self._server is not None:
                self._server.server_close()


def run_agent(host: str = "127.0.0.1", port: int = 8765) -> None:
    StockWatchAgent(host, port).run()
