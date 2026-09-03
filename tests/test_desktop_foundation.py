from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

from core.clock import MARKET_TZ, market_now
from core.instance_lock import AlreadyRunningError, InstanceLock
from core.scheduler import current_monitor_slot, latest_due_full_slot
from core.settings import FileSettingsStore, settings_path, stockwatch_home
from utils.storage import Storage


def test_market_clock_converts_aware_values_to_shanghai():
    value = datetime(2026, 9, 3, 1, 10, tzinfo=timezone.utc)
    converted = market_now(value)
    assert converted.tzinfo == MARKET_TZ
    assert (converted.hour, converted.minute) == (9, 10)


def test_scheduler_slots_are_market_timezone_aware():
    now = datetime(2026, 9, 3, 7, 20, tzinfo=timezone.utc)
    assert latest_due_full_slot(now).isoformat() == "2026-09-03T15:15:00+08:00"
    monitor = current_monitor_slot(datetime(2026, 9, 3, 1, 27, tzinfo=timezone.utc))
    assert monitor.isoformat() == "2026-09-03T09:25:00+08:00"


def test_scheduler_skips_weekends_without_calling_calendar(tmp_path):
    calls = []
    from core.scheduler import AgentScheduler

    scheduler = AgentScheduler(
        storage=Storage(tmp_path / "db.sqlite"),
        full_runner=lambda: calls.append("full"),
        monitor_runner=lambda **_kwargs: calls.append("monitor"),
        trading_day_checker=lambda: (_ for _ in ()).throw(AssertionError("calendar called")),
    )
    assert scheduler.tick(datetime(2026, 9, 5, 2, 0, tzinfo=timezone.utc)) == []
    assert calls == []


def test_settings_home_and_atomic_allowlist(tmp_path, monkeypatch):
    home = tmp_path / "runtime"
    monkeypatch.setenv("STOCKWATCH_HOME", str(home))
    monkeypatch.delenv("STOCKWATCH_ENV_PATH", raising=False)
    store = FileSettingsStore({"WATCHLIST": "", "ENABLE_AI": "auto"})
    store.save({"WATCHLIST": "600519,000858", "UNKNOWN": "ignored"})
    assert stockwatch_home() == home
    assert settings_path() == home / "settings.env"
    assert store.load()["WATCHLIST"] == "600519,000858"
    assert "UNKNOWN" not in store.path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert store.path.stat().st_mode & 0o777 == 0o600


def test_scheduled_job_claim_is_atomic(tmp_path):
    storage_a = Storage(tmp_path / "db.sqlite")
    storage_b = Storage(tmp_path / "db.sqlite")
    assert storage_a.claim_scheduled_job("full:slot", "full", "2026-09-03T09:10:00+08:00")
    assert not storage_b.claim_scheduled_job("full:slot", "full", "2026-09-03T09:10:00+08:00")
    storage_a.finish_scheduled_job("full:slot")
    assert storage_b.get_recent_scheduled_jobs(1)[0]["status"] == "completed"


def test_instance_lock_rejects_second_owner(tmp_path):
    first = InstanceLock(tmp_path / "agent.lock").acquire()
    try:
        try:
            InstanceLock(tmp_path / "agent.lock").acquire()
        except AlreadyRunningError:
            pass
        else:
            raise AssertionError("second lock unexpectedly succeeded")
    finally:
        first.release()


def test_local_agent_api_status_settings_and_run(tmp_path, monkeypatch):
    from dashboard import create_dashboard_server
    from desktop.client import AgentClient

    class Controller:
        action = None

        def status(self):
            return {"ok": True, "mode": "test-agent"}

        def submit(self, action):
            self.action = action
            return action in {"full", "monitor", "score"}

    monkeypatch.setenv("STOCKWATCH_ENV_PATH", str(tmp_path / "settings.env"))
    controller = Controller()
    server = create_dashboard_server("127.0.0.1", 0, agent_controller=controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = AgentClient(f"http://127.0.0.1:{server.server_address[1]}")
    try:
        assert client.status()["mode"] == "test-agent"
        assert client.save_settings({"WATCHLIST": "600519"})["ok"]
        assert client.settings()["WATCHLIST"] == "600519"
        assert client.run("full")["ok"]
        assert controller.action == "full"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
