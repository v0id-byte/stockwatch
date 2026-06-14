"""数据源健康 + 运行状态存储层测试。"""
from utils.storage import Storage


def _storage(tmp_path):
    return Storage(tmp_path / "health.sqlite")


def test_source_health_accumulates_and_tracks_latest(tmp_path):
    s = _storage(tmp_path)
    s.upsert_source_health("行情", True, 100.0, None, 5)
    s.upsert_source_health("行情", False, 0.0, "timeout", 0)
    s.upsert_source_health("行情", True, 120.0, None, 6)

    health = {r["source"]: r for r in s.get_source_health()}
    row = health["行情"]
    assert row["status"] == "ok"               # 反映最近一次
    assert row["ok_count"] == 2
    assert row["fail_count"] == 1
    assert row["last_ok_at"] is not None        # 成功时间被保留
    assert row["last_error_at"] is not None      # 失败时间也被保留
    assert row["last_count"] == 6


def test_last_successful_run_ignores_pushed_ok(tmp_path):
    """关键回归：安静但健康的一天 pushed_ok=0，仍应算作成功运行。"""
    s = _storage(tmp_path)
    assert s.get_last_successful_run_ts() is None

    # 数据拉取失败的运行不应被算作成功
    s.insert_run("r_fail", {
        "stocks_analyzed": 0, "llm_calls": 0, "tokens_used": 0,
        "pushed_count": 0, "pushed_ok": 0, "status": "failed", "error": "行情失败",
    })
    assert s.get_last_successful_run_ts() is None

    # 健康但无可推送（pushed_ok=0）的运行 = 成功
    s.insert_run("r_ok_quiet", {
        "stocks_analyzed": 8, "llm_calls": 8, "tokens_used": 0,
        "pushed_count": 0, "pushed_ok": 0, "status": "ok", "error": None,
    })
    assert s.get_last_successful_run_ts() is not None


def test_insert_run_defaults_status_ok(tmp_path):
    """旧调用方不传 status 时默认 ok，保持向后兼容。"""
    s = _storage(tmp_path)
    s.insert_run("r_legacy", {
        "stocks_analyzed": 3, "llm_calls": 3, "tokens_used": 0,
        "pushed_count": 1, "pushed_ok": 1,
    })
    runs = s.get_recent_runs()
    assert runs[0]["status"] == "ok"


def test_misreported_codes_only_when_bad_dominates(tmp_path):
    from datetime import datetime
    s = _storage(tmp_path)

    def fb(code, label):
        s.insert_alert_feedback({"code": code, "label": label, "source": "decision:1",
                                 "created_at": datetime.now().isoformat()})

    for _ in range(3):
        fb("AAA", "误报")          # 3 误报 0 有用 -> 抑制
    fb("BBB", "误报")              # 1 误报 -> 未达 min_count
    fb("BBB", "有用"); fb("BBB", "有用")
    fb("CCC", "误报"); fb("CCC", "误报"); fb("CCC", "有用")  # 2>1 -> 抑制

    muted = s.get_misreported_codes(days=30, min_count=2)
    assert set(muted) == {"AAA", "CCC"}
    assert muted["AAA"]["bad"] == 3
