"""盘中监控在 NOTIFY_CHANNEL=web（feishu=None）时，把触发的盯价/重大消息提醒
写进 web_alerts，由首页「盘中提醒」feed 渲染。规则模式（未启用 AI）只靠标题关键词。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")

import core.monitor as monitor
from utils.storage import Storage


class FakeMarket:
    def __init__(self, quotes: dict):
        self._quotes = quotes

    def get_realtime_quote(self, codes):
        return {c: self._quotes[c] for c in codes if c in self._quotes}


def _storage(tmp_path):
    return Storage(tmp_path / "monitor.sqlite")


def _no_health(monkeypatch):
    # 健康记录走模块级缓存的真实 db，测试里打成空操作保持 hermetic
    monkeypatch.setattr(monitor, "record_source_health", lambda *a, **k: None)


def test_record_web_alert_dedupes_by_event_key(tmp_path):
    s = _storage(tmp_path)
    s.record_web_alert("k1", "盯价", "info", "600519", "贵州茅台", "触达风险价", "现价 1490")
    s.record_web_alert("k1", "盯价", "info", "600519", "贵州茅台", "触达风险价", "现价 1490")
    assert len(s.get_recent_web_alerts()) == 1


def test_price_alert_below_records_web_alert_in_web_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("ALERT_LEVELS", "critical,warning,info")
    monkeypatch.delenv("MUST_SEE_MODE", raising=False)
    _no_health(monkeypatch)

    s = _storage(tmp_path)
    s.upsert_price_alert({
        "user_id": "web-ui", "code": "600519", "name": "贵州茅台",
        "trigger_price": 1500, "direction": "below",
    })
    market = FakeMarket({"600519": {
        "close": 1490.0, "name": "贵州茅台", "pct_change": -1.2,
        "buy_volume_5": 100, "sell_volume_5": 50,
    }})

    monitor._monitor_price_alerts(s, market, feishu=None)

    alerts = s.get_recent_web_alerts()
    assert len(alerts) == 1
    assert alerts[0]["code"] == "600519"
    assert alerts[0]["category"] == "盯价"
    assert "1490" in alerts[0]["body"]

    # 当日已提醒 -> 二次运行不重复写
    monitor._monitor_price_alerts(s, market, feishu=None)
    assert len(s.get_recent_web_alerts()) == 1


def test_price_alert_skips_when_not_triggered(tmp_path, monkeypatch):
    _no_health(monkeypatch)
    s = _storage(tmp_path)
    s.upsert_price_alert({"user_id": "web-ui", "code": "600519",
                          "trigger_price": 1500, "direction": "below"})
    market = FakeMarket({"600519": {"close": 1600.0, "name": "贵州茅台"}})  # 高于触发价

    monitor._monitor_price_alerts(s, market, feishu=None)
    assert s.get_recent_web_alerts() == []


def test_major_news_rule_mode_keyword_records_web_alert(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_AI", "off")          # 规则模式：不调 LLM
    monkeypatch.setenv("ALERT_LEVELS", "critical,warning,info")
    monkeypatch.setenv("WATCHLIST", "000001")        # 收敛分析池到单只，便于断言
    monkeypatch.delenv("MUST_SEE_MODE", raising=False)
    _no_health(monkeypatch)

    s = _storage(tmp_path)
    market = FakeMarket({"000001": {"name": "平安银行"}})

    monkeypatch.setattr(monitor.NewsData, "get_news", staticmethod(
        lambda code, days=1: [{
            "title": "某公司被证监会立案调查", "ts": "2026-06-14 10:00",
            "source": "测试源", "content": "",
        }]))

    def _boom(*a, **k):
        raise AssertionError("规则模式不应调用 LLM 打分")

    monkeypatch.setattr(monitor, "score_news_items", _boom)

    monitor._monitor_major_news(s, market, feishu=None)

    alerts = [a for a in s.get_recent_web_alerts() if a["category"] == "重大消息"]
    assert len(alerts) == 1
    assert "立案" in alerts[0]["title"]
    assert alerts[0]["level"] == "info"

    # 同一条新闻二次扫描已被 alert_events 去重，不重复入 feed
    monitor._monitor_major_news(s, market, feishu=None)
    assert len([a for a in s.get_recent_web_alerts() if a["category"] == "重大消息"]) == 1
