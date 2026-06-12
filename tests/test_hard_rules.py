"""Tests for hard-rule logic that must never regress."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.runner import _decision_alert_level, _tracked_alert_reason


class TestDecisionAlertLevel:
    def test_sell_high_confidence_is_critical(self):
        d = {"action": "SELL", "confidence": 0.80, "one_liner": ""}
        assert _decision_alert_level(d) == "critical"

    def test_sell_low_confidence_is_warning(self):
        d = {"action": "SELL", "confidence": 0.60, "one_liner": ""}
        assert _decision_alert_level(d) == "warning"

    def test_buy_is_warning(self):
        d = {"action": "BUY", "confidence": 0.90, "one_liner": ""}
        assert _decision_alert_level(d) == "warning"

    def test_hold_is_info(self):
        d = {"action": "HOLD", "confidence": 0.50, "one_liner": ""}
        assert _decision_alert_level(d) == "info"

    def test_stop_loss_breach_in_one_liner_is_critical(self):
        d = {"action": "HOLD", "confidence": 0.50, "one_liner": "跌破止损位"}
        assert _decision_alert_level(d) == "critical"

    def test_model_turn_sell_in_one_liner_is_critical(self):
        d = {"action": "HOLD", "confidence": 0.50, "one_liner": "模型转为 SELL"}
        assert _decision_alert_level(d) == "critical"


class TestTrackedAlertReason:
    def _base_decision(self):
        return {"_current_price": 10.0, "stop_loss": 9.0, "target_price": 12.0, "action": "HOLD", "confidence": 0.5}

    def test_stop_loss_breach_triggers(self):
        pos = {"last_notified_at": "2020-01-01", "stop_loss": 10.5, "target_price": None}
        dec = self._base_decision()
        reason = _tracked_alert_reason(pos, dec)
        assert "跌破止损" in reason

    def test_target_near_triggers(self):
        pos = {"last_notified_at": "2020-01-01", "stop_loss": None, "target_price": 10.1}
        dec = self._base_decision()
        reason = _tracked_alert_reason(pos, dec)
        assert "接近目标" in reason

    def test_already_notified_today_no_trigger(self):
        from datetime import date
        today = date.today().isoformat()
        pos = {"last_notified_at": today, "stop_loss": 10.5, "target_price": None}
        dec = self._base_decision()
        assert _tracked_alert_reason(pos, dec) == ""

    def test_no_position_no_trigger(self):
        assert _tracked_alert_reason(None, {"action": "SELL", "confidence": 0.9}) == ""
        assert _tracked_alert_reason({}, {"action": "SELL", "confidence": 0.9}) == ""
