import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_monitor_report import _realized_targets


def _bars(dates, opens, closes):
    return pd.DataFrame({
        "trade_date": pd.to_datetime(dates),
        "open": opens, "close": closes,
    })


def test_realized_targets_label_math_and_entry_lag(monkeypatch):
    dates = pd.bdate_range("2026-01-05", periods=30)
    opens = [10.0] * 30
    closes = [10.0] * 30
    closes[5] = 8.0  # worst close inside the window of a day-0 signal
    klines = {"600000": _bars(dates, opens, closes)}

    # long-suspension code: next bar 20 calendar days after the signal
    gap_dates = [pd.Timestamp("2026-01-05")] + list(pd.bdate_range("2026-01-26", periods=25))
    klines["000001"] = _bars(gap_dates, [10.0] * 26, [10.0] * 26)

    import core.model_scoring as ms
    monkeypatch.setattr(ms, "_download_history", lambda codes: klines)

    scores = pd.DataFrame({
        "trade_date": [dates[0], pd.Timestamp("2026-01-05")],
        "code": ["600000", "000001"],
        "risk_score": [0.5, 0.5],
        "risk_model_version": ["v", "v"],
    })
    out, dropped = _realized_targets(scores, max_entry_lag_days=5)
    # 600000: entry = day1 open 10.0, worst close in bars[1..20] = 8.0 -> -0.2
    assert len(out) == 1 and out.iloc[0]["code"] == "600000"
    assert abs(out.iloc[0]["realized_drawdown"] - (-0.2)) < 1e-12
    # 000001 dropped by the entry-lag guard
    assert dropped["entry_lag"] == 1


def test_realized_targets_incomplete_window(monkeypatch):
    dates = pd.bdate_range("2026-01-05", periods=10)  # too short for 20 bars
    klines = {"600000": _bars(dates, [10.0] * 10, [10.0] * 10)}
    import core.model_scoring as ms
    monkeypatch.setattr(ms, "_download_history", lambda codes: klines)
    scores = pd.DataFrame({
        "trade_date": [dates[0]], "code": ["600000"],
        "risk_score": [0.5], "risk_model_version": ["v"],
    })
    out, dropped = _realized_targets(scores, max_entry_lag_days=5)
    assert out.empty and dropped["window_incomplete"] == 1
