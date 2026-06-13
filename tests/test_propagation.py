"""Tests for lead-lag propagation features."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _synthetic_panel(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d")
    leader_ret = rng.normal(0.0005, 0.006, n)
    leader_events = [35, 50, 65, 80, 95]
    leader_ret[leader_events] = 0.06
    follower_ret = rng.normal(0.0002, 0.004, n) + np.roll(leader_ret, 1) * 0.75
    follower_ret[0] = 0.0
    random_ret = rng.normal(0.0002, 0.008, n)

    def close_from_returns(returns):
        return 10 * np.cumprod(1 + returns)

    rows = []
    for code, returns, volume_base in [
        ("000001", leader_ret, 100_000),
        ("000002", follower_ret, 90_000),
        ("000003", random_ret, 95_000),
    ]:
        volume = np.full(n, volume_base, dtype=float)
        if code == "000001":
            volume[leader_events] = volume_base * 4
        close = close_from_returns(returns)
        for i in range(n):
            rows.append({
                "trade_date": dates[i],
                "code": code,
                "close": close[i],
                "volume": volume[i],
            })
    return pd.DataFrame(rows)


def test_propagation_feature_frame_detects_lagged_follower():
    from analysis.propagation import compute_propagation_feature_frame

    panel = _synthetic_panel()
    features = compute_propagation_feature_frame(
        panel,
        leader_return_threshold=0.04,
        leader_volz_threshold=0.5,
        min_corr=0.05,
        lookback=50,
    )
    event_date = panel[panel["code"] == "000001"]["trade_date"].iloc[95]
    event_rows = features[features["trade_date"] == event_date].set_index("code")

    assert event_rows.loc["000002", "prop_score"] > 0
    assert event_rows.loc["000002", "prop_relation_corr_1d"] > 0
    assert event_rows.loc["000002", "prop_score"] >= event_rows.loc["000003", "prop_score"]


def test_latest_propagation_context_mentions_leader():
    from analysis.propagation import compute_latest_propagation_features

    panel = _synthetic_panel()
    kline_by_code = {
        code: group.sort_values("trade_date").assign(
            open=group["close"],
            high=group["close"],
            low=group["close"],
            amount=0,
        ).to_dict("records")
        for code, group in panel.groupby("code")
    }
    quotes = {
        "000001": {"name": "领涨A", "pct_change": 6.0},
        "000002": {"name": "跟随B", "pct_change": 0.5},
        "000003": {"name": "随机C", "pct_change": 0.0},
    }

    features, contexts = compute_latest_propagation_features(
        ["000001", "000002", "000003"],
        quotes,
        kline_by_code,
        leader_return_threshold=0.04,
        min_corr=0.05,
        lookback=50,
    )

    assert features["000002"]["prop_score"] > 0
    assert "领涨A(000001)" in contexts["000002"]
