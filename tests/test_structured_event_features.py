import pandas as pd

from scripts.build_sentiment_features import _base_frame
from scripts.build_structured_event_features import _add_event_features


def test_structured_event_rolls_forward_without_future_leakage():
    dates = pd.Index(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    grid = pd.DataFrame({"trade_date": dates, "code": ["000001"] * 3})
    frame = _base_frame(grid, dates)
    events = pd.DataFrame([{
        "code": "000001",
        "feature_date": pd.Timestamp("2025-01-03"),
        "category": "earnings",
        "direction": 1,
        "signed_score": 2.0,
        "novelty": 1.0,
        "confidence": 0.9,
        "magnitude_value": None,
        "magnitude_percent": 50.0,
    }])

    out = _add_event_features(frame, events).set_index("trade_date")

    assert out.loc[pd.Timestamp("2025-01-02"), "event_count_7d"] == 0
    assert out.loc[pd.Timestamp("2025-01-03"), "event_count_7d"] == 1
    assert out.loc[pd.Timestamp("2025-01-06"), "event_signed_score_7d"] == 2.0
    assert out.loc[pd.Timestamp("2025-01-06"), "event_earnings_count_20d"] == 1
