import pandas as pd

from scripts.evaluate_curve_model import _non_overlapping


def test_curve_evaluation_uses_non_overlapping_dates():
    dates = pd.date_range("2025-01-01", periods=45, freq="D")
    frame = pd.DataFrame({"trade_date": dates, "value": range(45)})

    sampled = _non_overlapping(frame, 20)

    assert sampled["trade_date"].tolist() == [dates[0], dates[20], dates[40]]
