import pandas as pd

from scripts.evaluate_structured_event_strategy import _strategy_records


def test_event_strategy_uses_cash_when_too_few_positive_events():
    date = pd.Timestamp("2025-01-02")
    data = pd.DataFrame([
        {"trade_date": date, "score": 1.0, "active": 1, "target": 0.10},
        {"trade_date": date, "score": -1.0, "active": 1, "target": -0.10},
        {"trade_date": date, "score": 0.0, "active": 0, "target": 0.00},
    ])

    records = _strategy_records(
        data, "score", "active", target="target", sampled_dates=[date],
        top_k=20, min_positive=2, cost=0.002, risk_free=0.03,
        benchmark={date: 0.01}, horizon=20,
    )

    assert records.loc[0, "invested"] == False
    assert records.loc[0, "selected_names"] == 0
    assert records.loc[0, "universe"] == 0.0
