import pandas as pd

from scripts.build_sequence_features import SEQUENCE_FEATURES, _numeric_sequence_frame


def _prices():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    close = pd.Series(range(100, 200), dtype=float)
    return pd.DataFrame({
        "trade_date": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": pd.Series(range(1000, 1100), dtype=float),
    })


def test_sequence_shape_and_numeric_columns():
    out = _numeric_sequence_frame(_prices())

    assert len(SEQUENCE_FEATURES) == 200
    assert list(out.columns) == ["trade_date", *SEQUENCE_FEATURES]
    assert out.loc[99, "SEQ_RET_00"] > 0
    assert out.loc[99, "SEQ_RET_01"] == out.loc[98, "SEQ_RET_00"]


def test_sequence_features_are_causal():
    base = _prices()
    changed = base.copy()
    changed.loc[90:, ["open", "high", "low", "close", "volume"]] *= 10

    before = _numeric_sequence_frame(base)
    after = _numeric_sequence_frame(changed)

    pd.testing.assert_series_equal(before.loc[80, SEQUENCE_FEATURES], after.loc[80, SEQUENCE_FEATURES])
