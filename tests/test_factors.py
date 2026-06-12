"""Tests for Alpha158 factor computation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


def _make_kline(n: int = 80) -> pd.DataFrame:
    """Generate a minimal synthetic K-line DataFrame."""
    rng = np.random.default_rng(42)
    closes = 10.0 + np.cumsum(rng.normal(0, 0.1, n))
    closes = np.maximum(closes, 0.5)
    df = pd.DataFrame({
        "trade_date": pd.date_range("2024-01-01", periods=n).strftime("%Y-%m-%d"),
        "open":   closes * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   closes * (1 + rng.uniform(0.000, 0.015, n)),
        "low":    closes * (1 - rng.uniform(0.000, 0.015, n)),
        "close":  closes,
        "volume": rng.integers(50_000, 500_000, n).astype(float),
        "amount": 0.0,
    })
    return df


class TestComputeTechScore:
    def test_returns_score_in_range(self):
        from analysis.technical import compute_tech_score
        kline = _make_kline(60).to_dict("records")
        result = compute_tech_score(kline)
        assert "score" in result
        assert -1.0 <= result["score"] <= 1.0

    def test_insufficient_data_returns_zero(self):
        from analysis.technical import compute_tech_score
        kline = _make_kline(10).to_dict("records")
        result = compute_tech_score(kline)
        assert result["score"] == 0

    def test_details_present(self):
        from analysis.technical import compute_tech_score
        kline = _make_kline(60).to_dict("records")
        result = compute_tech_score(kline)
        assert isinstance(result.get("details"), dict)


class TestComputeAlpha158:
    def test_basic_computation(self):
        from analysis.factors import compute_alpha158
        df = _make_kline(80)
        factors = compute_alpha158(df, df)
        assert isinstance(factors, dict)
        assert len(factors) > 0

    def test_returns_numeric_values(self):
        from analysis.factors import compute_alpha158
        df = _make_kline(80)
        factors = compute_alpha158(df, df)
        for k, v in factors.items():
            assert isinstance(v, (int, float)), f"Factor {k} is not numeric: {v}"

    def test_requires_minimum_rows(self):
        from analysis.factors import compute_alpha158
        df = _make_kline(5)
        factors = compute_alpha158(df, df)
        # Should not raise; may return empty dict or partial values
        assert isinstance(factors, dict)
