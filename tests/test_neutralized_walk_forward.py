"""Tests for neutralized label and walk-forward diagnostics."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_neutralized_walk_forward import (
    _long_only_stats,
    _merge_sector_exposure,
    _negative_screen_stats,
    _non_overlapping,
    _primary_negative_screen_metrics,
    make_walk_forward_folds,
    neutralize_by_date,
)


def test_neutralize_by_date_removes_numeric_exposure():
    rng = np.random.default_rng(7)
    rows = []
    for day in pd.date_range("2024-01-01", periods=4):
        exposure = np.linspace(-1, 1, 80)
        noise = rng.normal(0, 0.01, len(exposure))
        target = 0.03 + 0.4 * exposure + noise
        for i, (x, y) in enumerate(zip(exposure, target)):
            rows.append({
                "trade_date": day,
                "code": f"{i:06d}",
                "target": y,
                "style": x,
            })
    data = pd.DataFrame(rows)

    residual = neutralize_by_date(data, "target", ["style"], winsor_tail=0)
    checked = data.assign(residual=residual)

    for _date, group in checked.groupby("trade_date"):
        assert abs(group["residual"].corr(group["style"])) < 1e-10
        assert abs(group["residual"].mean()) < 1e-10


def test_walk_forward_folds_leave_purge_gap():
    dates = pd.Index(pd.bdate_range("2023-01-02", periods=420))

    folds = make_walk_forward_folds(
        dates,
        horizon_days=20,
        min_train_months=6,
        fold_months=3,
        val_months=2,
        max_folds=3,
    )

    assert len(folds) == 3
    for fold in folds:
        assert fold["train_end_exclusive"] == fold["val_start"]
        assert fold["val_start"] < fold["val_end_exclusive"] == fold["purge_start"]
        assert fold["purge_start"] < fold["test_start"]
        assert fold["val_start"] >= dates[0] + pd.DateOffset(months=6)
        gap = dates.searchsorted(fold["test_start"]) - dates.searchsorted(fold["purge_start"])
        assert gap >= 20


def test_non_overlapping_keeps_every_nth_trade_date():
    data = pd.DataFrame({
        "trade_date": pd.bdate_range("2024-01-01", periods=7).repeat(2),
        "code": ["000001", "000002"] * 7,
    })

    sampled = _non_overlapping(data, 3)

    assert sampled["trade_date"].drop_duplicates().tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-04"),
        pd.Timestamp("2024-01-09"),
    ]


def test_long_only_stats_reports_top_bucket_net_excess():
    data = pd.DataFrame({
        "trade_date": [pd.Timestamp("2024-01-01")] * 100,
        "pred": np.arange(100),
        "target": np.linspace(-0.05, 0.05, 100),
    })

    stats = _long_only_stats(data, "pred", "target", min_per_date=30, top_k=5, round_trip_cost=0.002)

    assert stats is not None
    ranks = data["pred"].rank(method="first", pct=True)
    deciles = (ranks * 10).clip(upper=9).astype(int)
    expected_top = data[deciles == 9]["target"].mean()
    expected_universe = data["target"].mean()
    assert stats["top_decile_excess"] == expected_top - expected_universe
    assert stats["top_decile_net_excess"] == expected_top - expected_universe - 0.002
    assert stats["top_k"] == 5


def test_negative_screen_stats_excludes_low_score_losers():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-30", "2024-02-27"]):
        for idx in range(100):
            rows.append({
                "trade_date": date,
                "pred": float(idx),
                "target": -0.10 if idx < 10 else 0.02,
            })
    data = pd.DataFrame(rows)

    stats = _negative_screen_stats(
        data,
        "pred",
        "target",
        min_per_date=30,
        exclude_bottom_fraction=0.10,
        round_trip_cost=0.002,
        horizon_days=20,
        risk_free_annual=0.03,
    )

    assert stats is not None
    assert stats["period_count"] == 3
    assert stats["average_excluded_fraction"] == pytest.approx(0.10)
    assert stats["filtered_excess_vs_universe"]["mean"] == pytest.approx(0.012)
    assert stats["excluded_bucket_excess_vs_universe"]["mean"] == pytest.approx(-0.108)
    assert stats["filtered"]["cagr"] > stats["baseline"]["cagr"]
    assert stats["acceptance"]["beats_risk_free"] is True


def test_negative_screen_stats_rejects_non_predictive_exclusion():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-30", "2024-02-27"]):
        for idx in range(100):
            rows.append({
                "trade_date": date,
                "pred": float(idx),
                "target": 0.01,
            })
    data = pd.DataFrame(rows)

    stats = _negative_screen_stats(
        data,
        "pred",
        "target",
        min_per_date=30,
        exclude_bottom_fraction=0.10,
        round_trip_cost=0.002,
        horizon_days=20,
        risk_free_annual=0.03,
    )

    assert stats is not None
    assert stats["filtered_excess_vs_universe"]["mean"] == pytest.approx(0.0)
    assert stats["acceptance"]["positive_net_excess"] is False


def test_primary_negative_screen_requires_recent_frozen_oos_to_pass():
    passing_screen = {
        "period_count": 35,
        "annualized_delta": 0.01,
        "acceptance": {
            "passes_performance_gate": True,
            "passes_period_gate": True,
        },
    }
    failing_oos = {
        "period_count": 18,
        "annualized_delta": -0.02,
        "acceptance": {
            "passes_performance_gate": False,
            "passes_period_gate": False,
            "positive_net_excess": False,
        },
    }
    metrics = {
        "neutral_label_model": {
            "non_overlapping": {"forward_20d_return": {"negative_screen": passing_screen}},
        },
        "negative_screen_fold_stability": {
            "neutral_label_model": {"forward_20d_return": {"positive_rate": 2 / 3}},
        },
        "frozen_oos_2025_2026": {
            "neutral_label_model": {
                "non_overlapping": {"forward_20d_return": {"negative_screen": failing_oos}},
            },
        },
    }

    result = _primary_negative_screen_metrics(metrics, "forward_20d_return")

    assert result["status"] == "REJECTED"
    assert result["gate"]["frozen_oos_passed"] is False


def test_merge_sector_exposure_uses_historical_asof_dates(tmp_path):
    data = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-01", "2024-07-01", "2024-01-01"]),
        "code": ["000001", "000001", "000002"],
    })
    sectors = pd.DataFrame({
        "code": ["000001", "000001", "000002"],
        "start_date": pd.to_datetime(["2023-01-01", "2024-06-01", "2023-01-01"]),
        "sector": ["银行", "非银金融", "电子"],
    })
    path = tmp_path / "sector.parquet"
    sectors.to_parquet(path, index=False)

    merged, meta = _merge_sector_exposure(data, str(path))

    assert meta["kind"] == "point_in_time"
    by_key = dict(zip(zip(merged["code"], merged["trade_date"]), merged["sector"]))
    assert by_key[("000001", pd.Timestamp("2024-01-01"))] == "银行"
    assert by_key[("000001", pd.Timestamp("2024-07-01"))] == "非银金融"
    assert by_key[("000002", pd.Timestamp("2024-01-01"))] == "电子"
