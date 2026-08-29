"""Tests for strict frozen OOS baselines and common portfolio accounting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.oos_baselines import (
    DataContractError,
    EnhancementConstraints,
    TradingCosts,
    _return_stats,
    assign_size_bucket,
    factor_diagnostics,
    index_enhancement_builder,
    make_expanding_folds,
    simulate_portfolio,
    summarize_baseline,
    topk_dropout_builder,
    validate_pit_panel,
)
from scripts.evaluate_frozen_oos_baselines import (
    CH4_STYLE_FEATURES,
    CH4_WITH_ILLIQUIDITY_FEATURES,
)


def _panel(dates: int = 2, names: int = 6) -> pd.DataFrame:
    rows = []
    for day in range(dates):
        signal = pd.Timestamp("2025-01-02") + pd.offsets.BDay(day)
        execution = signal + pd.offsets.BDay(1) + pd.Timedelta(hours=9, minutes=30)
        exit_at = signal + pd.offsets.BDay(2) + pd.Timedelta(hours=9, minutes=30)
        for index in range(names):
            rows.append({
                "signal_date": signal,
                "execution_at": execution,
                "exit_at": exit_at,
                "feature_available_at": signal + pd.Timedelta(hours=15),
                "code": f"{index + 1:06d}",
                "next_open_return": 0.001 * (index + 1),
                "buyable": True,
                "sellable": True,
                "universe_member": True,
                "benchmark_return": 0.002,
                "feature": float(index),
                "score": float(index if day == 0 else names - index),
                "fold": 1,
            })
    return pd.DataFrame(rows)


def test_validate_panel_fails_closed_on_unknown_timing_and_low_coverage():
    frame = _panel()
    frame.loc[0, "feature_available_at"] = frame.loc[0, "execution_at"]
    with pytest.raises(DataContractError, match="strictly before"):
        validate_pit_panel(frame, ["feature"])


def test_validate_panel_rejects_stock_specific_execution_calendar():
    frame = _panel()
    frame.loc[0, "execution_at"] += pd.offsets.BDay(1)
    frame.loc[0, "exit_at"] += pd.offsets.BDay(1)
    with pytest.raises(DataContractError, match="one market-calendar execution"):
        validate_pit_panel(frame, ["feature"])


def test_validate_panel_rejects_unparseable_and_zero_stock_codes():
    frame = _panel()
    frame.loc[0, "code"] = "not-a-stock"
    with pytest.raises(DataContractError, match="invalid stock codes"):
        validate_pit_panel(frame, ["feature"])
    frame = _panel()
    frame.loc[0, "code"] = "000000"
    with pytest.raises(DataContractError, match="invalid stock codes"):
        validate_pit_panel(frame, ["feature"])


def test_ch4_style_uses_pmo_and_keeps_amihud_as_separate_increment():
    assert "abnormal_turnover_20_240" in CH4_STYLE_FEATURES
    assert "amihud_20d" not in CH4_STYLE_FEATURES
    assert "amihud_20d" in CH4_WITH_ILLIQUIDITY_FEATURES


def test_size_bucket_and_factor_diagnostics_ignore_nonmembers():
    date = pd.Timestamp("2025-01-02")
    members = pd.DataFrame({
        "signal_date": date,
        "code": [f"{index:06d}" for index in range(30)],
        "universe_member": True,
        "log_market_cap": np.arange(30.0),
        "score": np.arange(30.0),
        "next_open_return": np.arange(30.0) / 1000,
        "fold": 1,
    })
    outsiders = pd.DataFrame({
        "signal_date": date,
        "code": [f"9{index:05d}" for index in range(30)],
        "universe_member": False,
        "log_market_cap": np.arange(-100.0, -70.0),
        "score": np.arange(30.0),
        "next_open_return": -np.arange(30.0),
        "fold": 1,
    })
    frame = pd.concat([members, outsiders], ignore_index=True)
    bucket = assign_size_bucket(frame)
    assert (bucket.loc[~frame["universe_member"]] == "not_in_universe").all()
    diagnostics = factor_diagnostics(frame)
    assert diagnostics["mean_ic"] == pytest.approx(1.0)

    frame = _panel()
    frame.loc[0:7, "feature"] = np.nan
    with pytest.raises(DataContractError, match="coverage below 50%"):
        validate_pit_panel(frame, ["feature"])


def test_expanding_folds_have_purges_on_both_sides_of_validation():
    dates = pd.bdate_range("2020-01-01", periods=100)
    folds = make_expanding_folds(
        dates,
        min_train_days=40,
        validation_days=10,
        test_days=10,
        purge_days=2,
    )
    assert folds
    first = folds[0]
    ordered = pd.Index(dates)
    train_end = ordered.get_loc(first["train_end_exclusive"])
    validation_start = ordered.get_loc(first["validation_start"])
    validation_end = ordered.get_loc(first["validation_end_exclusive"])
    test_start = ordered.get_loc(first["test_start"])
    assert validation_start - train_end == 2
    assert test_start - validation_end == 2


def test_topk_dropout_and_asymmetric_a_share_costs_are_accounted():
    frame = _panel(dates=2, names=4)
    records = simulate_portfolio(
        frame,
        topk_dropout_builder(top_k=2, drop_n=1),
        costs=TradingCosts(commission_bps=3, stamp_duty_bps=5, slippage_bps=2),
    )
    first = records.iloc[0]
    second = records.iloc[1]
    assert first["holding_count"] == 2
    assert first["buy_notional"] == pytest.approx(1 / (1 + 5 / 10_000))
    assert first["sell_notional"] == 0
    assert first["trading_cost"] == pytest.approx(first["buy_notional"] * 5 / 10_000)
    assert first["cash_weight"] >= 0
    assert second["sell_notional"] > 0
    buy_only_cost = second["buy_notional"] * 5 / 10_000
    assert second["trading_cost"] > buy_only_cost


def test_topk_dropout_does_not_sell_when_existing_names_stay_above_new_candidates():
    frame = _panel(dates=2, names=4)
    first_scores = frame[frame["signal_date"] == frame["signal_date"].min()].set_index("code")["score"]
    second_date = frame["signal_date"].max()
    frame.loc[frame["signal_date"] == second_date, "score"] = (
        frame.loc[frame["signal_date"] == second_date, "code"].map(first_scores)
    )
    records = simulate_portfolio(frame, topk_dropout_builder(top_k=2, drop_n=1))
    assert records.iloc[1]["buy_notional"] == pytest.approx(0)
    assert records.iloc[1]["sell_notional"] == pytest.approx(0)


def test_topk_dropout_never_selects_nonmember_even_with_highest_score():
    frame = _panel(dates=1, names=4)
    best = frame["score"].idxmax()
    frame["next_open_return"] = 0.0
    frame.loc[best, "universe_member"] = False
    frame.loc[best, "next_open_return"] = 1.0
    records = simulate_portfolio(frame, topk_dropout_builder(top_k=2, drop_n=1))
    assert records.iloc[0]["gross_return"] == pytest.approx(0)


def test_unbuyable_name_is_not_added_to_topk_portfolio():
    frame = _panel(dates=1, names=4)
    best = frame["score"].idxmax()
    frame.loc[best, "buyable"] = False
    records = simulate_portfolio(frame, topk_dropout_builder(top_k=2, drop_n=1))
    assert records.iloc[0]["holding_count"] == 1
    assert records.iloc[0]["cash_weight"] == pytest.approx(0.5 - 0.5 * 8 / 10_000)


def test_index_enhancement_weights_respect_neutrality_and_risk_caps():
    group = _panel(dates=1, names=6)
    group["benchmark_weight"] = 1 / 6
    group["sector"] = ["A", "A", "B", "B", "C", "C"]
    group["log_market_cap"] = [8.0, 8.4, 8.1, 8.8, 8.3, 9.0]
    group["beta"] = [0.8, 1.2, 0.9, 1.1, 0.7, 1.3]
    group["volatility_20d"] = 0.25
    constraints = EnhancementConstraints(
        max_tracking_error=0.05,
        max_one_way_turnover=1.0,
        max_active_weight=0.03,
        max_stock_weight=0.30,
        exposure_tolerance=1e-8,
    )
    weights = index_enhancement_builder(constraints)(group, pd.Series(dtype=float))
    members = group.set_index("code").loc[weights.index]
    benchmark = members["benchmark_weight"] / members["benchmark_weight"].sum()
    active = weights - benchmark
    assert weights.sum() == pytest.approx(1.0)
    assert weights.min() >= 0
    assert active.abs().max() <= constraints.max_active_weight + 1e-10
    assert np.sqrt(np.sum(np.square(active * members["volatility_20d"]))) <= 0.05 + 1e-10
    assert abs(float(active @ members["log_market_cap"])) <= 1e-8
    assert abs(float(active @ members["beta"])) <= 1e-8
    for sector in members["sector"].unique():
        assert abs(float(active[members["sector"] == sector].sum())) <= 1e-8


def test_index_panel_rejects_missing_benchmark_weight_mass():
    frame = _panel(dates=1, names=6)
    frame["benchmark_weight"] = 0.1
    frame["sector"] = "A"
    frame["log_market_cap"] = 8.0
    frame["beta"] = 1.0
    frame["volatility_20d"] = 0.2
    with pytest.raises(DataContractError, match="sum to one"):
        validate_pit_panel(frame, ["feature"], require_index_columns=True)


def test_index_panel_allows_explicit_rolling_warmup_nans():
    frame = _panel(dates=2, names=6)
    frame["benchmark_weight"] = 1 / 6
    frame["sector"] = "A"
    frame["log_market_cap"] = 8.0
    frame["beta"] = 1.0
    frame["volatility_20d"] = 0.2
    first_date = frame["signal_date"].min()
    frame.loc[frame["signal_date"] == first_date, ["beta", "volatility_20d"]] = np.nan
    result = validate_pit_panel(frame, ["feature"], require_index_columns=True)
    assert result["beta"].notna().mean() == pytest.approx(0.5)


def test_index_enhancement_rejects_benchmark_weight_above_stock_cap():
    group = _panel(dates=1, names=5)
    group["benchmark_weight"] = [0.25, 0.20, 0.20, 0.20, 0.15]
    group["sector"] = ["A", "A", "B", "B", "C"]
    group["log_market_cap"] = np.arange(5.0)
    group["beta"] = np.linspace(0.8, 1.2, 5)
    group["volatility_20d"] = 0.2
    constraints = EnhancementConstraints(
        max_tracking_error=0.1,
        max_one_way_turnover=1.0,
        max_active_weight=0.1,
        max_stock_weight=0.18,
    )
    with pytest.raises(DataContractError, match="no grandfathering"):
        index_enhancement_builder(constraints)(group, pd.Series(dtype=float))


def test_return_stats_counts_first_day_loss_in_drawdown():
    stats = _return_stats(pd.Series([-0.10, 0.0]))
    assert stats["max_drawdown"] == pytest.approx(-0.10)


def test_acceptance_gate_requires_absolute_five_percent_cagr():
    dates = pd.bdate_range("2025-01-02", periods=2)
    predictions = pd.DataFrame([
        {
            "signal_date": day,
            "code": f"{index + 1:06d}",
            "score": float(index),
            "next_open_return": float(index) / 10_000,
            "universe_member": True,
            "fold": 1,
        }
        for day in dates for index in range(30)
    ])
    portfolio = pd.DataFrame({
        "signal_date": dates,
        "net_return": [0.0001, 0.0001],
        "benchmark_return": [0.0, 0.0],
        "excess_vs_benchmark": [0.0001, 0.0001],
        "fold": [1, 1],
        "one_way_turnover": [0.0, 0.0],
        "trading_cost": [0.0, 0.0],
    })
    universe = pd.DataFrame({
        "signal_date": dates,
        "net_return": [0.0, 0.0],
    })

    report = summarize_baseline(
        predictions, portfolio, universe,
        frozen_start="2025-01-01", min_frozen_days=2,
    )

    assert report["portfolio"]["cagr"] < 0.05
    assert not report["acceptance"]["portfolio_cagr_at_least_target"]
    assert report["status"] == "REJECTED"
