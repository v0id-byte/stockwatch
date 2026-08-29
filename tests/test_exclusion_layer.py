import pandas as pd
import pytest

from scripts.evaluate_exclusion_layer import (
    _attach_filters,
    _build_pead_flags,
    _entry_date_after,
    _evaluate_filter,
    _preregistered_controls,
    _preregistered_summary,
)


def test_entry_date_after_weekend():
    trade_dates = pd.Index(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]))

    assert _entry_date_after(pd.Timestamp("2024-01-06"), trade_dates) == pd.Timestamp("2024-01-08")


def test_build_pead_flags_expands_negative_event_window():
    trade_dates = pd.Index(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    events = pd.DataFrame([{
        "code": "000001",
        "entry_date": pd.Timestamp("2024-01-03"),
        "sign": -1,
        "strong_negative": True,
        "is_turnaround": False,
    }])

    flags = _build_pead_flags(events, trade_dates)
    active = flags[flags["pead_negative_20d"]].sort_values("trade_date")

    assert active["trade_date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")]
    assert flags["pead_strong_negative_20d"].sum() == 3


def test_attach_filters_builds_combo_flags():
    train = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-02"),
        "code": "000001",
        "forward_20d_return": 0.01,
    }])
    sentiment = pd.DataFrame([{
        "trade_date": pd.Timestamp("2024-01-02"),
        "code": "000001",
        "ann_count_20d": 3,
        "ann_holding_count_20d": 1,
        "ann_risk_count_20d": 0,
        "ann_capital_action_count_20d": 0,
    }])
    pead_flags = pd.DataFrame(columns=["trade_date", "code"])

    data = _attach_filters(train, sentiment, pead_flags)

    assert bool(data.loc[0, "ann_holding_20d"]) is True
    assert bool(data.loc[0, "combo_narrow_negative"]) is True
    assert bool(data.loc[0, "combo_broad_negative"]) is True


def test_evaluate_filter_improves_when_flagged_names_underperform():
    rows = []
    for date in pd.to_datetime(["2024-01-02", "2024-01-22", "2024-02-19"]):
        for idx in range(120):
            flagged = idx < 10
            rows.append({
                "trade_date": date,
                "code": f"{idx:06d}",
                "forward_20d_return": -0.10 if flagged else 0.02,
                "bad_flag": flagged,
            })
    data = pd.DataFrame(rows)
    market = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-22", "2024-02-19"]),
        "csi300_forward_20d_return": [0.0, 0.0, 0.0],
    })

    result = _evaluate_filter(data, market, "bad_flag", horizon=20, step=1, min_names=100, cost=0.002)

    assert result["rebalances"] == 3
    assert result["active_rebalances"] == 3
    assert result["filtered_excess_vs_universe"]["mean"] == pytest.approx(0.01)
    assert result["excluded_bucket_excess_vs_universe"]["mean"] == pytest.approx(-0.11)


def test_evaluate_filter_reports_year_and_oos_breakdowns():
    rows = []
    for date in pd.to_datetime(["2023-01-03", "2024-01-03", "2025-01-03", "2026-01-05"]):
        for idx in range(120):
            flagged = idx < 10
            rows.append({
                "trade_date": date,
                "code": f"{idx:06d}",
                "forward_20d_return": -0.05 if flagged else 0.01,
                "bad_flag": flagged,
            })
    data = pd.DataFrame(rows)
    market = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2024-01-03", "2025-01-03", "2026-01-05"]),
        "csi300_forward_20d_return": [0.0, 0.0, 0.0, 0.0],
    })

    result = _evaluate_filter(data, market, "bad_flag", horizon=20, step=1, min_names=100, cost=0.002)

    assert set(result["by_year"]) == {"2023", "2024", "2025", "2026"}
    assert result["oos_split"]["train_2022_2024"]["rebalances"] == 2
    assert result["oos_split"]["test_2025_2026"]["rebalances"] == 2
    assert result["downside"]["excess_positive_rate"] == 1.0


def test_preregistered_summary_uses_frozen_candidate_only():
    rows = []
    for date in pd.to_datetime(["2023-01-03", "2024-01-03", "2025-01-03", "2026-01-05"]):
        for idx in range(120):
            flagged = idx < 10
            rows.append({
                "trade_date": date,
                "code": f"{idx:06d}",
                "forward_20d_return": -0.05 if flagged else 0.01,
                "combo_broad_negative": flagged,
            })
    data = pd.DataFrame(rows)
    market = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03", "2024-01-03", "2025-01-03", "2026-01-05"]),
        "csi300_forward_20d_return": [0.0, 0.0, 0.0, 0.0],
    })
    result = _evaluate_filter(data, market, "combo_broad_negative", horizon=20, step=1, min_names=100, cost=0.002)

    summary = _preregistered_summary({"combo_broad_negative": {"20": result}, "better_looking_filter": {"5": {}}})

    assert summary["filter"] == "combo_broad_negative"
    assert summary["horizon"] == 20
    assert summary["stability"]["positive_excess_years"] == 4
    assert summary["stability"]["oos_2025_2026"]["rebalances"] == 2


def test_preregistered_controls_compare_same_count_exclusions():
    rows = []
    dates = pd.to_datetime(["2023-01-03", "2024-01-03", "2025-01-03", "2026-01-05"])
    for date in dates:
        for idx in range(120):
            flagged = idx < 10
            rows.append({
                "trade_date": date,
                "code": f"{idx:06d}",
                "forward_20d_return": -0.05 if flagged else 0.01,
                "combo_broad_negative": flagged,
                "STD20": 10.0 if flagged else 1.0,
            })
    data = pd.DataFrame(rows)
    market = pd.DataFrame({
        "trade_date": dates,
        "csi300_forward_20d_return": [0.0, 0.0, 0.0, 0.0],
    })

    controls = _preregistered_controls(
        data,
        market,
        horizon=20,
        step=1,
        min_names=100,
        cost=0.002,
        reps=5,
        seed=7,
    )

    assert controls["status"] == "ok"
    assert controls["random_same_count"]["completed_reps"] == 5
    assert controls["high_volatility_same_count"]["records"] == 4
    assert controls["actual"]["max_drawdown_delta"] is not None
    assert controls["random_same_count"]["actual_percentile"]["max_drawdown_delta"] is not None
