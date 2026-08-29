import pandas as pd

from scripts.evaluate_dual_model import _passes_gate, _selection_stats


def test_dual_selection_excludes_predicted_risk_before_alpha_ranking():
    rows = []
    for index in range(10):
        rows.append({
            "trade_date": pd.Timestamp("2026-01-02"),
            "code": f"{index:06d}",
            "fold": 1,
            "alpha": 100 - index,
            "risk_safe": index,
            "forward_20d_return": -0.50 if index == 0 else 0.10,
        })
    data = pd.DataFrame(rows)

    plain = _selection_stats(
        data, "alpha", "forward_20d_return", horizon=20, top_k=1, cost=0,
        risk_free=0.03, benchmark={}, min_names=5,
    )
    dual = _selection_stats(
        data, "alpha", "forward_20d_return", horizon=20, top_k=1, cost=0,
        risk_free=0.03, benchmark={}, risk_col="risk_safe", exclude_fraction=0.10,
        min_names=5,
    )

    assert plain["selected"]["mean_period_return"] == -0.50
    assert dual["selected"]["mean_period_return"] == 0.10
    assert dual["average_excluded_names"] == 1


def test_gate_requires_benchmark_universe_and_stability():
    passing = {
        "period_count": 12,
        "selected": {"cagr": 0.08},
        "annualized_delta_vs_universe": 0.02,
        "annualized_delta_vs_csi500": 0.01,
        "positive_fold_rate": 0.75,
    }

    assert _passes_gate(passing, 12)
    assert not _passes_gate({**passing, "annualized_delta_vs_universe": 0.005}, 12)


def test_regime_off_uses_cash_instead_of_stock_selection():
    data = pd.DataFrame({
        "trade_date": [pd.Timestamp("2026-01-02")] * 5,
        "code": [f"{index:06d}" for index in range(5)],
        "fold": [1] * 5,
        "alpha": [5, 4, 3, 2, 1],
        "forward_20d_return": [-0.5] * 5,
    })

    timed = _selection_stats(
        data, "alpha", "forward_20d_return", horizon=20, top_k=1, cost=0,
        risk_free=0.03, benchmark={}, min_names=5,
        risk_on={pd.Timestamp("2026-01-02"): False},
    )

    assert timed["selected"]["mean_period_return"] > 0
    assert timed["risk_on_rate"] == 0
    assert timed["average_selected_names"] == 0
