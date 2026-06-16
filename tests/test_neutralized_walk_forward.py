"""Tests for neutralized label and walk-forward diagnostics."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_neutralized_walk_forward import (
    _non_overlapping,
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
