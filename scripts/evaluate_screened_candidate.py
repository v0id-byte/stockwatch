#!/usr/bin/env python3
"""EXPLORATORY: IC-stability feature screening, then the standard seven gates.

Hypothesis: the 181-feature deploy set drowns weak signal in noise; a small
subset chosen for training-window IC stability may rank better out of sample.

Screening discipline (no lookahead): per-feature daily Spearman IC is computed
ONLY on dates strictly before fold 1's train_end_exclusive — a window that
precedes every validation and test date of every fold.  The top-K features by
|mean IC| / std(IC) are then fed through the unchanged candidate pipeline
(same folds, purge assertions, costs, simulator, summarize_baseline).

This is a post-hoc exploratory variant: the retrospective window has already
been observed repeatedly, so even a PASS here is weak evidence, and the
LOCKBOX stays out of reach for this candidate (see docs/lockbox_go_no_go_v1.md
section 2 — the one-shot batch is frozen).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.alpha158 import QLIB_ALPHA158_FEATURES  # noqa: E402
from analysis.oos_baselines import (  # noqa: E402
    DataContractError,
    TradingCosts,
    equal_weight_builder,
    fit_lightgbm_oos,
    make_expanding_folds,
    simulate_portfolio,
    summarize_baseline,
    topk_dropout_builder,
    validate_pit_panel,
)
from scripts.build_training_panel_v2 import CUSTOM_FEATURES  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

TARGET_RANK = "target_rank_20d"


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--output", default=str(root / "screened_candidate_report.json"))
    parser.add_argument("--top-features", type=int, default=40)
    parser.add_argument("--retrospective-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=22)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--drop-n", type=int, default=5)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    features = [*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES]
    panel = pd.read_parquet(Path(args.panel).expanduser())
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to evaluate")
    panel = validate_pit_panel(panel, QLIB_ALPHA158_FEATURES)

    labeled = panel["next_open_return_20d"].notna()
    panel.loc[labeled, TARGET_RANK] = (
        panel[labeled].groupby("signal_date")["next_open_return_20d"].rank(pct=True)
    )

    # Training-parity normalization (same block as the primary evaluator).
    member = panel["universe_member"].to_numpy()
    ranked = (
        panel.loc[member].groupby("signal_date", sort=False)[features].rank(pct=True) - 0.5
    ).astype(np.float32)
    normalized = pd.DataFrame(
        np.zeros((len(panel), len(features)), dtype=np.float32),
        index=panel.index, columns=features,
    )
    normalized.loc[member] = ranked.to_numpy()
    panel[features] = normalized.fillna(0.0)

    folds = make_expanding_folds(
        panel["signal_date"],
        min_train_days=args.min_train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
    )
    if not folds:
        raise DataContractError("not enough history for expanding folds")
    end_dates = panel[["signal_date", "label_end_date_20d"]].dropna()
    end_dates["label_end_date_20d"] = pd.to_datetime(end_dates["label_end_date_20d"])
    for fold_id, fold in enumerate(folds, start=1):
        train_max = end_dates.loc[
            end_dates["signal_date"] < fold["train_end_exclusive"], "label_end_date_20d"].max()
        val_max = end_dates.loc[
            (end_dates["signal_date"] >= fold["validation_start"])
            & (end_dates["signal_date"] < fold["validation_end_exclusive"]),
            "label_end_date_20d"].max()
        if pd.notna(train_max) and train_max >= pd.Timestamp(fold["validation_start"]):
            raise DataContractError(f"fold {fold_id} violates the train->val purge invariant")
        if pd.notna(val_max) and val_max >= pd.Timestamp(fold["test_start"]):
            raise DataContractError(f"fold {fold_id} violates the val->test purge invariant")

    # --- screening window: strictly before every evaluation date -------------
    screen_end = pd.Timestamp(folds[0]["train_end_exclusive"])
    screen = panel[
        panel["universe_member"] & (panel["signal_date"] < screen_end) & panel[TARGET_RANK].notna()
    ]
    if screen.empty:
        raise DataContractError("screening window is empty")

    # Features are already per-day pct ranks and so is the target, so per-day
    # Pearson on these columns IS the Spearman IC — computed vectorized.
    day = screen["signal_date"]
    f_dm = screen[features] - screen.groupby("signal_date")[features].transform("mean")
    t_dm = screen[TARGET_RANK] - screen.groupby("signal_date")[TARGET_RANK].transform("mean")
    num = f_dm.mul(t_dm, axis=0).groupby(day).sum()
    den = np.sqrt(
        f_dm.pow(2).groupby(day).sum().mul(t_dm.pow(2).groupby(day).sum(), axis=0)
    )
    daily_ic = (num / den).replace([np.inf, -np.inf], np.nan)
    stability = {}
    for name in features:
        ics = daily_ic[name].dropna()
        if len(ics) < 60 or ics.std() == 0:
            continue
        stability[name] = float(abs(ics.mean()) / ics.std())
    if len(stability) < args.top_features:
        raise DataContractError("too few screenable features")
    selected = sorted(stability, key=stability.get, reverse=True)[: args.top_features]
    print(f"screened {len(features)} -> {len(selected)} features "
          f"(window < {screen_end.date()}, {screen['signal_date'].nunique()} days)", flush=True)

    costs = TradingCosts(
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )
    predictions = fit_lightgbm_oos(panel, selected, TARGET_RANK, folds, num_threads=args.num_threads)
    portfolio = simulate_portfolio(predictions, topk_dropout_builder(args.top_k, args.drop_n), costs=costs)
    universe = simulate_portfolio(predictions, equal_weight_builder(), costs=costs)
    summary = summarize_baseline(
        predictions, portfolio, universe,
        frozen_start=args.retrospective_start,
        min_frozen_days=126, min_portfolio_cagr=0.05,
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate": f"lgbm_v2_screened_top{args.top_features}",
        "exploratory": True,
        "not_deployable_reason": "post-hoc exploratory variant; retrospective window reused",
        "screening": {
            "method": "abs(mean daily IC) / std over pre-fold1 training window",
            "window_end_exclusive": str(screen_end.date()),
            "selected": {name: stability[name] for name in selected},
        },
        "result": summary,
    }
    Path(args.output).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": summary.get("status"),
                      "mean_ic": (summary.get("factor") or {}).get("mean_ic"),
                      "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
