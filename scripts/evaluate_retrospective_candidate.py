#!/usr/bin/env python3
"""Run the seven acceptance gates on a candidate feature set (retrospective OOS).

Same machinery as evaluate_frozen_oos_baselines (purged expanding folds,
topk-dropout, TradingCosts, summarize_baseline) but with:
* the candidate feature set (deploy = Qlib158 + robust custom 40; research
  additionally includes the llm_* family, flagged NOT_DEPLOYABLE_V1);
* per-day cross-sectional rank-normalized features (training parity);
* a 20-trading-day executable rank target;
* purge asserted through label_end_date_20d, not assumed from purge_days.

Naming discipline: the 2025+ window is *retrospective* OOS — it has been
observed by earlier research.  A PASS here means "historical evidence", never
"confirmed on unseen data"; that is the lockbox / prospective gate's job.
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
    parser.add_argument("--feature-set", choices=("deploy", "research"), default="deploy")
    parser.add_argument("--output", default=str(root / "retrospective_candidate_report.json"))
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
    parser.add_argument("--min-frozen-days", type=int, default=126)
    parser.add_argument("--min-portfolio-cagr", type=float, default=0.05)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def _candidate_features(args: argparse.Namespace, columns: set[str]) -> list[str]:
    features = [f for f in (*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES) if f in columns]
    missing = [f for f in (*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES) if f not in columns]
    if missing:
        raise DataContractError(f"candidate panel lacks features: {missing[:8]}")
    if args.feature_set == "research":
        llm = sorted(c for c in columns if c.startswith(("llm_", "announcement_count", "prefilter_selected")))
        if not llm:
            raise DataContractError("research set requested but no llm columns present")
        features += llm
    return features


def main() -> None:
    args = _parse_args()
    panel = pd.read_parquet(Path(args.panel).expanduser())
    panel = panel.rename(columns={c: c for c in panel.columns})
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to evaluate")
    features = _candidate_features(args, set(panel.columns))
    panel = validate_pit_panel(panel, QLIB_ALPHA158_FEATURES)

    # Executable 20d rank target across every labeled row (non-members keep a
    # value so the simulator can still account for liquidations).
    labeled = panel["next_open_return_20d"].notna()
    panel.loc[labeled, TARGET_RANK] = (
        panel[labeled].groupby("signal_date")["next_open_return_20d"].rank(pct=True)
    )

    # Training-parity normalization: per-day rank over universe members.
    # Whole-column reassignment sidesteps pandas' float32-preserving setitem.
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
    # Purge invariant: the label window of every training row must close before
    # the first validation/test signal date of its fold.
    end_dates = panel[["signal_date", "label_end_date_20d"]].dropna()
    end_dates["label_end_date_20d"] = pd.to_datetime(end_dates["label_end_date_20d"])
    for fold_id, fold in enumerate(folds, start=1):
        train_max = end_dates.loc[
            end_dates["signal_date"] < fold["train_end_exclusive"], "label_end_date_20d"
        ].max()
        val_max = end_dates.loc[
            (end_dates["signal_date"] >= fold["validation_start"])
            & (end_dates["signal_date"] < fold["validation_end_exclusive"]),
            "label_end_date_20d",
        ].max()
        if pd.notna(train_max) and train_max >= pd.Timestamp(fold["validation_start"]):
            raise DataContractError(f"fold {fold_id} violates the train->val purge invariant")
        if pd.notna(val_max) and val_max >= pd.Timestamp(fold["test_start"]):
            raise DataContractError(f"fold {fold_id} violates the val->test purge invariant")

    costs = TradingCosts(
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )
    # Rows without a 20d label stay in the panel (score becomes NaN inside
    # fit_lightgbm_oos) so the simulator can still see and unwind them.
    predictions = fit_lightgbm_oos(
        panel,
        features,
        TARGET_RANK,
        folds,
        num_threads=args.num_threads,
    )
    portfolio = simulate_portfolio(predictions, topk_dropout_builder(args.top_k, args.drop_n), costs=costs)
    universe = simulate_portfolio(predictions, equal_weight_builder(), costs=costs)
    summary = summarize_baseline(
        predictions,
        portfolio,
        universe,
        frozen_start=args.retrospective_start,
        min_frozen_days=args.min_frozen_days,
        min_portfolio_cagr=args.min_portfolio_cagr,
    )

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate": f"lgbm_v2_{args.feature_set}",
        "feature_set": args.feature_set,
        "feature_count": len(features),
        "deployable": args.feature_set == "deploy",
        "not_deployable_reason": None if args.feature_set == "deploy"
        else "NOT_DEPLOYABLE_V1_llm_features_lack_online_parity",
        "target": "next_open_return_20d cross-sectional pct rank",
        "normalization": "cross_sectional_rank_pct_centered_over_members",
        "oos_naming": {
            "retrospective_oos_start": args.retrospective_start,
            "caveat": (
                "retrospective window was observed by prior research; PASS is "
                "historical evidence only — final confirmation comes from the "
                "sealed lockbox one-shot and the prospective paper-monitor gate"
            ),
        },
        "folds": [
            {key: str(value) for key, value in fold.items()} for fold in folds
        ],
        "result": summary,
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "candidate": report["candidate"],
        "status": summary.get("status"),
        "gates": summary.get("gates"),
        "output": str(output),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
