#!/usr/bin/env python3
"""EXPLORATORY: three-arm LLM announcement-feature ablation (spec v1 §preregistration).

Arms, run through the identical candidate pipeline (folds, purge, costs, gates):
  A  deploy features only (replicates the rejected primary — sanity anchor)
  B  deploy + the three count columns (announcement activity, no semantics)
  C  arm B + the LLM semantic columns (severity sums/decay, direction, families)

The B arm is the falsification control: announcement-count features were
rejected in earlier research, so any C-over-A gain must also beat B before it
can be read as "semantic scoring adds information" rather than "announcement
activity recovered".  All arms are exploratory (NOT_DEPLOYABLE_V1): LLM
features have no online scoring path yet, and the retrospective window has
been reused heavily.  Conclusions steer phase-2 planning only.
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
COUNT_COLUMNS = ["announcement_count_20d", "prefilter_selected_count_20d", "llm_scored_count_20d"]


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--llm-features", default=str(root / "llm_event_features.parquet"))
    parser.add_argument("--output", default=str(root / "llm_ablation_report.json"))
    parser.add_argument("--retrospective-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=22)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--drop-n", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_features = [*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES]
    panel = pd.read_parquet(Path(args.panel).expanduser())
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to evaluate")
    panel = validate_pit_panel(panel, QLIB_ALPHA158_FEATURES)

    llm = pd.read_parquet(Path(args.llm_features).expanduser())
    llm["trade_date"] = pd.to_datetime(llm["trade_date"])
    if (llm["trade_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        # PIT-keyed to the signal date, so lockbox-dated rows are simply
        # dropped rather than failing the run (the file may legitimately
        # cover the full scored corpus).
        llm = llm[llm["trade_date"] < pd.Timestamp(LOCKBOX_START)]
    semantic_columns = sorted(c for c in llm.columns if c.startswith("llm_") and c not in COUNT_COLUMNS)
    missing = [c for c in COUNT_COLUMNS if c not in llm.columns]
    if missing:
        raise DataContractError(f"llm feature file lacks count columns: {missing}")

    rows_before = len(panel)
    panel = panel.merge(
        llm.rename(columns={"trade_date": "signal_date"}),
        on=["signal_date", "code"], how="left", validate="one_to_one",
    )
    if len(panel) != rows_before:
        raise DataContractError("llm feature merge changed the panel row count")
    added = [*COUNT_COLUMNS, *semantic_columns]
    panel[added] = panel[added].fillna(0.0).astype(np.float32)
    member = panel["universe_member"]
    coverage = float((panel.loc[member, "llm_scored_count_20d"] > 0).mean())
    print(f"llm columns: {len(added)}; member-day 20d-event coverage {coverage:.3f}", flush=True)

    labeled = panel["next_open_return_20d"].notna()
    panel.loc[labeled, TARGET_RANK] = (
        panel[labeled].groupby("signal_date")["next_open_return_20d"].rank(pct=True)
    )

    # Training-parity normalization over every feature any arm uses.
    all_features = [*base_features, *added]
    member_mask = member.to_numpy()
    ranked = (
        panel.loc[member_mask].groupby("signal_date", sort=False)[all_features].rank(pct=True) - 0.5
    ).astype(np.float32)
    normalized = pd.DataFrame(
        np.zeros((len(panel), len(all_features)), dtype=np.float32),
        index=panel.index, columns=all_features,
    )
    normalized.loc[member_mask] = ranked.to_numpy()
    panel[all_features] = normalized.fillna(0.0)

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

    arms = {
        "A_deploy_only": base_features,
        "B_plus_counts": [*base_features, *COUNT_COLUMNS],
        "C_plus_semantic": [*base_features, *COUNT_COLUMNS, *semantic_columns],
    }
    costs = TradingCosts(commission_bps=3.0, stamp_duty_bps=5.0, slippage_bps=5.0)
    results = {}
    for arm, features in arms.items():
        predictions = fit_lightgbm_oos(panel, features, TARGET_RANK, folds, num_threads=args.num_threads)
        portfolio = simulate_portfolio(predictions, topk_dropout_builder(args.top_k, args.drop_n), costs=costs)
        universe = simulate_portfolio(predictions, equal_weight_builder(), costs=costs)
        summary = summarize_baseline(
            predictions, portfolio, universe,
            frozen_start=args.retrospective_start,
            min_frozen_days=126, min_portfolio_cagr=0.05,
        )
        results[arm] = summary
        diag = summary.get("factor") or {}
        print(f"{arm}: status={summary.get('status')} mean_ic={diag.get('mean_ic')}", flush=True)

    def mean_ic(arm):
        return ((results[arm].get("factor") or {}).get("mean_ic"))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "exploratory": True,
        "not_deployable_reason": "NOT_DEPLOYABLE_V1: no online LLM scoring path; retro window reused",
        "spec": "docs/llm_scoring_spec_v1.md",
        "member_day_event_coverage_20d": coverage,
        "semantic_column_count": len(semantic_columns),
        "arms": results,
        "deltas": {
            "B_minus_A_mean_ic": (mean_ic("B_plus_counts") - mean_ic("A_deploy_only"))
            if mean_ic("B_plus_counts") is not None and mean_ic("A_deploy_only") is not None else None,
            "C_minus_A_mean_ic": (mean_ic("C_plus_semantic") - mean_ic("A_deploy_only"))
            if mean_ic("C_plus_semantic") is not None and mean_ic("A_deploy_only") is not None else None,
            "C_minus_B_mean_ic": (mean_ic("C_plus_semantic") - mean_ic("B_plus_counts"))
            if mean_ic("C_plus_semantic") is not None and mean_ic("B_plus_counts") is not None else None,
        },
        "reading_rule": "semantic increment requires C > B and C > A on the same folds; C > A alone is ambiguous",
    }
    Path(args.output).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({arm: {"status": s.get("status"), "mean_ic": (s.get("factor") or {}).get("mean_ic")}
                      for arm, s in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
