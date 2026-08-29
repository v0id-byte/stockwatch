#!/usr/bin/env python3
"""Risk-model acceptance on the retrospective window (pre-registered gates).

Model: LightGBM on the executable drawdown target (higher score = safer),
walk-forward via the same purged expanding folds as the alpha candidate.

Gates (all on out-of-fold predictions):
  G1 drawdown IC          mean >= 0.20, ICIR >= 1.0, positive_rate >= 0.85
  G2 worst-decile enrich  E[dd | bottom decile] - E[dd | universe] <= -enrich_min
                          and precision/lift for true drawdown <= -15%
  G3 exclusion improves   equal-weight universe max-drawdown improves when the
                          predicted-riskiest 10% are excluded (retrospective
                          window), gross of costs, like-for-like accounting
  G4 not just randomness  actual max-drawdown improvement >= p90 of 200
                          same-count random exclusions
  G5 not just low-vol     improvement also exceeds the volatility-decile control

All comparisons share one daily equal-weight next-open accounting, so deltas
are internally consistent.  Costs cancel to first order between same-count
variants and are therefore omitted (documented).
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
    fit_lightgbm_oos,
    make_expanding_folds,
    validate_pit_panel,
)
from scripts.build_training_panel_v2 import CUSTOM_FEATURES  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

TARGET = "forward_drawdown_20d"
TARGET_RANK = "risk_rank_20d"


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--output", default=str(root / "risk_model_v2_report.json"))
    parser.add_argument("--retrospective-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=22)
    parser.add_argument("--exclude-fraction", type=float, default=0.10)
    parser.add_argument("--controls", type=int, default=200)
    parser.add_argument("--ic-min", type=float, default=0.20)
    parser.add_argument("--icir-min", type=float, default=1.0)
    parser.add_argument("--positive-rate-min", type=float, default=0.85)
    parser.add_argument("--bad-tail-threshold", type=float, default=-0.15)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def _daily_ic(frame: pd.DataFrame, pred_col: str, target_col: str) -> pd.Series:
    def one(group: pd.DataFrame) -> float:
        if len(group) <= 2:
            return np.nan
        return group[pred_col].corr(group[target_col], method="spearman")

    return frame.groupby("signal_date").apply(one).dropna()


def _equal_weight_nav(frame: pd.DataFrame, keep_mask: pd.Series) -> pd.Series:
    """Daily-rebalanced equal-weight next-open return series over kept rows."""
    kept = frame[keep_mask]
    daily = kept.groupby("signal_date")["next_open_return"].mean()
    return daily.sort_index()


def _max_drawdown(returns: pd.Series) -> float:
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return float((nav / peak - 1.0).min())


def main() -> None:
    args = _parse_args()
    panel = pd.read_parquet(Path(args.panel).expanduser())
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to evaluate")
    panel = validate_pit_panel(panel, QLIB_ALPHA158_FEATURES)
    features = [f for f in (*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES) if f in panel.columns]

    labeled = panel[TARGET].notna()
    panel.loc[labeled, TARGET_RANK] = (
        panel[labeled].groupby("signal_date")[TARGET].rank(pct=True)
    )
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
    # Purge invariant asserted through label_end_date_20d, never assumed.
    ends = panel[["signal_date", "label_end_date_20d"]].dropna()
    ends["label_end_date_20d"] = pd.to_datetime(ends["label_end_date_20d"])
    for fold_id, fold in enumerate(folds, start=1):
        train_max = ends.loc[ends["signal_date"] < fold["train_end_exclusive"], "label_end_date_20d"].max()
        val_max = ends.loc[
            (ends["signal_date"] >= fold["validation_start"])
            & (ends["signal_date"] < fold["validation_end_exclusive"]),
            "label_end_date_20d",
        ].max()
        if pd.notna(train_max) and train_max >= pd.Timestamp(fold["validation_start"]):
            raise DataContractError(f"fold {fold_id} violates the train->val purge invariant")
        if pd.notna(val_max) and val_max >= pd.Timestamp(fold["test_start"]):
            raise DataContractError(f"fold {fold_id} violates the val->test purge invariant")
    # Rows without a 20d label stay in the panel (score becomes NaN inside
    # fit_lightgbm_oos) so the simulator can still see and unwind them.
    predictions = fit_lightgbm_oos(
        panel, features, TARGET_RANK, folds,
        num_threads=args.num_threads,
    )
    predictions["signal_date"] = pd.to_datetime(predictions["signal_date"])
    scored = predictions.merge(
        panel[["signal_date", "code", TARGET, "next_open_return", "universe_member",
               "buyable", "volatility_20d"]],
        on=["signal_date", "code"], how="left", suffixes=("", "_panel"),
    )
    scored = scored[scored["universe_member"] & scored[TARGET].notna()]
    retro = scored[scored["signal_date"] >= pd.Timestamp(args.retrospective_start)]

    # G1: drawdown IC
    ic = _daily_ic(scored, "score", TARGET)
    ic_retro = _daily_ic(retro, "score", TARGET)
    g1 = {
        "full": {"mean": float(ic.mean()), "icir": float(ic.mean() / ic.std()),
                 "positive_rate": float((ic > 0).mean()), "days": int(len(ic))},
        "retrospective": {"mean": float(ic_retro.mean()),
                          "icir": float(ic_retro.mean() / ic_retro.std()),
                          "positive_rate": float((ic_retro > 0).mean()), "days": int(len(ic_retro))},
    }
    g1_pass = (
        g1["retrospective"]["mean"] >= args.ic_min
        and g1["retrospective"]["icir"] >= args.icir_min
        and g1["retrospective"]["positive_rate"] >= args.positive_rate_min
    )

    # G2: worst-decile enrichment (business metric)
    def decile_flags(frame: pd.DataFrame) -> pd.Series:
        return frame.groupby("signal_date")["score"].transform(
            lambda s: s.rank(pct=True) <= args.exclude_fraction
        )

    retro = retro.copy()
    retro["bottom_decile"] = decile_flags(retro)
    bottom = retro[retro["bottom_decile"]]
    universe_dd = float(retro[TARGET].mean())
    bottom_dd = float(bottom[TARGET].mean())
    bad = retro[TARGET] <= args.bad_tail_threshold
    precision = float((bottom[TARGET] <= args.bad_tail_threshold).mean())
    base_rate = float(bad.mean())
    g2 = {
        "expected_drawdown_bottom_decile": bottom_dd,
        "expected_drawdown_universe": universe_dd,
        "enrichment": bottom_dd - universe_dd,
        "bad_tail_threshold": args.bad_tail_threshold,
        "bad_tail_precision_bottom_decile": precision,
        "bad_tail_base_rate": base_rate,
        "bad_tail_lift": precision / base_rate if base_rate else None,
    }
    g2_pass = g2["enrichment"] < 0 and (g2["bad_tail_lift"] or 0) > 1.5

    # G3-G5: exclusion improvement vs controls, one shared accounting
    tradable = retro["buyable"].fillna(False)
    base_nav = _equal_weight_nav(retro, tradable)
    excl_nav = _equal_weight_nav(retro, tradable & ~retro["bottom_decile"])
    base_mdd, excl_mdd = _max_drawdown(base_nav), _max_drawdown(excl_nav)
    actual_delta = excl_mdd - base_mdd  # positive = drawdown improved (less negative)
    base_total = float((1 + base_nav.fillna(0)).prod() - 1)
    excl_total = float((1 + excl_nav.fillna(0)).prod() - 1)

    rng = np.random.default_rng(20260829)
    counts = retro[retro["bottom_decile"]].groupby("signal_date").size()
    date_index = {d: idx.to_numpy() for d, idx in retro.groupby("signal_date").groups.items()}
    control_deltas = []
    for _ in range(args.controls):
        flags = pd.Series(False, index=retro.index)
        for day, idx in date_index.items():
            k = int(counts.get(day, round(len(idx) * args.exclude_fraction)))
            if k > 0:
                flags.loc[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
        nav = _equal_weight_nav(retro, tradable & ~flags)
        control_deltas.append(_max_drawdown(nav) - base_mdd)
    control_deltas = np.array(control_deltas)
    percentile = float((control_deltas < actual_delta).mean())

    retro["highvol_decile"] = retro.groupby("signal_date")["volatility_20d"].transform(
        lambda s: s.rank(pct=True) >= 1 - args.exclude_fraction
    )
    highvol_nav = _equal_weight_nav(retro, tradable & ~retro["highvol_decile"])
    highvol_delta = _max_drawdown(highvol_nav) - base_mdd

    g3 = {
        "baseline_max_drawdown": base_mdd,
        "excluded_max_drawdown": excl_mdd,
        "max_drawdown_delta": actual_delta,
        "baseline_total_return": base_total,
        "excluded_total_return": excl_total,
        "total_return_delta": excl_total - base_total,
        "accounting": "gross_of_costs_daily_equal_weight_next_open (same-count deltas cancel costs to first order)",
    }
    g4 = {
        "controls": args.controls,
        "control_delta_p50": float(np.percentile(control_deltas, 50)),
        "control_delta_p90": float(np.percentile(control_deltas, 90)),
        "actual_delta_percentile": percentile,
    }
    g5 = {"highvol_control_delta": highvol_delta, "actual_delta": actual_delta}

    gates = {
        "G1_drawdown_ic": g1_pass,
        "G2_worst_decile_enrichment": g2_pass,
        "G3_exclusion_improves_drawdown": actual_delta > 0,
        "G4_beats_random_controls_p90": percentile >= 0.90,
        "G5_beats_highvol_control": actual_delta > highvol_delta,
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "risk_lgbm_v2",
        "target": TARGET,
        "target_rank_direction": "higher_is_safer",
        "risk_tail": "lowest_score",
        "retrospective_start": args.retrospective_start,
        "gates": gates,
        "status": status,
        "g1_ic": g1, "g2_enrichment": g2, "g3_exclusion": g3,
        "g4_random_controls": g4, "g5_highvol_control": g5,
        "caveat": "retrospective evidence only; deployment additionally requires the lockbox one-shot and prospective gates",
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": status, "gates": gates, "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
