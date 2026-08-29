#!/usr/bin/env python3
"""Train the v2 LightGBM alpha/risk models on the immutable training panel.

Contracts:
* executable labels only (next-open entry), sample filter uses signal-day
  information exclusively;
* per-day cross-sectional rank normalization, then fillna(0.0) to match the
  online ``.get(name, 0.0)`` semantics exactly;
* chronological train/val/test split whose purge is asserted via
  ``label_end_date_20d`` (never a bare day count);
* meta.json is a full feature contract and passes through
  ``analysis.lgbm.evaluate_model_health`` before anything is deployable;
* refuses to touch lockbox-period rows.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.alpha158 import QLIB_ALPHA158_FEATURES  # noqa: E402
from analysis.lgbm import evaluate_model_health  # noqa: E402
from scripts.build_training_panel_v2 import CUSTOM_FEATURES  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402
from scripts.train_lgbm import (  # noqa: E402
    _cross_sectional_rank_normalize,
    _evaluate_split,
    _per_year_ic,
    _summary,
)

FEATURE_CONTRACT_VERSION = "v2.1"
RSS_LIMIT_GB = 12.0

TARGETS = {
    "alpha": "next_open_return_20d",
    "risk": "forward_drawdown_20d",
}


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--model-kind", choices=("alpha", "risk"), required=True)
    parser.add_argument("--feature-set", choices=("deploy", "research"), default="deploy")
    parser.add_argument("--output", required=True, help="e.g. models/lgbm_v2.txt")
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping", type=int, default=100)
    return parser.parse_args()


def _rss_guard() -> None:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KB on Linux, bytes on macOS
    rss_gb = raw / (1024 ** 2) if sys.platform == "linux" else raw / (1024 ** 3)
    if rss_gb > RSS_LIMIT_GB:
        raise MemoryError(f"RSS {rss_gb:.1f}GB exceeds the {RSS_LIMIT_GB}GB budget")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _feature_names(args: argparse.Namespace, panel_columns: set[str]) -> list[str]:
    deploy = [f for f in (*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES) if f in panel_columns]
    missing = [f for f in (*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES) if f not in panel_columns]
    if missing:
        raise RuntimeError(f"panel lacks deploy features: {missing[:8]}")
    if args.feature_set == "deploy":
        return deploy
    llm = sorted(c for c in panel_columns if c.startswith(("llm_", "announcement_count", "prefilter_selected")))
    if not llm:
        raise RuntimeError("research feature set requested but no llm_* columns in panel")
    return deploy + llm


def _split_dates(dates: pd.DatetimeIndex, args: argparse.Namespace) -> tuple:
    n = len(dates)
    test_n = max(int(n * args.test_frac), 40)
    val_n = max(int(n * args.val_frac), 40)
    purge = 22  # >= label horizon 20 + next-open execution lag, asserted below
    test_start = n - test_n
    val_start = test_start - purge - val_n
    train_end = val_start - purge
    if train_end < 200:
        raise RuntimeError("not enough trade dates for a purged split")
    return dates[:train_end], dates[val_start:val_start + val_n], dates[test_start:]


def main() -> None:
    import lightgbm as lgb

    args = _parse_args()
    target_col = TARGETS[args.model_kind]
    meta_cols = [
        "signal_date", "code", "universe_member", "signal_is_suspended", "signal_is_limit_up",
        "label_end_date_20d", "next_open_return_20d", "forward_drawdown_20d",
    ]
    import pyarrow.parquet as pq

    panel_columns = set(pq.ParquetFile(args.panel).schema_arrow.names)
    features = _feature_names(args, panel_columns)
    df = pd.read_parquet(args.panel, columns=list(dict.fromkeys(meta_cols + features)))
    _rss_guard()
    df = df.rename(columns={"signal_date": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    if (df["trade_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise RuntimeError("panel contains lockbox rows; refusing to train")

    # Signal-day-known filter only (no execution-outcome conditioning).
    df = df[df["universe_member"] & ~df["signal_is_suspended"] & ~df["signal_is_limit_up"]]
    df = df.dropna(subset=[target_col, "label_end_date_20d"])
    df = df.sort_values(["trade_date", "code"]).reset_index(drop=True)
    df["label_end_date_20d"] = pd.to_datetime(df["label_end_date_20d"])

    # Cross-sectional rank label; for risk the target is "higher is safer".
    df["label"] = df.groupby("trade_date", sort=False)[target_col].rank(pct=True) * 10.0
    # The reused v1 evaluation helpers read a raw "label_score" column.
    df["label_score"] = df[target_col]
    df = _cross_sectional_rank_normalize(df, features)
    df[features] = df[features].fillna(0.0)

    dates = pd.DatetimeIndex(df["trade_date"].unique()).sort_values()
    train_dates, val_dates, test_dates = _split_dates(dates, args)
    train = df[df["trade_date"].isin(train_dates)]
    val = df[df["trade_date"].isin(val_dates)]
    test = df[df["trade_date"].isin(test_dates)]
    for earlier, later, name in ((train, val, "train->val"), (val, test, "val->test")):
        if earlier["label_end_date_20d"].max() >= later["trade_date"].min():
            raise RuntimeError(f"purge invariant violated at {name}")

    params = {
        "objective": "regression_l2", "metric": "l2", "learning_rate": 0.03,
        "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 200,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l1": 0.1, "lambda_l2": 1.0, "verbosity": -1,
        "seed": 20260829, "feature_fraction_seed": 20260829, "bagging_seed": 20260829,
        "deterministic": True, "force_col_wise": True, "num_threads": args.num_threads,
    }
    train_set = lgb.Dataset(train[features], label=train["label"], feature_name=features)
    val_set = lgb.Dataset(val[features], label=val["label"], reference=train_set)
    booster = lgb.train(
        params, train_set, num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
    )
    _rss_guard()

    return_col = "next_open_return_20d"
    splits = {}
    preds = {}
    for name, frame in (("validation", val), ("test", test)):
        pred = booster.predict(frame[features], num_iteration=booster.best_iteration)
        preds[name] = pred
        metrics = _evaluate_split(frame, pred, return_col, 20)
        if args.model_kind == "risk":
            from scripts.train_lgbm import _spearman_ic_stats

            dd = _spearman_ic_stats(frame, pred, "forward_drawdown_20d")
            metrics["drawdown_spearman_ic"] = dd["mean"] if dd else None
            metrics["drawdown_ic_detail"] = {k: dd[k] for k in ("mean", "icir", "positive_rate", "count")} if dd else None
        splits[name] = metrics

    test_metrics = {
        "return_spearman_ic": splits["test"]["return_spearman_ic"],
        "decile_returns": splits["test"]["decile_returns"],
    }
    if args.model_kind == "risk":
        test_metrics["drawdown_spearman_ic"] = splits["test"]["drawdown_spearman_ic"]
        test_metrics["drawdown_ic_detail"] = splits["test"]["drawdown_ic_detail"]

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_kind": args.model_kind,
        "features": features,
        "feature_set": args.feature_set,
        "feature_normalization": "cross_sectional_rank_pct_centered",
        "feature_contract_version": FEATURE_CONTRACT_VERSION,
        "feature_code_git_sha": _git_sha(),
        "reference_universe": "CSI500_PIT",
        "signal_time": "after_close",
        "execution_time": "next_open",
        "entry_policy": "cancel_if_not_buyable_next_open",
        "label_horizon_days": 20,
        "return_col": return_col,
        "target_name": target_col,
        "target_definition": (
            "open[t+21]/open[t+1]-1" if args.model_kind == "alpha"
            else "min_close[t+1..t+20]/open[t+1]-1"
        ),
        "target_rank_direction": "higher_is_better" if args.model_kind == "alpha" else "higher_is_safer",
        "risk_tail": None if args.model_kind == "alpha" else "lowest_score",
        "deployable": args.feature_set == "deploy",
        "not_deployable_reason": None if args.feature_set == "deploy" else "NOT_DEPLOYABLE_V1_llm_features_lack_online_parity",
        "panel": str(args.panel),
        "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
        "split": {
            "train_end": str(train["trade_date"].max().date()),
            "val_start": str(val["trade_date"].min().date()),
            "val_end": str(val["trade_date"].max().date()),
            "test_start": str(test["trade_date"].min().date()),
            "purge": "label_end_date_20d invariant (>=22 trading days)",
        },
        "best_iteration": int(booster.best_iteration),
        "validation_metrics": _summary(splits["validation"]),
        "test_metrics": test_metrics,
        "per_year_ic_test_only": _per_year_ic(test, preds["test"], return_col),
        "feature_importance_top30": sorted(
            zip(features, booster.feature_importance("gain").tolist()),
            key=lambda item: item[1], reverse=True,
        )[:30],
    }
    health = evaluate_model_health(meta)
    meta["validation_status"] = health.get("status", "UNKNOWN")
    meta["validation_failures"] = health.get("failures", [])
    meta["validation_thresholds"] = health.get("thresholds", {})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_path), num_iteration=booster.best_iteration)
    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": str(out_path), "meta": str(meta_path),
        "validation_status": meta["validation_status"],
        "validation_failures": meta["validation_failures"],
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "decile_returns"},
        "test_decile_spread": (test_metrics["decile_returns"] or {}).get("spread_9_minus_0"),
        "best_iteration": meta["best_iteration"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
