#!/usr/bin/env python3
"""EXPLORATORY DIAGNOSTIC: does return alpha exist at shorter horizons?

Public China-equity benchmarks (e.g. Microsoft Qlib's Alpha158 suites) report
their LightGBM/NN rank IC on ~2-day labels; our product constraint fixes the
deployed horizon at 20 days, where four research rounds found nothing.  This
diagnostic trains the same LightGBM recipe on executable 5d and 10d labels
(open[t+1] -> open[t+H+1], same folds — the 22-day purge is conservative for
any H <= 20) and reports rank IC only.  No portfolio simulation, no gates:
the deployed horizon cannot change without a product decision, so this only
answers "is the failure horizon-specific or absolute".
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


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--stocks-dir", default=str(root / "stocks"))
    parser.add_argument("--output", default=str(root / "alpha_horizon_diagnostic.json"))
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--retrospective-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=22)
    parser.add_argument("--num-threads", type=int, default=os.cpu_count() or 8)
    return parser.parse_args()


def _horizon_labels(stocks_dir: Path, codes: list[str], horizons: list[int]) -> pd.DataFrame:
    frames = []
    for index, code in enumerate(codes):
        bars = pd.read_parquet(stocks_dir / f"{code}.parquet", columns=["trade_date", "open"])
        bars["trade_date"] = pd.to_datetime(bars["trade_date"])
        bars = bars.sort_values("trade_date").reset_index(drop=True)
        open_px = pd.to_numeric(bars["open"], errors="coerce")
        out = {"signal_date": bars["trade_date"], "code": code}
        for horizon in horizons:
            out[f"ret_{horizon}d"] = (open_px.shift(-(horizon + 1)) / open_px.shift(-1) - 1.0)
        frames.append(pd.DataFrame(out))
        if (index + 1) % 200 == 0:
            print(f"labels {index + 1}/{len(codes)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = _parse_args()
    features = [*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES]
    panel = pd.read_parquet(Path(args.panel).expanduser())
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to evaluate")
    panel = validate_pit_panel(panel, QLIB_ALPHA158_FEATURES)

    labels = _horizon_labels(
        Path(args.stocks_dir).expanduser(), sorted(panel["code"].unique()), args.horizons)
    panel = panel.merge(labels, on=["signal_date", "code"], how="left", validate="one_to_one")

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

    retro_start = pd.Timestamp(args.retrospective_start)
    results = {}
    for horizon in args.horizons:
        return_col = f"ret_{horizon}d"
        rank_col = f"rank_{horizon}d"
        labeled = panel[return_col].notna()
        panel.loc[labeled, rank_col] = (
            panel[labeled].groupby("signal_date")[return_col].rank(pct=True)
        )
        predictions = fit_lightgbm_oos(panel, features, rank_col, folds, num_threads=args.num_threads)
        scored = predictions[
            predictions["universe_member"]
            & predictions["score"].notna()
            & predictions[return_col].notna()
        ]

        def daily(frame):
            def one(group):
                if len(group) <= 2:
                    return np.nan
                return group["score"].corr(group[return_col], method="spearman")
            return frame.groupby("signal_date").apply(one).dropna()

        ic_all = daily(scored)
        ic_retro = ic_all[ic_all.index >= retro_start]
        results[f"{horizon}d"] = {
            "full_oos": {
                "mean_ic": float(ic_all.mean()),
                "icir": float(ic_all.mean() / ic_all.std()) if ic_all.std() else None,
                "positive_rate": float((ic_all > 0).mean()),
                "days": int(len(ic_all)),
            },
            "retrospective": {
                "mean_ic": float(ic_retro.mean()),
                "icir": float(ic_retro.mean() / ic_retro.std()) if ic_retro.std() else None,
                "positive_rate": float((ic_retro > 0).mean()),
                "days": int(len(ic_retro)),
            },
        }
        print(f"horizon {horizon}d: retro mean IC {ic_retro.mean():.4f} "
              f"({len(ic_retro)} days)", flush=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "diagnostic": "alpha_horizon",
        "exploratory": True,
        "note": "IC only; deployed horizon stays 20d absent a product decision",
        "label": "open[t+1] -> open[t+H+1] executable next-open return, rank target",
        "results": results,
    }
    Path(args.output).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({h: r["retrospective"]["mean_ic"] for h, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
