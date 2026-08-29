#!/usr/bin/env python3
"""Replay parity: the production scoring path must reproduce offline features.

For sampled historical dates (development period ONLY — never the lockbox):
  0. assert the day's PIT universe == the replayed universe snapshot;
  1. run core.model_scoring.compute_deploy_features on bars truncated to the
     replay date (production code path, "as of" data);
  2. compare per feature against training_panel_v2 rows for that date.

Research pipelines and production dying apart on "the same" feature is the
classic quant failure mode; this test is the contract that prevents it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.alpha158 import QLIB_ALPHA158_FEATURES  # noqa: E402
from core.model_scoring import compute_deploy_features  # noqa: E402
from scripts.build_frozen_oos_panel import _load_benchmark  # noqa: E402
from scripts.build_training_panel_v2 import CUSTOM_FEATURES  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

ABS_TOL = 1e-8
REL_TOL = 1e-6


def main() -> None:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--pit", default=str(root / "pit_universe_daily.parquet"))
    parser.add_argument("--stocks-dir", default=str(root / "stocks"))
    parser.add_argument("--benchmark", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--market", default=str(root / "market_sh000300.parquet"))
    parser.add_argument("--dates", type=int, default=20)
    parser.add_argument("--codes-per-date", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    features = [*QLIB_ALPHA158_FEATURES, *CUSTOM_FEATURES]
    panel_cols = ["signal_date", "code", "universe_member", *features]
    panel = pd.read_parquet(args.panel, columns=panel_cols)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    pit = pd.read_parquet(
        args.pit, columns=["trade_date", "code", "is_member", "is_listed", "is_st"])
    pit["trade_date"] = pd.to_datetime(pit["trade_date"])
    # Same definition as the panel: investable universe = member & listed & not-ST.
    pit["investable"] = pit["is_member"] & pit["is_listed"] & ~pit["is_st"]

    rng = np.random.default_rng(args.seed)
    candidate_dates = sorted(
        d for d in panel["signal_date"].unique()
        if pd.Timestamp("2023-01-01") <= pd.Timestamp(d) < pd.Timestamp(LOCKBOX_START)
    )
    sampled = sorted(rng.choice(len(candidate_dates), size=args.dates, replace=False))
    sampled_dates = [pd.Timestamp(candidate_dates[i]) for i in sampled]

    benchmark = _load_benchmark(Path(args.benchmark))
    market_300 = pd.read_parquet(args.market)
    market_300["trade_date"] = pd.to_datetime(market_300["trade_date"])

    summary = {"dates": [], "features_compared": 0, "mismatches": []}
    for trade_date in sampled_dates:
        pit_members = set(pit[(pit["trade_date"] == trade_date) & pit["investable"]]["code"])
        offline = panel[(panel["signal_date"] == trade_date) & panel["universe_member"]]
        offline_codes = set(offline["code"])
        if pit_members != offline_codes:
            raise SystemExit(
                f"universe mismatch at {trade_date.date()}: pit={len(pit_members)} "
                f"panel={len(offline_codes)} diff={sorted(pit_members ^ offline_codes)[:5]}"
            )
        codes = sorted(rng.choice(sorted(offline_codes), size=min(args.codes_per_date, len(offline_codes)), replace=False))

        klines = {}
        for code in codes:
            bars = pd.read_parquet(Path(args.stocks_dir) / f"{code}.parquet")
            bars["trade_date"] = pd.to_datetime(bars["trade_date"])
            klines[code] = bars[bars["trade_date"] <= trade_date].reset_index(drop=True)
        bench_cut = benchmark[benchmark["trade_date"] <= trade_date]
        market_cut = market_300[market_300["trade_date"] <= trade_date]

        produced = compute_deploy_features(klines, bench_cut, market_cut, trade_date)
        offline_rows = offline.set_index("code")
        for code in codes:
            if code not in produced:
                summary["mismatches"].append({"date": str(trade_date.date()), "code": code, "feature": "<missing row>"})
                continue
            for name in features:
                off = offline_rows.at[code, name]
                on = produced[code].get(name)
                off_nan = pd.isna(off)
                on_nan = on is None or pd.isna(on)
                summary["features_compared"] += 1
                if off_nan and on_nan:
                    continue
                if off_nan != on_nan or not np.isclose(float(off), float(on), rtol=REL_TOL, atol=ABS_TOL):
                    summary["mismatches"].append({
                        "date": str(trade_date.date()), "code": code, "feature": name,
                        "offline": None if off_nan else float(off),
                        "online": None if on_nan else float(on),
                    })
        summary["dates"].append(str(trade_date.date()))
        print(f"replayed {trade_date.date()}: codes={len(codes)} mismatches_so_far={len(summary['mismatches'])}", flush=True)

    summary["status"] = "PASS" if not summary["mismatches"] else "FAIL"
    summary["mismatch_count"] = len(summary["mismatches"])
    summary["mismatches"] = summary["mismatches"][:40]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
