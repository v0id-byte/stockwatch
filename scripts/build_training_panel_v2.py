#!/usr/bin/env python3
"""Assemble the immutable training panel v2 from the development panel.

development_panel.parquet (Qlib Alpha158 + executable 1d/20d labels, PIT flags)
  + 40 robust custom-technical features (ROBUST_FEATURES + BEAR extras,
    computed exactly like the proven risk model: adj basis, sh000300 market)
  + LLM event features (llm_event_features.parquet), missing days = 0
-> training_panel_v2.parquet (float32 features) + report with sha256.

The output is immutable: every experiment must reference the report's sha256.
This script refuses to read any file containing lockbox-period rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.alpha158 import QLIB_ALPHA158_FEATURES  # noqa: E402
from analysis.factors import BEAR_FEATURES, ROBUST_FEATURES, compute_alpha158_frame  # noqa: E402
from scripts.build_training_set import _factor_input  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

# Only custom features whose NAME does not collide with exact Qlib Alpha158:
# same-named pairs are near-duplicate definitions and a merge would suffix
# them into ambiguity.  The custom set's unique value is the 120/250-day
# windows and the TURN/ILLIQ/RELV/VOLZ/AMTMA families Qlib158 lacks.
CUSTOM_FEATURES = [
    f for f in dict.fromkeys(ROBUST_FEATURES + BEAR_FEATURES)
    if f not in set(QLIB_ALPHA158_FEATURES)
]


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-panel", default=str(root / "development_panel.parquet"))
    parser.add_argument("--stocks-dir", default=str(root / "stocks"))
    parser.add_argument("--market", default=str(root / "market_sh000300.parquet"))
    parser.add_argument("--llm-features", default=str(root / "llm_event_features.parquet"))
    parser.add_argument("--output", default=str(root / "training_panel_v2.parquet"))
    parser.add_argument("--skip-llm", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict:
    panel = pd.read_parquet(args.dev_panel)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise RuntimeError("input panel contains lockbox-period rows; refusing to build")

    market = pd.read_parquet(args.market)
    codes = sorted(panel["code"].unique())
    frames = []
    failed: list[str] = []
    for index, code in enumerate(codes):
        path = Path(args.stocks_dir) / f"{code}.parquet"
        if not path.exists():
            failed.append(code)
            continue
        kline = pd.read_parquet(path)
        factors = compute_alpha158_frame(_factor_input(kline, code), market)
        keep = factors[["trade_date", *CUSTOM_FEATURES]].copy()
        keep["trade_date"] = pd.to_datetime(keep["trade_date"])
        keep["code"] = code
        keep[CUSTOM_FEATURES] = keep[CUSTOM_FEATURES].astype(np.float32)
        frames.append(keep)
        if (index + 1) % 100 == 0:
            print(f"custom factors {index + 1}/{len(codes)}", flush=True)
    if failed:
        raise RuntimeError(f"stocks missing for panel codes: {failed[:10]}")
    custom = pd.concat(frames, ignore_index=True)
    out = panel.merge(
        custom, left_on=["signal_date", "code"], right_on=["trade_date", "code"],
        how="left", validate="one_to_one",
    ).drop(columns="trade_date")

    llm_columns: list[str] = []
    if not args.skip_llm:
        llm = pd.read_parquet(args.llm_features)
        llm["trade_date"] = pd.to_datetime(llm["trade_date"])
        llm_columns = [c for c in llm.columns if c not in ("trade_date", "code")]
        out = out.merge(
            llm, left_on=["signal_date", "code"], right_on=["trade_date", "code"],
            how="left", validate="one_to_one",
        ).drop(columns="trade_date")
        out[llm_columns] = out[llm_columns].fillna(0.0).astype(np.float32)

    feature_like = [c for c in out.columns if out[c].dtype == np.float64
                    and c not in ("next_open_return", "next_open_return_20d",
                                  "forward_drawdown_20d", "forward_drawdown_20d_low",
                                  "benchmark_return")]
    out[feature_like] = out[feature_like].astype(np.float32)

    out_path = Path(args.output).expanduser()
    out.to_parquet(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    custom_coverage = float(out[CUSTOM_FEATURES[0]].notna().mean())
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(len(out)),
        "codes": len(codes),
        "signal_date_range": [str(out["signal_date"].min().date()), str(out["signal_date"].max().date())],
        "custom_features": CUSTOM_FEATURES,
        "custom_feature_coverage": custom_coverage,
        "llm_features": llm_columns,
        "columns": int(out.shape[1]),
        "immutable_sha256": sha,
        "output": str(out_path),
        "lockbox_start_excluded": LOCKBOX_START,
    }
    Path(str(out_path) + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = build(_parse_args())
    slim = {k: v for k, v in report.items() if k not in ("custom_features", "llm_features")}
    print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
