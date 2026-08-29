#!/usr/bin/env python3
"""Cheap diagnostic: how much of the old proven drawdown construct survives in
the new executable risk target?  (Not a model experiment; no test budget used.)

old target: min(close[t+1..t+20]) / close[t] - 1        (unexecutable entry)
new target: min(close[t+1..t+20]) / open[t+1] - 1       (executable entry)

Reports daily cross-sectional Spearman between the two, overall correlation,
decile overlap and bad-tail overlap on the development panel.  If correlation
is low, the old IC 0.27-0.29 must not be used as a prior for the new task.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-panel", default=str(root / "development_panel.parquet"))
    parser.add_argument("--stocks-dir", default=str(root / "stocks"))
    args = parser.parse_args()

    panel = pd.read_parquet(args.dev_panel, columns=[
        "signal_date", "code", "forward_drawdown_20d", "universe_member",
    ])
    panel = panel[panel["universe_member"] & panel["forward_drawdown_20d"].notna()]
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])

    # Fair comparison: the old target must use the same trading-calendar
    # reindex + ffill as the panel, so entry price is the only variable.
    calendar = pd.DatetimeIndex(sorted(panel["signal_date"].unique()))
    old_frames = []
    for code in panel["code"].unique():
        path = Path(args.stocks_dir) / f"{code}.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path, columns=["trade_date", "adj_close"])
        bars["trade_date"] = pd.to_datetime(bars["trade_date"])
        bars = bars.set_index("trade_date").sort_index()
        cal = calendar[calendar >= bars.index.min()]
        close = bars["adj_close"].reindex(cal).ffill()
        fwd_min = close.iloc[::-1].rolling(20, min_periods=20).min().iloc[::-1].shift(-1)
        old = pd.DataFrame({
            "trade_date": cal,
            "old_drawdown_20d": (fwd_min / close - 1).to_numpy(),
        })
        old["code"] = code
        old_frames.append(old)
    old = pd.concat(old_frames, ignore_index=True)
    merged = panel.merge(
        old, left_on=["signal_date", "code"], right_on=["trade_date", "code"], how="inner",
    ).dropna(subset=["old_drawdown_20d"])

    def daily_spearman(group: pd.DataFrame) -> float:
        if len(group) <= 2:
            return np.nan
        return group["forward_drawdown_20d"].corr(group["old_drawdown_20d"], method="spearman")

    daily = merged.groupby("signal_date").apply(daily_spearman).dropna()

    def decile_overlap(frame: pd.DataFrame, tail: float) -> float:
        overlaps = []
        for _d, group in frame.groupby("signal_date"):
            if len(group) < 50:
                continue
            k = max(int(len(group) * tail), 1)
            worst_new = set(group.nsmallest(k, "forward_drawdown_20d")["code"])
            worst_old = set(group.nsmallest(k, "old_drawdown_20d")["code"])
            overlaps.append(len(worst_new & worst_old) / k)
        return float(np.mean(overlaps)) if overlaps else float("nan")

    report = {
        "rows": int(len(merged)),
        "daily_cross_sectional_spearman": {
            "mean": float(daily.mean()), "p10": float(daily.quantile(0.10)),
            "min": float(daily.min()),
        },
        "overall_pearson": float(merged["forward_drawdown_20d"].corr(merged["old_drawdown_20d"])),
        "worst_decile_overlap": decile_overlap(merged, 0.10),
        "bad_tail_overlap_5pct": decile_overlap(merged, 0.05),
        "verdict_hint": "if mean daily spearman < 0.6, do NOT carry the old IC prior to the new target",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
