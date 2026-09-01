#!/usr/bin/env python3
"""Prospective paper-monitor report for the deployed risk model.

Pre-registered acceptance (frozen with the lockbox criteria; see
docs/lockbox_go_no_go_v1.md and the deployment memo):
  the gate can PASS only after >= --min-days (default 63, ~3 months) of
  evaluable scored trade dates, AND prospective mean daily Spearman IC > 0,
  AND worst-decile enrichment < 0.  Until enough days accumulate the status
  is ACCUMULATING — never an early PASS.

A scored date is evaluable once its full outcome window has elapsed:
realized target = min(close[t+1..t+20]) / open[t+1] - 1 on the stock's own
traded bars (the model meta's target_definition).  Using per-code bars means
a suspension shifts entry to the next traded bar; rows whose entry bar lags
the signal by more than --max-entry-lag-days calendar days are dropped and
counted, so long suspensions cannot silently distort the IC.

Run on the production host (reads its model_scores table and downloads the
needed histories with the same fallback chain scoring uses):

    .venv/bin/python scripts/paper_monitor_report.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TARGET_HORIZON = 20  # traded bars, per lgbm_v2_risk meta target_definition


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(Path("~/.stockwatch/db.sqlite").expanduser()))
    parser.add_argument("--output", default=str(Path("~/.stockwatch/paper_monitor_report.json").expanduser()))
    parser.add_argument("--min-days", type=int, default=63)
    parser.add_argument("--min-names-per-day", type=int, default=30)
    parser.add_argument("--exclude-fraction", type=float, default=0.10)
    parser.add_argument("--bad-tail-threshold", type=float, default=-0.15)
    parser.add_argument("--max-entry-lag-days", type=int, default=5)
    return parser.parse_args()


def _load_scores(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        frame = pd.read_sql_query(
            "SELECT trade_date, code, risk_score, risk_model_version "
            "FROM model_scores WHERE risk_score IS NOT NULL",
            conn,
        )
    finally:
        conn.close()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def _realized_targets(scores: pd.DataFrame, max_entry_lag_days: int) -> tuple[pd.DataFrame, dict]:
    """Join realized drawdown outcomes onto evaluable scored rows."""
    from core.model_scoring import _download_history

    codes = sorted(scores["code"].unique())
    klines = _download_history(codes)
    dropped = {"missing_history": 0, "window_incomplete": 0, "entry_lag": 0}
    rows = []
    for code, group in scores.groupby("code"):
        bars = klines.get(code)
        if bars is None or len(bars) == 0:
            dropped["missing_history"] += len(group)
            continue
        bars = bars.sort_values("trade_date").reset_index(drop=True)
        dates = pd.to_datetime(bars["trade_date"]).reset_index(drop=True)
        opens = pd.to_numeric(bars["open"], errors="coerce")
        closes = pd.to_numeric(bars["close"], errors="coerce")
        positions = pd.Series(np.arange(len(dates)), index=dates)
        for record in group.itertuples():
            pos = positions[positions.index > record.trade_date]
            if pos.empty:
                dropped["window_incomplete"] += len(pos) or 1
                continue
            entry = int(pos.iloc[0])
            if (dates.iloc[entry] - record.trade_date).days > max_entry_lag_days:
                dropped["entry_lag"] += 1
                continue
            window_end = entry + TARGET_HORIZON  # closes[entry .. entry+19]
            if window_end > len(bars):
                dropped["window_incomplete"] += 1
                continue
            entry_open = opens.iloc[entry]
            worst_close = closes.iloc[entry:window_end].min()
            if not np.isfinite(entry_open) or entry_open <= 0 or not np.isfinite(worst_close):
                dropped["window_incomplete"] += 1
                continue
            rows.append({
                "trade_date": record.trade_date,
                "code": code,
                "risk_score": record.risk_score,
                "realized_drawdown": float(worst_close / entry_open - 1.0),
            })
    return pd.DataFrame(rows), dropped


def main() -> None:
    args = _parse_args()
    scores = _load_scores(args.db)
    if scores.empty:
        print(json.dumps({"status": "NO_SCORES"}))
        return

    # Evaluable ceiling before any download: a 20-bar window cannot have
    # closed within ~20 trading days (~30 calendar days) of the score.
    horizon_cutoff = pd.Timestamp(datetime.now().date()) - timedelta(days=30)
    candidates = scores[scores["trade_date"] <= horizon_cutoff]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "monitor": "prospective_paper_monitor",
        "model_versions": sorted(scores["risk_model_version"].dropna().unique()),
        "scored_dates_total": int(scores["trade_date"].nunique()),
        "first_scored_date": str(scores["trade_date"].min().date()),
        "preregistered_gate": {
            "min_evaluable_days": args.min_days,
            "mean_ic_positive": None,
            "worst_decile_enrichment_negative": None,
        },
    }
    if candidates.empty:
        report["status"] = "ACCUMULATING"
        report["evaluable_days"] = 0
        report["note"] = "no scored date old enough for a completed 20d outcome window"
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    evaluated, dropped = _realized_targets(candidates, args.max_entry_lag_days)
    report["dropped_rows"] = dropped
    daily_groups = [g for _, g in evaluated.groupby("trade_date") if len(g) >= args.min_names_per_day]
    if not daily_groups:
        report["status"] = "ACCUMULATING"
        report["evaluable_days"] = 0
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    ics = pd.Series({
        g["trade_date"].iloc[0]: g["risk_score"].corr(g["realized_drawdown"], method="spearman")
        for g in daily_groups
    }).dropna().sort_index()

    bottom_frames = []
    for g in daily_groups:
        cutoff = g["risk_score"].quantile(args.exclude_fraction)
        bottom_frames.append(g[g["risk_score"] <= cutoff])
    bottom = pd.concat(bottom_frames)
    universe_dd = float(evaluated["realized_drawdown"].mean())
    bottom_dd = float(bottom["realized_drawdown"].mean())
    enrichment = bottom_dd - universe_dd
    bad_tail_base = float((evaluated["realized_drawdown"] <= args.bad_tail_threshold).mean())
    bad_tail_bottom = float((bottom["realized_drawdown"] <= args.bad_tail_threshold).mean())

    mean_ic = float(ics.mean())
    days = int(len(ics))
    gate = report["preregistered_gate"]
    gate["mean_ic_positive"] = mean_ic > 0
    gate["worst_decile_enrichment_negative"] = enrichment < 0
    enough = days >= args.min_days
    report.update({
        "status": ("PASS" if (enough and gate["mean_ic_positive"] and gate["worst_decile_enrichment_negative"])
                   else "FAIL" if enough else "ACCUMULATING"),
        "evaluable_days": days,
        "evaluated_rows": int(len(evaluated)),
        "metrics": {
            "mean_daily_spearman_ic": mean_ic,
            "icir": float(mean_ic / ics.std(ddof=1)) if days > 1 and ics.std(ddof=1) > 0 else None,
            "positive_ic_day_rate": float((ics > 0).mean()),
            "worst_decile_enrichment": enrichment,
            "expected_drawdown_bottom_decile": bottom_dd,
            "expected_drawdown_universe": universe_dd,
            "bad_tail_precision_bottom_decile": bad_tail_bottom,
            "bad_tail_base_rate": bad_tail_base,
            "bad_tail_lift": float(bad_tail_bottom / bad_tail_base) if bad_tail_base > 0 else None,
        },
        "daily_ic_series": {str(k.date()): float(v) for k, v in ics.items()},
    })
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {k: report[k] for k in ("status", "evaluable_days")}
    concise.update(report.get("metrics", {}))
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
