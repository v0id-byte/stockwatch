#!/usr/bin/env python3
"""Evaluate sparse structured events as an event-driven long-only strategy."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.backfill_announcement_documents import CATEGORY_PRIORITY
from scripts.evaluate_dual_model import _benchmark_returns
from scripts.evaluate_neutralized_walk_forward import _portfolio_stats, _series_stats


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Evaluate rule-based structured event scores.")
    parser.add_argument("--training-set", default=str(root / "training_set.parquet"))
    parser.add_argument("--event-features", default=str(root / "structured_event_features.parquet"))
    parser.add_argument("--benchmark", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--target", default="forward_20d_return")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-positive", type=int, default=5)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--risk-free-annual", type=float, default=0.03)
    parser.add_argument("--frozen-start", default="2025-01-01")
    parser.add_argument("--latest-start", default="2026-01-01")
    parser.add_argument("--output", default=str(root / "structured_event_strategy_report.json"))
    return parser.parse_args()


def _sample_dates(frame: pd.DataFrame, horizon: int) -> list[pd.Timestamp]:
    dates = list(frame["trade_date"].drop_duplicates())
    return dates[::max(1, horizon)]


def _strategy_records(data: pd.DataFrame, score_col: str, active_col: str, *,
                      target: str, sampled_dates: list[pd.Timestamp], top_k: int,
                      min_positive: int, cost: float, risk_free: float,
                      benchmark: dict[pd.Timestamp, float], horizon: int) -> pd.DataFrame:
    cash_return = (1 + risk_free) ** (horizon / 252) - 1
    rows = []
    sampled = data[data["trade_date"].isin(set(sampled_dates))]
    for date, group in sampled.groupby("trade_date", sort=False):
        universe = float(group[target].mean())
        active = group[group[active_col] > 0]
        positive = active[active[score_col] > 0].sort_values(score_col, ascending=False)
        selected = positive.head(min(top_k, len(positive)))
        if len(selected) >= min_positive:
            selected_net = float(selected[target].mean()) - cost
            invested = True
        else:
            selected_net = float(cash_return)
            invested = False
        rows.append({
            "trade_date": pd.Timestamp(date),
            "selected_net": selected_net,
            "universe": universe,
            "active_universe": float(active[target].mean()) if not active.empty else None,
            "benchmark": benchmark.get(pd.Timestamp(date)),
            "selected_names": int(len(selected)) if invested else 0,
            "active_names": int(len(active)),
            "positive_names": int(len(positive)),
            "invested": invested,
        })
    return pd.DataFrame(rows)


def _metrics(records: pd.DataFrame, horizon: int, risk_free: float) -> dict | None:
    if records.empty:
        return None
    selected = _portfolio_stats(records["selected_net"].tolist(), horizon, risk_free)
    universe = _portfolio_stats(records["universe"].tolist(), horizon, risk_free)
    active_values = records["active_universe"].dropna().tolist()
    active = _portfolio_stats(active_values, horizon, risk_free) if active_values else None
    benchmark_rows = records.dropna(subset=["benchmark"])
    benchmark = _portfolio_stats(benchmark_rows["benchmark"].tolist(), horizon, risk_free) if len(benchmark_rows) else None
    selected_aligned = _portfolio_stats(benchmark_rows["selected_net"].tolist(), horizon, risk_free) if len(benchmark_rows) else None
    by_year = records.groupby(records["trade_date"].dt.year).apply(
        lambda group: float((group["selected_net"] - group["universe"]).mean()),
        include_groups=False,
    )
    delta_universe = selected["cagr"] - universe["cagr"] if selected and universe else None
    delta_benchmark = selected_aligned["cagr"] - benchmark["cagr"] if selected_aligned and benchmark else None
    return {
        "period_count": int(len(records)),
        "selected": selected,
        "universe": universe,
        "active_event_universe": active,
        "benchmark": benchmark,
        "net_excess_vs_universe": _series_stats((records["selected_net"] - records["universe"]).tolist()),
        "net_excess_vs_active_universe": _series_stats(
            (records["selected_net"] - records["active_universe"]).dropna().tolist()
        ),
        "annualized_delta_vs_universe": float(delta_universe) if delta_universe is not None else None,
        "annualized_delta_vs_csi500": float(delta_benchmark) if delta_benchmark is not None else None,
        "positive_year_rate": float((by_year > 0).mean()) if len(by_year) else None,
        "year_count": int(len(by_year)),
        "invested_rate": float(records["invested"].mean()),
        "average_selected_names": float(records["selected_names"].mean()),
        "average_active_names": float(records["active_names"].mean()),
    }


def _passes(metrics: dict | None, minimum_periods: int) -> bool:
    metrics = metrics or {}
    selected = metrics.get("selected") or {}
    return bool(
        (metrics.get("period_count") or 0) >= minimum_periods
        and (selected.get("cagr") or -1) > 0.05
        and (metrics.get("annualized_delta_vs_universe") or -1) > 0.01
        and (metrics.get("annualized_delta_vs_csi500") or -1) > 0
        and (metrics.get("positive_year_rate") or 0) >= 0.60
    )


def main() -> None:
    args = _parse_args()
    horizon = int("".join(character for character in args.target if character.isdigit()) or 20)
    feature_path = Path(args.event_features).expanduser()
    score_pairs = {
        "total_7d": ("event_signed_score_7d", "event_count_7d"),
        "total_20d": ("event_signed_score_20d", "event_count_20d"),
        **{
            category: (f"event_{category}_score_20d", f"event_{category}_count_20d")
            for category in CATEGORY_PRIORITY
        },
    }
    feature_columns = sorted({name for pair in score_pairs.values() for name in pair})
    targets = pd.read_parquet(args.training_set, columns=["trade_date", "code", args.target])
    features = pd.read_parquet(feature_path, columns=["trade_date", "code", *feature_columns])
    for frame in (targets, features):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["code"] = frame["code"].astype(str).str.zfill(6)
    data = targets.merge(features, on=["trade_date", "code"], how="inner", validate="one_to_one")
    sampled_dates = _sample_dates(data, horizon)
    benchmark = _benchmark_returns(Path(args.benchmark).expanduser(), horizon)
    cost = args.round_trip_cost_bps / 10000
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "REJECTED",
        "research_only": True,
        "data": {
            "rows": int(len(data)),
            "codes": int(data["code"].nunique()),
            "date_start": str(data["trade_date"].min().date()),
            "date_end": str(data["trade_date"].max().date()),
            "event_feature_report": json.loads(feature_path.with_suffix(".report.json").read_text()),
        },
        "validation": {
            "kind": "rule-based event scores; no return-fitted model",
            "non_overlapping_horizon": horizon,
            "round_trip_cost_bps": args.round_trip_cost_bps,
            "top_k": args.top_k,
            "minimum_positive_names": args.min_positive,
            "multiple_score_warning": "Eight predeclared score families are shown; the best retrospective score is not fresh OOS evidence.",
        },
        "scores": {},
        "acceptance": {},
    }
    for name, (score_col, active_col) in score_pairs.items():
        records = _strategy_records(
            data, score_col, active_col, target=args.target, sampled_dates=sampled_dates,
            top_k=args.top_k, min_positive=args.min_positive, cost=cost,
            risk_free=args.risk_free_annual, benchmark=benchmark, horizon=horizon,
        )
        sections = {
            "full": records,
            "frozen": records[records["trade_date"] >= pd.Timestamp(args.frozen_start)],
            "latest": records[records["trade_date"] >= pd.Timestamp(args.latest_start)],
        }
        report["scores"][name] = {
            section: _metrics(frame, horizon, args.risk_free_annual) for section, frame in sections.items()
        }
        frozen_pass = _passes(report["scores"][name]["frozen"], 12)
        latest_pass = _passes(report["scores"][name]["latest"], 6)
        report["acceptance"][name] = {"frozen_pass": frozen_pass, "latest_pass": latest_pass}
    if any(item["frozen_pass"] and item["latest_pass"] for item in report["acceptance"].values()):
        report["status"] = "EXPLORATORY_PASS_NOT_DEPLOYABLE"
    report["limitations"] = [
        "The document library is time-stratified and incomplete; missing downloads are encoded as zero.",
        "Current index membership and static-current sector limitations remain.",
        "The 2025/2026 windows have already been inspected and cannot serve as a fresh final test.",
    ]
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"structured event strategy report saved: {output}")
    print(f"status={report['status']} acceptance={json.dumps(report['acceptance'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
