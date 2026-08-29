#!/usr/bin/env python3
"""Evaluate passive equal-weight core portfolios with event-driven exclusion filters."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


SENTIMENT_COLUMNS = [
    "ann_count_20d",
    "ann_holding_count_20d",
    "ann_risk_count_20d",
    "ann_capital_action_count_20d",
]
PREREGISTERED_FILTER = "combo_broad_negative"
PREREGISTERED_HORIZON = 20
RISK_CONTROL_COLUMN = "STD20"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate passive-core exclusion overlays.")
    parser.add_argument("--root", default=os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history"))
    parser.add_argument("--training-set", default="")
    parser.add_argument("--sentiment-features", default="")
    parser.add_argument("--pead-events", default="")
    parser.add_argument("--horizons", default="5,20,60")
    parser.add_argument("--rebalance-step", type=int, default=0, help="0 means non-overlap by horizon.")
    parser.add_argument("--min-names", type=int, default=100)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--control-reps", type=int, default=200)
    parser.add_argument("--control-seed", type=int, default=42)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _json_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _root(raw: str) -> Path:
    return Path(raw).expanduser()


def _latest_sentiment_path(root: Path) -> Path | None:
    paths = sorted(root.glob("sentiment_features*.parquet"))
    return paths[-1] if paths else None


def _load_training(path: Path, horizons: list[int]) -> pd.DataFrame:
    columns = ["trade_date", "code", *[f"forward_{h}d_return" for h in horizons], RISK_CONTROL_COLUMN]
    data = pd.read_parquet(path, columns=columns)
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    return data


def _load_sentiment(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["trade_date", "code", *SENTIMENT_COLUMNS])
    columns = ["trade_date", "code", *SENTIMENT_COLUMNS]
    available = pd.read_parquet(path, columns=None)
    keep = [col for col in columns if col in available.columns]
    data = available[keep].copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.normalize()
    data["code"] = data["code"].astype(str).str.zfill(6)
    return data


def _entry_date_after(disclosure_date: pd.Timestamp, trade_dates: pd.Index) -> pd.Timestamp | None:
    if pd.isna(disclosure_date):
        return None
    idx = int(trade_dates.searchsorted(pd.Timestamp(disclosure_date).normalize(), side="right"))
    if idx >= len(trade_dates):
        return None
    return pd.Timestamp(trade_dates[idx]).normalize()


def _load_pead_events(path: Path, trade_dates: pd.Index) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    events = pd.read_parquet(path)
    events["code"] = events["code"].astype(str).str.zfill(6)
    events["available_at"] = pd.to_datetime(events["available_at"], errors="coerce").dt.normalize()
    events["sign"] = pd.to_numeric(events["sign"], errors="coerce")
    for col in ["strong_negative", "is_turnaround"]:
        if col not in events.columns:
            events[col] = False
        events[col] = events[col].fillna(False).astype(bool)
    events["entry_date"] = events["available_at"].map(lambda value: _entry_date_after(value, trade_dates))
    return events.dropna(subset=["entry_date", "code", "sign"]).copy()


def _expand_event_flag(events: pd.DataFrame, trade_dates: pd.Index, mask: pd.Series,
                       flag_name: str, lookback_days: int) -> pd.DataFrame:
    date_to_idx = {pd.Timestamp(value): idx for idx, value in enumerate(trade_dates)}
    records = []
    subset = events[mask].copy()
    for row in subset.itertuples(index=False):
        start_idx = date_to_idx.get(pd.Timestamp(row.entry_date))
        if start_idx is None:
            continue
        end_idx = min(start_idx + lookback_days, len(trade_dates))
        for idx in range(start_idx, end_idx):
            records.append({"trade_date": pd.Timestamp(trade_dates[idx]), "code": row.code, flag_name: True})
    if not records:
        return pd.DataFrame(columns=["trade_date", "code", flag_name])
    return pd.DataFrame(records).drop_duplicates(["trade_date", "code"])


def _build_pead_flags(events: pd.DataFrame, trade_dates: pd.Index) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["trade_date", "code"])
    specs = [
        ("pead_negative_20d", events["sign"] < 0, 20),
        ("pead_strong_negative_20d", events["strong_negative"], 20),
        ("pead_negative_turnaround_60d", (events["sign"] < 0) & events["is_turnaround"], 60),
        ("pead_negative_60d", events["sign"] < 0, 60),
    ]
    out = None
    for flag_name, mask, lookback in specs:
        frame = _expand_event_flag(events, trade_dates, mask, flag_name, lookback)
        out = frame if out is None else out.merge(frame, on=["trade_date", "code"], how="outer")
    if out is None:
        return pd.DataFrame(columns=["trade_date", "code"])
    for flag_name, _, _ in specs:
        out[flag_name] = out[flag_name].fillna(False).astype(bool)
    return out


def _load_market(root: Path, horizons: list[int]) -> pd.DataFrame:
    path = root / "market_sh000300.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["trade_date"])
    market = pd.read_parquet(path)
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    market = market.sort_values("trade_date").drop_duplicates("trade_date")
    close = pd.to_numeric(market["close"], errors="coerce")
    out = market[["trade_date"]].copy()
    for horizon in horizons:
        out[f"csi300_forward_{horizon}d_return"] = close.shift(-horizon) / close - 1
    return out


def _attach_filters(train: pd.DataFrame, sentiment: pd.DataFrame, pead_flags: pd.DataFrame) -> pd.DataFrame:
    data = train.merge(sentiment, on=["trade_date", "code"], how="left")
    for col in SENTIMENT_COLUMNS:
        if col not in data.columns:
            data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    if not pead_flags.empty:
        data = data.merge(pead_flags, on=["trade_date", "code"], how="left")
    for col in ["pead_negative_20d", "pead_strong_negative_20d", "pead_negative_turnaround_60d", "pead_negative_60d"]:
        if col not in data.columns:
            data[col] = False
        data[col] = data[col].fillna(False).astype(bool)

    data["ann_holding_20d"] = data["ann_holding_count_20d"] > 0
    data["ann_risk_20d"] = data["ann_risk_count_20d"] > 0
    data["ann_capital_action_20d"] = data["ann_capital_action_count_20d"] > 0
    q90 = data.groupby("trade_date")["ann_count_20d"].transform(lambda values: values.quantile(0.90))
    data["ann_high_activity_q90_20d"] = (data["ann_count_20d"] > 0) & (data["ann_count_20d"] >= q90)
    data["combo_narrow_negative"] = (
        data["ann_risk_20d"]
        | data["ann_holding_20d"]
        | data["pead_strong_negative_20d"]
    )
    data["combo_broad_negative"] = (
        data["ann_risk_20d"]
        | data["ann_holding_20d"]
        | data["ann_capital_action_20d"]
        | data["pead_negative_20d"]
    )
    return data


def _distribution(values: pd.Series) -> dict | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "count": int(len(values)),
        "mean": _json_float(values.mean()),
        "median": _json_float(values.median()),
        "std": _json_float(values.std()),
        "positive_rate": float((values > 0).mean()),
        "q10": _json_float(values.quantile(0.10)),
        "q90": _json_float(values.quantile(0.90)),
    }


def _tstat(values: pd.Series) -> dict | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 3:
        return None
    std = values.std()
    if not std or pd.isna(std):
        return {"n": int(len(values)), "mean": _json_float(values.mean()), "t": None}
    se = std / math.sqrt(len(values))
    return {"n": int(len(values)), "mean": _json_float(values.mean()), "se": float(se), "t": float(values.mean() / se)}


def _annualized_return(period_returns: pd.Series, horizon: int) -> float | None:
    values = pd.to_numeric(period_returns, errors="coerce").dropna()
    if values.empty:
        return None
    total = float((1 + values).prod())
    if total <= 0:
        return None
    periods_per_year = 252.0 / horizon
    return float(total ** (periods_per_year / len(values)) - 1)


def _max_drawdown(period_returns: pd.Series) -> float | None:
    values = pd.to_numeric(period_returns, errors="coerce").dropna()
    if values.empty:
        return None
    equity = (1 + values).cumprod()
    drawdown = equity / equity.cummax() - 1
    return _json_float(drawdown.min())


def _downside_stats(records: pd.DataFrame) -> dict:
    baseline = pd.to_numeric(records["baseline_net"], errors="coerce").dropna()
    filtered = pd.to_numeric(records["filtered_net"], errors="coerce").dropna()
    excess = pd.to_numeric(records["filtered_excess_vs_universe"], errors="coerce").dropna()
    out = {
        "baseline_negative_rate": float((baseline < 0).mean()) if len(baseline) else None,
        "filtered_negative_rate": float((filtered < 0).mean()) if len(filtered) else None,
        "excess_positive_rate": float((excess > 0).mean()) if len(excess) else None,
        "baseline_worst_period": _json_float(baseline.min()) if len(baseline) else None,
        "filtered_worst_period": _json_float(filtered.min()) if len(filtered) else None,
    }
    if out["baseline_negative_rate"] is not None and out["filtered_negative_rate"] is not None:
        out["negative_rate_delta"] = out["filtered_negative_rate"] - out["baseline_negative_rate"]
    if out["baseline_worst_period"] is not None and out["filtered_worst_period"] is not None:
        out["worst_period_delta"] = out["filtered_worst_period"] - out["baseline_worst_period"]
    return out


def _summarize_records(records: pd.DataFrame, horizon: int, include_breakdowns: bool = True) -> dict:
    active = records[records["excluded_count"] > 0].copy()
    baseline_annualized = _annualized_return(records["baseline_net"], horizon)
    filtered_annualized = _annualized_return(records["filtered_net"], horizon)
    baseline_drawdown = _max_drawdown(records["baseline_net"])
    filtered_drawdown = _max_drawdown(records["filtered_net"])
    out = {
        "rebalances": int(len(records)),
        "active_rebalances": int(len(active)),
        "avg_total_names": _json_float(records["total_count"].mean()),
        "avg_excluded_names": _json_float(records["excluded_count"].mean()),
        "avg_excluded_rate": _json_float(records["excluded_rate"].mean()),
        "baseline_net": _distribution(records["baseline_net"]),
        "filtered_net": _distribution(records["filtered_net"]),
        "filtered_excess_vs_universe": _distribution(records["filtered_excess_vs_universe"]),
        "filtered_excess_vs_universe_t": _tstat(records["filtered_excess_vs_universe"]),
        "excluded_bucket_excess_vs_universe": _distribution(active["excluded_excess_vs_universe"]) if not active.empty else None,
        "excluded_bucket_excess_vs_universe_t": _tstat(active["excluded_excess_vs_universe"]) if not active.empty else None,
        "baseline_excess_vs_csi300": _distribution(records["baseline_excess_vs_csi300"]),
        "filtered_excess_vs_csi300": _distribution(records["filtered_excess_vs_csi300"]),
        "baseline_annualized": baseline_annualized,
        "filtered_annualized": filtered_annualized,
        "annualized_delta": (
            filtered_annualized - baseline_annualized
            if filtered_annualized is not None and baseline_annualized is not None else None
        ),
        "baseline_max_drawdown": baseline_drawdown,
        "filtered_max_drawdown": filtered_drawdown,
        "max_drawdown_delta": (
            filtered_drawdown - baseline_drawdown
            if filtered_drawdown is not None and baseline_drawdown is not None else None
        ),
        "downside": _downside_stats(records),
    }
    if not include_breakdowns:
        return out

    by_year = {}
    for year, group in records.groupby(records["trade_date"].dt.year):
        by_year[str(int(year))] = _summarize_records(group, horizon, include_breakdowns=False)
    train = records[records["trade_date"].dt.year <= 2024].copy()
    test = records[records["trade_date"].dt.year >= 2025].copy()
    out["by_year"] = by_year
    out["oos_split"] = {
        "train_2022_2024": _summarize_records(train, horizon, include_breakdowns=False) if not train.empty else None,
        "test_2025_2026": _summarize_records(test, horizon, include_breakdowns=False) if not test.empty else None,
        "note": "Single frozen split for regime sanity only; not a retrained model.",
    }
    return {
        **out,
    }


def _portfolio_record(trade_date, group: pd.DataFrame, excluded: pd.DataFrame, kept: pd.DataFrame,
                      target: str, csi_col: str, cost: float) -> dict:
    baseline = float(group[target].mean())
    filtered = float(kept[target].mean())
    excluded_return = float(excluded[target].mean()) if not excluded.empty else np.nan
    csi = float(group[csi_col].iloc[0]) if csi_col in group and pd.notna(group[csi_col].iloc[0]) else np.nan
    return {
        "trade_date": trade_date,
        "total_count": int(len(group)),
        "kept_count": int(len(kept)),
        "excluded_count": int(len(excluded)),
        "excluded_rate": float(len(excluded) / len(group)),
        "baseline_net": baseline - cost,
        "filtered_net": filtered - cost,
        "filtered_excess_vs_universe": filtered - baseline,
        "excluded_excess_vs_universe": excluded_return - baseline if not excluded.empty else np.nan,
        "baseline_excess_vs_csi300": baseline - csi - cost if pd.notna(csi) else np.nan,
        "filtered_excess_vs_csi300": filtered - csi - cost if pd.notna(csi) else np.nan,
    }


def _merged_for_horizon(data: pd.DataFrame, market: pd.DataFrame, horizon: int) -> pd.DataFrame:
    csi_col = f"csi300_forward_{horizon}d_return"
    if csi_col in market:
        return data.merge(market[["trade_date", csi_col]], on="trade_date", how="left")
    return data.copy()


def _selected_dates(data: pd.DataFrame, step: int) -> set[pd.Timestamp]:
    dates = sorted(data["trade_date"].drop_duplicates())
    return set(dates[::step])


def _filter_records(merged: pd.DataFrame, filter_col: str, horizon: int,
                    step: int, min_names: int, cost: float) -> pd.DataFrame:
    target = f"forward_{horizon}d_return"
    csi_col = f"csi300_forward_{horizon}d_return"
    selected_dates = _selected_dates(merged, step)
    rows = []
    for trade_date, group in merged[merged["trade_date"].isin(selected_dates)].dropna(subset=[target]).groupby("trade_date"):
        if len(group) < min_names:
            continue
        excluded = group[group[filter_col]]
        kept = group[~group[filter_col]]
        if len(kept) < min_names:
            continue
        rows.append(_portfolio_record(trade_date, group, excluded, kept, target, csi_col, cost))
    return pd.DataFrame(rows)


def _control_records(merged: pd.DataFrame, reference_records: pd.DataFrame, horizon: int,
                     step: int, min_names: int, cost: float, mode: str,
                     rng: np.random.Generator | None = None) -> pd.DataFrame:
    target = f"forward_{horizon}d_return"
    csi_col = f"csi300_forward_{horizon}d_return"
    selected_dates = _selected_dates(merged, step)
    exclude_counts = {
        pd.Timestamp(row.trade_date): int(row.excluded_count)
        for row in reference_records.itertuples(index=False)
    }
    rows = []
    for trade_date, group in merged[merged["trade_date"].isin(selected_dates)].dropna(subset=[target]).groupby("trade_date"):
        exclude_count = exclude_counts.get(pd.Timestamp(trade_date), 0)
        if exclude_count <= 0 or len(group) < min_names or len(group) - exclude_count < min_names:
            continue
        if mode == "random":
            if rng is None:
                raise ValueError("random control requires rng")
            selected = rng.choice(group.index.to_numpy(), size=exclude_count, replace=False)
            excluded = group.loc[selected]
        elif mode == "high_volatility":
            if RISK_CONTROL_COLUMN not in group.columns:
                return pd.DataFrame()
            ranked = group.assign(
                _risk_control=pd.to_numeric(group[RISK_CONTROL_COLUMN], errors="coerce").fillna(-np.inf)
            ).sort_values("_risk_control", ascending=False)
            excluded = ranked.head(exclude_count).drop(columns=["_risk_control"])
        else:
            raise ValueError(mode)
        kept = group.drop(index=excluded.index)
        rows.append(_portfolio_record(trade_date, group, excluded, kept, target, csi_col, cost))
    return pd.DataFrame(rows)


def _control_metric_row(summary: dict) -> dict:
    return {
        "filtered_excess_vs_universe_mean": _metric_mean(summary, "filtered_excess_vs_universe"),
        "excluded_bucket_excess_vs_universe_mean": _metric_mean(summary, "excluded_bucket_excess_vs_universe"),
        "annualized_delta": summary.get("annualized_delta"),
        "max_drawdown_delta": summary.get("max_drawdown_delta"),
        "negative_rate_delta": (summary.get("downside") or {}).get("negative_rate_delta"),
        "worst_period_delta": (summary.get("downside") or {}).get("worst_period_delta"),
    }


def _metric_percentile(actual: float | None, controls: pd.Series, higher_is_better: bool = True) -> float | None:
    values = pd.to_numeric(controls, errors="coerce").dropna()
    if actual is None or values.empty:
        return None
    if higher_is_better:
        return float((values <= actual).mean())
    return float((values >= actual).mean())


def _control_distribution(rows: pd.DataFrame) -> dict:
    return {
        col: _distribution(rows[col])
        for col in rows.columns
    }


def _preregistered_controls(data: pd.DataFrame, market: pd.DataFrame, horizon: int,
                            step: int, min_names: int, cost: float,
                            reps: int, seed: int) -> dict:
    merged = _merged_for_horizon(data, market, horizon)
    actual_records = _filter_records(merged, PREREGISTERED_FILTER, horizon, step, min_names, cost)
    if actual_records.empty:
        return {"status": "missing_actual_records"}
    actual_summary = _summarize_records(actual_records, horizon, include_breakdowns=False)
    actual_metrics = _control_metric_row(actual_summary)

    rng = np.random.default_rng(seed)
    random_rows = []
    for _ in range(max(0, reps)):
        records = _control_records(merged, actual_records, horizon, step, min_names, cost, "random", rng)
        if records.empty:
            continue
        random_rows.append(_control_metric_row(_summarize_records(records, horizon, include_breakdowns=False)))
    random_frame = pd.DataFrame(random_rows)

    high_vol_records = _control_records(merged, actual_records, horizon, step, min_names, cost, "high_volatility")
    high_vol_summary = (
        _summarize_records(high_vol_records, horizon, include_breakdowns=False)
        if not high_vol_records.empty else None
    )
    high_vol_metrics = _control_metric_row(high_vol_summary) if high_vol_summary else None

    percentile = {}
    if not random_frame.empty:
        percentile = {
            "annualized_delta": _metric_percentile(actual_metrics["annualized_delta"], random_frame["annualized_delta"]),
            "max_drawdown_delta": _metric_percentile(actual_metrics["max_drawdown_delta"], random_frame["max_drawdown_delta"]),
            "filtered_excess_vs_universe_mean": _metric_percentile(
                actual_metrics["filtered_excess_vs_universe_mean"],
                random_frame["filtered_excess_vs_universe_mean"],
            ),
            "negative_rate_delta": _metric_percentile(
                actual_metrics["negative_rate_delta"],
                random_frame["negative_rate_delta"],
                higher_is_better=False,
            ),
        }

    return {
        "status": "ok",
        "horizon": horizon,
        "excluded_count_source": PREREGISTERED_FILTER,
        "actual": actual_metrics,
        "random_same_count": {
            "seed": seed,
            "requested_reps": reps,
            "completed_reps": int(len(random_frame)),
            "metric_distribution": _control_distribution(random_frame) if not random_frame.empty else {},
            "actual_percentile": percentile,
        },
        "high_volatility_same_count": {
            "risk_column": RISK_CONTROL_COLUMN,
            "summary": high_vol_metrics,
            "records": int(len(high_vol_records)) if high_vol_summary else 0,
        },
        "interpretation": (
            "Controls remove the same number of names on each rebalance date. "
            "Random controls test mechanical de-risking; high-volatility controls test whether the filter only removes volatile names."
        ),
    }


def _evaluate_filter(data: pd.DataFrame, market: pd.DataFrame, filter_col: str, horizon: int,
                     step: int, min_names: int, cost: float) -> dict:
    merged = _merged_for_horizon(data, market, horizon)
    records = _filter_records(merged, filter_col, horizon, step, min_names, cost)
    if records.empty:
        return {"rebalances": 0}
    return _summarize_records(records, horizon)


def _filter_columns() -> list[str]:
    return [
        "ann_holding_20d",
        "ann_risk_20d",
        "ann_capital_action_20d",
        "ann_high_activity_q90_20d",
        "pead_negative_20d",
        "pead_strong_negative_20d",
        "pead_negative_turnaround_60d",
        "pead_negative_60d",
        "combo_narrow_negative",
        "combo_broad_negative",
    ]


def _metric_mean(summary: dict | None, key: str) -> float | None:
    if not summary:
        return None
    item = summary.get(key)
    if not item:
        return None
    return item.get("mean")


def _preregistered_summary(results: dict) -> dict:
    candidate = results.get(PREREGISTERED_FILTER, {}).get(str(PREREGISTERED_HORIZON))
    if not candidate or candidate.get("rebalances", 0) == 0:
        return {
            "filter": PREREGISTERED_FILTER,
            "horizon": PREREGISTERED_HORIZON,
            "status": "missing",
        }

    by_year = candidate.get("by_year", {})
    year_rows = {}
    positive_excess_years = 0
    drawdown_improved_years = 0
    downside_improved_years = 0
    for year, item in by_year.items():
        excess_mean = _metric_mean(item, "filtered_excess_vs_universe")
        drawdown_delta = item.get("max_drawdown_delta")
        downside_delta = (item.get("downside") or {}).get("negative_rate_delta")
        year_rows[year] = {
            "filtered_excess_vs_universe_mean": excess_mean,
            "max_drawdown_delta": drawdown_delta,
            "negative_rate_delta": downside_delta,
            "filtered_annualized": item.get("filtered_annualized"),
            "baseline_annualized": item.get("baseline_annualized"),
            "rebalances": item.get("rebalances"),
        }
        if excess_mean is not None and excess_mean > 0:
            positive_excess_years += 1
        if drawdown_delta is not None and drawdown_delta > 0:
            drawdown_improved_years += 1
        if downside_delta is not None and downside_delta < 0:
            downside_improved_years += 1

    test = (candidate.get("oos_split") or {}).get("test_2025_2026") or {}
    test_excess = _metric_mean(test, "filtered_excess_vs_universe")
    test_drawdown_delta = test.get("max_drawdown_delta")
    test_downside_delta = (test.get("downside") or {}).get("negative_rate_delta")
    full = {
        "filtered_excess_vs_universe_mean": _metric_mean(candidate, "filtered_excess_vs_universe"),
        "filtered_excess_vs_universe_t": (candidate.get("filtered_excess_vs_universe_t") or {}).get("t"),
        "excluded_bucket_excess_vs_universe_mean": _metric_mean(candidate, "excluded_bucket_excess_vs_universe"),
        "excluded_bucket_excess_vs_universe_t": (candidate.get("excluded_bucket_excess_vs_universe_t") or {}).get("t"),
        "annualized_delta": candidate.get("annualized_delta"),
        "max_drawdown_delta": candidate.get("max_drawdown_delta"),
        "negative_rate_delta": (candidate.get("downside") or {}).get("negative_rate_delta"),
        "excess_positive_rate": (candidate.get("downside") or {}).get("excess_positive_rate"),
    }
    pass_alpha_gate = (
        positive_excess_years >= 3
        and test_excess is not None and test_excess > 0
        and (test.get("annualized_delta") or 0) > 0.01
    )
    pass_drawdown_gate = (
        drawdown_improved_years >= 3
        and full["max_drawdown_delta"] is not None and full["max_drawdown_delta"] > 0
        and test_drawdown_delta is not None and test_drawdown_delta > 0
    )
    status = "research_only_fail"
    if pass_drawdown_gate and pass_alpha_gate:
        status = "research_only_alpha_and_drawdown_candidate"
    elif pass_drawdown_gate:
        status = "research_only_drawdown_candidate_not_alpha"
    return {
        "filter": PREREGISTERED_FILTER,
        "horizon": PREREGISTERED_HORIZON,
        "status": status,
        "gate": {
            "positive_excess_years_required": 3,
            "drawdown_improved_years_required": 3,
            "alpha_oos_2025_2026_annualized_delta_min": 0.01,
            "oos_2025_2026_drawdown_delta_must_be_positive": True,
        },
        "full_sample": full,
        "stability": {
            "years": year_rows,
            "positive_excess_years": positive_excess_years,
            "drawdown_improved_years": drawdown_improved_years,
            "downside_improved_years": downside_improved_years,
            "passed_alpha_gate": pass_alpha_gate,
            "passed_drawdown_gate": pass_drawdown_gate,
            "oos_2025_2026": {
                "filtered_excess_vs_universe_mean": test_excess,
                "max_drawdown_delta": test_drawdown_delta,
                "negative_rate_delta": test_downside_delta,
                "annualized_delta": test.get("annualized_delta"),
                "rebalances": test.get("rebalances"),
            },
        },
        "interpretation": "Use as a risk/exclusion-layer research candidate only; exploratory grid results are not deployment evidence.",
    }


def main() -> None:
    args = _parse_args()
    root = _root(args.root)
    horizons = _int_list(args.horizons)
    training_path = Path(args.training_set).expanduser() if args.training_set else root / "training_set.parquet"
    sentiment_path = Path(args.sentiment_features).expanduser() if args.sentiment_features else _latest_sentiment_path(root)
    pead_path = Path(args.pead_events).expanduser() if args.pead_events else root / "pead_events_structured.parquet"
    output = Path(args.output).expanduser() if args.output else root / "exclusion_layer_report.json"
    cost = args.cost_bps / 10000.0

    train = _load_training(training_path, horizons)
    trade_dates = pd.Index(sorted(train["trade_date"].drop_duplicates()))
    sentiment = _load_sentiment(sentiment_path)
    pead_events = _load_pead_events(pead_path, trade_dates)
    pead_flags = _build_pead_flags(pead_events, trade_dates)
    data = _attach_filters(train, sentiment, pead_flags)
    market = _load_market(root, horizons)

    results = {}
    for filter_col in _filter_columns():
        results[filter_col] = {}
        for horizon in horizons:
            step = args.rebalance_step or horizon
            results[filter_col][str(horizon)] = _evaluate_filter(
                data,
                market,
                filter_col,
                horizon,
                step,
                args.min_names,
                cost,
            )

    coverage = {}
    for filter_col in _filter_columns():
        values = data[filter_col]
        coverage[filter_col] = {
            "flagged_rows": int(values.sum()),
            "flagged_rate": float(values.mean()),
            "flagged_codes": int(data.loc[values, "code"].nunique()),
            "flagged_dates": int(data.loc[values, "trade_date"].nunique()),
        }

    preregistered = _preregistered_summary(results)
    if PREREGISTERED_HORIZON in horizons:
        preregistered["controls"] = _preregistered_controls(
            data,
            market,
            PREREGISTERED_HORIZON,
            args.rebalance_step or PREREGISTERED_HORIZON,
            args.min_names,
            cost,
            args.control_reps,
            args.control_seed,
        )
    else:
        preregistered["controls"] = {"status": "horizon_not_requested"}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_set": str(training_path),
        "sentiment_features": str(sentiment_path) if sentiment_path else None,
        "pead_events": str(pead_path),
        "rows": int(len(data)),
        "codes": int(data["code"].nunique()),
        "dates": int(data["trade_date"].nunique()),
        "horizons": horizons,
        "rebalance": "non_overlap_by_horizon" if args.rebalance_step == 0 else f"every_{args.rebalance_step}_trading_days",
        "cost_bps_applied_symmetrically": args.cost_bps,
        "primary_metric": "filtered_excess_vs_universe",
        "benchmark_note": "Universe is equal-weight training_set universe; CSI300 is price index and secondary only.",
        "multiple_testing_note": (
            "Full filter x horizon grid is exploratory. The preregistered candidate is "
            f"{PREREGISTERED_FILTER} at {PREREGISTERED_HORIZON} trading days."
        ),
        "filter_coverage": coverage,
        "results": results,
        "preregistered_candidate": preregistered,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"exclusion layer report saved: {output}")
    for filter_col in _filter_columns():
        h20 = results[filter_col].get("20") or {}
        excess = h20.get("filtered_excess_vs_universe") or {}
        excluded = h20.get("excluded_bucket_excess_vs_universe") or {}
        t = h20.get("filtered_excess_vs_universe_t") or {}
        print(
            f"{filter_col}: h20 active={h20.get('active_rebalances')} "
            f"excess_mean={excess.get('mean')} t={t.get('t')} "
            f"excluded_mean={excluded.get('mean')}"
        )
    candidate = report["preregistered_candidate"]
    full = candidate.get("full_sample") or {}
    stability = candidate.get("stability") or {}
    print(
        f"preregistered {candidate.get('filter')} h={candidate.get('horizon')} "
        f"status={candidate.get('status')} excess={full.get('filtered_excess_vs_universe_mean')} "
        f"mdd_delta={full.get('max_drawdown_delta')} "
        f"positive_years={stability.get('positive_excess_years')} "
        f"drawdown_years={stability.get('drawdown_improved_years')}"
    )
    controls = candidate.get("controls") or {}
    random_control = controls.get("random_same_count") or {}
    high_vol_control = controls.get("high_volatility_same_count") or {}
    print(
        f"controls status={controls.get('status')} random_reps={random_control.get('completed_reps')} "
        f"actual_mdd_percentile={((random_control.get('actual_percentile') or {}).get('max_drawdown_delta'))} "
        f"high_vol_mdd_delta={((high_vol_control.get('summary') or {}).get('max_drawdown_delta'))}"
    )


if __name__ == "__main__":
    main()
