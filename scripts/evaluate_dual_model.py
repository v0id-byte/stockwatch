#!/usr/bin/env python3
"""Evaluate separate risk-screen and residual-alpha models with PIT announcements.

Research only: this script writes a JSON report, never a production model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.evaluate_neutralized_walk_forward import (
    DEFAULT_STYLE_EXPOSURES,
    _decile_spread,
    _feature_names,
    _ic_stats,
    _merge_market_cap,
    _merge_sector_exposure,
    _negative_screen_stats,
    _portfolio_stats,
    _rank_label_by_date,
    _rank_normalize_features,
    _series_stats,
    _train_predict,
    _winsorize,
    make_walk_forward_folds,
    neutralize_by_date,
)


KEY_COLUMNS = {"trade_date", "code"}
FUNDAMENTAL_FEATURES = ("ocf_to_eps",)


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Evaluate split risk and alpha models.")
    parser.add_argument("--training-set", default=str(root / "training_set.parquet"))
    parser.add_argument("--announcement-features", default=str(root / "sentiment_features.parquet"))
    parser.add_argument("--sector-map", default=str(root / "sector_map_sw.parquet"))
    parser.add_argument("--market-cap", default=str(root / "market_cap_daily.parquet"))
    parser.add_argument("--benchmark", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--target", default="forward_20d_return")
    parser.add_argument("--drawdown-target", default="forward_20d_drawdown")
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--fold-months", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=180)
    parser.add_argument("--early-stopping", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--min-per-date", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--exclude-bottom-fraction", type=float, default=0.10)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--risk-free-annual", type=float, default=0.03)
    parser.add_argument("--frozen-start", default="2025-01-01")
    parser.add_argument("--latest-start", default="2026-01-01")
    parser.add_argument("--output", default=str(root / "dual_model_report.json"))
    return parser.parse_args()


def _schema_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def _announcement_feature_names(path: Path) -> list[str]:
    return [name for name in _schema_columns(path) if name not in KEY_COLUMNS]


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, list[str], list[str], list[str]]:
    training_path = Path(args.training_set).expanduser()
    announcement_path = Path(args.announcement_features).expanduser()
    if not training_path.exists():
        raise RuntimeError(f"training set missing: {training_path}")
    if not announcement_path.exists():
        raise RuntimeError(f"announcement features missing: {announcement_path}")

    available = set(_schema_columns(training_path))
    technical = _feature_names("robust", available)
    fundamental = [name for name in FUNDAMENTAL_FEATURES if name in available]
    meta = ["trade_date", "code", args.target, args.drawdown_target]
    missing = [name for name in meta if name not in available]
    if missing:
        raise RuntimeError(f"training set missing columns: {missing}")
    data = pd.read_parquet(training_path, columns=[*meta, *technical, *fundamental])
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["code"] = data["code"].astype(str).str.zfill(6)

    announcement_features = _announcement_feature_names(announcement_path)
    messages = pd.read_parquet(announcement_path, columns=["trade_date", "code", *announcement_features])
    messages["trade_date"] = pd.to_datetime(messages["trade_date"])
    messages["code"] = messages["code"].astype(str).str.zfill(6)
    if messages.duplicated(["trade_date", "code"]).any():
        raise RuntimeError("announcement features contain duplicate trade_date/code rows")
    data = data.merge(messages, on=["trade_date", "code"], how="left", validate="one_to_one")
    message_coverage = float(data[announcement_features].notna().any(axis=1).mean())
    data[announcement_features] = data[announcement_features].fillna(0.0)

    data, sector_meta = _merge_sector_exposure(data, args.sector_map)
    data, cap_meta = _merge_market_cap(data, args.market_cap)
    style = [name for name in DEFAULT_STYLE_EXPOSURES if name in data.columns]
    if "log_market_cap" in data.columns and data["log_market_cap"].notna().any():
        style.append("log_market_cap")
    sector_col = "sector" if "sector" in data.columns and data["sector"].notna().any() else None
    data["neutral_return"] = neutralize_by_date(data, args.target, style, sector_col, winsor_tail=0.01)
    data["raw_return_winsor"] = data.groupby("trade_date", sort=False)[args.target].transform(
        lambda values: _winsorize(values, 0.01)
    )
    data["neutral_label"] = _rank_label_by_date(data, "neutral_return")
    data["risk_safe_label"] = _rank_label_by_date(data, args.drawdown_target)

    combined = [*technical, *fundamental, *announcement_features]
    data = _rank_normalize_features(data, combined)
    needed = [args.target, args.drawdown_target, "neutral_return", "neutral_label", "risk_safe_label", *combined]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
    feature_report_path = announcement_path.with_suffix(".report.json")
    feature_report = None
    if feature_report_path.exists():
        try:
            feature_report = json.loads(feature_report_path.read_text())
        except (OSError, json.JSONDecodeError):
            feature_report = None
    metadata = {
        "training_set": str(training_path),
        "announcement_features": str(announcement_path),
        "announcement_feature_report": str(feature_report_path) if feature_report_path.exists() else None,
        "announcement_feature_metadata": feature_report,
        "message_row_coverage": message_coverage,
        "sector": sector_meta,
        "market_cap": cap_meta,
        "neutralization_exposures": style,
        "sector_kind": sector_meta.get("kind"),
    }
    return data, metadata, technical, announcement_features, combined


def _benchmark_returns(path: Path, horizon: int) -> dict[pd.Timestamp, float]:
    if not path.exists():
        return {}
    market = pd.read_parquet(path, columns=["trade_date", "close"]).sort_values("trade_date")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    close = pd.to_numeric(market["close"], errors="coerce")
    market["forward_return"] = close.shift(-horizon) / close - 1
    return dict(zip(market["trade_date"], market["forward_return"]))


def _risk_on_by_date(path: Path, window: int = 200) -> dict[pd.Timestamp, bool]:
    """Canonical prior-close trend gate; no window search is performed."""
    if not path.exists():
        return {}
    market = pd.read_parquet(path, columns=["trade_date", "close"]).sort_values("trade_date")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    close = pd.to_numeric(market["close"], errors="coerce")
    trend = close.rolling(window, min_periods=window).mean()
    market["risk_on"] = (close.shift(1) > trend.shift(1)).fillna(False)
    return dict(zip(market["trade_date"], market["risk_on"].astype(bool)))


def _run_folds(data: pd.DataFrame, folds: list[dict], technical: list[str],
               messages: list[str], combined: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for index, fold in enumerate(folds, start=1):
        train = data[data["trade_date"] < fold["val_start"]]
        val = data[(data["trade_date"] >= fold["val_start"]) & (data["trade_date"] < fold["val_end_exclusive"])]
        test = data[(data["trade_date"] >= fold["test_start"]) & (data["trade_date"] < fold["test_end_exclusive"])]
        if train.empty or val.empty or test.empty:
            continue
        print(
            f"fold {index}/{len(folds)} val={fold['val_start'].date()}~{fold['val_end_exclusive'].date()} "
            f"test={fold['test_start'].date()}~{fold['test_end_exclusive'].date()} rows={len(test)}",
            flush=True,
        )
        out = test[["trade_date", "code", args.target, args.drawdown_target, "neutral_return"]].copy()
        out["fold"] = index
        out["pred_alpha_technical"] = _train_predict(train, val, test, technical, "neutral_label", args)
        out["pred_alpha_announcements"] = _train_predict(train, val, test, messages, "neutral_label", args)
        out["pred_alpha_combined"] = _train_predict(train, val, test, combined, "neutral_label", args)
        out["pred_risk_safe"] = _train_predict(train, val, test, combined, "risk_safe_label", args)
        rows.append(out)
    if not rows:
        raise RuntimeError("walk-forward produced no predictions")
    return pd.concat(rows, ignore_index=True)


def _non_overlapping(data: pd.DataFrame, step: int) -> pd.DataFrame:
    dates = list(data["trade_date"].drop_duplicates())
    return data[data["trade_date"].isin(set(dates[::max(1, step)]))]


def _selection_stats(data: pd.DataFrame, score_col: str, target_col: str, *, horizon: int,
                     top_k: int, cost: float, risk_free: float,
                     benchmark: dict[pd.Timestamp, float], risk_col: str = "",
                     exclude_fraction: float = 0.10, min_names: int = 100,
                     risk_on: dict[pd.Timestamp, bool] | None = None) -> dict | None:
    rows = []
    periods_per_year = 252 / max(1, horizon)
    cash_return = (1 + max(-0.99, risk_free)) ** (1 / periods_per_year) - 1
    for date, group in data.dropna(subset=[score_col, target_col]).groupby("trade_date", sort=False):
        if len(group) < min_names or group[score_col].nunique() <= 1:
            continue
        date = pd.Timestamp(date)
        is_risk_on = True if risk_on is None else bool(risk_on.get(date, False))
        universe = float(group[target_col].mean())
        if not is_risk_on:
            rows.append({
                "date": date,
                "fold": int(group["fold"].iloc[0]) if "fold" in group else 0,
                "selected_net": float(cash_return),
                "universe": universe,
                "net_excess": float(cash_return - universe),
                "benchmark": benchmark.get(date),
                "selected_names": 0,
                "excluded_names": 0,
                "risk_on": False,
            })
            continue
        candidates = group
        excluded = 0
        if risk_col:
            candidates = candidates.dropna(subset=[risk_col]).sort_values(risk_col)
            excluded = max(1, int(np.floor(len(candidates) * exclude_fraction)))
            candidates = candidates.iloc[excluded:]
        selected = candidates.sort_values(score_col, ascending=False).head(min(top_k, len(candidates)))
        if selected.empty:
            continue
        selected_net = float(selected[target_col].mean()) - cost
        rows.append({
            "date": date,
            "fold": int(group["fold"].iloc[0]) if "fold" in group else 0,
            "selected_net": selected_net,
            "universe": universe,
            "net_excess": selected_net - universe,
            "benchmark": benchmark.get(pd.Timestamp(date)),
            "selected_names": int(len(selected)),
            "excluded_names": int(excluded),
            "risk_on": True,
        })
    if not rows:
        return None
    records = pd.DataFrame(rows)
    selected_stats = _portfolio_stats(records["selected_net"].tolist(), horizon, risk_free)
    universe_stats = _portfolio_stats(records["universe"].tolist(), horizon, risk_free)
    benchmark_records = records.dropna(subset=["benchmark"])
    benchmark_stats = (
        _portfolio_stats(benchmark_records["benchmark"].tolist(), horizon, risk_free)
        if not benchmark_records.empty else None
    )
    selected_benchmark_aligned = (
        _portfolio_stats(benchmark_records["selected_net"].tolist(), horizon, risk_free)
        if not benchmark_records.empty else None
    )
    by_fold = records.groupby("fold")["net_excess"].mean()
    annualized_delta_universe = None
    annualized_delta_benchmark = None
    if selected_stats and universe_stats and selected_stats["cagr"] is not None and universe_stats["cagr"] is not None:
        annualized_delta_universe = float(selected_stats["cagr"] - universe_stats["cagr"])
    if (
        selected_benchmark_aligned and benchmark_stats
        and selected_benchmark_aligned["cagr"] is not None
        and benchmark_stats["cagr"] is not None
    ):
        annualized_delta_benchmark = float(selected_benchmark_aligned["cagr"] - benchmark_stats["cagr"])
    return {
        "period_count": int(len(records)),
        "selected": selected_stats,
        "universe": universe_stats,
        "benchmark": benchmark_stats,
        "net_excess": _series_stats(records["net_excess"].tolist()),
        "annualized_delta_vs_universe": annualized_delta_universe,
        "annualized_delta_vs_csi500": annualized_delta_benchmark,
        "positive_fold_rate": float((by_fold > 0).mean()) if len(by_fold) else None,
        "fold_count": int(len(by_fold)),
        "average_selected_names": float(records["selected_names"].mean()),
        "average_excluded_names": float(records["excluded_names"].mean()),
        "risk_on_rate": float(records["risk_on"].mean()),
    }


def _slice_metrics(preds: pd.DataFrame, args: argparse.Namespace,
                   benchmark: dict[pd.Timestamp, float],
                   risk_on: dict[pd.Timestamp, bool]) -> dict:
    horizon = int("".join(ch for ch in args.target if ch.isdigit()) or 20)
    sampled = _non_overlapping(preds, horizon)
    cost = args.round_trip_cost_bps / 10000
    out = {"rows": int(len(preds)), "dates": int(preds["trade_date"].nunique()), "sampled_dates": int(sampled["trade_date"].nunique())}
    for name, score in {
        "technical_alpha": "pred_alpha_technical",
        "announcement_alpha": "pred_alpha_announcements",
        "combined_alpha": "pred_alpha_combined",
    }.items():
        out[name] = {
            "raw_return_ic": _ic_stats(preds, score, args.target, args.min_per_date),
            "neutral_return_ic": _ic_stats(preds, score, "neutral_return", args.min_per_date),
            "raw_decile": _decile_spread(preds, score, args.target, args.min_per_date),
            "portfolio": _selection_stats(
                sampled, score, args.target, horizon=horizon, top_k=args.top_k,
                cost=cost, risk_free=args.risk_free_annual, benchmark=benchmark,
                min_names=args.min_per_date,
            ),
        }
    out["risk_model"] = {
        "drawdown_ic": _ic_stats(preds, "pred_risk_safe", args.drawdown_target, args.min_per_date),
        "screen": _negative_screen_stats(
            sampled, "pred_risk_safe", args.target, args.min_per_date,
            args.exclude_bottom_fraction, cost, horizon, args.risk_free_annual,
        ),
    }
    out["dual_portfolio"] = _selection_stats(
        sampled, "pred_alpha_combined", args.target, horizon=horizon, top_k=args.top_k,
        cost=cost, risk_free=args.risk_free_annual, benchmark=benchmark,
        risk_col="pred_risk_safe", exclude_fraction=args.exclude_bottom_fraction,
        min_names=args.min_per_date,
    )
    out["regime_announcement_portfolio"] = _selection_stats(
        sampled, "pred_alpha_announcements", args.target, horizon=horizon, top_k=args.top_k,
        cost=cost, risk_free=args.risk_free_annual, benchmark=benchmark,
        min_names=args.min_per_date, risk_on=risk_on,
    )
    return out


def _passes_gate(metrics: dict, minimum_periods: int) -> bool:
    portfolio = metrics or {}
    selected = portfolio.get("selected") or {}
    return bool(
        (portfolio.get("period_count") or 0) >= minimum_periods
        and (selected.get("cagr") or -1) > 0.05
        and (portfolio.get("annualized_delta_vs_universe") or -1) > 0.01
        and (portfolio.get("annualized_delta_vs_csi500") or -1) > 0
        and (portfolio.get("positive_fold_rate") or 0) >= 0.60
    )


def main() -> None:
    args = _parse_args()
    data, metadata, technical, messages, combined = _load_data(args)
    horizon = int("".join(ch for ch in args.target if ch.isdigit()) or 20)
    folds = make_walk_forward_folds(
        pd.Index(data["trade_date"].drop_duplicates()), horizon,
        args.min_train_months, args.fold_months, args.val_months, args.max_folds,
    )
    if not folds:
        raise RuntimeError("no walk-forward folds")
    preds = _run_folds(data, folds, technical, messages, combined, args)
    benchmark_path = Path(args.benchmark).expanduser()
    benchmark = _benchmark_returns(benchmark_path, horizon)
    risk_on = _risk_on_by_date(benchmark_path)
    full = _slice_metrics(preds, args, benchmark, risk_on)
    frozen = _slice_metrics(
        preds[preds["trade_date"] >= pd.Timestamp(args.frozen_start)], args, benchmark, risk_on,
    )
    latest = _slice_metrics(
        preds[preds["trade_date"] >= pd.Timestamp(args.latest_start)], args, benchmark, risk_on,
    )
    announcement_pass = _passes_gate(frozen["announcement_alpha"]["portfolio"], minimum_periods=12)
    combined_pass = _passes_gate(frozen["combined_alpha"]["portfolio"], minimum_periods=12)
    dual_pass = _passes_gate(frozen["dual_portfolio"], minimum_periods=12)
    latest_announcement_pass = _passes_gate(latest["announcement_alpha"]["portfolio"], minimum_periods=6)
    latest_combined_pass = _passes_gate(latest["combined_alpha"]["portfolio"], minimum_periods=6)
    latest_dual_pass = _passes_gate(latest["dual_portfolio"], minimum_periods=6)
    status = "PASS" if any((
        announcement_pass and latest_announcement_pass,
        combined_pass and latest_combined_pass,
        dual_pass and latest_dual_pass,
    )) else "REJECTED"
    is_structured_fulltext = "structured_event" in str(metadata.get("announcement_features", ""))
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "model_roles": {
            "alpha": "rank neutralized forward return",
            "risk": "rank forward drawdown from worst to safest",
            "dual": "exclude bottom risk decile, then select alpha top-k",
            "regime_exploratory": "prior-close CSI500 above 200-day mean; otherwise 3% annual cash",
        },
        "feature_sets": {
            "technical": technical,
            "announcements": messages,
            "combined": combined,
        },
        "data": {
            **metadata,
            "rows": int(len(data)),
            "codes": int(data["code"].nunique()),
            "date_start": str(data["trade_date"].min().date()),
            "date_end": str(data["trade_date"].max().date()),
        },
        "validation": {
            "method": "expanding purged walk-forward with separate validation and test windows",
            "purge_trading_days": horizon,
            "fold_count": len(folds),
            "frozen_start": args.frozen_start,
            "latest_start": args.latest_start,
            "round_trip_cost_bps": args.round_trip_cost_bps,
            "top_k": args.top_k,
            "exclude_bottom_fraction": args.exclude_bottom_fraction,
        },
        "full_walk_forward": full,
        "frozen_oos": frozen,
        "latest_oos": latest,
        "acceptance": {
            "announcement_alpha_pass": announcement_pass,
            "combined_alpha_pass": combined_pass,
            "dual_portfolio_pass": dual_pass,
            "latest_announcement_pass": latest_announcement_pass,
            "latest_combined_pass": latest_combined_pass,
            "latest_dual_pass": latest_dual_pass,
            "gate": "CAGR>5%, annualized alpha vs universe>1%, beats CSI500, >=60% positive folds",
        },
        "dl_decision": (
            "RUN_SHALLOW_NN_CONTROL" if status == "PASS"
            else "SKIP_DL_NO_TRADITIONAL_MODEL_SIGNAL"
        ),
        "limitations": [
            "Index membership is a current snapshot, not historical point-in-time membership.",
            "Sector mapping is static-current unless a dated sector file is supplied.",
            (
                "Structured full-text event features come from a time-stratified partial document library; zero can mean not downloaded."
                if is_structured_fulltext else
                "Announcement features are title/category aggregates; no historical full-text news corpus is available."
            ),
            "The 200-day regime overlay was added after inspecting the first dual-model result and is exploratory, not fresh OOS evidence.",
        ],
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"dual-model report saved: {output}")
    print(f"status={status} dl_decision={report['dl_decision']}")


if __name__ == "__main__":
    main()
