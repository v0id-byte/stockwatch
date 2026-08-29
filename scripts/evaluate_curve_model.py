#!/usr/bin/env python3
"""Compare Alpha158 factors with causal numeric OHLCV sequences on identical folds."""
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

from scripts.evaluate_dual_model import (
    _benchmark_returns,
    _passes_gate,
    _schema_columns,
    _selection_stats,
)
from scripts.evaluate_neutralized_walk_forward import (
    DEFAULT_STYLE_EXPOSURES,
    _decile_spread,
    _feature_names,
    _ic_stats,
    _merge_market_cap,
    _merge_sector_exposure,
    _rank_label_by_date,
    _rank_normalize_features,
    _train_predict,
    _winsorize,
    make_walk_forward_folds,
    neutralize_by_date,
)


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Evaluate raw numeric price curves against Alpha158.")
    parser.add_argument("--training-set", default=str(root / "training_set.parquet"))
    parser.add_argument("--sequence-features", default=str(root / "sequence_features.parquet"))
    parser.add_argument("--sector-map", default=str(root / "sector_map_sw.parquet"))
    parser.add_argument("--market-cap", default=str(root / "market_cap_daily.parquet"))
    parser.add_argument("--benchmark", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--target", default="forward_20d_return")
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--fold-months", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int, default=180)
    parser.add_argument("--early-stopping", type=int, default=25)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--min-per-date", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--round-trip-cost-bps", type=float, default=20.0)
    parser.add_argument("--risk-free-annual", type=float, default=0.03)
    parser.add_argument("--frozen-start", default="2025-01-01")
    parser.add_argument("--latest-start", default="2026-01-01")
    parser.add_argument("--output", default=str(root / "curve_model_report.json"))
    return parser.parse_args()


def _load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict, list[str], list[str], list[str]]:
    training_path = Path(args.training_set).expanduser()
    sequence_path = Path(args.sequence_features).expanduser()
    if not training_path.exists() or not sequence_path.exists():
        raise RuntimeError("training and sequence feature files are required")
    training_columns = set(_schema_columns(training_path))
    technical = _feature_names("robust", training_columns)
    meta = ["trade_date", "code", args.target]
    data = pd.read_parquet(training_path, columns=[*meta, *technical])
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["code"] = data["code"].astype(str).str.zfill(6)
    sequence = [name for name in _schema_columns(sequence_path) if name.startswith("SEQ_")]
    if not sequence:
        raise RuntimeError("no SEQ_ features found")
    curves = pd.read_parquet(sequence_path, columns=["trade_date", "code", *sequence])
    curves["trade_date"] = pd.to_datetime(curves["trade_date"])
    curves["code"] = curves["code"].astype(str).str.zfill(6)
    if curves.duplicated(["trade_date", "code"]).any():
        raise RuntimeError("sequence feature keys are not unique")
    before = len(data)
    data = data.merge(curves, on=["trade_date", "code"], how="inner", validate="one_to_one")
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
    data = _rank_normalize_features(data, technical)
    combined = [*technical, *sequence]
    needed = [args.target, "neutral_return", "neutral_label", *combined]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)
    metadata = {
        "training_set": str(training_path),
        "sequence_features": str(sequence_path),
        "training_rows_before_sequence_merge": int(before),
        "rows_after_sequence_merge_and_exposures": int(len(data)),
        "sequence_feature_count": len(sequence),
        "technical_feature_count": len(technical),
        "sector": sector_meta,
        "market_cap": cap_meta,
        "neutralization_exposures": style,
    }
    return data, metadata, technical, sequence, combined


def _run_folds(data: pd.DataFrame, folds: list[dict], technical: list[str],
               sequence: list[str], combined: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for index, fold in enumerate(folds, start=1):
        train = data[data["trade_date"] < fold["val_start"]]
        val = data[(data["trade_date"] >= fold["val_start"]) & (data["trade_date"] < fold["val_end_exclusive"])]
        test = data[(data["trade_date"] >= fold["test_start"]) & (data["trade_date"] < fold["test_end_exclusive"])]
        if train.empty or val.empty or test.empty:
            continue
        print(
            f"curve fold {index}/{len(folds)} val={fold['val_start'].date()}~{fold['val_end_exclusive'].date()} "
            f"test={fold['test_start'].date()}~{fold['test_end_exclusive'].date()} rows={len(test)}",
            flush=True,
        )
        out = test[["trade_date", "code", args.target, "neutral_return"]].copy()
        out["fold"] = index
        out["pred_technical"] = _train_predict(train, val, test, technical, "neutral_label", args)
        out["pred_sequence"] = _train_predict(train, val, test, sequence, "neutral_label", args)
        out["pred_combined"] = _train_predict(train, val, test, combined, "neutral_label", args)
        rows.append(out)
    if not rows:
        raise RuntimeError("curve walk-forward produced no predictions")
    return pd.concat(rows, ignore_index=True)


def _non_overlapping(data: pd.DataFrame, step: int) -> pd.DataFrame:
    dates = list(data["trade_date"].drop_duplicates())
    return data[data["trade_date"].isin(set(dates[::max(1, step)]))]


def _slice_metrics(preds: pd.DataFrame, args: argparse.Namespace,
                   benchmark: dict[pd.Timestamp, float]) -> dict:
    horizon = int("".join(character for character in args.target if character.isdigit()) or 20)
    sampled = _non_overlapping(preds, horizon)
    cost = args.round_trip_cost_bps / 10000
    out = {
        "rows": int(len(preds)),
        "dates": int(preds["trade_date"].nunique()),
        "sampled_dates": int(sampled["trade_date"].nunique()),
    }
    for name, score in {
        "technical": "pred_technical",
        "sequence": "pred_sequence",
        "combined": "pred_combined",
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
    return out


def main() -> None:
    args = _parse_args()
    data, metadata, technical, sequence, combined = _load_data(args)
    horizon = int("".join(character for character in args.target if character.isdigit()) or 20)
    folds = make_walk_forward_folds(
        pd.Index(data["trade_date"].drop_duplicates()), horizon,
        args.min_train_months, args.fold_months, args.val_months, args.max_folds,
    )
    if not folds:
        raise RuntimeError("no walk-forward folds")
    predictions = _run_folds(data, folds, technical, sequence, combined, args)
    benchmark = _benchmark_returns(Path(args.benchmark).expanduser(), horizon)
    full = _slice_metrics(predictions, args, benchmark)
    frozen = _slice_metrics(
        predictions[predictions["trade_date"] >= pd.Timestamp(args.frozen_start)], args, benchmark,
    )
    latest = _slice_metrics(
        predictions[predictions["trade_date"] >= pd.Timestamp(args.latest_start)], args, benchmark,
    )
    acceptance = {}
    for name in ("technical", "sequence", "combined"):
        acceptance[f"{name}_frozen_pass"] = _passes_gate(frozen[name]["portfolio"], 12)
        acceptance[f"{name}_latest_pass"] = _passes_gate(latest[name]["portfolio"], 6)
    status = "PASS" if any(
        acceptance[f"{name}_frozen_pass"] and acceptance[f"{name}_latest_pass"]
        for name in ("sequence", "combined")
    ) else "REJECTED"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "data": {
            **metadata,
            "codes": int(data["code"].nunique()),
            "date_start": str(data["trade_date"].min().date()),
            "date_end": str(data["trade_date"].max().date()),
        },
        "validation": {
            "method": "identical expanding purged walk-forward folds",
            "purge_trading_days": horizon,
            "fold_count": len(folds),
            "round_trip_cost_bps": args.round_trip_cost_bps,
            "top_k": args.top_k,
            "frozen_start": args.frozen_start,
            "latest_start": args.latest_start,
        },
        "feature_sets": {
            "technical": technical,
            "sequence": sequence,
            "combined_count": len(combined),
        },
        "full_walk_forward": full,
        "frozen_oos": frozen,
        "latest_oos": latest,
        "acceptance": {
            **acceptance,
            "gate": "CAGR>5%, annualized alpha vs universe>1%, beats CSI500, >=60% positive folds",
        },
        "limitations": [
            "The 2025/2026 windows were already inspected in prior research and are retrospective OOS, not a fresh untouched test.",
            "Sequence and Alpha158 features share the same OHLCV source; this test measures representation increment, not a new information family.",
            "Index membership and sector history retain the limitations documented by the existing research pipeline.",
        ],
    }
    output = Path(args.output).expanduser()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"curve model report saved: {output}")
    print(f"status={status} acceptance={json.dumps(acceptance, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
