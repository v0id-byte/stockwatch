#!/usr/bin/env python3
"""Evaluate standalone sentiment/event features before model integration."""
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate daily sentiment features with IC/deciles.")
    parser.add_argument("--features-path", default=None, help="sentiment_features.parquet path.")
    parser.add_argument("--features", default="", help="Comma-separated feature columns. Defaults to all numeric features.")
    parser.add_argument("--target", default="", help="Target return column. Defaults to training label horizon.")
    parser.add_argument("--condition-col", default="", help="Optional condition flag column for conditional IC.")
    parser.add_argument("--min-per-date", type=int, default=30, help="Minimum non-null rows per date.")
    parser.add_argument("--output", default=None, help="Output JSON report path.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _target_col(root: Path, args: argparse.Namespace) -> str:
    if args.target:
        return args.target
    report_path = root / "training_set_report.json"
    label_horizon = 20
    if report_path.exists():
        label_horizon = int(json.loads(report_path.read_text()).get("label_horizon_days", 20))
    return f"forward_{label_horizon}d_return"


def _json_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _ic_stats(df: pd.DataFrame, feature: str, target: str, min_per_date: int) -> dict | None:
    rows = []
    for date, group in df.dropna(subset=[feature, target]).groupby("trade_date", sort=False):
        if len(group) < min_per_date or group[feature].nunique() <= 1 or group[target].nunique() <= 1:
            continue
        ic = group[feature].corr(group[target], method="spearman")
        if pd.notna(ic):
            rows.append({"date": str(date), "ic": float(ic), "n": int(len(group))})
    if not rows:
        return None
    values = pd.Series([row["ic"] for row in rows])
    std = values.std()
    return {
        "mean": _json_float(values.mean()),
        "std": _json_float(std),
        "icir": float(values.mean() / std) if pd.notna(std) and std else None,
        "median": _json_float(values.median()),
        "positive_rate": float((values > 0).mean()),
        "count": int(len(values)),
        "worst5": sorted(rows, key=lambda row: row["ic"])[:5],
        "best5": sorted(rows, key=lambda row: row["ic"], reverse=True)[:5],
    }


def _decile_returns(df: pd.DataFrame, feature: str, target: str, min_per_date: int) -> dict | None:
    rows = []
    universe = []
    for _date, group in df.dropna(subset=[feature, target]).groupby("trade_date", sort=False):
        if len(group) < min_per_date or group[feature].nunique() <= 1:
            continue
        ranks = group[feature].rank(method="first", pct=True)
        deciles = (ranks * 10).clip(upper=9).astype(int)
        means = group.assign(decile=deciles).groupby("decile")[target].mean()
        rows.append({decile: means.get(decile) for decile in range(10)})
        universe.append(group[target].mean())
    if not rows:
        return None
    result = pd.DataFrame(rows)
    universe_return = float(pd.Series(universe).mean())
    low = result[0].mean() if 0 in result else None
    high = result[9].mean() if 9 in result else None
    return {
        "note": "decile 0 is the lowest feature value; decile 9 is the highest feature value",
        "universe_return": universe_return,
        "deciles": [
            {
                "decile": int(decile),
                "mean_return": _json_float(result[decile].mean()) if decile in result else None,
                "excess_return": (
                    _json_float(result[decile].mean() - universe_return)
                    if decile in result else None
                ),
            }
            for decile in range(10)
        ],
        "spread_9_minus_0": _json_float(high - low) if low is not None and high is not None else None,
    }


def _condition_col(data: pd.DataFrame, feature: str, requested: str) -> str | None:
    if requested:
        return requested if requested in data.columns else None
    if feature.startswith("news_") and "has_news_7d" in data.columns:
        return "has_news_7d"
    if feature.startswith("ann_") and "has_ann_7d" in data.columns:
        return "has_ann_7d"
    return None


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
        "q25": _json_float(values.quantile(0.25)),
        "q75": _json_float(values.quantile(0.75)),
        "q90": _json_float(values.quantile(0.90)),
    }


def _event_study(data: pd.DataFrame, target: str) -> dict:
    out = {}
    universe_by_date = data.groupby("trade_date")[target].mean()
    ann_cols = [
        col for col in data.columns
        if col.startswith("ann_")
        and col.endswith("_count_20d")
        and col not in {"ann_count_20d"}
    ]
    for col in ann_cols:
        subset = data[data[col].fillna(0) > 0].copy()
        if subset.empty:
            continue
        subset["date_universe_return"] = subset["trade_date"].map(universe_by_date)
        subset["excess_return"] = subset[target] - subset["date_universe_return"]
        out[col.removeprefix("ann_").removesuffix("_count_20d")] = {
            "rows": int(len(subset)),
            "dates": int(subset["trade_date"].nunique()),
            "codes": int(subset["code"].nunique()),
            "return": _distribution(subset[target]),
            "date_excess_return": _distribution(subset["excess_return"]),
        }
    return out


def main() -> None:
    args = _parse_args()
    root = _root()
    features_path = Path(args.features_path).expanduser() if args.features_path else root / "sentiment_features.parquet"
    if not features_path.exists():
        raise RuntimeError(f"情绪特征文件不存在: {features_path}")
    target = _target_col(root, args)

    train = pd.read_parquet(root / "training_set.parquet", columns=["trade_date", "code", target])
    features = pd.read_parquet(features_path)
    data = train.merge(features, on=["trade_date", "code"], how="inner")
    if args.features:
        feature_cols = [col.strip() for col in args.features.split(",") if col.strip()]
    else:
        ignored = {"trade_date", "code", target}
        feature_cols = [
            col for col in data.select_dtypes(include="number").columns
            if col not in ignored
        ]
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "features_path": str(features_path),
        "target": target,
        "rows": int(len(data)),
        "features": {},
    }
    for feature in feature_cols:
        if feature not in data.columns:
            continue
        valid = data[[feature, target]].dropna()
        condition_col = _condition_col(data, feature, args.condition_col)
        conditional = None
        if condition_col:
            conditional_data = data[data[condition_col].fillna(0) > 0]
            conditional_valid = conditional_data[[feature, target]].dropna()
            conditional = {
                "condition_col": condition_col,
                "rows": int(len(conditional_data)),
                "coverage": float(len(conditional_valid) / len(data)) if len(data) else 0.0,
                "ic": _ic_stats(conditional_data, feature, target, args.min_per_date),
                "decile_returns": _decile_returns(conditional_data, feature, target, args.min_per_date),
            }
        report["features"][feature] = {
            "coverage": float(len(valid) / len(data)) if len(data) else 0.0,
            "non_null_rows": int(len(valid)),
            "ic": _ic_stats(data, feature, target, args.min_per_date),
            "decile_returns": _decile_returns(data, feature, target, args.min_per_date),
            "conditional": conditional,
        }
    report["announcement_event_study"] = _event_study(data, target)

    output = Path(args.output).expanduser() if args.output else root / "sentiment_feature_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"sentiment feature report saved: {output}")
    for feature, item in report["features"].items():
        ic = item["ic"] or {}
        decile = item["decile_returns"] or {}
        print(
            f"{feature}: coverage={item['coverage']:.2%}, "
            f"IC={ic.get('mean')}, ICIR={ic.get('icir')}, "
            f"spread_9_minus_0={decile.get('spread_9_minus_0')}"
        )
        conditional = item.get("conditional") or {}
        if conditional:
            cond_ic = conditional.get("ic") or {}
            print(
                f"  conditional[{conditional.get('condition_col')}]: "
                f"rows={conditional.get('rows')}, "
                f"IC={cond_ic.get('mean')}, ICIR={cond_ic.get('icir')}"
            )


if __name__ == "__main__":
    main()
