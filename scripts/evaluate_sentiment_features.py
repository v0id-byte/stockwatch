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
        "mean": float(values.mean()),
        "std": float(std) if pd.notna(std) else None,
        "icir": float(values.mean() / std) if pd.notna(std) and std else None,
        "median": float(values.median()),
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
    return {
        "note": "decile 0 is the lowest feature value; decile 9 is the highest feature value",
        "universe_return": universe_return,
        "deciles": [
            {
                "decile": int(decile),
                "mean_return": float(result[decile].mean()),
                "excess_return": float(result[decile].mean() - universe_return),
            }
            for decile in range(10)
        ],
        "spread_9_minus_0": float(result[9].mean() - result[0].mean()),
    }


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
        report["features"][feature] = {
            "coverage": float(len(valid) / len(data)) if len(data) else 0.0,
            "non_null_rows": int(len(valid)),
            "ic": _ic_stats(data, feature, target, args.min_per_date),
            "decile_returns": _decile_returns(data, feature, target, args.min_per_date),
        }

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


if __name__ == "__main__":
    main()
