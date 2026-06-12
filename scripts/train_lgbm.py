#!/usr/bin/env python3
"""Train LightGBM LambdaRank model from the offline training set."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ALPHA158_FEATURES

STABLE_FEATURE_PREFIXES = (
    "ILLIQ", "BETA", "RSV", "DD", "RET", "ROC", "RELV", "STD",
    "RSQR", "CORR", "VMA", "WVMA", "TURN", "VOLZ", "MOM", "SHARPE",
)


def _group_sizes(df):
    return df.groupby("trade_date", sort=False).size().tolist()


def _split_by_date_with_purge(df, label_horizon: int):
    import pandas as pd

    max_date = df["trade_date"].max()
    raw_test_start = max_date - pd.DateOffset(months=6)
    raw_val_start = raw_test_start - pd.DateOffset(months=6)
    dates = pd.Index(sorted(df["trade_date"].drop_duplicates()))
    val_idx = int(dates.searchsorted(raw_val_start, side="left"))
    test_idx = int(dates.searchsorted(raw_test_start, side="left"))
    if val_idx >= len(dates) or test_idx >= len(dates):
        raise RuntimeError("训练/验证/测试切分为空，请检查历史数据跨度")

    purge_days = max(0, label_horizon)
    train_end_idx = max(0, val_idx - purge_days)
    val_end_idx = max(val_idx, test_idx - purge_days)
    train_end = dates[train_end_idx]
    val_start = dates[val_idx]
    val_end = dates[val_end_idx]
    test_start = dates[test_idx]

    train = df[df["trade_date"] < train_end]
    val = df[(df["trade_date"] >= val_start) & (df["trade_date"] < val_end)]
    test = df[df["trade_date"] >= test_start]
    split = {
        "raw_val_start": str(raw_val_start.date()),
        "raw_test_start": str(raw_test_start.date()),
        "val_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "train_end_exclusive": str(train_end.date()),
        "val_end_exclusive": str(val_end.date()),
        "purge_trading_days": purge_days,
    }
    return train, val, test, split


def _feature_names() -> tuple[str, list[str]]:
    feature_set = os.getenv("STOCKWATCH_LGBM_FEATURE_SET", "stable").strip().lower()
    if feature_set == "all":
        return "all", ALPHA158_FEATURES
    if feature_set != "stable":
        print(f"unknown STOCKWATCH_LGBM_FEATURE_SET={feature_set}, fallback to stable")
    return "stable", [
        name for name in ALPHA158_FEATURES
        if name.startswith(STABLE_FEATURE_PREFIXES)
    ]


def _mean_ndcg(df, pred, k: int) -> float | None:
    try:
        from sklearn.metrics import ndcg_score
        import numpy as np
    except Exception:
        return None
    scores = []
    offset = 0
    for _date, group in df.groupby("trade_date", sort=False):
        n = len(group)
        if n <= 1:
            offset += n
            continue
        y_true = group["label"].to_numpy().reshape(1, -1)
        y_score = np.asarray(pred[offset:offset+n]).reshape(1, -1)
        scores.append(float(ndcg_score(y_true, y_score, k=min(k, n))))
        offset += n
    return sum(scores) / len(scores) if scores else None


def _spearman_ic_stats(df, pred, target_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", target_col]].copy()
    tmp["pred"] = pred
    rows = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) <= 2 or group["pred"].nunique() <= 1 or group[target_col].nunique() <= 1:
            continue
        value = group["pred"].corr(group[target_col], method="spearman")
        if pd.notna(value):
            rows.append({"date": str(pd.Timestamp(_date).date()), "ic": float(value)})
    if not rows:
        return None
    values = pd.Series([row["ic"] for row in rows]).dropna()
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
        "by_date": rows,
    }


def _mean_topk_return(df, pred, k: int, return_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", return_col]].copy()
    tmp["pred"] = pred
    rows = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) < k:
            continue
        top = group.sort_values("pred", ascending=False).head(k)
        rows.append({
            "top_return": top[return_col].mean(),
            "universe_return": group[return_col].mean(),
        })
    if not rows:
        return None
    result = pd.DataFrame(rows)
    excess = result["top_return"] - result["universe_return"]
    return {
        "top_return": float(result["top_return"].mean()),
        "universe_return": float(result["universe_return"].mean()),
        "excess_return": float(excess.mean()),
        "excess_std": float(excess.std()),
        "excess_positive_rate": float((excess > 0).mean()),
        "period_count": int(len(result)),
    }


def _decile_returns(df, pred, return_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", return_col]].copy()
    tmp["pred"] = pred
    rows = []
    universe = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) < 10 or group["pred"].nunique() <= 1:
            continue
        ranks = group["pred"].rank(method="first", pct=True)
        deciles = (ranks * 10).clip(upper=9).astype(int)
        means = group.assign(decile=deciles).groupby("decile")[return_col].mean()
        rows.append({decile: means.get(decile) for decile in range(10)})
        universe.append(group[return_col].mean())
    if not rows:
        return None
    result = pd.DataFrame(rows)
    universe_return = float(pd.Series(universe).mean())
    deciles = [
        {
            "decile": int(decile),
            "mean_return": float(result[decile].mean()),
            "excess_return": float(result[decile].mean() - universe_return),
        }
        for decile in range(10)
    ]
    return {
        "note": "decile 0 is the lowest model score; decile 9 is the highest model score",
        "universe_return": universe_return,
        "deciles": deciles,
        "spread_9_minus_0": float(result[9].mean() - result[0].mean()),
    }


def _evaluate_split_core(df, pred, return_col: str) -> dict:
    label_ic_stats = _spearman_ic_stats(df, pred, "label_score")
    return_ic_stats = _spearman_ic_stats(df, pred, return_col)
    return {
        "ndcg5": _mean_ndcg(df, pred, 5),
        "ndcg10": _mean_ndcg(df, pred, 10),
        "spearman_ic": label_ic_stats["mean"] if label_ic_stats else None,
        "spearman_ic_target": "label_score",
        "spearman_ic_stats": label_ic_stats,
        "return_spearman_ic": return_ic_stats["mean"] if return_ic_stats else None,
        "return_spearman_ic_stats": return_ic_stats,
        "top5": _mean_topk_return(df, pred, 5, return_col),
        "top10": _mean_topk_return(df, pred, 10, return_col),
        "top30": _mean_topk_return(df, pred, 30, return_col),
        "top50": _mean_topk_return(df, pred, 50, return_col),
        "decile_returns": _decile_returns(df, pred, return_col),
    }


def _non_overlapping_sample(df, pred, step: int):
    import numpy as np

    if step <= 1:
        return df, pred
    dates = list(df["trade_date"].drop_duplicates())
    keep_dates = set(dates[::step])
    mask = df["trade_date"].isin(keep_dates).to_numpy()
    return df[mask], np.asarray(pred)[mask]


def _evaluate_split(df, pred, return_col: str, label_horizon: int) -> dict:
    metrics = _evaluate_split_core(df, pred, return_col)
    sampled_df, sampled_pred = _non_overlapping_sample(df, pred, label_horizon)
    metrics["non_overlapping"] = (
        _evaluate_split_core(sampled_df, sampled_pred, return_col)
        if len(sampled_df) else None
    )
    return metrics


def _summary(metrics: dict) -> dict:
    return {
        "ndcg5": metrics["ndcg5"],
        "ndcg10": metrics["ndcg10"],
        "spearman_ic": metrics["spearman_ic"],
        "return_spearman_ic": metrics["return_spearman_ic"],
        "top5_excess": metrics["top5"]["excess_return"] if metrics["top5"] else None,
        "top30_excess": metrics["top30"]["excess_return"] if metrics["top30"] else None,
        "decile_spread_9_minus_0": (
            metrics["decile_returns"]["spread_9_minus_0"]
            if metrics["decile_returns"] else None
        ),
    }


def main():
    import lightgbm as lgb
    import pandas as pd

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")
    df = pd.read_parquet(data_path).sort_values(["trade_date", "code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    report_path = root / "training_set_report.json"
    training_report = {}
    if report_path.exists():
        training_report = json.loads(report_path.read_text())
    label_horizon = int(training_report.get("label_horizon_days", 20))
    return_col = f"forward_{label_horizon}d_return"
    if return_col not in df.columns:
        return_col = "forward_5d_return"

    train, val, test, split = _split_by_date_with_purge(df, label_horizon)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("训练/验证/测试切分为空，请检查历史数据跨度")
    feature_set, feature_names = _feature_names()
    if not feature_names:
        raise RuntimeError("训练特征为空，请检查 STOCKWATCH_LGBM_FEATURE_SET")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    train_set = lgb.Dataset(
        train[feature_names],
        label=train["label"],
        group=_group_sizes(train),
        feature_name=feature_names,
    )
    val_set = lgb.Dataset(
        val[feature_names],
        label=val["label"],
        group=_group_sizes(val),
        feature_name=feature_names,
        reference=train_set,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=300,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(25), lgb.log_evaluation(20)],
    )

    val_pred = model.predict(val[feature_names], num_iteration=model.best_iteration)
    test_pred = model.predict(test[feature_names], num_iteration=model.best_iteration)
    validation_metrics = _evaluate_split(val, val_pred, return_col, label_horizon)
    test_metrics = _evaluate_split(test, test_pred, return_col, label_horizon)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "lgbm.txt"
    model.save_model(str(model_path))
    importance = sorted(
        zip(feature_names, model.feature_importance(importance_type="gain")),
        key=lambda item: item[1],
        reverse=True,
    )
    meta = {
        "trained_at": datetime.now().isoformat(),
        "features": feature_names,
        "feature_set": feature_set,
        "feature_count": len(feature_names),
        "available_feature_count": len(ALPHA158_FEATURES),
        "label_horizon_days": label_horizon,
        "return_col": return_col,
        "ndcg5": test_metrics["ndcg5"],
        "ndcg10": test_metrics["ndcg10"],
        "spearman_ic": test_metrics["spearman_ic"],
        "top5": test_metrics["top5"],
        "top10": test_metrics["top10"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "best_iteration": model.best_iteration,
        "split": split,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_codes": int(train["code"].nunique()),
        "val_codes": int(val["code"].nunique()),
        "test_codes": int(test["code"].nunique()),
        "date_start": str(df["trade_date"].min().date()),
        "date_end": str(df["trade_date"].max().date()),
        "training_report": training_report,
        "feature_importance_top30": [
            {"feature": name, "gain": float(gain)}
            for name, gain in importance[:30]
        ],
    }
    with open(out_dir / "lgbm_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"model saved: {model_path}")
    print(f"validation={_summary(validation_metrics)}")
    print(f"test={_summary(test_metrics)}")


if __name__ == "__main__":
    main()
