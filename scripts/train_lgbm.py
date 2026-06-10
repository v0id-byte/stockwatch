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


def _group_sizes(df):
    return df.groupby("trade_date", sort=False).size().tolist()


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


def _mean_spearman_ic(df, pred) -> float | None:
    import pandas as pd

    tmp = df[["trade_date", "label_score"]].copy()
    tmp["pred"] = pred
    values = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) <= 2 or group["pred"].nunique() <= 1 or group["label_score"].nunique() <= 1:
            continue
        values.append(group["pred"].corr(group["label_score"], method="spearman"))
    values = pd.Series(values).dropna()
    return float(values.mean()) if len(values) else None


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
    return {
        "top_return": float(result["top_return"].mean()),
        "universe_return": float(result["universe_return"].mean()),
        "excess_return": float((result["top_return"] - result["universe_return"]).mean()),
    }


def _evaluate_split(df, pred, return_col: str) -> dict:
    return {
        "ndcg5": _mean_ndcg(df, pred, 5),
        "ndcg10": _mean_ndcg(df, pred, 10),
        "spearman_ic": _mean_spearman_ic(df, pred),
        "top5": _mean_topk_return(df, pred, 5, return_col),
        "top10": _mean_topk_return(df, pred, 10, return_col),
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

    max_date = df["trade_date"].max()
    test_start = max_date - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=6)
    train = df[df["trade_date"] < val_start]
    val = df[(df["trade_date"] >= val_start) & (df["trade_date"] < test_start)]
    test = df[df["trade_date"] >= test_start]
    if train.empty or val.empty or test.empty:
        raise RuntimeError("训练/验证/测试切分为空，请检查历史数据跨度")

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }
    train_set = lgb.Dataset(
        train[ALPHA158_FEATURES],
        label=train["label"],
        group=_group_sizes(train),
        feature_name=ALPHA158_FEATURES,
    )
    val_set = lgb.Dataset(
        val[ALPHA158_FEATURES],
        label=val["label"],
        group=_group_sizes(val),
        feature_name=ALPHA158_FEATURES,
        reference=train_set,
    )
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)],
    )

    val_pred = model.predict(val[ALPHA158_FEATURES], num_iteration=model.best_iteration)
    test_pred = model.predict(test[ALPHA158_FEATURES], num_iteration=model.best_iteration)
    validation_metrics = _evaluate_split(val, val_pred, return_col)
    test_metrics = _evaluate_split(test, test_pred, return_col)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "lgbm.txt"
    model.save_model(str(model_path))
    importance = sorted(
        zip(ALPHA158_FEATURES, model.feature_importance(importance_type="gain")),
        key=lambda item: item[1],
        reverse=True,
    )
    meta = {
        "trained_at": datetime.now().isoformat(),
        "features": ALPHA158_FEATURES,
        "feature_count": len(ALPHA158_FEATURES),
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
    print(f"validation={validation_metrics}")
    print(f"test={test_metrics}")


if __name__ == "__main__":
    main()
