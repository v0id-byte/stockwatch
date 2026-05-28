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


def main():
    import lightgbm as lgb
    import pandas as pd

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")
    df = pd.read_parquet(data_path).sort_values(["trade_date", "code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])

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
        "num_leaves": 63,
        "max_depth": 7,
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

    pred = model.predict(test[ALPHA158_FEATURES], num_iteration=model.best_iteration)
    ndcg5 = _mean_ndcg(test, pred, 5)
    ndcg10 = _mean_ndcg(test, pred, 10)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "lgbm.txt"
    model.save_model(str(model_path))
    meta = {
        "trained_at": datetime.now().isoformat(),
        "features": ALPHA158_FEATURES,
        "ndcg5": ndcg5,
        "ndcg10": ndcg10,
        "best_iteration": model.best_iteration,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    with open(out_dir / "lgbm_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"model saved: {model_path}")
    print(f"test ndcg@5={ndcg5}, ndcg@10={ndcg10}")


if __name__ == "__main__":
    main()
