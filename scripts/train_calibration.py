#!/usr/bin/env python3
"""Resolve historical decisions and train BUY/SELL confidence calibration."""
from __future__ import annotations

import sys
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from analysis.calibration import make_model_row, resolve_decisions
from utils.storage import Storage

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def train_action(storage: Storage, action: str, min_samples: int):
    samples = storage.get_calibration_samples(action)
    sample_size = len(samples)
    if sample_size < min_samples:
        storage.insert_calibration_model(make_model_row(
            action, sample_size, 1.0, 0.0, None,
            f"样本不足（{sample_size}/{min_samples}），pass-through",
        ))
        print(f"{action}: samples {sample_size}/{min_samples}, pass-through")
        return

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception as e:
        raise RuntimeError("训练校准模型需要 scikit-learn，请安装 requirements-train.txt") from e

    x = np.array([
        [float(row["raw_confidence"] if row["raw_confidence"] is not None else row["confidence"])]
        for row in samples
    ])
    y = np.array([int(row["success"]) for row in samples])
    if len(set(y.tolist())) < 2:
        storage.insert_calibration_model(make_model_row(
            action, sample_size, 1.0, 0.0, None,
            "样本只有单一类别，pass-through",
        ))
        print(f"{action}: single-class samples, pass-through")
        return

    split = max(1, int(sample_size * 0.8))
    if split >= sample_size:
        split = sample_size - 1
    clf = LogisticRegression()
    clf.fit(x[:split], y[:split])
    pred = clf.predict_proba(x[split:])[:, 1]
    auc = float(roc_auc_score(y[split:], pred)) if len(set(y[split:].tolist())) > 1 else None
    coef = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    storage.insert_calibration_model(make_model_row(
        action, sample_size, coef, intercept, auc,
        "trained logistic calibration",
    ))
    logger.info(f"{action} calibration trained: samples={sample_size}, auc={auc}")
    print(f"{action}: samples={sample_size}, coef={coef:.4f}, intercept={intercept:.4f}, auc={auc}")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def parse_args():
    parser = argparse.ArgumentParser(description="Resolve decisions and train confidence calibration.")
    parser.add_argument(
        "--db",
        default=os.getenv("STOCKWATCH_DB_PATH", "~/.stockwatch/db.sqlite"),
        help="SQLite DB path, default: ~/.stockwatch/db.sqlite",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=_env_int("CALIBRATION_LOOKBACK_DAYS", 5),
        help="Forward trading days used to resolve a decision, default: 5",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=_env_int("CALIBRATION_MIN_SAMPLES", 50),
        help="Minimum resolved samples per action before fitting, default: 50",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    storage = Storage(Path(args.db).expanduser())
    resolved = resolve_decisions(storage, args.lookback_days)
    print(f"resolved decisions: {resolved}")
    train_action(storage, "BUY", args.min_samples)
    train_action(storage, "SELL", args.min_samples)


if __name__ == "__main__":
    main()
