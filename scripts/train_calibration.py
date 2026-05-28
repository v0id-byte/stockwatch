#!/usr/bin/env python3
"""Resolve historical decisions and train BUY/SELL confidence calibration."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from loguru import logger

from analysis.calibration import make_model_row
from config import get_config
from utils.storage import Storage


def _run_date(run_ts: str) -> datetime:
    return datetime.fromisoformat(run_ts.replace("Z", "+00:00")).replace(tzinfo=None)


def _resolve_one(storage: Storage, decision: dict, lookback_days: int) -> int | None:
    run_dt = _run_date(decision["run_ts"])
    end_dt = run_dt + timedelta(days=lookback_days * 3 + 7)
    rows = storage.get_kline(
        decision["code"],
        run_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
    )
    if len(rows) < lookback_days + 1:
        return None

    current = float(rows[0]["close"] or 0)
    future = rows[1:lookback_days + 1]
    if current <= 0 or not future:
        return None

    max_high = max(float(row["high"] or 0) for row in future)
    min_low = min(float(row["low"] or 0) for row in future)
    end_close = float(future[-1]["close"] or 0)
    target = float(decision.get("target_price") or 0)
    stop = float(decision.get("stop_loss") or 0)
    action = decision["action"]

    if action == "BUY":
        success = (target > 0 and max_high >= target) or end_close > current * 1.03
        fail = (stop > 0 and min_low <= stop) or end_close < current * 0.98
    elif action == "SELL":
        success = (target > 0 and min_low <= target) or end_close < current * 0.98
        fail = (stop > 0 and max_high >= stop) or end_close > current * 1.03
    else:
        return None

    if success and fail:
        return None
    if success:
        return 1
    if fail:
        return 0
    return None


def resolve_decisions(storage: Storage, lookback_days: int) -> int:
    resolved = 0
    for decision in storage.get_unresolved_action_decisions():
        run_dt = _run_date(decision["run_ts"])
        if datetime.now() < run_dt + timedelta(days=lookback_days):
            continue
        success = _resolve_one(storage, decision, lookback_days)
        if success is None:
            continue
        storage.mark_decision_resolved(decision["id"], success)
        resolved += 1
    return resolved


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


def main():
    cfg = get_config()
    storage = Storage()
    resolved = resolve_decisions(storage, cfg.calibration_lookback_days)
    print(f"resolved decisions: {resolved}")
    train_action(storage, "BUY", cfg.calibration_min_samples)
    train_action(storage, "SELL", cfg.calibration_min_samples)


if __name__ == "__main__":
    main()
