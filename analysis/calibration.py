"""Confidence calibration utilities."""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta

from loguru import logger

from config import get_config
from utils.storage import Storage


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


class ConfidenceCalibrator:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.cfg = get_config()
        self.models = {
            "BUY": storage.get_latest_calibration_model("BUY"),
            "SELL": storage.get_latest_calibration_model("SELL"),
        }
        self._logged_shortage = set()

    def calibrate(self, action: str, raw_confidence: float) -> float:
        action = action.upper()
        if action not in {"BUY", "SELL"}:
            return raw_confidence
        model = self.models.get(action)
        if not model or (model.get("sample_size") or 0) < self.cfg.calibration_min_samples:
            samples = int(model.get("sample_size") or 0) if model else 0
            if action not in self._logged_shortage:
                logger.info(
                    f"置信度校准：{action} 样本不足（{samples}/{self.cfg.calibration_min_samples}），使用原始置信度"
                )
                self._logged_shortage.add(action)
            return raw_confidence
        coef = float(model.get("coef") or 0.0)
        intercept = float(model.get("intercept") or 0.0)
        return max(0.0, min(1.0, sigmoid(coef * raw_confidence + intercept)))


def _run_date(run_ts: str) -> datetime:
    return datetime.fromisoformat(str(run_ts).replace("Z", "+00:00")).replace(tzinfo=None)


def _resolve_one(storage: Storage, decision: dict, lookback_days: int) -> int | None:
    """根据决策后 lookback_days 个交易日的走势判定 BUY/SELL 是否达标。
    返回 1=成功 / 0=失败 / None=数据不足或多空模糊，无法判定。"""
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
    use_llm_targets = _env_bool("CALIBRATION_USE_LLM_TARGETS", False)

    if action == "BUY":
        if use_llm_targets and target > 0:
            success = max_high >= target or end_close > current * 1.03
        else:
            success = end_close > current * 1.03
        if use_llm_targets and stop > 0:
            fail = min_low <= stop or end_close < current * 0.98
        else:
            fail = end_close < current * 0.98
    elif action == "SELL":
        if use_llm_targets and target > 0:
            success = min_low <= target or end_close < current * 0.98
        else:
            success = end_close < current * 0.98
        if use_llm_targets and stop > 0:
            fail = max_high >= stop or end_close > current * 1.03
        else:
            fail = end_close > current * 1.03
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
    """结算所有窗口已到期、尚未结算的 BUY/SELL 决策，写回 decisions.success。
    幂等：已结算的不会重复处理；窗口未到期的跳过。供 daemon 与 train_calibration 复用。"""
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


def make_model_row(action: str, sample_size: int, coef: float, intercept: float,
                   auc: float | None, notes: str = "") -> dict:
    return {
        "action": action,
        "trained_at": datetime.now().isoformat(),
        "sample_size": sample_size,
        "coef": coef,
        "intercept": intercept,
        "auc": auc,
        "notes": notes,
    }
