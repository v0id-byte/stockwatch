"""Confidence calibration utilities."""
from __future__ import annotations

import math
from datetime import datetime

from loguru import logger

from config import get_config
from utils.storage import Storage


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
