"""LightGBM ranker inference wrapper."""
from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class LgbmRanker:
    def __init__(self, model_path: Path):
        self.model = None
        self.meta = {"features": []}
        self.model_path = model_path
        if not model_path.exists():
            logger.info("LightGBM 模型未加载，跳过")
            return
        try:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(model_path))
            meta_path = model_path.parent / "lgbm_meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    self.meta = json.load(f)
            else:
                self.meta = {"features": self.model.feature_name()}
            logger.info(f"LightGBM 模型已加载: {model_path}")
        except Exception as e:
            logger.warning(f"LightGBM 模型加载失败: {e}")
            self.model = None

    def predict(self, factors_dict: dict) -> float | None:
        if self.model is None:
            return None
        features = self.meta.get("features", [])
        if not features:
            return None
        x = [[float(factors_dict.get(name, 0.0) or 0.0) for name in features]]
        return float(self.model.predict(x)[0])


def format_lgbm_context(scores_by_code: dict[str, float | None]) -> dict[str, str]:
    valid = {code: score for code, score in scores_by_code.items() if score is not None}
    if not valid:
        return {code: "LightGBM 排序模型预测: 未加载，跳过" for code in scores_by_code}
    ordered = sorted(valid.items(), key=lambda item: item[1])
    denom = max(1, len(ordered) - 1)
    contexts = {}
    for rank, (code, _score) in enumerate(ordered):
        percentile = rank / denom
        display_score = percentile * 9
        top_pct = 1 - percentile
        contexts[code] = (
            f"LightGBM 排序模型预测: {display_score:.1f}/9 "
            f"（Top {max(1, int(top_pct * 100))}%）"
        )
    for code, score in scores_by_code.items():
        if score is None:
            contexts[code] = "LightGBM 排序模型预测: 未加载，跳过"
    return contexts
