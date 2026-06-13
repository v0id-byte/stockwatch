"""LightGBM ranker inference wrapper.

References:
    LightGBM: A Highly Efficient Gradient Boosting Decision Tree
        Ke et al. (2017, NeurIPS) — https://papers.nips.cc/paper/6907-lightgbm
    Learning to Rank with LambdaMART / LambdaRank
        Burges (2010) — https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
    Cross-sectional stock ranking framing follows Qlib LightGBM examples.
"""
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

    def _needs_rank_normalize(self) -> bool:
        return self.meta.get("feature_normalization") == "cross_sectional_rank_pct_centered"

    def predict_batch(self, factors_by_code: dict[str, dict]) -> dict[str, float | None]:
        """Score a cross-section of stocks together.

        Models trained with cross_sectional_rank normalization MUST be scored on
        rank-normalized features (matching training); doing this per stock with raw
        features — as the old code did — fed the model an out-of-distribution input.
        With a single stock the ranking is undefined, so every feature collapses to
        the neutral 0.0 and the caller's formatter notes it cannot be ranked.
        """
        if self.model is None or not factors_by_code:
            return {code: None for code in factors_by_code}
        features = self.meta.get("features", [])
        if not features:
            return {code: None for code in factors_by_code}
        codes = list(factors_by_code)
        rows = [[float(factors_by_code[code].get(name, 0.0) or 0.0) for name in features] for code in codes]
        if self._needs_rank_normalize():
            import pandas as pd
            frame = pd.DataFrame(rows, columns=features)
            if len(frame) > 1:
                frame = frame.rank(pct=True) - 0.5
            else:
                frame[:] = 0.0
            rows = frame.to_numpy().tolist()
        preds = self.model.predict(rows)
        return {code: float(value) for code, value in zip(codes, preds)}

    def predict(self, factors_dict: dict) -> float | None:
        return self.predict_batch({"_": factors_dict}).get("_")


def format_lgbm_context(scores_by_code: dict[str, float | None]) -> dict[str, str]:
    valid = {code: score for code, score in scores_by_code.items() if score is not None}
    if not valid:
        return {code: "LightGBM 排序模型预测: 未加载，跳过" for code in scores_by_code}
    if len(valid) == 1:
        code, score = next(iter(valid.items()))
        contexts = {
            code: f"LightGBM 排序模型预测: 原始分 {score:.4f}（单票无法横向排名）"
        }
        for item_code, item_score in scores_by_code.items():
            if item_score is None:
                contexts[item_code] = "LightGBM 排序模型预测: 未加载，跳过"
        return contexts

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
