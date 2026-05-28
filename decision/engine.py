"""最终决策引擎：技术分 + 情绪分 → LLM决策 + 程序级兜底"""
import json
import math
from loguru import logger

from config import get_config
from utils.llm import get_llm_client
from analysis.calibration import ConfidenceCalibrator


_HARD_RULES_PROMPT = """【硬规则】以下标的必须输出 HOLD，无论其他信号多强：
1. 当日已涨停（涨幅≥9.5%）→ HOLD，禁止 BUY
2. ST/*ST/退市风险股 → HOLD
3. 上市不足60个交易日 → HOLD
4. BUY 信号必须同时输出 stop_loss（若 LLM 未给出，程序自动填当前价×0.93）
5. BUY/SELL 信号必须输出 trade_price、support_price、resistance_price、position_basis
6. 置信度 < 0.6 → 仅记录，不推送"""


_ENGINE_PROMPT = """你是一个A股智能决策助手。请对以下股票给出投资建议。

【股票信息】
代码: {code}
名称: {name}
当前价: {price}元
技术面分数: {tech_score:.3f}（-1到+1，越高越强势）
情绪面分数: {sentiment:.3f}（-1到+1，越高越乐观）
大盘上下文：
  - 北向资金: {north_money}
  - 上证涨跌: {sht000001_pct:.2f}%
  - 板块氛围: {sector_sentiment}

【价格位置参考】
{price_context}

【输出格式】严格JSON，无markdown围栏，无前言：
{{
  "code": "{code}",
  "name": "{name}",
  "action": "BUY|SELL|HOLD",
  "confidence": 0.0到1.0,
  "trade_price": 0.0,
  "target_price": 0.0,
  "stop_loss": 0.0,
  "support_price": 0.0,
  "resistance_price": 0.0,
  "position_basis": "为什么这个买/卖位置合理，30字以内",
  "reasons": ["原因1", "原因2"],
  "risks": ["风险1", "风险2"],
  "one_liner": "一句话给非技术用户看，30字以内"
}}

{hard_rules}

不要输出 JSON 以外的内容。"""


_ENGINE_PROMPT_V2 = """你是一个A股智能决策助手。请对以下股票给出投资建议。

【股票信息】
代码: {code} 名称: {name} 当前价: {price}元

【技术信号】
- 技术面综合分: {tech_score:.3f}
{alpha_summary}
{lgbm_context}

【基本面信号】
- 情绪面: {sentiment:.3f}

【市场结构】
- {regime_context}
- 上证当日: {sht000001_pct:.2f}%
- 北向资金: {north_money}
- {sector_context}

【价格位置参考】
{price_context}

【输出格式】严格JSON，无markdown围栏，无前言：
{{
  "code": "{code}",
  "name": "{name}",
  "action": "BUY|SELL|HOLD",
  "confidence": 0.0到1.0,
  "trade_price": 0.0,
  "target_price": 0.0,
  "stop_loss": 0.0,
  "support_price": 0.0,
  "resistance_price": 0.0,
  "position_basis": "为什么这个买/卖位置合理，30字以内",
  "reasons": ["原因1", "原因2"],
  "risks": ["风险1", "风险2"],
  "one_liner": "一句话给非技术用户看，30字以内"
}}

{hard_rules}

不要输出 JSON 以外的内容。"""


def _safe_float(value) -> float:
    try:
        num = float(value)
        return num if math.isfinite(num) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _round_price(value) -> float:
    value = _safe_float(value)
    return round(value, 2) if value > 0 else 0


def _derive_price_levels(kline: list[dict], current_price: float) -> dict:
    recent = kline[-20:] if kline else []
    lows = [_safe_float(row.get("low")) for row in recent]
    highs = [_safe_float(row.get("high")) for row in recent]
    lows = [price for price in lows if price > 0]
    highs = [price for price in highs if price > 0]
    if not lows or not highs:
        return {"support_price": 0, "resistance_price": 0, "low_20d": 0, "high_20d": 0}

    support_candidates = [price for price in lows if current_price <= 0 or price <= current_price]
    resistance_candidates = [price for price in highs if current_price <= 0 or price >= current_price]
    support = max(support_candidates) if support_candidates else min(lows)
    resistance = min(resistance_candidates) if resistance_candidates else max(highs)
    return {
        "support_price": _round_price(support),
        "resistance_price": _round_price(resistance),
        "low_20d": _round_price(min(lows)),
        "high_20d": _round_price(max(highs)),
    }


def _format_price_context(levels: dict) -> str:
    if not levels.get("support_price") and not levels.get("resistance_price"):
        return "- 近20日支撑/压力: 数据不足"
    return (
        f"- 近20日参考支撑: {levels.get('support_price', 0)}元\n"
        f"- 近20日参考压力: {levels.get('resistance_price', 0)}元\n"
        f"- 近20日区间: {levels.get('low_20d', 0)}元 ~ {levels.get('high_20d', 0)}元"
    )


def _default_position_basis(action: str, support: float, resistance: float) -> str:
    if action == "BUY":
        return "买点靠近支撑，目标看上方压力"
    if action == "SELL":
        return "卖点靠近压力，下方支撑作观察位"
    return ""


class DecisionEngine:
    """决策引擎：技术分析 → LLM → 兜底"""

    def __init__(self, storage):
        self.storage = storage
        self.cfg = get_config()
        self.llm = get_llm_client()
        self.calibrator = ConfidenceCalibrator(storage) if self.cfg.enable_calibration else None

    def decide_one(self, code: str, name: str, tech_score: float,
                   sentiment: float, kline: list[dict],
                   north_context: str = "未知", sht_pct: float = 0.0,
                   alpha_summary: str = "", lgbm_context: str = "",
                   regime_context: str = "大盘 regime: normal",
                   sector_context: str = "所属板块: 未知",
                   confidence_floor: float | None = None) -> dict:
        """对单只股票做最终决策，返回决策 dict"""
        current_price = _safe_float(kline[-1].get("close")) if kline else 0
        floor = confidence_floor if confidence_floor is not None else self.cfg.min_confidence_to_push
        price_levels = _derive_price_levels(kline, current_price)
        price_context = _format_price_context(price_levels)

        # ---- 涨停/ST/次新：强制 HOLD ----
        if kline and len(kline) >= 2:
            pct = (kline[-1]["close"] / kline[-2]["close"] - 1) * 100
            if pct >= 9.5:
                logger.info(f"{code}({name}) 涨停，强制 HOLD")
                return self._hold_decision(code, name, current_price, "涨停禁止买入")
        if "ST" in name or "*ST" in name or "退" in name:
            logger.info(f"{code}({name}) ST/*ST/退市，强制 HOLD")
            return self._hold_decision(code, name, current_price, "ST/*ST/退市风险")
        if kline and len(kline) < 60:
            logger.info(f"{code}({name}) 上市不足60个交易日，强制 HOLD")
            return self._hold_decision(code, name, current_price, "上市不足60个交易日")

        # ---- LLM 决策 ----
        try:
            if self.cfg.any_v2_context_enabled:
                prompt = _ENGINE_PROMPT_V2.format(
                    code=code, name=name,
                    price=current_price,
                    tech_score=tech_score,
                    sentiment=sentiment,
                    alpha_summary=alpha_summary or "- Alpha158 摘要: 未启用",
                    lgbm_context=lgbm_context or "- LightGBM 排序预测: 未启用",
                    regime_context=regime_context,
                    north_money=north_context,
                    sht000001_pct=sht_pct,
                    sector_context=sector_context or "所属板块: 未知",
                    price_context=price_context,
                    hard_rules=_HARD_RULES_PROMPT,
                )
                logger.debug(f"{code} v2 prompt context:\n{prompt}")
            else:
                prompt = _ENGINE_PROMPT.format(
                    code=code, name=name,
                    price=current_price,
                    tech_score=tech_score,
                    sentiment=sentiment,
                    north_money=north_context,
                    sht000001_pct=sht_pct,
                    sector_sentiment="中性",
                    price_context=price_context,
                    hard_rules=_HARD_RULES_PROMPT,
                )
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "请给出决策建议"}
            ]
            result = self.llm.chat_json(messages)
        except Exception as e:
            logger.warning(f"LLM决策异常 {code}: {e}")
            result = {}

        action = result.get("action", "HOLD").upper()
        raw_confidence = float(result.get("confidence", 0))
        confidence = (
            self.calibrator.calibrate(action, raw_confidence)
            if self.calibrator else raw_confidence
        )
        if self.calibrator:
            logger.info(f"{code} 置信度校准: raw={raw_confidence:.3f}, calibrated={confidence:.3f}")
        target_price = _safe_float(result.get("target_price"))
        stop_loss = _safe_float(result.get("stop_loss"))
        trade_price = _safe_float(result.get("trade_price"))
        support_price = _safe_float(result.get("support_price"))
        resistance_price = _safe_float(result.get("resistance_price"))
        position_basis = str(result.get("position_basis") or "").strip()
        reasons = result.get("reasons", [])
        risks = result.get("risks", [])
        one_liner = result.get("one_liner", "")

        if action in {"BUY", "SELL"}:
            if trade_price <= 0:
                trade_price = current_price
            if support_price <= 0:
                support_price = price_levels.get("support_price", 0)
            if resistance_price <= 0:
                resistance_price = price_levels.get("resistance_price", 0)
            if target_price <= 0 and action == "BUY":
                target_price = resistance_price if resistance_price > current_price else current_price * 1.05
            if target_price <= 0 and action == "SELL":
                target_price = support_price if 0 < support_price < current_price else current_price * 0.97
            if stop_loss <= 0 and action == "SELL":
                stop_loss = resistance_price if resistance_price > current_price else current_price * 1.03
            if not position_basis:
                position_basis = _default_position_basis(action, support_price, resistance_price)

        # ---- 兜底：stop_loss ----
        if action == "BUY" and stop_loss <= 0 and current_price > 0:
            stop_loss = round(current_price * (1 - self.cfg.stop_loss_fallback_pct), 2)
            logger.info(f"兜底 stop_loss: {code} → {stop_loss}")

        # ---- 兜底：confidence 阈值 ----
        will_push = confidence >= floor

        return {
            "code": code,
            "name": name,
            "action": action,
            "confidence": round(confidence, 3),
            "raw_confidence": round(raw_confidence, 3),
            "calibrated_confidence": round(confidence, 3),
            "trade_price": _round_price(trade_price),
            "target_price": round(target_price, 2) if target_price else 0,
            "stop_loss": round(stop_loss, 2) if stop_loss else 0,
            "support_price": _round_price(support_price),
            "resistance_price": _round_price(resistance_price),
            "position_basis": position_basis[:50],
            "reasons_json": json.dumps(reasons, ensure_ascii=False),
            "risks_json": json.dumps(risks, ensure_ascii=False),
            "one_liner": one_liner[:50],
            "_will_push": will_push,
            "_current_price": current_price,
        }

    def _hold_decision(self, code, name, price, reason):
        return {
            "code": code, "name": name, "action": "HOLD",
            "confidence": 0.5, "raw_confidence": 0.5, "calibrated_confidence": 0.5,
            "trade_price": 0, "target_price": 0, "stop_loss": 0,
            "support_price": 0, "resistance_price": 0, "position_basis": "",
            "reasons_json": json.dumps([reason], ensure_ascii=False),
            "risks_json": json.dumps(["风险提示：规则强制HOLD"], ensure_ascii=False),
            "one_liner": f"{name} {reason}，保持观望",
            "_will_push": False, "_current_price": price,
        }
