"""最终决策引擎：技术分 + 情绪分 → LLM决策 + 程序级兜底"""
import json
from loguru import logger

from config import get_config
from utils.llm import get_llm_client
from data.market import MarketData


_HARD_RULES_PROMPT = """【硬规则】以下标的必须输出 HOLD，无论其他信号多强：
1. 当日已涨停（涨幅≥9.5%）→ HOLD，禁止 BUY
2. ST/*ST/退市风险股 → HOLD
3. 上市不足60个交易日 → HOLD
4. BUY 信号必须同时输出 stop_loss（若 LLM 未给出，程序自动填当前价×0.93）
5. 置信度 < 0.6 → 仅记录，不推送"""


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

【输出格式】严格JSON，无markdown围栏，无前言：
{{
  "code": "{code}",
  "name": "{name}",
  "action": "BUY|SELL|HOLD",
  "confidence": 0.0到1.0,
  "target_price": 0.0,
  "stop_loss": 0.0,
  "reasons": ["原因1", "原因2"],
  "risks": ["风险1", "风险2"],
  "one_liner": "一句话给非技术用户看，30字以内"
}}

{hard_rules}

不要输出 JSON 以外的内容。"""


class DecisionEngine:
    """决策引擎：技术分析 → LLM → 兜底"""

    def __init__(self, storage):
        self.storage = storage
        self.cfg = get_config()
        self.llm = get_llm_client()

    def decide_one(self, code: str, name: str, tech_score: float,
                   sentiment: float, kline: list[dict],
                   north_context: str = "未知", sht_pct: float = 0.0) -> dict:
        """对单只股票做最终决策，返回决策 dict"""
        current_price = kline[-1]["close"] if kline else 0

        # ---- 涨停/ST/次新：强制 HOLD ----
        if tech_score > 0.5 and kline and len(kline) >= 2:
            pct = (kline[-1]["close"] / kline[-2]["close"] - 1) * 100
            if pct >= 9.5:
                logger.info(f"{code}({name}) 涨停，强制 HOLD")
                return self._hold_decision(code, name, current_price, "涨停禁止买入")
        if "ST" in name or "*ST" in name or "退" in name:
            logger.info(f"{code}({name}) ST/*ST/退市，强制 HOLD")
            return self._hold_decision(code, name, current_price, "ST/*ST/退市风险")

        # ---- LLM 决策 ----
        try:
            messages = [
                {"role": "system", "content": _ENGINE_PROMPT.format(
                    code=code, name=name,
                    price=current_price,
                    tech_score=tech_score,
                    sentiment=sentiment,
                    north_money=north_context,
                    sht000001_pct=sht_pct,
                    sector_sentiment="中性",
                    hard_rules=_HARD_RULES_PROMPT,
                )},
                {"role": "user", "content": "请给出决策建议"}
            ]
            result = self.llm.chat_json(messages)
        except Exception as e:
            logger.warning(f"LLM决策异常 {code}: {e}")
            result = {}

        action = result.get("action", "HOLD").upper()
        confidence = float(result.get("confidence", 0))
        target_price = float(result.get("target_price") or 0)
        stop_loss = float(result.get("stop_loss") or 0)
        reasons = result.get("reasons", [])
        risks = result.get("risks", [])
        one_liner = result.get("one_liner", "")

        # ---- 兜底：stop_loss ----
        if action == "BUY" and stop_loss <= 0 and current_price > 0:
            stop_loss = round(current_price * (1 - self.cfg.stop_loss_fallback_pct), 2)
            logger.info(f"兜底 stop_loss: {code} → {stop_loss}")

        # ---- 兜底：confidence 阈值 ----
        will_push = confidence >= self.cfg.min_confidence_to_push

        return {
            "code": code,
            "name": name,
            "action": action,
            "confidence": round(confidence, 3),
            "target_price": round(target_price, 2) if target_price else 0,
            "stop_loss": round(stop_loss, 2) if stop_loss else 0,
            "reasons_json": json.dumps(reasons, ensure_ascii=False),
            "risks_json": json.dumps(risks, ensure_ascii=False),
            "one_liner": one_liner[:50],
            "_will_push": will_push,
            "_current_price": current_price,
        }

    def _hold_decision(self, code, name, price, reason):
        return {
            "code": code, "name": name, "action": "HOLD",
            "confidence": 0.5, "target_price": 0, "stop_loss": 0,
            "reasons_json": json.dumps([reason], ensure_ascii=False),
            "risks_json": json.dumps(["风险提示：规则强制HOLD"], ensure_ascii=False),
            "one_liner": f"{name} {reason}，保持观望",
            "_will_push": False, "_current_price": price,
        }