"""Bot command handlers built on the existing StockWatch analysis pipeline."""
from __future__ import annotations

import uuid
from datetime import datetime

from loguru import logger

from analysis.sentiment import batch_sentiment_details
from analysis.technical import compute_tech_score
from bot.research import answer_market_question, answer_stock_question, resolve_stock
from config import get_config
from data.market import MarketData
from decision.engine import DecisionEngine
from push.feishu import render_single_decision_card, render_text_card
from utils.llm import reset_token_usage
from utils.storage import Storage


class BotService:
    def __init__(self, storage: Storage | None = None):
        self.cfg = get_config()
        self.storage = storage or Storage()
        self.market = MarketData()

    def query_stock(self, code: str) -> dict:
        decision, quote = self._analyze_stock(code)
        extra = self._quote_lines(quote)
        return render_single_decision_card(decision, title="股票即时分析", extra_lines=extra)

    def research_stock(self, text: str, code: str = "") -> dict:
        query = text or code
        stock = resolve_stock(query, self.market)
        if code and (not stock or stock.code != code):
            quote = self.market.get_realtime_quote([code]).get(code, {})
            stock = resolve_stock(code, self.market)
            if stock and quote.get("name"):
                stock.name = quote["name"]
        if not stock:
            answer = answer_market_question(query, self.market, self.storage)
            return render_text_card("行情问答", answer.splitlines())
        answer = answer_stock_question(query, stock, self.market, self.storage)
        return render_text_card(f"{stock.name}({stock.code}) 深度问答", answer.splitlines())

    def open_position(self, user_id: str, chat_id: str, code: str,
                      buy_price: float, quantity: float | None = None) -> dict:
        decision, quote = self._analyze_stock(code)
        self.storage.upsert_tracked_position({
            "user_id": user_id,
            "chat_id": chat_id,
            "code": code,
            "name": decision.get("name", code),
            "buy_price": buy_price,
            "quantity": quantity,
            "stop_loss": decision.get("stop_loss", 0),
            "target_price": decision.get("target_price", 0),
        })
        extra = [f"已开始跟踪：成本价 {buy_price:.2f}元"]
        if quantity:
            extra.append(f"数量：{quantity:g}股")
        extra.extend(self._position_pnl_lines(quote, buy_price))
        return render_single_decision_card(decision, title="已开始持仓跟踪", extra_lines=extra)

    def close_position(self, user_id: str, code: str) -> dict:
        count = self.storage.close_tracked_position(user_id, code)
        if count:
            return render_text_card("已停止跟踪", [f"已停止跟踪 `{code}`。"], template="green")
        return render_text_card("未找到跟踪", [f"`{code}` 没有正在跟踪的持仓。"], template="orange")

    def open_price_alert(self, user_id: str, chat_id: str, code: str,
                         trigger_price: float, quantity: float | None = None) -> dict:
        quotes = self.market.get_realtime_quote([code])
        quote = quotes.get(code, {})
        name = quote.get("name", code)
        self.storage.upsert_price_alert({
            "user_id": user_id,
            "chat_id": chat_id,
            "code": code,
            "name": name,
            "trigger_price": trigger_price,
            "quantity": quantity,
            "note": "跌到关键价提醒观察",
        })
        lines = [f"已盯 `{name}({code})`：跌到 {trigger_price:.2f} 元提醒观察。"]
        if quantity:
            lines.append(f"计划数量：{quantity:g}股")
        if quote:
            lines.append(f"当前价：{quote.get('close', 0):.2f}元，今日涨跌：{quote.get('pct_change', 0):+.2f}%")
        lines.append("触价时会顺带看盘口卖压，卖压重会提示先复核风险。")
        return render_text_card("关键价提醒已设置", lines, template="green")

    def cancel_price_alert(self, user_id: str, code: str) -> dict:
        count = self.storage.close_price_alert(user_id, code)
        if count:
            return render_text_card("已取消盯价", [f"已取消 `{code}` 的关键价提醒。"], template="green")
        return render_text_card("未找到盯价", [f"`{code}` 没有正在生效的关键价提醒。"], template="orange")

    def _analyze_stock(self, code: str) -> tuple[dict, dict]:
        reset_token_usage()
        quotes = self.market.get_realtime_quote([code])
        quote = quotes.get(code, {})
        name = quote.get("name", code)
        if not self.storage.kline_cached_today(code):
            for row in self.market.get_daily_kline(code):
                self.storage.upsert_kline(code, row["trade_date"], row)
        kline = self.storage.get_kline(code, "2020-01-01", datetime.now().strftime("%Y-%m-%d"))
        if len(kline) < 20:
            raise RuntimeError(f"{code} K线不足，暂时无法分析")

        tech_result = compute_tech_score(kline)
        sentiment_detail = batch_sentiment_details([(code, name)], storage=self.storage).get(code, {})
        sentiment = sentiment_detail.get("score", 0.0)
        alpha_summary, lgbm_context = self._factor_contexts(code, kline)
        regime_context = self._regime_context()
        sector_context = self._sector_context(code)

        decision = DecisionEngine(self.storage).decide_one(
            code, name,
            tech_result["score"], sentiment,
            kline,
            tech_details=tech_result.get("details"),
            sentiment_context=sentiment_detail.get("context", ""),
            north_context="即时查询跳过北向Top5",
            sht_pct=self.market.get_index_pct(),
            alpha_summary=alpha_summary,
            lgbm_context=lgbm_context,
            regime_context=regime_context,
            sector_context=sector_context,
            confidence_floor=0.0,
        )
        run_id = "bot_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.storage.insert_decision(run_id, datetime.now().isoformat(), decision)
        return decision, quote

    def _factor_contexts(self, code: str, kline: list[dict]) -> tuple[str, str]:
        if not (self.cfg.enable_alpha158 or self.cfg.enable_lgbm):
            return "", ""
        try:
            import pandas as pd
            from analysis.factors import compute_alpha158

            market_df = pd.DataFrame(self.market.get_index_kline("sh000001", limit=320))
            factors = compute_alpha158(pd.DataFrame(kline), market_df)
            alpha_summary = ""
            lgbm_context = ""
            if self.cfg.enable_alpha158:
                from analysis.factors import summarize_alpha158_cross_section
                alpha_summary = summarize_alpha158_cross_section({code: factors}).get(code, "")
            if self.cfg.enable_lgbm:
                from analysis.lgbm import LgbmRanker, format_lgbm_context
                ranker = LgbmRanker(self.cfg.lgbm_model_path)
                lgbm_context = format_lgbm_context({code: ranker.predict(factors)}).get(code, "")
            return alpha_summary, lgbm_context
        except Exception as e:
            logger.warning(f"即时查询因子上下文失败 {code}: {e}")
            return "", ""

    def _regime_context(self) -> str:
        if not self.cfg.enable_regime:
            return "大盘 regime: normal (未启用)"
        try:
            from analysis.regime import get_market_regime
            return get_market_regime(self.market, self.storage).get("context", "大盘 regime: normal")
        except Exception as e:
            logger.warning(f"即时查询 regime 获取失败: {e}")
            return "大盘 regime: normal"

    def _sector_context(self, code: str) -> str:
        if not self.cfg.enable_sector:
            return "所属板块: 未启用"
        try:
            from analysis.sector import get_sector_contexts
            return get_sector_contexts([code], self.market, self.storage).get(code, "所属板块: 未知")
        except Exception as e:
            logger.warning(f"即时查询板块上下文失败 {code}: {e}")
            return "所属板块: 未知"

    @staticmethod
    def _quote_lines(quote: dict) -> list[str]:
        if not quote:
            return []
        close = quote.get("close", 0)
        pct = quote.get("pct_change", 0)
        return [f"现价：{close:.2f}元，今日涨跌：{pct:+.2f}%"]

    @staticmethod
    def _position_pnl_lines(quote: dict, buy_price: float) -> list[str]:
        current = float(quote.get("close") or 0)
        if current <= 0 or buy_price <= 0:
            return []
        pct = current / buy_price - 1
        return [f"当前价：{current:.2f}元，浮盈亏：{pct:+.2%}"]
