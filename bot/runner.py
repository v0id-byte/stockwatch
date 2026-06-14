"""Feishu SDK long-connection runner."""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass

from loguru import logger

from bot.financial_report import parse_user_reply
from bot.parser import help_lines, parse_command
from bot.research import StockRef, resolve_stock
from bot.service import BotService
from config import get_config
from data.market import MarketData
from push.feishu import FeishuClient, render_text_card
from utils.storage import Storage


_FINANCIAL_FLOW_TTL_SEC = 600
_FINANCIAL_FLOW_LOCK = threading.Lock()
_FINANCIAL_FLOW_STATE: dict[str, "_FinancialFlow"] = {}


@dataclass
class _FinancialFlow:
    code: str
    name: str
    created_at: float

    def is_expired(self) -> bool:
        return time.time() - self.created_at > _FINANCIAL_FLOW_TTL_SEC


def _payload_to_dict(data) -> dict:
    import lark_oapi as lark

    return json.loads(lark.JSON.marshal(data))


def _message_text(message: dict) -> str:
    content = message.get("content") or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {}
    text = parsed.get("text", "")
    return re.sub(r"<at[^>]*>.*?</at>", "", text).strip()


def _sender_open_id(payload: dict) -> str:
    sender_id = payload.get("event", {}).get("sender", {}).get("sender_id", {})
    return sender_id.get("open_id") or sender_id.get("user_id") or ""


def _flow_key(user_id: str, chat_id: str) -> str:
    return f"{user_id}:{chat_id}"


def _prune_expired_flows() -> None:
    with _FINANCIAL_FLOW_LOCK:
        stale = [k for k, v in _FINANCIAL_FLOW_STATE.items() if v.is_expired()]
        for k in stale:
            _FINANCIAL_FLOW_STATE.pop(k, None)


def _set_financial_flow(user_id: str, chat_id: str, stock: StockRef) -> None:
    _prune_expired_flows()
    with _FINANCIAL_FLOW_LOCK:
        _FINANCIAL_FLOW_STATE[_flow_key(user_id, chat_id)] = _FinancialFlow(
            code=stock.code, name=stock.name, created_at=time.time(),
        )


def _claim_financial_flow(user_id: str, chat_id: str) -> _FinancialFlow | None:
    """Atomically pop-and-return the flow if it's still valid.

    Critical: ``on_message`` can be invoked concurrently for the same user by
    the lark WebSocket worker (e.g. fast double-tap). Peek-then-pop is a TOCTOU
    race — both threads could observe the pending state and both invoke the
    LLM, wasting a token call. This single-lock pop+validate avoids that.
    """
    with _FINANCIAL_FLOW_LOCK:
        flow = _FINANCIAL_FLOW_STATE.pop(_flow_key(user_id, chat_id), None)
        if flow and not flow.is_expired():
            return flow
        return None


def _discard_financial_flow(user_id: str, chat_id: str) -> None:
    """Pop without checking expiry — used for explicit cancel."""
    with _FINANCIAL_FLOW_LOCK:
        _FINANCIAL_FLOW_STATE.pop(_flow_key(user_id, chat_id), None)


def run_bot():
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
    except Exception as e:
        raise RuntimeError("飞书长连接需要安装 lark-oapi，请先 pip install -r requirements.txt") from e

    cfg = get_config()
    storage = Storage()
    service = BotService(storage)
    market = MarketData()
    feishu = FeishuClient()

    def on_message(data: P2ImMessageReceiveV1):
        payload = _payload_to_dict(data)
        header = payload.get("header", {})
        message = payload.get("event", {}).get("message", {})
        event_id = header.get("event_id") or message.get("message_id", "")
        message_id = message.get("message_id", "")
        chat_id = message.get("chat_id", "")
        user_id = _sender_open_id(payload)
        if storage.is_bot_event_handled(event_id):
            logger.info(f"跳过重复飞书事件: {event_id}")
            return

        try:
            if message.get("message_type") != "text":
                card = render_text_card("暂不支持", help_lines(), template="orange")
            else:
                raw_text = _message_text(message)

                # 1) 财报解析多轮：先尝试把当前消息当作上一条菜单的回复
                pending = _claim_financial_flow(user_id, chat_id)
                selections = parse_user_reply(raw_text)
                if pending is not None and selections is not None:
                    stock = StockRef(code=pending.code, name=pending.name)
                    card = service.financial_report_answer(stock, selections)
                else:
                    # 没匹配到菜单回复时，按普通命令解析。
                    # 如果上面 _claim 拿到了 pending 但本条不是有效回复，把状态
                    # 放回去，下条消息继续等回复。
                    if pending is not None:
                        _set_financial_flow(user_id, chat_id, pending)
                    command = parse_command(raw_text)
                    if command.action == "cancel_pending_flow":
                        _discard_financial_flow(user_id, chat_id)
                        card = render_text_card(
                            "已取消财报解析",
                            ["已清空待选的财报菜单。你可以重新发送 `财报解析 600519` 开始。"],
                            template="green",
                        )
                    elif command.action == "query":
                        card = service.query_stock(command.code)
                    elif command.action == "research":
                        card = service.research_stock(command.text, command.code)
                    elif command.action == "financial_report_menu":
                        card = service.financial_report_menu(command.code)
                        if command.code:
                            stock = resolve_stock(command.code, market)
                            if stock:
                                _set_financial_flow(user_id, chat_id, stock)
                    elif command.action == "buy":
                        if command.price is None or command.price <= 0:
                            card = render_text_card("买入命令缺少价格", [
                                f"请发送：`买入 {command.code} 买入价`，例如 `买入 {command.code} 1680`",
                            ], template="orange")
                        else:
                            card = service.open_position(user_id, chat_id, command.code, command.price, command.quantity)
                    elif command.action == "price_alert":
                        if command.price is None or command.price <= 0:
                            card = render_text_card("盯价命令缺少价格", [
                                f"请发送：`盯买 {command.code} 目标价`，例如 `盯买 {command.code} 1500`",
                            ], template="orange")
                        else:
                            card = service.open_price_alert(user_id, chat_id, command.code, command.price, command.quantity)
                    elif command.action == "cancel_price_alert":
                        card = service.cancel_price_alert(user_id, command.code)
                    elif command.action == "sell":
                        card = service.close_position(user_id, command.code)
                    else:
                        card = render_text_card("StockWatch 助手", help_lines())
        except Exception as e:
            logger.exception(f"处理飞书消息失败: {e}")
            card = render_text_card("处理失败", [str(e)], template="red")

        sent = feishu.send_card_to(chat_id, card, "chat_id") if chat_id else False
        if sent:
            storage.mark_bot_event_handled(event_id, message_id)

    event_handler = (
        lark.EventDispatcherHandler.builder(
            cfg.feishu_verification_token,
            cfg.feishu_encrypt_key,
        )
        .register_p2_im_message_receive_v1(on_message)
        .build()
    )
    client = lark.ws.Client(
        cfg.feishu_app_id,
        cfg.feishu_app_secret,
        event_handler=event_handler,
        log_level=lark.LogLevel.INFO,
    )
    logger.info("飞书长连接机器人启动")
    client.start()
