"""Feishu SDK long-connection runner."""
from __future__ import annotations

import json
import re

from loguru import logger

from bot.parser import help_lines, parse_command
from bot.service import BotService
from config import get_config
from push.feishu import FeishuClient, render_text_card
from utils.storage import Storage


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


def run_bot():
    try:
        import lark_oapi as lark
        from lark_oapi.api.im.v1 import P2ImMessageReceiveV1
    except Exception as e:
        raise RuntimeError("飞书长连接需要安装 lark-oapi，请先 pip install -r requirements.txt") from e

    cfg = get_config()
    storage = Storage()
    service = BotService(storage)
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
        storage.mark_bot_event_handled(event_id, message_id)

        try:
            if message.get("message_type") != "text":
                card = render_text_card("暂不支持", help_lines(), template="orange")
            else:
                command = parse_command(_message_text(message))
                if command.action == "query":
                    card = service.query_stock(command.code)
                elif command.action == "research":
                    card = service.research_stock(command.text, command.code)
                elif command.action == "buy":
                    if command.price is None or command.price <= 0:
                        card = render_text_card("买入命令缺少价格", [
                            f"请发送：`买入 {command.code} 买入价`，例如 `买入 {command.code} 1680`",
                        ], template="orange")
                    else:
                        card = service.open_position(user_id, chat_id, command.code, command.price, command.quantity)
                elif command.action == "sell":
                    card = service.close_position(user_id, command.code)
                else:
                    card = render_text_card("StockWatch 助手", help_lines())
        except Exception as e:
            logger.exception(f"处理飞书消息失败: {e}")
            card = render_text_card("处理失败", [str(e)], template="red")

        if chat_id:
            feishu.send_card_to(chat_id, card, "chat_id")

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
