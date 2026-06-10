"""Parse simple stock bot commands from Feishu text messages."""
from __future__ import annotations

import re
from dataclasses import dataclass


_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_PRICE_ALERT_RE = re.compile(r"^(盯买|盯价|加仓提醒|买点|挂单)[+\s,，:：]*(\d{6})(?:[+\s,，:：]*(\d+(?:\.\d+)?))?(?:[+\s,，]*(\d+(?:\.\d+)?)(?:股)?)?")
_CANCEL_PRICE_ALERT_RE = re.compile(r"^(取消盯价|取消盯买|停止盯价|撤销盯价|不盯了)[+\s,，:：]*(\d{6})")
_BUY_RE = re.compile(r"^(买入|买|跟踪|建仓)[+\s,，:：]*(\d{6})(?:[+\s,，:：]*(\d+(?:\.\d+)?))?(?:[+\s,，]*(\d+(?:\.\d+)?)(?:股)?)?")
_SELL_RE = re.compile(r"^(卖出|卖|停止跟踪|取消跟踪|不跟了)[+\s,，:：]*(\d{6})")


@dataclass
class BotCommand:
    action: str
    code: str = ""
    price: float | None = None
    quantity: float | None = None
    text: str = ""


def _to_float(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def normalize_text(text: str) -> str:
    text = re.sub(r"<at[^>]*>.*?</at>", "", text or "")
    return text.replace("\u3000", " ").strip()


def parse_command(text: str) -> BotCommand:
    text = normalize_text(text)
    if not text or text.lower() in {"help", "帮助", "?"}:
        return BotCommand("help")

    cancel_alert = _CANCEL_PRICE_ALERT_RE.search(text)
    if cancel_alert:
        return BotCommand("cancel_price_alert", code=cancel_alert.group(2))

    price_alert = _PRICE_ALERT_RE.search(text)
    if price_alert:
        return BotCommand(
            "price_alert",
            code=price_alert.group(2),
            price=_to_float(price_alert.group(3)),
            quantity=_to_float(price_alert.group(4)),
        )

    sell = _SELL_RE.search(text)
    if sell:
        return BotCommand("sell", code=sell.group(2))

    buy = _BUY_RE.search(text)
    if buy:
        return BotCommand(
            "buy",
            code=buy.group(2),
            price=_to_float(buy.group(3)),
            quantity=_to_float(buy.group(4)),
        )

    code = _CODE_RE.search(text)
    if code:
        bare_query = re.fullmatch(r"(?:查|查询)?\s*(\d{6})\s*", text)
        if bare_query:
            return BotCommand("query", code=bare_query.group(1))
        return BotCommand("research", code=code.group(1), text=text)

    return BotCommand("research", text=text)


def help_lines() -> list[str]:
    return [
        "**可用命令**",
        "查股票：`600519` 或 `查 600519`",
        "问近况：`宁夏建材重组怎么样`、`600449 最近一周走势如何`",
        "开始跟踪：`买入 600519 1680`，可追加数量：`买入 600519 1680 100股`",
        "盯加仓价：`盯买 600519 1500`，可追加数量：`盯买 600519 1500 100股`",
        "取消盯价：`取消盯价 600519`",
        "停止跟踪：`卖出 600519` 或 `停止跟踪 600519`",
    ]
