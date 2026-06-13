"""Small local dashboard and settings panel for StockWatch."""
from __future__ import annotations

import argparse
import email.policy
import html
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime
from email.parser import BytesParser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from analysis.report import build_report
from utils.storage import Storage


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CUSTOM_FACTORS_DIR = Path.home() / ".stockwatch" / "custom_factors"
_ENV_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")
WEB_USER_ID = "web-ui"
WEB_CHAT_ID = "web-ui"
FEEDBACK_URL = "https://github.com/v0id-byte/stockwatch/issues"
AUTH_COOKIE = "stockwatch_auth"

DEFAULT_SETTINGS = {
    "NOTIFY_CHANNEL": "feishu",
    "LLM_PROVIDER": "openai",
    "LLM_API_KEY": "",
    "LLM_BASE_URL": "https://api.minimaxi.com/v1",
    "LLM_MODEL": "MiniMax-M2.7",
    "WATCHLIST": "600519,000858,510300,510500,159915",
    "MAX_STOCKS_PER_RUN": "50",
    "MIN_CONFIDENCE_TO_PUSH": "0.6",
    "ALERT_LEVELS": "critical,warning,info",
    "ENABLE_REASSURANCE_MODE": "false",
    "ENABLE_FAMILY_BRIEF": "false",
    "ENABLE_AFTER_CLOSE_SUMMARY": "false",
    "AI_RESPONSE_STYLE": "balanced",
    "ENABLE_CALIBRATION": "false",
    "ENABLE_ALPHA158": "false",
    "ENABLE_LGBM": "false",
    "ENABLE_REGIME": "false",
    "ENABLE_SECTOR": "false",
    "FEISHU_APP_ID": "",
    "FEISHU_APP_SECRET": "",
    "FEISHU_RECEIVE_ID": "",
    "FEISHU_RECEIVE_ID_TYPE": "open_id",
    "FEISHU_RECEIVE_ID_2": "",
    "FEISHU_VERIFICATION_TOKEN": "",
    "FEISHU_ENCRYPT_KEY": "",
    "WEB_AUTH_TOKEN": "",
}

BUILTIN_FACTORS = [
    {
        "id": "regime",
        "env_key": "ENABLE_REGIME",
        "name": "市场状态因子",
        "category": "风控",
        "horizon": "短线/波动",
        "source": "官方",
        "description": "识别大盘波动状态，在高波动阶段抬高提醒阈值，减少情绪化打扰。",
        "data_requirements": "指数 K 线、20 日波动率",
        "status": "builtin",
    },
    {
        "id": "sector",
        "env_key": "ENABLE_SECTOR",
        "name": "板块强弱因子",
        "category": "板块轮动",
        "horizon": "1-5 日",
        "source": "官方",
        "description": "比较所属板块近期强弱，把个股提醒放回板块环境里解释。",
        "data_requirements": "个股-板块映射、板块 5 日收益",
        "status": "builtin",
    },
    {
        "id": "alpha158",
        "env_key": "ENABLE_ALPHA158",
        "name": "Alpha158 技术因子",
        "category": "技术面",
        "horizon": "5-20 日",
        "source": "官方",
        "description": "计算动量、波动、成交量结构、相对强弱和回撤等量化特征。",
        "data_requirements": "个股日 K、指数日 K",
        "status": "builtin",
    },
    {
        "id": "calibration",
        "env_key": "ENABLE_CALIBRATION",
        "name": "提醒校准因子",
        "category": "提醒质量",
        "horizon": "复盘",
        "source": "官方",
        "description": "用历史提醒结果校准置信度，让后续提醒更克制。",
        "data_requirements": "历史 decisions、后验成功标记",
        "status": "builtin",
    },
    {
        "id": "lgbm",
        "env_key": "ENABLE_LGBM",
        "name": "LightGBM 排序因子",
        "category": "机器学习",
        "horizon": "20 日",
        "source": "官方",
        "description": "离线训练横截面排序模型，线上只作为解释和排序辅助。",
        "data_requirements": "训练好的 LGBM 模型、Alpha 因子",
        "status": "builtin",
    },
]


def _rows(conn: sqlite3.Connection, sql: str, params: list | None = None) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params or []).fetchall()]


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value or value.startswith("#"):
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1]
        except json.JSONDecodeError:
            return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    return value


def _read_env_values() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        match = _ENV_ASSIGN_RE.match(line)
        if match:
            values[match.group(2)] = _parse_env_value(match.group(4))
    return values


def load_settings() -> dict[str, str]:
    values = dict(DEFAULT_SETTINGS)
    for key in values:
        if os.getenv(key) is not None:
            values[key] = os.getenv(key, "")
    env_values = _read_env_values()
    for key in values:
        if key in env_values:
            values[key] = env_values[key]
    if not values["LLM_API_KEY"]:
        if values["LLM_PROVIDER"] == "anthropic":
            values["LLM_API_KEY"] = env_values.get("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
        else:
            values["LLM_API_KEY"] = env_values.get("MINIMAX_API_KEY", os.getenv("MINIMAX_API_KEY", ""))
    return values


def _web_auth_token() -> str:
    return (load_settings().get("WEB_AUTH_TOKEN") or "").strip()


def _cookie_value(headers, name: str) -> str:
    raw = headers.get("Cookie", "")
    if not raw:
        return ""
    cookie = SimpleCookie()
    try:
        cookie.load(raw)
    except Exception:
        return ""
    morsel = cookie.get(name)
    return morsel.value if morsel else ""


def _request_token(headers, params: dict[str, list[str]]) -> str:
    return (
        headers.get("X-StockWatch-Token", "").strip()
        or _first(params, "token").strip()
        or _cookie_value(headers, AUTH_COOKIE).strip()
    )


def _is_authorized(headers, params: dict[str, list[str]] | None = None) -> bool:
    token = _web_auth_token()
    if not token:
        return True
    supplied = _request_token(headers, params or {})
    return bool(supplied) and secrets.compare_digest(supplied, token)


def _serialize_env_value(value: str) -> str:
    value = str(value)
    if value == "" or re.fullmatch(r"[A-Za-z0-9_./:@,+~=-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


def save_settings(updates: dict[str, str]) -> None:
    allowed = set(DEFAULT_SETTINGS)
    cleaned = {key: str(value).strip() for key, value in updates.items() if key in allowed}
    if not cleaned:
        return
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines(keepends=True) if ENV_PATH.exists() else []
    written: set[str] = set()
    next_lines: list[str] = []
    for line in lines:
        match = _ENV_ASSIGN_RE.match(line)
        if match and match.group(2) in cleaned:
            key = match.group(2)
            newline = match.group(5) or "\n"
            next_lines.append(f"{key}={_serialize_env_value(cleaned[key])}{newline}")
            written.add(key)
        else:
            next_lines.append(line)
    missing = [key for key in cleaned if key not in written]
    if missing:
        if next_lines and not next_lines[-1].endswith(("\n", "\r\n")):
            next_lines[-1] += "\n"
        if next_lines:
            next_lines.append("\n")
        next_lines.append("# ===== Web UI settings =====\n")
        for key in missing:
            next_lines.append(f"{key}={_serialize_env_value(cleaned[key])}\n")
    ENV_PATH.write_text("".join(next_lines), encoding="utf-8")
    for key, value in cleaned.items():
        os.environ[key] = value


def _safe_filename(raw: str, fallback: str = "factor.py") -> str:
    name = Path(raw or fallback).name
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-")
    return name[:96] or fallback


def _parse_form_data(headers, body: bytes) -> tuple[dict[str, list[str]], dict[str, tuple[str, bytes]]]:
    content_type = headers.get("Content-Type", "")
    if content_type.startswith("multipart/form-data"):
        raw = (
            f"Content-Type: {content_type}\n"
            "MIME-Version: 1.0\n\n"
        ).encode("utf-8") + body
        message = BytesParser(policy=email.policy.default).parsebytes(raw)
        fields: dict[str, list[str]] = {}
        files: dict[str, tuple[str, bytes]] = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            if filename:
                files[name] = (_safe_filename(filename), payload)
            else:
                charset = part.get_content_charset() or "utf-8"
                fields.setdefault(name, []).append(payload.decode(charset, errors="replace"))
        return fields, files
    params = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return params, {}


def _factor_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-").lower()
    return slug[:48] or "custom-factor"


def save_custom_factor(params: dict[str, list[str]], files: dict[str, tuple[str, bytes]]) -> str:
    filename, payload = files.get("factor_file", ("", b""))
    if not filename or not payload:
        raise ValueError("请选择要上传的因子文件")
    if len(payload) > 512 * 1024:
        raise ValueError("因子文件不能超过 512KB")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".py", ".json", ".md", ".txt"}:
        raise ValueError("仅支持 .py / .json / .md / .txt 因子文件")

    name = _first(params, "factor_name", Path(filename).stem).strip() or Path(filename).stem
    description = _first(params, "factor_description").strip()
    category = _first(params, "factor_category", "自定义").strip() or "自定义"
    horizon = _first(params, "factor_horizon", "自定义").strip() or "自定义"
    data_requirements = _first(params, "factor_data_requirements").strip()
    license_name = _first(params, "factor_license", "MIT").strip() or "MIT"
    share = _bool_param(params, "share_to_community") == "true"
    CUSTOM_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = f"{stamp}_{_factor_slug(name)}_{filename}"
    target = CUSTOM_FACTORS_DIR / target_name
    target.write_bytes(payload)
    manifest = {
        "id": _factor_slug(name),
        "name": name,
        "category": category,
        "horizon": horizon,
        "description": description,
        "data_requirements": data_requirements,
        "license": license_name,
        "source": "用户上传",
        "share_to_community": share,
        "status": "contribution_ready" if share else "local_draft",
        "file": str(target),
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    manifest_path = target.with_name(target.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"因子已保存到 {target}"


def load_custom_factors() -> list[dict]:
    if not CUSTOM_FACTORS_DIR.exists():
        return []
    items = []
    for path in sorted(CUSTOM_FACTORS_DIR.glob("*.manifest.json"), reverse=True):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        item["_manifest"] = str(path)
        items.append(item)
    return items[:40]


def _factor_enabled(settings: dict[str, str], item: dict) -> bool:
    env_key = item.get("env_key")
    return bool(env_key and str(settings.get(env_key, "false")).lower() in {"1", "true", "yes", "on"})


def _normalize_custom_factor(item: dict) -> dict:
    name = item.get("name") or Path(item.get("file", "custom-factor")).stem
    return {
        "id": item.get("id") or _factor_slug(str(name)),
        "name": name,
        "category": item.get("category") or "自定义",
        "horizon": item.get("horizon") or "自定义",
        "source": item.get("source") or "用户上传",
        "description": item.get("description") or "用户上传的本地因子，默认只入库展示，不自动执行。",
        "data_requirements": item.get("data_requirements") or "见上传文件说明",
        "license": item.get("license") or "未声明",
        "status": item.get("status") or "local_draft",
        "file": item.get("file") or "",
        "uploaded_at": item.get("uploaded_at") or "",
        "share_to_community": bool(item.get("share_to_community")),
    }


def load_factor_catalog(settings: dict[str, str], filters: dict[str, str] | None = None) -> list[dict]:
    filters = filters or {}
    catalog = [dict(item, enabled=_factor_enabled(settings, item), executable=True) for item in BUILTIN_FACTORS]
    catalog.extend(dict(_normalize_custom_factor(item), enabled=False, executable=False) for item in load_custom_factors())

    query = filters.get("q", "").strip().lower()
    category = filters.get("category", "").strip()
    source = filters.get("source", "").strip()
    status = filters.get("status", "").strip()

    def matches(item: dict) -> bool:
        haystack = " ".join(str(item.get(key, "")) for key in ("name", "category", "horizon", "source", "description")).lower()
        if query and query not in haystack:
            return False
        if category and item.get("category") != category:
            return False
        if source and item.get("source") != source:
            return False
        if status == "enabled" and not item.get("enabled"):
            return False
        if status == "local" and item.get("source") != "用户上传":
            return False
        return True

    return [item for item in catalog if matches(item)]


def _card_to_result(card: dict) -> dict:
    title = (((card.get("header") or {}).get("title") or {}).get("content") or "StockWatch")
    parts = []
    for element in card.get("elements", []):
        text = element.get("text") or {}
        if text.get("content"):
            parts.append(str(text["content"]))
    text = "\n".join(parts)
    if "帮助" in title:
        summary, actions = [], []
    else:
        summary, actions = _console_summary_and_actions(title, text)
    return {"title": title, "text": text, "ok": True, "summary": summary, "actions": actions}


def _error_result(message: str) -> dict:
    return {"title": "操作失败", "text": message, "ok": False, "summary": [], "actions": []}


def _is_market_question(text: str) -> bool:
    if re.search(r"(?<!\d)\d{6}(?!\d)", text):
        return False
    stripped = re.sub(r"\s+", "", text or "")
    if any(word in stripped for word in ("大盘", "市场", "指数")):
        return True
    return stripped in {"现在行情怎么样", "今天行情怎么样", "行情怎么样", "今天怎么样", "现在怎么样"}


def _market_snapshot_result(service) -> dict:
    from bot.research import build_market_snapshot, format_market_snapshot

    snapshot = build_market_snapshot(
        service.market,
        service.storage,
        include_news=False,
        include_north=False,
        include_regime=False,
    )
    text = format_market_snapshot(snapshot)
    indexes = snapshot.get("indexes", [])
    valid_pcts = [item.get("pct_change") for item in indexes if item.get("pct_change") is not None]
    avg_pct = sum(valid_pcts) / len(valid_pcts) if valid_pcts else 0
    tone = "偏强" if avg_pct >= 0.4 else "偏弱" if avg_pct <= -0.4 else "震荡"
    summary = [{"label": "市场状态", "value": tone}]
    for item in indexes[:4]:
        pct = item.get("pct_change")
        value = f"{pct:+.2f}%" if pct is not None else "暂无"
        summary.append({"label": item.get("name", "指数"), "value": value})
    return {"title": "行情快照", "text": text, "ok": True, "summary": summary, "actions": []}


def _positive_float(value: str, label: str, required: bool = True) -> float | None:
    value = (value or "").strip().replace(",", "")
    if not value:
        if required:
            raise ValueError(f"请输入{label}")
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{label}需要是数字") from exc
    if number <= 0:
        raise ValueError(f"{label}需要大于 0")
    return number


def _stock_code(value: str) -> str:
    match = re.search(r"(?<!\d)(\d{6})(?!\d)", value or "")
    if not match:
        raise ValueError("请输入 6 位股票代码")
    return match.group(1)


def _row_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _average_true_range(kline: list[dict], limit: int = 14) -> float:
    rows = kline[-(limit + 1):]
    if len(rows) < 2:
        return 0.0
    prev_close = _row_float(rows[0], "close")
    values = []
    for row in rows[1:]:
        high = _row_float(row, "high")
        low = _row_float(row, "low")
        close = _row_float(row, "close")
        if high <= 0 or low <= 0 or prev_close <= 0:
            prev_close = close
            continue
        values.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
        prev_close = close
    return sum(values[-limit:]) / len(values[-limit:]) if values else 0.0


def _recent_support_resistance(kline: list[dict], current_price: float) -> tuple[float, float]:
    recent = kline[-20:]
    lows = sorted({_row_float(row, "low") for row in recent if _row_float(row, "low") > 0})
    highs = sorted({_row_float(row, "high") for row in recent if _row_float(row, "high") > 0})
    support = max([price for price in lows if price <= current_price], default=(min(lows) if lows else 0.0))
    resistance = min([price for price in highs if price >= current_price], default=(max(highs) if highs else 0.0))
    return support, resistance


def _dedupe_candidates(items: list[tuple[str, float]], current_price: float, side: str) -> list[tuple[str, float]]:
    result = []
    seen = set()
    for label, price in items:
        if price <= 0:
            continue
        if side == "below" and price >= current_price:
            continue
        if side == "above" and price <= current_price:
            continue
        rounded = round(price, 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        result.append((label, rounded))
    return result


def _format_candidates(items: list[tuple[str, float]]) -> str:
    if not items:
        return "暂无可用候选位"
    return "\n".join(f"- {label}: {price:.2f} 元" for label, price in items)


def _risk_plan_result(params: dict[str, list[str]], storage: Storage) -> dict:
    from data.market import MarketData

    code = _stock_code(_first(params, "code"))
    cost_price = _positive_float(_first(params, "cost_price"), "成本价")
    quantity = _positive_float(_first(params, "quantity"), "数量", required=False)

    market = MarketData()
    quote = market.get_realtime_quote([code]).get(code, {})
    name = quote.get("name") or code

    try:
        if not storage.kline_cached_today(code):
            for row in market.get_daily_kline(code, limit=120):
                storage.upsert_kline(code, row["trade_date"], row)
    except Exception:
        pass
    kline = storage.get_kline(code, "2020-01-01", datetime.now().strftime("%Y-%m-%d"))[-60:]

    current_price = _row_float(quote, "close")
    if current_price <= 0 and kline:
        current_price = _row_float(kline[-1], "close")
    if current_price <= 0:
        current_price = cost_price

    atr = _average_true_range(kline)
    support, resistance = _recent_support_resistance(kline, current_price)

    risk_items = [
        ("成本下方 7%", cost_price * 0.93),
        ("成本下方 10%", cost_price * 0.90),
        ("现价下方 3%", current_price * 0.97),
        ("近 20 日支撑下方 1%", support * 0.99),
        ("2 倍 ATR 风险线", current_price - 2 * atr if atr > 0 else 0),
    ]
    pressure_items = [
        ("成本上方 10%", cost_price * 1.10),
        ("固定 2R 观察位", cost_price + 2 * (cost_price - cost_price * 0.93)),
        ("近 20 日压力位", resistance),
        ("2 倍 ATR 观察位", current_price + 2 * atr if atr > 0 else 0),
        ("现价上方 5%", current_price * 1.05),
    ]
    risk_candidates = _dedupe_candidates(risk_items, current_price, "below")
    pressure_candidates = _dedupe_candidates(pressure_items, current_price, "above")
    risk_price = risk_candidates[0][1] if risk_candidates else round(current_price * 0.97, 2)
    pressure_price = pressure_candidates[0][1] if pressure_candidates else round(current_price * 1.05, 2)

    data_note = "已结合实时价、近 20 日支撑压力和 ATR" if kline else "行情/K 线不足，先按成本价和现价固定比例生成"
    text = (
        f"{name}({code}) 参考风险线\n"
        f"{data_note}。\n\n"
        "候选风险价（向下触发）：\n"
        f"{_format_candidates(risk_candidates)}\n\n"
        "候选观察压力位（向上触发）：\n"
        f"{_format_candidates(pressure_candidates)}\n\n"
        "这是参考风控线和观察压力位，不是买卖指令；请自行确认后再保存提醒。"
    )
    actions = [
        {
            "label": "保存风险价提醒",
            "params": {"console_action": "price_alert", "code": code, "price": risk_price, "direction": "below", "quantity": quantity or ""},
        },
        {
            "label": "保存压力位提醒",
            "params": {"console_action": "price_alert", "code": code, "price": pressure_price, "direction": "above", "quantity": quantity or ""},
        },
        {
            "label": "按成本价跟踪",
            "params": {"console_action": "track_position", "code": code, "price": cost_price, "quantity": quantity or ""},
        },
    ]
    return {
        "title": "参考风险线",
        "text": text,
        "ok": True,
        "summary": [
            {"label": "标的", "value": f"{name} {code}"},
            {"label": "成本价", "value": f"{cost_price:.2f}"},
            {"label": "现价", "value": f"{current_price:.2f}"},
            {"label": "风险价", "value": f"{risk_price:.2f}"},
            {"label": "压力位", "value": f"{pressure_price:.2f}"},
        ],
        "actions": actions,
    }


def _extract_price(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _console_summary_and_actions(title: str, text: str) -> tuple[list[dict], list[dict]]:
    code_match = re.search(r"\((\d{6})\)", text) or re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    code = code_match.group(1) if code_match else ""
    action = "观察"
    if re.search(r"\bBUY\b|推荐买入|建议买入|可关注机会|买入", text):
        action = "机会观察"
    if re.search(r"\bSELL\b|推荐卖出|建议卖出|减仓|离场|风险变高", text):
        action = "风险复核"
    if "HOLD" in text or "暂不操作" in text:
        action = "继续观察"

    confidence = _extract_price(text, [r"置信度\s*([0-9]+%)"])
    buy_price = _extract_price(text, [r"参考买入价\s*([0-9]+(?:\.[0-9]+)?)", r"买入\s*([0-9]+(?:\.[0-9]+)?)\s*元"])
    stop_loss = _extract_price(text, [r"止损\s*([0-9]+(?:\.[0-9]+)?)", r"跌破\s*([0-9]+(?:\.[0-9]+)?)\s*止损"])
    target = _extract_price(text, [r"目标卖出\s*([0-9]+(?:\.[0-9]+)?)", r"目标\s*([0-9]+(?:\.[0-9]+)?)"])
    support = _extract_price(text, [r"支撑\s*([0-9]+(?:\.[0-9]+)?)"])
    resistance = _extract_price(text, [r"压力\s*([0-9]+(?:\.[0-9]+)?)"])
    current = _extract_price(text, [r"现价：\s*([0-9]+(?:\.[0-9]+)?)", r"当前价：\s*([0-9]+(?:\.[0-9]+)?)"])

    summary = [{"label": "提醒类型", "value": action}]
    if code:
        summary.append({"label": "标的", "value": code})
    if confidence:
        summary.append({"label": "置信度", "value": confidence})
    for label, value in [
        ("现价", current),
        ("观察价", buy_price),
        ("风险价", stop_loss),
        ("压力位", target),
        ("支撑", support),
        ("压力", resistance),
    ]:
        if value:
            summary.append({"label": label, "value": value})

    actions = []
    track_price = buy_price or current
    if code and track_price and action in {"机会观察", "继续观察"}:
        actions.append({
            "label": "按观察价跟踪",
            "params": {"console_action": "track_position", "code": code, "price": track_price},
        })
    alert_price = support or stop_loss or buy_price
    if code and alert_price:
        actions.append({
            "label": "跌到关键价提醒",
            "params": {"console_action": "price_alert", "code": code, "price": alert_price, "direction": "below"},
        })
    if code:
        actions.append({
            "label": "停止跟踪这只",
            "params": {"console_action": "close_position", "code": code},
        })
    return summary[:8], actions[:3]


def _run_web_command(text: str, storage: Storage) -> dict:
    text = (text or "").strip()
    if not text:
        return _error_result("请输入问题或控制命令")
    try:
        from bot.parser import help_lines, parse_command
        from bot.service import BotService
        from push.feishu import render_text_card

        service = BotService(storage)
        command = parse_command(text)
        if command.action == "help":
            return _card_to_result(render_text_card("Web 控制台帮助", help_lines()))
        if command.action == "query":
            return _card_to_result(service.query_stock(command.code))
        if command.action == "research":
            if not command.code and _is_market_question(command.text or text):
                return _market_snapshot_result(service)
            return _card_to_result(service.research_stock(command.text or text, command.code))
        if command.action == "buy":
            if not command.price:
                return _error_result("持仓跟踪需要成本价，例如：买入 600519 1680")
            return _card_to_result(service.open_position(WEB_USER_ID, WEB_CHAT_ID, command.code, command.price, command.quantity))
        if command.action == "sell":
            return _card_to_result(service.close_position(WEB_USER_ID, command.code))
        if command.action == "price_alert":
            if not command.price:
                return _error_result("盯价需要关键价，例如：盯买 600519 1500")
            return _card_to_result(service.open_price_alert(WEB_USER_ID, WEB_CHAT_ID, command.code, command.price, command.quantity))
        if command.action == "cancel_price_alert":
            return _card_to_result(service.cancel_price_alert(WEB_USER_ID, command.code))
        return _error_result("暂不支持这个命令")
    except Exception as exc:
        return _error_result(str(exc))


def _run_web_action(params: dict[str, list[str]], storage: Storage) -> dict:
    try:
        action = _first(params, "console_action")
        if action == "risk_plan":
            return _risk_plan_result(params, storage)
        if action == "track_position":
            command = " ".join(filter(None, [
                "买入",
                _first(params, "code"),
                _first(params, "price"),
                _first(params, "quantity"),
            ]))
            return _run_web_command(command, storage)
        if action == "price_alert":
            from bot.service import BotService

            direction = _first(params, "direction", "below")
            if direction not in {"below", "above"}:
                direction = "below"
            code = _stock_code(_first(params, "code"))
            price = _positive_float(_first(params, "price"), "关键价")
            quantity = _positive_float(_first(params, "quantity"), "数量", required=False)
            service = BotService(storage)
            return _card_to_result(service.open_price_alert(
                WEB_USER_ID, WEB_CHAT_ID, code, price, quantity, direction=direction,
            ))
        if action == "close_position":
            return _run_web_command(f"停止跟踪 {_first(params, 'code')}", storage)
        if action == "cancel_price_alert":
            return _run_web_command(f"取消盯价 {_first(params, 'code')}", storage)
        return _run_web_command(_first(params, "question"), storage)
    except Exception as exc:
        return _error_result(str(exc))


def load_dashboard_data(storage: Storage) -> dict:
    with storage._conn() as conn:
        runs = _rows(conn, """
            SELECT run_id, run_ts, stocks_analyzed, llm_calls, tokens_used, pushed_count, pushed_ok
            FROM runs ORDER BY run_ts DESC LIMIT 12
        """)
        decisions = _rows(conn, """
            SELECT id, run_ts, code, name, action, confidence, target_price, stop_loss, one_liner, pushed
            FROM decisions ORDER BY run_ts DESC, id DESC LIMIT 40
        """)
        positions = _rows(conn, """
            SELECT code, name, buy_price, quantity, stop_loss, target_price, opened_at, last_notified_at
            FROM tracked_positions WHERE status='active' ORDER BY opened_at DESC LIMIT 40
        """)
        price_alerts = _rows(conn, """
            SELECT code, name, trigger_price, direction, quantity, created_at, last_notified_at
            FROM price_alerts WHERE status='active' ORDER BY created_at DESC LIMIT 40
        """)
    report = build_report(storage, horizon=5, limit=500)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
        "decisions": decisions,
        "positions": positions,
        "price_alerts": price_alerts,
        "feedback": storage.get_recent_alert_feedback(20),
        "report": report,
    }


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _date(value) -> str:
    return _e(str(value or "")[:16].replace("T", " "))


def _pct(value) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "-"


def _price(value) -> str:
    try:
        num = float(value or 0)
    except (TypeError, ValueError):
        num = 0
    return f"{num:.2f}" if num > 0 else "-"


def _stat(value, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _action_label(action: str) -> str:
    labels = {
        "BUY": "机会观察",
        "SELL": "风险复核",
        "HOLD": "继续观察",
    }
    return labels.get(str(action or "").upper(), str(action or "-"))


def _summary_cards(data: dict) -> str:
    latest_run = data["runs"][0] if data["runs"] else {}
    report = data.get("report", {})
    overall = report.get("overall", {})
    cards = [
        ("最近分析", _date(latest_run.get("run_ts")) if latest_run else "暂无"),
        ("分析股票", _stat(latest_run.get("stocks_analyzed"))),
        ("活跃持仓", _stat(len(data.get("positions", [])))),
        ("盯价提醒", _stat(len(data.get("price_alerts", [])))),
        ("5日样本", _stat(report.get("resolved_samples"))),
        ("5日命中率", _pct(overall.get("hit_rate"))),
    ]
    return "".join(
        f"<div class='metric'><span>{_e(label)}</span><strong>{_e(value)}</strong></div>"
        for label, value in cards
    )


def _feedback_buttons(source: str, code: str, source_id) -> str:
    buttons = []
    for label in ("有用", "误报", "看不懂"):
        buttons.append(
            "<button type='submit' class='tiny secondary' name='label' "
            f"value='{_e(label)}'>{_e(label)}</button>"
        )
    return (
        "<form method='post' action='/feedback' class='feedback-buttons'>"
        f"<input type='hidden' name='source' value='{_e(source)}'>"
        f"<input type='hidden' name='source_id' value='{_e(source_id)}'>"
        f"<input type='hidden' name='code' value='{_e(code)}'>"
        f"{''.join(buttons)}"
        "</form>"
    )


def _decision_cards(rows: list[dict]) -> str:
    cards = []
    for row in rows[:8]:
        cards.append(
            "<article class='decision-card'>"
            f"<div><code>{_e(row.get('code'))}</code><span class='pill {str(row.get('action', '')).lower()}'>{_e(_action_label(row.get('action')))}</span></div>"
            f"<h3>{_e(row.get('name') or row.get('code'))}</h3>"
            f"<p>{_e(row.get('one_liner') or '继续观察')}</p>"
            f"<dl><div><dt>置信度</dt><dd>{_pct(row.get('confidence'))}</dd></div>"
            f"<div><dt>压力位</dt><dd>{_price(row.get('target_price'))}</dd></div>"
            f"<div><dt>风险价</dt><dd>{_price(row.get('stop_loss'))}</dd></div></dl>"
            f"{_feedback_buttons('decision', row.get('code', ''), row.get('id', ''))}"
            "</article>"
        )
    if not cards:
        return ""
    return f"<section class='mobile-cards'><h2>今日提醒</h2><div class='decision-card-list'>{''.join(cards)}</div></section>"


def _runs_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_date(row.get('run_ts'))}</td>"
        f"<td><code>{_e(row.get('run_id'))}</code></td>"
        f"<td>{_e(row.get('stocks_analyzed'))}</td>"
        f"<td>{_e(row.get('llm_calls'))}</td>"
        f"<td>{_e(row.get('tokens_used'))}</td>"
        f"<td>{_e(row.get('pushed_count'))}</td>"
        f"<td>{'成功' if row.get('pushed_ok') else '-'}</td>"
        "</tr>"
        for row in rows
    )
    return _table("运行记录", ["时间", "Run ID", "股票", "LLM", "Tokens", "推送", "状态"], body)


def _decisions_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_date(row.get('run_ts'))}</td>"
        f"<td><code>{_e(row.get('code'))}</code></td>"
        f"<td>{_e(row.get('name'))}</td>"
        f"<td><span class='pill {str(row.get('action', '')).lower()}'>{_e(_action_label(row.get('action')))}</span></td>"
        f"<td>{_pct(row.get('confidence'))}</td>"
        f"<td>{_price(row.get('target_price'))}</td>"
        f"<td>{_price(row.get('stop_loss'))}</td>"
        f"<td>{_e(row.get('one_liner'))}</td>"
        f"<td>{_feedback_buttons('decision', row.get('code', ''), row.get('id', ''))}</td>"
        "</tr>"
        for row in rows
    )
    return _table("最近提醒", ["时间", "代码", "名称", "类型", "置信度", "压力位", "风险价", "一句话", "反馈"], body)


def _positions_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td><code>{_e(row.get('code'))}</code></td>"
        f"<td>{_e(row.get('name'))}</td>"
        f"<td>{_price(row.get('buy_price'))}</td>"
        f"<td>{_e(row.get('quantity') or '-')}</td>"
        f"<td>{_price(row.get('stop_loss'))}</td>"
        f"<td>{_price(row.get('target_price'))}</td>"
        f"<td>{_date(row.get('opened_at'))}</td>"
        "</tr>"
        for row in rows
    )
    return _table("持仓跟踪", ["代码", "名称", "成本价", "数量", "风险价", "压力位", "开始时间"], body)


def _direction_label(direction: str) -> str:
    return "涨到提醒" if direction == "above" else "跌到提醒"


def _alerts_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td><code>{_e(row.get('code'))}</code></td>"
        f"<td>{_e(row.get('name'))}</td>"
        f"<td>{_price(row.get('trigger_price'))}</td>"
        f"<td>{_e(_direction_label(row.get('direction')))}</td>"
        f"<td>{_e(row.get('quantity') or '-')}</td>"
        f"<td>{_date(row.get('created_at'))}</td>"
        "</tr>"
        for row in rows
    )
    return _table("盯价提醒", ["代码", "名称", "触发价", "方向", "数量", "创建时间"], body)


def _report_table(report: dict) -> str:
    rows = report.get("by_action", {})
    body = "".join(
        "<tr>"
        f"<td>{_e(_action_label(action))}</td>"
        f"<td>{_e(stats.get('count', 0))}</td>"
        f"<td>{_pct(stats.get('hit_rate'))}</td>"
        f"<td>{float(stats.get('avg_return') or 0):+.2f}%</td>"
        f"<td>{float(stats.get('median_return') or 0):+.2f}%</td>"
        "</tr>"
        for action, stats in rows.items()
    )
    return _table("5日提醒复盘", ["类型", "样本", "命中率", "平均收益", "中位收益"], body)


def _feedback_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td>{_date(row.get('created_at'))}</td>"
        f"<td><code>{_e(row.get('code'))}</code></td>"
        f"<td>{_e(row.get('label'))}</td>"
        f"<td>{_e(row.get('source'))}</td>"
        "</tr>"
        for row in rows
    )
    return _table("最近反馈", ["时间", "代码", "反馈", "来源"], body)


def _table(title: str, headers: list[str], body: str) -> str:
    header = "".join(f"<th>{_e(item)}</th>" for item in headers)
    empty = f"<tr><td colspan='{len(headers)}' class='empty'>暂无数据</td></tr>"
    return (
        f"<section><h2>{_e(title)}</h2><div class='table-wrap'><table>"
        f"<thead><tr>{header}</tr></thead><tbody>{body or empty}</tbody>"
        "</table></div></section>"
    )


def _console_result_html(result: dict | None) -> str:
    if not result:
        return """
        <section id="console-result" class="answer-card empty-answer">
          <h2>回答</h2>
          <p>提问后，回答会直接显示在这里。</p>
        </section>
        """
    cls = "ok" if result.get("ok") else "error"
    summary = "".join(
        f"<div class='summary-chip'><span>{_e(item.get('label'))}</span><strong>{_e(item.get('value'))}</strong></div>"
        for item in result.get("summary", [])
    )
    actions = "".join(
        "<button type='button' class='secondary action-button' "
        f"data-action='{_e(json.dumps(action.get('params', {}), ensure_ascii=False))}'>{_e(action.get('label'))}</button>"
        for action in result.get("actions", [])
    )
    summary_html = f"<div class='summary-grid'>{summary}</div>" if summary else ""
    actions_html = f"<div class='suggested-actions'>{actions}</div>" if actions else ""
    return (
        f"<section id='console-result' class='answer-card {cls}'>"
        f"<h2>{_e(result.get('title') or '回答')}</h2>"
        f"{summary_html}<div class='result-text'>{_e(result.get('text'))}</div>{actions_html}"
        "</section>"
    )


def _select(name: str, current: str, options: list[tuple[str, str]]) -> str:
    items = []
    for value, label in options:
        selected = " selected" if value == current else ""
        items.append(f"<option value='{_e(value)}'{selected}>{_e(label)}</option>")
    return f"<select name='{_e(name)}'>{''.join(items)}</select>"


def _checked(value: str) -> str:
    return " checked" if str(value).strip().lower() in {"1", "true", "yes", "on"} else ""


def _mask_secret(value: str) -> str:
    value = str(value or "")
    if not value:
        return "未配置"
    return "已配置"


def _watchlist_text(value: str) -> str:
    codes = re.findall(r"\d{6}", value or "")
    return "\n".join(codes) if codes else value


def _field(label: str, control: str, hint: str = "", wide: bool = False) -> str:
    hint_html = f"<div class='hint'>{_e(hint)}</div>" if hint else ""
    cls = "field wide" if wide else "field"
    return f"<label class='{cls}'><span>{_e(label)}</span>{control}{hint_html}</label>"


def _save_button(label: str = "保存设置") -> str:
    return f"<div class='actions'><button type='submit'>{_e(label)}</button></div>"


def _nav(active: str) -> str:
    items = [
        ("home", "/", "主页"),
        ("console", "/console", "AI 控制台"),
        ("watchlist", "/settings/watchlist", "自选股"),
        ("model", "/settings/model", "模型配置"),
        ("channels", "/settings/channels", "通知渠道"),
        ("remote", "/settings/remote", "远程访问"),
        ("features", "/settings/features", "功能开关"),
        ("personalization", "/settings/personalization", "个性化"),
        ("factors", "/settings/factors", "因子市场"),
    ]
    links = []
    for key, href, label in items:
        cls = "active" if key == active else ""
        current = " aria-current='page'" if key == active else ""
        links.append(f"<a class='{cls}' href='{href}'{current}>{_e(label)}</a>")
    links.append(f"<a href='{FEEDBACK_URL}' target='_blank' rel='noopener noreferrer'>反馈</a>")
    return "".join(links)


def _mobile_tabbar(active: str) -> str:
    items = [
        ("home", "/", "今日"),
        ("console", "/console", "问 AI"),
        ("watchlist", "/settings/watchlist", "自选"),
        ("factors", "/settings/factors", "因子"),
        ("features", "/settings/features", "设置"),
    ]
    links = []
    for key, href, label in items:
        cls = "active" if key == active else ""
        links.append(f"<a class='{cls}' href='{href}'>{_e(label)}</a>")
    return f"<div class='mobile-tabbar'>{''.join(links)}</div>"


def _onboarding_panel(data: dict, settings: dict[str, str]) -> str:
    watch_count = len(re.findall(r"\d{6}", settings.get("WATCHLIST", "")))
    base_url = settings.get("LLM_BASE_URL", "")
    model_ready = bool(settings.get("LLM_API_KEY")) or base_url.startswith(("http://127.0.0.1", "http://localhost"))
    channel = settings.get("NOTIFY_CHANNEL", "feishu")
    channel_ready = channel == "web" or all([
        settings.get("FEISHU_APP_ID"),
        settings.get("FEISHU_APP_SECRET"),
        settings.get("FEISHU_RECEIVE_ID"),
    ])
    levels_ready = bool(settings.get("ALERT_LEVELS"))
    watch_ready = bool(data.get("positions") or data.get("price_alerts"))

    # (label, done, hint, href, detail)
    steps = [
        (
            "第1步：添加自选股",
            watch_count > 0,
            f"当前 {watch_count} 只" if watch_count > 0 else "填入6位股票代码，逗号分隔",
            "/settings/watchlist",
            "约1分钟 · 这是系统监控的标的范围",
        ),
        (
            "第2步：配置 AI 模型",
            model_ready,
            "已配置" if model_ready else "推荐：免费 Ollama 本地模型 或 DeepSeek API",
            "/settings/model",
            "约3分钟 · 用于解读公告新闻、生成建议",
        ),
        (
            "第3步：选择通知方式",
            channel_ready,
            "已配置" if channel_ready else "推荐先选「仅 Web 控制台」，无需配飞书",
            "/settings/channels",
            "约1分钟 · 决定提醒发到哪里",
        ),
        (
            "第4步：设置打扰级别",
            levels_ready,
            "已配置" if levels_ready else "推荐只收「必须看」和「建议看」",
            "/settings/features",
            "约1分钟 · 控制哪些提醒推给你",
        ),
        (
            "第5步：添加持仓或关键价",
            watch_ready,
            "已添加" if watch_ready else "让系统围绕你真实关注的点提醒，而不是泛泛扫描",
            "/console",
            "按需 · 可随时在 AI 控制台里添加",
        ),
    ]
    rows = []
    for label, done, hint, href, detail in steps:
        cls = "step-row done" if done else "step-row"
        status = "✓" if done else "→"
        rows.append(
            f"<a class='{cls}' href='{href}'>"
            f"<span class='step-status'>{status}</span>"
            f"<div class='step-body'>"
            f"<strong>{_e(label)}</strong>"
            f"<span class='step-hint'>{_e(hint)}</span>"
            f"<span class='step-detail'>{_e(detail)}</span>"
            f"</div>"
            "</a>"
        )
    ready_count = sum(1 for _, done, _, _, _ in steps if done)
    all_done = ready_count == len(steps)
    done_msg = "全部完成！可以运行一次 <code>python main.py once</code> 或点下方「运行一次」试试。" if all_done else ""
    return f"""
    <section class="panel onboarding">
      <div>
        <h2>开始使用</h2>
        <p>按顺序完成这5步，大约10分钟，就能把"持续盯盘"变成"有事再看"。</p>
        {f"<p class='onboarding-done'>{done_msg}</p>" if done_msg else ""}
      </div>
      <div class="setup-progress">{ready_count}/{len(steps)} 已完成</div>
      <div class="step-list">{''.join(rows)}</div>
    </section>
    """


def _compliance_notice() -> str:
    return f"""
    <section class="feedback">
      <h2>反馈入口</h2>
      <p>遇到数据异常、提醒误报、配置卡住或想提新功能，可以到 <a href="{FEEDBACK_URL}" target="_blank" rel="noopener noreferrer">GitHub Issues</a> 留下问题。</p>
    </section>
    <section class="compliance">
      <h2>合规边界</h2>
      <p>StockWatch 是自选股盯盘提醒、公开信息聚合和持仓风险复核工具，不是证券投资咨询服务，也不是荐股软件。</p>
      <p>"机会观察""风险复核"等提示只表示值得进一步查看，不构成买入、卖出或收益承诺；请自行核验交易所公告、上市公司披露和券商行情。</p>
      <p>"风险价""观察压力位"只是用户确认后的提醒线，不代表止盈止损建议或自动交易指令。</p>
    </section>
    """


def _layout(active: str, title: str, subtitle: str, content: str,
            settings: dict[str, str], notice: str = "") -> str:
    notice_html = f"<div class='notice'>{_e(notice)}</div>" if notice else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="theme-color" content="#ffffff">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <meta name="apple-mobile-web-app-title" content="StockWatch">
  <link rel="manifest" href="/manifest.json">
  <link rel="icon" href="/icon.svg" type="image/svg+xml">
  <title>{_e(title)} · StockWatch</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f5f7;
      --text: #1d1d1f;
      --muted: #6e6e73;
      --line: #d6d6dc;
      --panel: #ffffff;
      --panel-soft: #fbfbfd;
      --green: #16845b;
      --red: #b9473f;
      --orange: #a76a18;
      --blue: #0a84ff;
      --nav: #2c2c2e;
      --nav-active: #eef5ff;
      --shadow: 0 18px 45px rgba(0, 0, 0, 0.07);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background:
        linear-gradient(180deg, #ffffff 0%, var(--bg) 240px);
    }}
    .shell {{ display: flex; min-height: 100vh; }}
    aside {{
      width: 224px;
      flex: 0 0 224px;
      padding: 20px 14px;
      border-right: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
    }}
    .brand {{ padding: 0 8px 18px; border-bottom: 1px solid var(--line); }}
    .brand strong {{ display: block; font-size: 18px; }}
    .brand span {{ display: block; margin-top: 4px; color: var(--muted); font-size: 12px; }}
    nav {{ display: grid; gap: 4px; margin-top: 16px; }}
    nav a {{
      display: block;
      padding: 10px 11px;
      border-radius: 8px;
      color: var(--nav);
      text-decoration: none;
      font-weight: 560;
    }}
    nav a:hover, nav a.active {{ background: var(--nav-active); color: #0057d9; }}
    .content {{ flex: 1; min-width: 0; }}
    header {{
      padding: 22px clamp(16px, 4vw, 42px) 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
    }}
    h1 {{ margin: 0; font-size: clamp(23px, 3vw, 32px); letter-spacing: 0; }}
    .sub {{ margin-top: 6px; color: var(--muted); }}
    main {{ padding: 20px clamp(16px, 4vw, 42px) 44px; }}
    .notice {{
      margin-bottom: 16px;
      padding: 10px 12px;
      border: 1px solid #b8dcc8;
      border-radius: 8px;
      background: #eef8f2;
      color: #17613f;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .metric {{ padding: 13px 14px; min-height: 76px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 20px; font-weight: 650; }}
    section {{ margin-top: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
    h3 {{ margin: 0; font-size: 16px; letter-spacing: 0; }}
    .panel {{ padding: 18px; max-width: 920px; }}
    .onboarding {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 14px;
      max-width: 1080px;
      margin-top: 0;
      margin-bottom: 20px;
    }}
    .onboarding p {{ margin: 0; color: var(--muted); }}
    .setup-progress {{
      align-self: start;
      padding: 6px 10px;
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .step-list {{
      grid-column: 1 / -1;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .step-row {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      min-height: 90px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--text);
      text-decoration: none;
      transition: border-color .15s;
    }}
    .step-row:hover {{ border-color: var(--blue); background: var(--panel); }}
    .step-body {{ display: flex; flex-direction: column; gap: 3px; flex: 1; }}
    .step-body strong {{ font-size: 14px; line-height: 1.3; }}
    .step-hint {{ color: var(--muted); font-size: 12px; }}
    .step-detail {{ color: var(--muted); font-size: 11px; margin-top: 2px; opacity: .7; }}
    .step-status {{
      flex-shrink: 0;
      width: 22px;
      height: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 50%;
      background: #fff4e5;
      color: var(--orange);
      font-size: 13px;
      font-weight: 700;
    }}
    .step-row.done .step-status {{
      background: #eef8f2;
      color: var(--green);
    }}
    .onboarding-done {{ color: var(--green) !important; font-weight: 600; margin-top: 6px !important; }}
    .feedback, .compliance {{
      max-width: 1080px;
      margin-top: 28px;
      padding: 14px 16px;
      border-radius: 8px;
    }}
    .feedback {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
    }}
    .feedback h2, .compliance h2 {{ margin-bottom: 6px; font-size: 15px; }}
    .feedback p, .compliance p {{ margin: 5px 0 0; font-size: 12px; }}
    .feedback p {{ color: var(--muted); }}
    .feedback a {{ color: var(--blue); font-weight: 650; }}
    .compliance {{
      border: 1px solid #f0d2a5;
      background: #fffaf2;
      color: #4b3a23;
    }}
    .compliance p {{ color: #6a5430; }}
    .form-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 14px; }}
    .field {{ display: grid; gap: 7px; }}
    .field.wide {{ grid-column: 1 / -1; }}
    .field span, .group-title {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    input, select, textarea {{
      width: 100%;
      min-height: 39px;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
      font-size: 16px;
    }}
    input:focus, select:focus, textarea:focus {{
      border-color: var(--blue);
      box-shadow: 0 0 0 3px rgba(10, 132, 255, 0.16);
      outline: none;
    }}
    textarea {{ min-height: 168px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .options {{ display: grid; gap: 10px; margin-top: 8px; }}
    .console-grid {{ display: grid; grid-template-columns: minmax(300px, 1.1fr) minmax(280px, 0.9fr); gap: 16px; align-items: start; }}
    .answer-card {{
      max-width: 1080px;
      margin-top: 18px;
      padding: 0;
      border: 0;
      border-radius: 8px;
      background: transparent;
      box-shadow: none;
    }}
    .answer-card h2 {{ margin-bottom: 12px; }}
    .empty-answer {{ color: var(--muted); }}
    .result-text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }}
    .answer-card.ok {{ border-color: #b8dcc8; }}
    .answer-card.error {{ border-color: #e6b7b0; }}
    .answer-card.error .result-text {{ background: #fff7f5; }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .summary-chip {{
      min-height: 62px;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }}
    .summary-chip span {{ display: block; color: var(--muted); font-size: 12px; }}
    .summary-chip strong {{ display: block; margin-top: 5px; font-size: 17px; }}
    .suggested-actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .mini-grid {{ display: grid; grid-template-columns: repeat(2, minmax(160px, 1fr)); gap: 10px; }}
    .mobile-cards {{ display: none; }}
    .decision-card-list {{ display: grid; gap: 10px; }}
    .decision-card {{
      display: grid;
      gap: 10px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    .decision-card > div:first-child {{ display: flex; align-items: center; justify-content: space-between; gap: 10px; }}
    .decision-card p {{ margin: 0; color: var(--muted); }}
    .decision-card dl {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 0; }}
    .decision-card dl div {{ padding: 8px; border-radius: 8px; background: var(--panel-soft); }}
    .decision-card dt {{ color: var(--muted); font-size: 12px; }}
    .decision-card dd {{ margin: 3px 0 0; font-weight: 650; }}
    .note-list {{ margin: 8px 0 0; padding-left: 18px; color: var(--muted); }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; overflow-wrap: anywhere; }}
    .option-row, .segment-row {{
      display: flex;
      align-items: center;
      gap: 9px;
      min-height: 42px;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }}
    .option-row input, .segment-row input {{ width: auto; min-height: auto; }}
    .segments {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 10px; }}
    .factor-market {{ max-width: 1080px; }}
    .factor-filters {{
      display: grid;
      grid-template-columns: minmax(220px, 1.2fr) repeat(3, minmax(130px, 0.7fr)) auto;
      gap: 10px;
      align-items: end;
      margin-bottom: 14px;
    }}
    .factor-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .factor-card {{
      display: grid;
      gap: 10px;
      min-height: 260px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
    }}
    .factor-card-top {{ display: flex; justify-content: space-between; gap: 8px; align-items: center; }}
    .factor-source, .factor-state, .factor-tags span {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #fff;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}
    .factor-state.enabled {{ background: #eef8f2; color: var(--green); }}
    .factor-state.local {{ background: #eef5ff; color: #0057d9; }}
    .factor-card p {{ margin: 0; color: var(--muted); }}
    .factor-tags {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .factor-card summary {{ cursor: pointer; font-weight: 650; }}
    .market-empty {{ padding: 24px; border: 1px solid var(--line); border-radius: 8px; }}
    .actions {{ margin-top: 16px; }}
    button {{
      min-height: 40px;
      padding: 0 16px;
      border: 0;
      border-radius: 8px;
      background: #1d1d1f;
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }}
    button:hover {{ background: #3a3a3c; }}
    button:disabled {{ cursor: wait; opacity: 0.64; }}
    button.secondary {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
    }}
    button.secondary:hover {{ background: var(--panel-soft); }}
    button.tiny {{
      min-height: 28px;
      padding: 0 9px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
    }}
    .feedback-buttons {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .mobile-tabbar {{ display: none; }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; background: var(--panel-soft); }}
    td code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    tr:last-child td {{ border-bottom: 0; }}
    .pill {{
      display: inline-block;
      min-width: 48px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 12px;
      text-align: center;
      color: #fff;
      background: var(--blue);
    }}
    .pill.buy {{ background: var(--red); }}
    .pill.sell {{ background: var(--orange); }}
    .pill.hold {{ background: var(--blue); }}
    .empty {{ color: var(--muted); text-align: center; }}
    footer {{ margin-top: 22px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 820px) {{
      .shell {{ display: block; }}
      aside {{ width: auto; border-right: 0; border-bottom: 1px solid var(--line); padding: 14px 12px; }}
      .brand {{ padding-bottom: 12px; }}
      nav {{
        display: flex;
        gap: 8px;
        overflow-x: auto;
        padding-bottom: 2px;
        -webkit-overflow-scrolling: touch;
      }}
      nav a {{ flex: 0 0 auto; min-width: 88px; text-align: center; }}
      header {{ padding-top: 18px; }}
      main {{ padding-bottom: 28px; }}
      .panel {{ max-width: none; }}
      .onboarding {{ grid-template-columns: 1fr; }}
      .setup-progress {{ width: fit-content; }}
      .form-grid, .segments, .console-grid, .mini-grid {{ grid-template-columns: 1fr; }}
      .factor-filters {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      button {{ width: 100%; }}
      button.tiny {{ width: auto; }}
      .suggested-actions {{ display: grid; grid-template-columns: 1fr; }}
      .mobile-cards {{ display: block; }}
      .mobile-tabbar {{
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 20;
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 4px;
        padding: 7px 8px calc(7px + env(safe-area-inset-bottom));
        border-top: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.96);
      }}
      .mobile-tabbar a {{
        min-height: 42px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        color: var(--muted);
        text-decoration: none;
        font-size: 12px;
        font-weight: 650;
      }}
      .mobile-tabbar a.active {{ background: var(--nav-active); color: #0057d9; }}
      main {{ padding-bottom: calc(96px + env(safe-area-inset-bottom)); }}
    }}
    @media (max-width: 480px) {{
      .metrics {{ grid-template-columns: 1fr; }}
      .step-list {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand"><strong>StockWatch</strong><span>{_e(settings.get('LLM_MODEL', ''))}</span></div>
      <nav>{_nav(active)}</nav>
    </aside>
    <div class="content">
      <header>
        <h1>{_e(title)}</h1>
        <div class="sub">{_e(subtitle)}</div>
      </header>
      <main>
        {notice_html}
        {content}
        {_compliance_notice()}
      </main>
    </div>
  </div>
  {_mobile_tabbar(active)}
  <script>
  (() => {{
    const resultBox = document.getElementById("console-result");
    if (!resultBox) return;

    const escapeHtml = (value) => String(value || "").replace(/[&<>"']/g, (ch) => ({{
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    }}[ch]));

    const renderResult = (payload) => {{
      const ok = payload.ok !== false;
      const summary = (payload.summary || []).map((item) => `
        <div class="summary-chip"><span>${{escapeHtml(item.label)}}</span><strong>${{escapeHtml(item.value)}}</strong></div>
      `).join("");
      const actions = (payload.actions || []).map((action) => `
        <button type="button" class="secondary action-button" data-action='${{escapeHtml(JSON.stringify(action.params || {{}}))}}'>${{escapeHtml(action.label)}}</button>
      `).join("");
      resultBox.className = `answer-card ${{ok ? "ok" : "error"}}`;
      resultBox.innerHTML = `
        <h2>${{escapeHtml(payload.title || "回答")}}</h2>
        ${{summary ? `<div class="summary-grid">${{summary}}</div>` : ""}}
        <div class="result-text">${{escapeHtml(payload.text || "")}}</div>
        ${{actions ? `<div class="suggested-actions">${{actions}}</div>` : ""}}
      `;
    }};

    const setLoading = (message) => {{
      resultBox.className = "answer-card ok";
      resultBox.innerHTML = `
        <h2>正在分析</h2>
        <div class="result-text">${{escapeHtml(message || "正在获取行情、公告和模型回答，页面仍可继续操作。")}}</div>
      `;
    }};

    const postConsole = async (params, button) => {{
      const originalText = button ? button.textContent : "";
      if (button) {{
        button.disabled = true;
        button.textContent = "处理中";
      }}
      setLoading();
      try {{
        const response = await fetch("/api/console", {{
          method: "POST",
          headers: {{"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}},
          body: new URLSearchParams(params),
        }});
        const payload = await response.json();
        renderResult(payload);
      }} catch (error) {{
        renderResult({{ok: false, title: "操作失败", text: String(error), summary: [], actions: []}});
      }} finally {{
        if (button) {{
          button.disabled = false;
          button.textContent = originalText;
        }}
      }}
    }};

    document.querySelectorAll("form[data-console-form]").forEach((form) => {{
      form.addEventListener("submit", (event) => {{
        event.preventDefault();
        const button = event.submitter || form.querySelector("button[type=submit]");
        postConsole(new FormData(form), button);
      }});
    }});

    resultBox.addEventListener("click", (event) => {{
      const button = event.target.closest(".action-button");
      if (!button) return;
      try {{
        postConsole(JSON.parse(button.dataset.action || "{{}}"), button);
      }} catch (error) {{
        renderResult({{ok: false, title: "操作失败", text: "操作解析失败", summary: [], actions: []}});
      }}
    }});
  }})();
  if ("serviceWorker" in navigator) {{
    navigator.serviceWorker.register("/service-worker.js").catch(() => {{}});
  }}
  </script>
</body>
</html>"""


def render_home(data: dict, settings: dict[str, str], notice: str = "") -> str:
    content = f"""
    {_onboarding_panel(data, settings)}
    <div class="metrics">{_summary_cards(data)}</div>
    {_decision_cards(data['decisions'])}
    {_report_table(data['report'])}
    {_decisions_table(data['decisions'])}
    {_positions_table(data['positions'])}
    {_alerts_table(data['price_alerts'])}
    {_feedback_table(data['feedback'])}
    {_runs_table(data['runs'])}
    <footer>仅供研究复盘，不构成投资建议。数据来自本地 SQLite 记录。</footer>
    """
    return _layout("home", "主页", f"更新时间 {_e(data['generated_at'])}", content, settings, notice)


def render_console(storage: Storage, settings: dict[str, str], result: dict | None = None,
                   question: str = "", notice: str = "") -> str:
    data = load_dashboard_data(storage)
    result_html = _console_result_html(result)
    content = f"""
    <div class="console-grid">
      <section class="panel">
        <h2>直接提问</h2>
        <form method="post" action="/console" data-console-form>
          {_field("问题或命令", f"<textarea name='question' placeholder='例如：现在行情怎么样 / 600519 最近怎么样 / 盯买 600519 1500'>{_e(question)}</textarea>", "也可以输入：买入 600519 1680，用成本价开始持仓跟踪", wide=True)}
          {_save_button("发送")}
        </form>
        {result_html}
      </section>
      <section class="panel">
        <h2>盯盘控制</h2>
        <form method="post" action="/console" class="options" data-console-form>
          <input type="hidden" name="console_action" value="risk_plan">
          <div class="mini-grid">
            <input name="code" placeholder="代码 600519">
            <input name="cost_price" placeholder="成本价">
            <input name="quantity" placeholder="数量，可选">
          </div>
          <button type="submit">生成参考风险线</button>
          <div class="hint">基于成本价、近 20 日支撑压力和 ATR 生成候选提醒位，需你确认后保存。</div>
        </form>
        <form method="post" action="/console" class="options" data-console-form>
          <input type="hidden" name="console_action" value="track_position">
          <div class="mini-grid">
            <input name="code" placeholder="代码 600519">
            <input name="price" placeholder="成本价">
            <input name="quantity" placeholder="数量，可选">
          </div>
          <button type="submit">开始持仓跟踪</button>
        </form>
        <form method="post" action="/console" class="options" data-console-form>
          <input type="hidden" name="console_action" value="price_alert">
          <div class="mini-grid">
            <input name="code" placeholder="代码 600519">
            <input name="price" placeholder="关键价">
            <input name="quantity" placeholder="数量，可选">
            <select name="direction">
              <option value="below">跌到提醒</option>
              <option value="above">涨到提醒</option>
            </select>
          </div>
          <button type="submit">新增盯价提醒</button>
        </form>
        <form method="post" action="/console" class="options" data-console-form>
          <input type="hidden" name="console_action" value="close_position">
          <input name="code" placeholder="代码 600519">
          <button type="submit">停止持仓跟踪</button>
        </form>
        <form method="post" action="/console" class="options" data-console-form>
          <input type="hidden" name="console_action" value="cancel_price_alert">
          <input name="code" placeholder="代码 600519">
          <button type="submit">取消盯价提醒</button>
        </form>
      </section>
    </div>
    {_positions_table(data['positions'])}
    {_alerts_table(data['price_alerts'])}
    """
    return _layout("console", "AI 控制台", "网页里直接问行情、查股票和控制盯盘", content, settings, notice)


def render_watchlist_settings(settings: dict[str, str], notice: str = "") -> str:
    content = f"""
    <section class="panel">
      <h2>自选股</h2>
      <form method="post" action="/settings/watchlist">
        <div class="form-grid">
          {_field("自选股代码", f"<textarea name='watchlist'>{_e(_watchlist_text(settings.get('WATCHLIST', '')))}</textarea>", "每行一个或逗号分隔，支持股票和 ETF 六位代码", wide=True)}
          {_field("单次最多分析", f"<input name='max_stocks_per_run' inputmode='numeric' value='{_e(settings.get('MAX_STOCKS_PER_RUN', '50'))}'>")}
          {_field("推送最低置信度", f"<input name='min_confidence_to_push' inputmode='decimal' value='{_e(settings.get('MIN_CONFIDENCE_TO_PUSH', '0.6'))}'>", "0 到 1 之间，例如 0.6")}
        </div>
        {_save_button()}
      </form>
    </section>
    """
    return _layout("watchlist", "自选股", "决定每天分析哪些标的", content, settings, notice)


def render_model_settings(settings: dict[str, str], notice: str = "") -> str:
    provider = settings.get("LLM_PROVIDER", "openai")
    provider_select = _select("llm_provider", provider, [
        ("openai", "OpenAI-compatible"),
        ("anthropic", "Anthropic"),
        ("custom", "自定义 OpenAI-compatible"),
        ("local", "本地 OpenAI-compatible"),
        ("minimax", "MiniMax 兼容"),
    ])
    # Quick-select presets for common free/low-cost models
    presets = [
        {
            "id": "ollama",
            "name": "🏠 Ollama 本地",
            "badge": "完全免费",
            "badge_cls": "green",
            "desc": "运行在自己电脑上，数据不出本机",
            "install": "需先安装 Ollama：<a href='https://ollama.com' target='_blank' rel='noopener'>ollama.com</a>，然后运行 <code>ollama pull qwen2.5:7b</code>",
            "provider": "openai",
            "base_url": "http://127.0.0.1:11434/v1",
            "model": "qwen2.5:7b",
            "api_key": "",
        },
        {
            "id": "deepseek",
            "name": "⚡ DeepSeek",
            "badge": "低价",
            "badge_cls": "blue",
            "desc": "国内直连，价格极低，质量不错",
            "install": "注册 <a href='https://platform.deepseek.com' target='_blank' rel='noopener'>platform.deepseek.com</a> 获取 API Key",
            "provider": "openai",
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "api_key": "",
        },
        {
            "id": "minimax",
            "name": "🇨🇳 MiniMax",
            "badge": "国内可用",
            "badge_cls": "blue",
            "desc": "国内服务，稳定，中文理解好",
            "install": "注册 <a href='https://www.minimaxi.com' target='_blank' rel='noopener'>minimaxi.com</a> 获取 API Key",
            "provider": "openai",
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.7",
            "api_key": "",
        },
        {
            "id": "claude",
            "name": "🌟 Claude",
            "badge": "质量最高",
            "badge_cls": "orange",
            "desc": "Anthropic 出品，金融文本理解最强",
            "install": "注册 <a href='https://console.anthropic.com' target='_blank' rel='noopener'>console.anthropic.com</a> 获取 API Key",
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet-4-6",
            "api_key": "",
        },
    ]
    preset_cards = []
    for p in presets:
        badge_color = {"green": "#2db55d", "blue": "#2563eb", "orange": "#e8930a"}.get(p["badge_cls"], "#888")
        preset_cards.append(f"""
        <div class="preset-card" onclick="applyPreset({_e(p['provider'])!r},{_e(p['base_url'])!r},{_e(p['model'])!r})">
          <div class="preset-header">
            <strong>{p['name']}</strong>
            <span class="preset-badge" style="background:{badge_color}20;color:{badge_color}">{p['badge']}</span>
          </div>
          <div class="preset-desc">{p['desc']}</div>
          <div class="preset-install">{p['install']}</div>
        </div>
        """)

    content = f"""
    <section class="panel">
      <h2>推荐配置（点击自动填入）</h2>
      <p style="margin:0 0 12px;color:var(--muted);font-size:13px;">选一个适合你的方案，填好 API Key 后点保存。</p>
      <div class="preset-grid">{''.join(preset_cards)}</div>
    </section>
    <section class="panel" style="margin-top:16px">
      <h2>手动配置</h2>
      <form method="post" action="/settings/model">
        <div class="form-grid">
          {_field("模型提供商", provider_select)}
          {_field("模型代号", f"<input id='f-model' name='llm_model' value='{_e(settings.get('LLM_MODEL', ''))}' placeholder='MiniMax-M2.7'>")}
          {_field("接口地址", f"<input id='f-base-url' name='llm_base_url' value='{_e(settings.get('LLM_BASE_URL', ''))}' placeholder='https://api.example.com/v1'>", "Anthropic 可用 https://api.anthropic.com；本地模型可用 http://127.0.0.1:11434/v1")}
          {_field("API Key", f"<input id='f-api-key' type='password' name='llm_api_key' placeholder='留空不修改，当前：{_e(_mask_secret(settings.get('LLM_API_KEY', '')))}'>", "本地 OpenAI-compatible 服务通常可以留空")}
        </div>
        {_save_button()}
      </form>
    </section>
    <style>
    .preset-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}}
    .preset-card{{padding:14px;border:1px solid var(--line);border-radius:8px;cursor:pointer;background:var(--panel-soft);transition:border-color .15s}}
    .preset-card:hover{{border-color:var(--blue);background:var(--panel)}}
    .preset-header{{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-bottom:6px}}
    .preset-header strong{{font-size:14px}}
    .preset-badge{{font-size:11px;padding:2px 7px;border-radius:999px;font-weight:600;white-space:nowrap}}
    .preset-desc{{font-size:12px;color:var(--muted);margin-bottom:6px}}
    .preset-install{{font-size:11px;color:var(--muted)}}
    .preset-install a{{color:var(--blue)}}
    .preset-install code{{background:var(--panel-soft);padding:1px 4px;border-radius:3px;font-size:11px}}
    </style>
    <script>
    function applyPreset(provider, baseUrl, model) {{
      var sel = document.querySelector('select[name="llm_provider"]');
      if (sel) sel.value = provider;
      var bu = document.getElementById('f-base-url');
      if (bu) bu.value = baseUrl;
      var m = document.getElementById('f-model');
      if (m) m.value = model;
      document.getElementById('f-api-key').placeholder = '请在此输入该服务的 API Key';
      document.getElementById('f-api-key').focus();
    }}
    </script>
    """
    return _layout("model", "模型配置", "选一个免费方案或填入自己的 API Key", content, settings, notice)


def render_channel_settings(settings: dict[str, str], notice: str = "") -> str:
    channel_select = _select("notify_channel", settings.get("NOTIFY_CHANNEL", "feishu"), [
        ("feishu", "飞书/Lark"),
        ("web", "仅 Web 控制台"),
    ])
    receive_type = _select("feishu_receive_id_type", settings.get("FEISHU_RECEIVE_ID_TYPE", "open_id"), [
        ("open_id", "open_id"),
        ("user_id", "user_id"),
        ("email", "email"),
        ("chat_id", "chat_id"),
    ])
    content = f"""
    <section class="panel">
      <h2>通知渠道</h2>
      <form method="post" action="/settings/channels">
        <div class="form-grid">
          {_field("当前渠道", channel_select, "以后可以在这里接企业微信、钉钉、Telegram 或自定义 Webhook")}
          {_field("接收 ID 类型", receive_type)}
          {_field("App ID", f"<input name='feishu_app_id' value='{_e(settings.get('FEISHU_APP_ID', ''))}' placeholder='cli_xxxxxxxxxx'>")}
          {_field("App Secret", f"<input type='password' name='feishu_app_secret' placeholder='留空不修改，当前：{_e(_mask_secret(settings.get('FEISHU_APP_SECRET', '')))}'>")}
          {_field("接收人 ID", f"<input name='feishu_receive_id' value='{_e(settings.get('FEISHU_RECEIVE_ID', ''))}' placeholder='open_id / user_id / email / chat_id'>")}
          {_field("备用接收人", f"<input name='feishu_receive_id_2' value='{_e(settings.get('FEISHU_RECEIVE_ID_2', ''))}' placeholder='可选'>")}
          {_field("事件订阅 Token", f"<input type='password' name='feishu_verification_token' placeholder='留空不修改，当前：{_e(_mask_secret(settings.get('FEISHU_VERIFICATION_TOKEN', '')))}'>")}
          {_field("事件订阅 Encrypt Key", f"<input type='password' name='feishu_encrypt_key' placeholder='留空不修改，当前：{_e(_mask_secret(settings.get('FEISHU_ENCRYPT_KEY', '')))}'>")}
        </div>
        {_save_button()}
      </form>
    </section>
    <section class="panel">
      <h2>飞书注意事项</h2>
      <ul class="note-list">
        <li>在飞书开放平台创建自建应用，保存 App ID 和 App Secret。</li>
        <li>发送卡片消息需要开通发送消息相关权限，例如 im:message:send_as_bot。</li>
        <li>如果要用飞书长连接机器人接收用户消息，需要配置事件订阅，并订阅接收消息事件。</li>
        <li>修改权限后需要发布/重新安装应用到目标企业或群聊。</li>
        <li><a href="https://open.feishu.cn/document/server-docs/im-v1/message/create?lang=zh-CN">飞书发送消息接口文档</a></li>
      </ul>
    </section>
    """
    return _layout("channels", "通知渠道", "配置飞书或切换为仅 Web 控制台", content, settings, notice)


def render_remote_settings(settings: dict[str, str], notice: str = "") -> str:
    token_state = _mask_secret(settings.get("WEB_AUTH_TOKEN", ""))
    content = f"""
    <section class="panel">
      <h2>远程访问保护</h2>
      <form method="post" action="/settings/remote">
        <div class="form-grid">
          {_field("Web 访问 Token", f"<input type='password' name='web_auth_token' placeholder='留空不修改，当前：{_e(token_state)}'>", "设置后访问 Web UI 需要先登录；也可用 ?token= 或 X-StockWatch-Token 访问 API", wide=True)}
        </div>
        <label class="option-row"><input type="checkbox" name="clear_web_auth_token" value="true">清空 Web 访问 Token</label>
        {_save_button("保存远程访问设置")}
      </form>
    </section>
    <section class="panel">
      <h2>监听地址</h2>
      <ul class="note-list">
        <li>默认本机访问：<code>python main.py dashboard</code>，只监听 <code>127.0.0.1:8765</code>。</li>
        <li>局域网手机访问：先设置 Web Token，再运行 <code>python main.py dashboard --host 0.0.0.0</code>。</li>
        <li>不要把没有 Token 的 Web UI 暴露到公网；配置页里可能有模型 Key、通知凭证和持仓成本。</li>
      </ul>
    </section>
    <section class="panel">
      <h2>出门在外访问</h2>
      <ul class="note-list">
        <li>自用优先：Tailscale Serve，把访问限制在自己的 tailnet 设备里。</li>
        <li>公开域名：Cloudflare Tunnel + Cloudflare Access，在 Web UI 前加身份验证。</li>
        <li>临时演示：ngrok 隧道，适合短时间测试，不建议长期裸奔。</li>
      </ul>
    </section>
    """
    return _layout("remote", "远程访问", "设置访问保护，并选择局域网或穿透方案", content, settings, notice)


def render_feature_settings(settings: dict[str, str], notice: str = "") -> str:
    levels = {item.strip() for item in settings.get("ALERT_LEVELS", "").split(",") if item.strip()}
    def level_checkbox(value: str, label: str) -> str:
        checked = " checked" if value in levels else ""
        return f"<label class='option-row'><input type='checkbox' name='alert_levels' value='{value}'{checked}>{_e(label)}</label>"

    content = f"""
    <section class="panel">
      <h2>功能开关</h2>
      <form method="post" action="/settings/features">
        <div class="group-title">异动提醒分级</div>
        <div class="options">
          {level_checkbox("critical", "必须看：止损、强风险、重大负面")}
          {level_checkbox("warning", "建议看：机会观察、风险复核、明显异动")}
          {level_checkbox("info", "普通提醒：观察、记录、正面信息")}
        </div>
        <div class="options">
          <label class="option-row"><input type="checkbox" name="enable_reassurance_mode" value="true"{_checked(settings.get('ENABLE_REASSURANCE_MODE', 'false'))}>安心模式</label>
          <label class="option-row"><input type="checkbox" name="enable_after_close_summary" value="true"{_checked(settings.get('ENABLE_AFTER_CLOSE_SUMMARY', 'false'))}>休市后"不用盯盘"总结</label>
          <label class="option-row"><input type="checkbox" name="enable_family_brief" value="true"{_checked(settings.get('ENABLE_FAMILY_BRIEF', 'false'))}>一句话家庭版提醒</label>
        </div>
        {_save_button()}
      </form>
    </section>
    """
    return _layout("features", "功能开关", "控制哪些提醒会打扰你", content, settings, notice)


def render_personalization_settings(settings: dict[str, str], notice: str = "") -> str:
    current = settings.get("AI_RESPONSE_STYLE", "balanced")
    def segment(value: str, label: str) -> str:
        checked = " checked" if value == current else ""
        return f"<label class='segment-row'><input type='radio' name='ai_response_style' value='{value}'{checked}>{_e(label)}</label>"

    content = f"""
    <section class="panel">
      <h2>个性化</h2>
      <form method="post" action="/settings/personalization">
        <div class="group-title">AI 回复长度</div>
        <div class="segments">
          {segment("concise", "精简")}
          {segment("balanced", "均衡")}
          {segment("detailed", "详细")}
          {segment("expert", "专家视角")}
        </div>
        {_save_button()}
      </form>
    </section>
    """
    return _layout("personalization", "个性化", "调整机器人回答方式", content, settings, notice)


def _option_values(items: list[dict], key: str) -> list[str]:
    return sorted({str(item.get(key) or "") for item in items if item.get(key)})


def _market_select(name: str, current: str, values: list[str], empty_label: str) -> str:
    options = [(value, value) for value in values]
    return _select(name, current, [("", empty_label), *options])


def _factor_cards(items: list[dict]) -> str:
    cards = []
    for item in items:
        enabled = bool(item.get("enabled"))
        status = "已启用" if enabled else "本地入库" if item.get("source") == "用户上传" else "未启用"
        status_cls = "enabled" if enabled else "local" if item.get("source") == "用户上传" else ""
        extra = ""
        if item.get("file"):
            extra = f"<div class='path'>{_e(item.get('file'))}</div>"
        executable_note = "可通过上方开关参与分析" if item.get("executable") else "已进入本地因子库，默认不执行上传代码"
        cards.append(f"""
        <article class="factor-card">
          <div class="factor-card-top">
            <span class="factor-source">{_e(item.get('source'))}</span>
            <span class="factor-state {status_cls}">{_e(status)}</span>
          </div>
          <h3>{_e(item.get('name'))}</h3>
          <p>{_e(item.get('description'))}</p>
          <div class="factor-tags">
            <span>{_e(item.get('category'))}</span>
            <span>{_e(item.get('horizon'))}</span>
            <span>{_e(item.get('license', '内置'))}</span>
          </div>
          <details>
            <summary>查看介绍和数据要求</summary>
            <p>{_e(executable_note)}</p>
            <p>数据要求：{_e(item.get('data_requirements'))}</p>
            {extra}
          </details>
        </article>
        """)
    if not cards:
        return "<div class='empty market-empty'>没有匹配的因子。</div>"
    return "".join(cards)


def render_factor_settings(settings: dict[str, str], notice: str = "", filters: dict[str, str] | None = None) -> str:
    filters = filters or {}
    full_catalog = load_factor_catalog(settings)
    catalog = load_factor_catalog(settings, filters)
    category_select = _market_select("category", filters.get("category", ""), _option_values(full_catalog, "category"), "全部类型")
    source_select = _market_select("source", filters.get("source", ""), _option_values(full_catalog, "source"), "全部来源")
    status_select = _select("status", filters.get("status", ""), [
        ("", "全部状态"),
        ("enabled", "只看已启用"),
        ("local", "只看本地上传"),
    ])
    content = f"""
    <section class="panel factor-market">
      <h2>因子市场</h2>
      <form method="get" action="/settings/factors" class="factor-filters">
        <input name="q" value="{_e(filters.get('q', ''))}" placeholder="搜索名称、分类、说明">
        {category_select}
        {source_select}
        {status_select}
        <button type="submit">筛选</button>
      </form>
      <div class="factor-grid">
        {_factor_cards(catalog)}
      </div>
    </section>
    <section class="panel">
      <h2>启用内置因子</h2>
      <form method="post" action="/settings/factors">
        <div class="options">
          <label class="option-row"><input type="checkbox" name="enable_regime" value="true"{_checked(settings.get('ENABLE_REGIME', 'false'))}>市场状态因子</label>
          <label class="option-row"><input type="checkbox" name="enable_sector" value="true"{_checked(settings.get('ENABLE_SECTOR', 'false'))}>板块强弱因子</label>
          <label class="option-row"><input type="checkbox" name="enable_alpha158" value="true"{_checked(settings.get('ENABLE_ALPHA158', 'false'))}>Alpha158 因子</label>
          <label class="option-row"><input type="checkbox" name="enable_calibration" value="true"{_checked(settings.get('ENABLE_CALIBRATION', 'false'))}>信号校准因子</label>
          <label class="option-row"><input type="checkbox" name="enable_lgbm" value="true"{_checked(settings.get('ENABLE_LGBM', 'false'))}>LightGBM 模型因子</label>
        </div>
        {_save_button()}
      </form>
    </section>
    <section class="panel">
      <h2>上传因子到本地库</h2>
      <form method="post" action="/settings/factors/upload" enctype="multipart/form-data">
        <div class="form-grid">
          {_field("因子名称", "<input name='factor_name' placeholder='例如：资金流反转因子'>")}
          {_field("分类", _select("factor_category", "自定义", [("技术面", "技术面"), ("资金流", "资金流"), ("消息面", "消息面"), ("基本面", "基本面"), ("风控", "风控"), ("板块轮动", "板块轮动"), ("自定义", "自定义")]))}
          {_field("适用周期", _select("factor_horizon", "自定义", [("日内", "日内"), ("1-5 日", "1-5 日"), ("5-20 日", "5-20 日"), ("中线", "中线"), ("复盘", "复盘"), ("自定义", "自定义")]))}
          {_field("开源协议", _select("factor_license", "MIT", [("MIT", "MIT"), ("Apache-2.0", "Apache-2.0"), ("BSD-3-Clause", "BSD-3-Clause"), ("Proprietary", "暂不公开")]))}
          {_field("数据要求", "<input name='factor_data_requirements' placeholder='例如：日 K、成交额、北向资金'>", wide=True)}
          {_field("说明", "<textarea name='factor_description' placeholder='简要说明输入数据、计算逻辑和适用场景'></textarea>", wide=True)}
          {_field("因子文件", "<input type='file' name='factor_file' accept='.py,.json,.md,.txt'>", "仅保存为本地贡献包，不会自动执行未审核代码", wide=True)}
        </div>
        <label class="option-row"><input type="checkbox" name="share_to_community" value="true">标记为可贡献给社区</label>
        {_save_button("上传因子")}
      </form>
    </section>
    <section class="panel">
      <h2>社区贡献</h2>
      <ul class="note-list">
        <li>上传文件会和分类、周期、说明一起进入本地因子库。</li>
        <li>准备贡献的因子可以从本地目录提交 PR，后续可做一键发布到社区市场。</li>
        <li>为了安全，Web UI 不会直接执行用户上传的 Python 因子代码。</li>
      </ul>
    </section>
    """
    return _layout("factors", "因子市场", "选择参与分析的增强因子", content, settings, notice)


def _first(params: dict[str, list[str]], name: str, default: str = "") -> str:
    values = params.get(name)
    return values[0] if values else default


def _safe_next(value: str) -> str:
    """只允许站内相对路径跳转，防止 ?next=https://evil.com 这类开放重定向钓鱼。"""
    if (
        value
        and value.startswith("/")
        and not value.startswith("//")
        and "://" not in value
        and "\\" not in value
    ):
        return value
    return "/"


def _bool_param(params: dict[str, list[str]], name: str) -> str:
    return "true" if _first(params, name) == "true" else "false"


def _clamp_float(value: str, default: float, min_value: float, max_value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    number = min(max(number, min_value), max_value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _clamp_int(value: str, default: int, min_value: int, max_value: int) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return str(min(max(number, min_value), max_value))


def _updates_for_route(route: str, params: dict[str, list[str]]) -> dict[str, str]:
    if route == "/settings/model":
        updates = {
            "LLM_PROVIDER": _first(params, "llm_provider", "openai"),
            "LLM_BASE_URL": _first(params, "llm_base_url"),
            "LLM_MODEL": _first(params, "llm_model"),
        }
        api_key = _first(params, "llm_api_key").strip()
        if api_key:
            updates["LLM_API_KEY"] = api_key
        return updates
    if route == "/settings/channels":
        channel = _first(params, "notify_channel", "feishu")
        if channel not in {"feishu", "web"}:
            channel = "feishu"
        updates = {
            "NOTIFY_CHANNEL": channel,
            "FEISHU_APP_ID": _first(params, "feishu_app_id"),
            "FEISHU_RECEIVE_ID": _first(params, "feishu_receive_id"),
            "FEISHU_RECEIVE_ID_TYPE": _first(params, "feishu_receive_id_type", "open_id"),
            "FEISHU_RECEIVE_ID_2": _first(params, "feishu_receive_id_2"),
        }
        secret_map = {
            "feishu_app_secret": "FEISHU_APP_SECRET",
            "feishu_verification_token": "FEISHU_VERIFICATION_TOKEN",
            "feishu_encrypt_key": "FEISHU_ENCRYPT_KEY",
        }
        for form_key, env_key in secret_map.items():
            value = _first(params, form_key).strip()
            if value:
                updates[env_key] = value
        return updates
    if route == "/settings/remote":
        updates = {}
        if _bool_param(params, "clear_web_auth_token") == "true":
            updates["WEB_AUTH_TOKEN"] = ""
        else:
            token = _first(params, "web_auth_token").strip()
            if token:
                updates["WEB_AUTH_TOKEN"] = token
        return updates
    if route == "/settings/watchlist":
        codes = re.findall(r"\d{6}", _first(params, "watchlist"))
        return {
            "WATCHLIST": ",".join(dict.fromkeys(codes)),
            "MAX_STOCKS_PER_RUN": _clamp_int(_first(params, "max_stocks_per_run"), 50, 1, 500),
            "MIN_CONFIDENCE_TO_PUSH": _clamp_float(_first(params, "min_confidence_to_push"), 0.6, 0, 1),
        }
    if route == "/settings/features":
        levels = [item for item in params.get("alert_levels", []) if item in {"critical", "warning", "info"}]
        return {
            "ALERT_LEVELS": ",".join(levels or ["critical"]),
            "ENABLE_REASSURANCE_MODE": _bool_param(params, "enable_reassurance_mode"),
            "ENABLE_AFTER_CLOSE_SUMMARY": _bool_param(params, "enable_after_close_summary"),
            "ENABLE_FAMILY_BRIEF": _bool_param(params, "enable_family_brief"),
        }
    if route == "/settings/personalization":
        style = _first(params, "ai_response_style", "balanced")
        if style not in {"concise", "balanced", "detailed", "expert"}:
            style = "balanced"
        return {"AI_RESPONSE_STYLE": style}
    if route == "/settings/factors":
        return {
            "ENABLE_REGIME": _bool_param(params, "enable_regime"),
            "ENABLE_SECTOR": _bool_param(params, "enable_sector"),
            "ENABLE_ALPHA158": _bool_param(params, "enable_alpha158"),
            "ENABLE_CALIBRATION": _bool_param(params, "enable_calibration"),
            "ENABLE_LGBM": _bool_param(params, "enable_lgbm"),
        }
    return {}


def _render_route(route: str, storage: Storage, notice: str = "", query_params: dict[str, list[str]] | None = None) -> str:
    settings = load_settings()
    query_params = query_params or {}
    if route == "/":
        return render_home(load_dashboard_data(storage), settings, notice)
    if route == "/console":
        return render_console(storage, settings, notice=notice)
    if route == "/settings/watchlist":
        return render_watchlist_settings(settings, notice)
    if route == "/settings/model":
        return render_model_settings(settings, notice)
    if route == "/settings/channels":
        return render_channel_settings(settings, notice)
    if route == "/settings/remote":
        return render_remote_settings(settings, notice)
    if route == "/settings/features":
        return render_feature_settings(settings, notice)
    if route == "/settings/personalization":
        return render_personalization_settings(settings, notice)
    if route == "/settings/factors":
        filters = {
            "q": _first(query_params, "q"),
            "category": _first(query_params, "category"),
            "source": _first(query_params, "source"),
            "status": _first(query_params, "status"),
        }
        return render_factor_settings(settings, notice, filters)
    raise KeyError(route)


def render_login(next_url: str = "/", error: str = "") -> str:
    error_html = f"<div class='error-box'>{_e(error)}</div>" if error else ""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>登录 · StockWatch</title>
  <style>
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f5f5f7; color: #1d1d1f; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .login {{ width: min(420px, calc(100vw - 32px)); padding: 22px; border: 1px solid #d6d6dc; border-radius: 8px; background: #fff; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    p {{ margin: 0 0 16px; color: #6e6e73; }}
    input {{ width: 100%; min-height: 42px; box-sizing: border-box; padding: 8px 10px; border: 1px solid #d6d6dc; border-radius: 8px; font: inherit; font-size: 16px; }}
    button {{ width: 100%; min-height: 42px; margin-top: 12px; border: 0; border-radius: 8px; background: #1d1d1f; color: #fff; font-weight: 650; }}
    .error-box {{ margin-bottom: 12px; padding: 10px 12px; border-radius: 8px; background: #fff4e5; color: #8a4f00; }}
  </style>
</head>
<body>
  <form class="login" method="post" action="/login">
    <h1>StockWatch</h1>
    <p>输入 Web 访问 Token 后继续。</p>
    {error_html}
    <input type="hidden" name="next" value="{_e(next_url or '/')}">
    <input type="password" name="token" placeholder="Web 访问 Token" autofocus>
    <button type="submit">登录</button>
  </form>
</body>
</html>"""


def manifest_json() -> bytes:
    return json.dumps({
        "name": "StockWatch",
        "short_name": "StockWatch",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#f5f5f7",
        "theme_color": "#ffffff",
        "description": "A 股自选股盯盘提醒和风险复核工具",
        "icons": [
            {"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any maskable"},
        ],
    }, ensure_ascii=False).encode("utf-8")


def service_worker_js() -> bytes:
    return b"""self.addEventListener('install', event => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));
"""


def icon_svg() -> bytes:
    return b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128">
<rect width="128" height="128" rx="24" fill="#1d1d1f"/>
<path d="M24 84h18l13-30 19 44 16-34h14" fill="none" stroke="#6ee7b7" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="92" cy="38" r="10" fill="#0a84ff"/>
</svg>"""


def _record_feedback(params: dict[str, list[str]], storage: Storage) -> str:
    label = _first(params, "label").strip()
    if label not in {"有用", "误报", "看不懂"}:
        raise ValueError("未知反馈类型")
    source = _first(params, "source", "decision")[:40]
    source_id = _first(params, "source_id")
    source = f"{source}:{source_id}" if source_id else source
    storage.insert_alert_feedback({
        "user_id": WEB_USER_ID,
        "source": source,
        "code": _first(params, "code")[:16],
        "label": label,
        "note": _first(params, "note")[:200],
    })
    return "反馈已记录，会用于后续提醒质量复盘。"


def make_handler(storage: Storage):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, body: bytes, content_type: str, status: int = 200, headers: dict[str, str] | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, location: str, headers: dict[str, str] | None = None):
            self.send_response(303)
            self.send_header("Location", location)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()

        def _require_auth(self, parsed) -> bool:
            params = parse_qs(parsed.query)
            if _is_authorized(self.headers, params):
                return True
            if parsed.path.startswith("/api"):
                body = json.dumps({"ok": False, "error": "需要登录"}, ensure_ascii=False).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8", status=401)
                return False
            next_url = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            self._redirect(f"/login?next={quote(next_url, safe='')}")
            return False

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/health":
                self._send_bytes(b"ok", "text/plain; charset=utf-8")
                return
            if route == "/manifest.json":
                self._send_bytes(manifest_json(), "application/manifest+json; charset=utf-8")
                return
            if route == "/service-worker.js":
                self._send_bytes(service_worker_js(), "text/javascript; charset=utf-8")
                return
            if route == "/icon.svg":
                self._send_bytes(icon_svg(), "image/svg+xml")
                return
            if route == "/login":
                params = parse_qs(parsed.query)
                next_url = _safe_next(_first(params, "next", "/") or "/")
                if not _web_auth_token():
                    self._redirect(next_url)
                    return
                if _web_auth_token() and _is_authorized(self.headers, params):
                    self._redirect(next_url)
                    return
                self._send_bytes(render_login(next_url).encode("utf-8"), "text/html; charset=utf-8")
                return
            if route == "/logout":
                self._redirect("/login", {"Set-Cookie": f"{AUTH_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"})
                return
            if route not in {
                "/",
                "/console",
                "/api.json",
                "/settings/watchlist",
                "/settings/model",
                "/settings/channels",
                "/settings/remote",
                "/settings/features",
                "/settings/personalization",
                "/settings/factors",
            }:
                self.send_error(404)
                return
            if not self._require_auth(parsed):
                return
            if route == "/api.json":
                data = load_dashboard_data(storage)
                body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                params = parse_qs(parsed.query)
                notice = ""
                if params.get("saved"):
                    notice = "设置已保存；已运行的守护进程或 Bot 需要重启后读取新配置。"
                if params.get("feedback"):
                    notice = "反馈已记录，会用于后续提醒质量复盘。"
                body = _render_route(route, storage, notice, params).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self._send_bytes(body, content_type)

        def do_POST(self):
            parsed = urlparse(self.path)
            route = parsed.path
            if route == "/login":
                length = int(self.headers.get("Content-Length") or "0")
                raw = self.rfile.read(length)
                params, _ = _parse_form_data(self.headers, raw)
                token = _web_auth_token()
                supplied = _first(params, "token").strip()
                next_url = _safe_next(_first(params, "next", "/") or "/")
                if token and secrets.compare_digest(supplied, token):
                    self._redirect(next_url, {"Set-Cookie": f"{AUTH_COOKIE}={supplied}; Path=/; HttpOnly; SameSite=Lax"})
                    return
                body = render_login(next_url, "Token 不正确").encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8", status=401)
                return
            if route not in {
                "/api/console",
                "/console",
                "/feedback",
                "/settings/watchlist",
                "/settings/model",
                "/settings/channels",
                "/settings/remote",
                "/settings/features",
                "/settings/personalization",
                "/settings/factors",
                "/settings/factors/upload",
            }:
                self.send_error(404)
                return
            if not self._require_auth(parsed):
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            params, files = _parse_form_data(self.headers, raw)
            if route == "/feedback":
                try:
                    _record_feedback(params, storage)
                    referer = self.headers.get("Referer") or "/"
                    ref = urlparse(referer)
                    location = ref.path or "/"
                    if ref.query:
                        location += f"?{ref.query}"
                    sep = "&" if "?" in location else "?"
                    self._redirect(f"{location}{sep}feedback=1")
                except ValueError as exc:
                    self._send_bytes(str(exc).encode("utf-8"), "text/plain; charset=utf-8", status=400)
                return
            if route == "/api/console":
                result = _run_web_action(params, storage)
                body = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
                self._send_bytes(body, "application/json; charset=utf-8")
                return
            if route == "/console":
                result = _run_web_action(params, storage)
                settings = load_settings()
                body = render_console(
                    storage, settings, result=result,
                    question=_first(params, "question"),
                ).encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return
            if route == "/settings/factors/upload":
                try:
                    notice = save_custom_factor(params, files)
                except ValueError as exc:
                    notice = str(exc)
                body = _render_route("/settings/factors", storage, notice).encode("utf-8")
                self._send_bytes(body, "text/html; charset=utf-8")
                return
            updates = _updates_for_route(route, params)
            save_settings(updates)
            headers = {}
            if route == "/settings/remote" and updates.get("WEB_AUTH_TOKEN"):
                headers["Set-Cookie"] = f"{AUTH_COOKIE}={updates['WEB_AUTH_TOKEN']}; Path=/; HttpOnly; SameSite=Lax"
            if route == "/settings/remote" and "WEB_AUTH_TOKEN" in updates and not updates["WEB_AUTH_TOKEN"]:
                headers["Set-Cookie"] = f"{AUTH_COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"
            self._redirect(f"{route}?saved=1", headers)

        def log_message(self, fmt, *args):
            return

    return Handler


def run_dashboard(host: str = "127.0.0.1", port: int = 8765):
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    storage = Storage()
    server = ThreadingHTTPServer((host, port), make_handler(storage))
    print(f"StockWatch dashboard: http://{host}:{port}")
    if host in {"0.0.0.0", "::"} and not _web_auth_token():
        print("WARNING: dashboard is listening on all interfaces without WEB_AUTH_TOKEN.")
        print("Set WEB_AUTH_TOKEN before exposing this service to LAN or tunnels.")
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local StockWatch dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    run_dashboard(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
