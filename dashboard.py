"""Small local dashboard and settings panel for StockWatch."""
from __future__ import annotations

import argparse
import email.policy
import html
import json
import os
import re
import sqlite3
from datetime import datetime
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analysis.report import build_report
from utils.storage import Storage


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CUSTOM_FACTORS_DIR = Path.home() / ".stockwatch" / "custom_factors"
_ENV_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")
WEB_USER_ID = "web-ui"
WEB_CHAT_ID = "web-ui"

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
}


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
    license_name = _first(params, "factor_license", "MIT").strip() or "MIT"
    share = _bool_param(params, "share_to_community") == "true"
    CUSTOM_FACTORS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_name = f"{stamp}_{_factor_slug(name)}_{filename}"
    target = CUSTOM_FACTORS_DIR / target_name
    target.write_bytes(payload)
    manifest = {
        "name": name,
        "description": description,
        "license": license_name,
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
            "params": {"console_action": "price_alert", "code": code, "price": alert_price},
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
    action = _first(params, "console_action")
    if action == "track_position":
        command = " ".join(filter(None, [
            "买入",
            _first(params, "code"),
            _first(params, "price"),
            _first(params, "quantity"),
        ]))
        return _run_web_command(command, storage)
    if action == "price_alert":
        command = " ".join(filter(None, [
            "盯买",
            _first(params, "code"),
            _first(params, "price"),
            _first(params, "quantity"),
        ]))
        return _run_web_command(command, storage)
    if action == "close_position":
        return _run_web_command(f"停止跟踪 {_first(params, 'code')}", storage)
    if action == "cancel_price_alert":
        return _run_web_command(f"取消盯价 {_first(params, 'code')}", storage)
    return _run_web_command(_first(params, "question"), storage)


def load_dashboard_data(storage: Storage) -> dict:
    with storage._conn() as conn:
        runs = _rows(conn, """
            SELECT run_id, run_ts, stocks_analyzed, llm_calls, tokens_used, pushed_count, pushed_ok
            FROM runs ORDER BY run_ts DESC LIMIT 12
        """)
        decisions = _rows(conn, """
            SELECT run_ts, code, name, action, confidence, target_price, stop_loss, one_liner, pushed
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
        "</tr>"
        for row in rows
    )
    return _table("最近提醒", ["时间", "代码", "名称", "类型", "置信度", "压力位", "风险价", "一句话"], body)


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


def _alerts_table(rows: list[dict]) -> str:
    body = "".join(
        "<tr>"
        f"<td><code>{_e(row.get('code'))}</code></td>"
        f"<td>{_e(row.get('name'))}</td>"
        f"<td>{_price(row.get('trigger_price'))}</td>"
        f"<td>{_e(row.get('direction'))}</td>"
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
        ("features", "/settings/features", "功能开关"),
        ("personalization", "/settings/personalization", "个性化"),
        ("factors", "/settings/factors", "因子市场"),
    ]
    links = []
    for key, href, label in items:
        cls = "active" if key == active else ""
        current = " aria-current='page'" if key == active else ""
        links.append(f"<a class='{cls}' href='{href}'{current}>{_e(label)}</a>")
    return "".join(links)


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
    steps = [
        ("添加自选股", watch_count > 0, f"当前 {watch_count} 只", "/settings/watchlist"),
        ("配置模型", model_ready, "用于解释公告、新闻和问答", "/settings/model"),
        ("选择通知方式", channel_ready, "可先用仅 Web 控制台", "/settings/channels"),
        ("设置打扰级别", levels_ready, "先只收必须看和建议看", "/settings/features"),
        ("添加持仓或关键价", watch_ready, "让系统围绕你的真实关注点提醒", "/console"),
    ]
    rows = []
    for label, done, hint, href in steps:
        cls = "step-row done" if done else "step-row"
        status = "已完成" if done else "待设置"
        rows.append(
            f"<a class='{cls}' href='{href}'>"
            f"<span class='step-status'>{status}</span>"
            f"<strong>{_e(label)}</strong>"
            f"<span>{_e(hint)}</span>"
            "</a>"
        )
    ready_count = sum(1 for _, done, _, _ in steps if done)
    return f"""
    <section class="panel onboarding">
      <div>
        <h2>开始使用</h2>
        <p>先完成这几步，就能把“持续盯盘”变成“有事再看”。</p>
      </div>
      <div class="setup-progress">{ready_count}/{len(steps)} 已完成</div>
      <div class="step-list">{''.join(rows)}</div>
    </section>
    """


def _compliance_notice() -> str:
    return """
    <section class="compliance">
      <h2>合规边界</h2>
      <p>StockWatch 是自选股盯盘提醒、公开信息聚合和持仓风险复核工具，不是证券投资咨询服务，也不是荐股软件。</p>
      <p>“机会观察”“风险复核”等提示只表示值得进一步查看，不构成买入、卖出或收益承诺；请自行核验交易所公告、上市公司披露和券商行情。</p>
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
      display: grid;
      gap: 5px;
      min-height: 104px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--text);
      text-decoration: none;
    }}
    .step-row:hover {{ border-color: var(--blue); }}
    .step-row strong {{ font-size: 15px; }}
    .step-row span:last-child {{ color: var(--muted); font-size: 12px; }}
    .step-status {{
      width: fit-content;
      padding: 2px 7px;
      border-radius: 999px;
      background: #fff4e5;
      color: var(--orange);
      font-size: 12px;
      font-weight: 650;
    }}
    .step-row.done .step-status {{
      background: #eef8f2;
      color: var(--green);
    }}
    .compliance {{
      max-width: 1080px;
      margin-top: 28px;
      padding: 14px 16px;
      border: 1px solid #f0d2a5;
      border-radius: 8px;
      background: #fffaf2;
      color: #4b3a23;
    }}
    .compliance h2 {{ margin-bottom: 6px; font-size: 15px; }}
    .compliance p {{ margin: 5px 0 0; color: #6a5430; font-size: 12px; }}
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
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      button {{ width: 100%; }}
      .suggested-actions {{ display: grid; grid-template-columns: 1fr; }}
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
  </script>
</body>
</html>"""


def render_home(data: dict, settings: dict[str, str], notice: str = "") -> str:
    content = f"""
    {_onboarding_panel(data, settings)}
    <div class="metrics">{_summary_cards(data)}</div>
    {_report_table(data['report'])}
    {_decisions_table(data['decisions'])}
    {_positions_table(data['positions'])}
    {_alerts_table(data['price_alerts'])}
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
    content = f"""
    <section class="panel">
      <h2>模型配置</h2>
      <form method="post" action="/settings/model">
        <div class="form-grid">
          {_field("模型提供商", provider_select)}
          {_field("模型代号", f"<input name='llm_model' value='{_e(settings.get('LLM_MODEL', ''))}' placeholder='MiniMax-M2.7'>")}
          {_field("接口地址", f"<input name='llm_base_url' value='{_e(settings.get('LLM_BASE_URL', ''))}' placeholder='https://api.example.com/v1'>", "Anthropic 可用 https://api.anthropic.com；本地模型可用 http://127.0.0.1:11434/v1")}
          {_field("API Key", f"<input type='password' name='llm_api_key' placeholder='留空不修改，当前：{_e(_mask_secret(settings.get('LLM_API_KEY', '')))}'>", "本地 OpenAI-compatible 服务通常可以留空")}
        </div>
        {_save_button()}
      </form>
    </section>
    """
    return _layout("model", "模型配置", "配置任意模型服务或本地模型", content, settings, notice)


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
          <label class="option-row"><input type="checkbox" name="enable_after_close_summary" value="true"{_checked(settings.get('ENABLE_AFTER_CLOSE_SUMMARY', 'false'))}>休市后“不用盯盘”总结</label>
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


def render_factor_settings(settings: dict[str, str], notice: str = "") -> str:
    custom_rows = "".join(
        "<tr>"
        f"<td>{_e(item.get('uploaded_at'))}</td>"
        f"<td>{_e(item.get('name'))}</td>"
        f"<td>{_e(item.get('license'))}</td>"
        f"<td>{_e('准备贡献' if item.get('share_to_community') else '本地草稿')}</td>"
        f"<td class='path'>{_e(item.get('file'))}</td>"
        "</tr>"
        for item in load_custom_factors()
    )
    upload_table = _table("已上传因子", ["时间", "名称", "协议", "状态", "文件"], custom_rows)
    content = f"""
    <section class="panel">
      <h2>因子市场</h2>
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
      <h2>上传因子</h2>
      <form method="post" action="/settings/factors/upload" enctype="multipart/form-data">
        <div class="form-grid">
          {_field("因子名称", "<input name='factor_name' placeholder='例如：资金流反转因子'>")}
          {_field("开源协议", _select("factor_license", "MIT", [("MIT", "MIT"), ("Apache-2.0", "Apache-2.0"), ("BSD-3-Clause", "BSD-3-Clause"), ("Proprietary", "暂不公开")]))}
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
        <li>上传文件会和说明一起保存为本地贡献包。</li>
        <li>准备贡献的因子可以从本地目录提交 PR，后续可做一键发布到社区市场。</li>
        <li>为了安全，Web UI 不会直接执行用户上传的 Python 因子代码。</li>
      </ul>
    </section>
    {upload_table}
    """
    return _layout("factors", "因子市场", "选择参与分析的增强因子", content, settings, notice)


def _first(params: dict[str, list[str]], name: str, default: str = "") -> str:
    values = params.get(name)
    return values[0] if values else default


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


def _render_route(route: str, storage: Storage, notice: str = "") -> str:
    settings = load_settings()
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
    if route == "/settings/features":
        return render_feature_settings(settings, notice)
    if route == "/settings/personalization":
        return render_personalization_settings(settings, notice)
    if route == "/settings/factors":
        return render_factor_settings(settings, notice)
    raise KeyError(route)


def make_handler(storage: Storage):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            if route not in {
                "/",
                "/console",
                "/api.json",
                "/health",
                "/settings/watchlist",
                "/settings/model",
                "/settings/channels",
                "/settings/features",
                "/settings/personalization",
                "/settings/factors",
            }:
                self.send_error(404)
                return
            if route == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/api.json":
                data = load_dashboard_data(storage)
                body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                notice = "设置已保存；已运行的守护进程或 Bot 需要重启后读取新配置。" if parse_qs(parsed.query).get("saved") else ""
                body = _render_route(route, storage, notice).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            parsed = urlparse(self.path)
            route = parsed.path
            if route not in {
                "/api/console",
                "/console",
                "/settings/watchlist",
                "/settings/model",
                "/settings/channels",
                "/settings/features",
                "/settings/personalization",
                "/settings/factors",
                "/settings/factors/upload",
            }:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length)
            params, files = _parse_form_data(self.headers, raw)
            if route == "/api/console":
                result = _run_web_action(params, storage)
                body = json.dumps(result, ensure_ascii=False, default=str).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/console":
                result = _run_web_action(params, storage)
                settings = load_settings()
                body = render_console(
                    storage, settings, result=result,
                    question=_first(params, "question"),
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route == "/settings/factors/upload":
                try:
                    notice = save_custom_factor(params, files)
                except ValueError as exc:
                    notice = str(exc)
                body = _render_route("/settings/factors", storage, notice).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            save_settings(_updates_for_route(route, params))
            self.send_response(303)
            self.send_header("Location", f"{route}?saved=1")
            self.end_headers()

        def log_message(self, fmt, *args):
            return

    return Handler


def run_dashboard(host: str = "127.0.0.1", port: int = 8765):
    os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")
    storage = Storage()
    server = ThreadingHTTPServer((host, port), make_handler(storage))
    print(f"StockWatch dashboard: http://{host}:{port}")
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
