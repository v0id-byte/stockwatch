"""Small local dashboard and settings panel for StockWatch."""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from analysis.report import build_report
from utils.storage import Storage


PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
_ENV_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")

DEFAULT_SETTINGS = {
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
        f"<td><span class='pill {str(row.get('action', '')).lower()}'>{_e(row.get('action'))}</span></td>"
        f"<td>{_pct(row.get('confidence'))}</td>"
        f"<td>{_price(row.get('target_price'))}</td>"
        f"<td>{_price(row.get('stop_loss'))}</td>"
        f"<td>{_e(row.get('one_liner'))}</td>"
        "</tr>"
        for row in rows
    )
    return _table("最近信号", ["时间", "代码", "名称", "动作", "置信度", "目标", "止损", "一句话"], body)


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
    return _table("持仓跟踪", ["代码", "名称", "买入价", "数量", "止损", "目标", "开始时间"], body)


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
        f"<td>{_e(action)}</td>"
        f"<td>{_e(stats.get('count', 0))}</td>"
        f"<td>{_pct(stats.get('hit_rate'))}</td>"
        f"<td>{float(stats.get('avg_return') or 0):+.2f}%</td>"
        f"<td>{float(stats.get('median_return') or 0):+.2f}%</td>"
        "</tr>"
        for action, stats in rows.items()
    )
    return _table("5日信号复盘", ["动作", "样本", "命中率", "平均收益", "中位收益"], body)


def _table(title: str, headers: list[str], body: str) -> str:
    header = "".join(f"<th>{_e(item)}</th>" for item in headers)
    empty = f"<tr><td colspan='{len(headers)}' class='empty'>暂无数据</td></tr>"
    return (
        f"<section><h2>{_e(title)}</h2><div class='table-wrap'><table>"
        f"<thead><tr>{header}</tr></thead><tbody>{body or empty}</tbody>"
        "</table></div></section>"
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


def _save_button() -> str:
    return "<div class='actions'><button type='submit'>保存设置</button></div>"


def _nav(active: str) -> str:
    items = [
        ("home", "/", "主页"),
        ("watchlist", "/settings/watchlist", "自选股"),
        ("model", "/settings/model", "模型配置"),
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
      --bg: #f4f6f8;
      --text: #1d2429;
      --muted: #66737c;
      --line: #d8dee4;
      --panel: #ffffff;
      --panel-soft: #f9fafb;
      --green: #16845b;
      --red: #b9473f;
      --orange: #a76a18;
      --blue: #246b9f;
      --nav: #26323a;
      --nav-active: #e8f2ed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .shell {{ display: flex; min-height: 100vh; }}
    aside {{
      width: 224px;
      flex: 0 0 224px;
      padding: 20px 14px;
      border-right: 1px solid var(--line);
      background: var(--panel);
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
    nav a:hover, nav a.active {{ background: var(--nav-active); color: #0f5f40; }}
    .content {{ flex: 1; min-width: 0; }}
    header {{
      padding: 22px clamp(16px, 4vw, 42px) 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
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
    }}
    .metric {{ padding: 13px 14px; min-height: 76px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 20px; font-weight: 650; }}
    section {{ margin-top: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
    .panel {{ padding: 18px; max-width: 920px; }}
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
    }}
    textarea {{ min-height: 168px; resize: vertical; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    .options {{ display: grid; gap: 10px; margin-top: 8px; }}
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
      background: #176b4c;
      color: #fff;
      font-weight: 650;
      cursor: pointer;
    }}
    button:hover {{ background: #0f5f40; }}
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
      aside {{ width: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      nav {{ grid-template-columns: repeat(3, 1fr); }}
      .form-grid, .segments {{ grid-template-columns: 1fr; }}
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
      </main>
    </div>
  </div>
</body>
</html>"""


def render_home(data: dict, settings: dict[str, str], notice: str = "") -> str:
    content = f"""
    <div class="metrics">{_summary_cards(data)}</div>
    {_report_table(data['report'])}
    {_decisions_table(data['decisions'])}
    {_positions_table(data['positions'])}
    {_alerts_table(data['price_alerts'])}
    {_runs_table(data['runs'])}
    <footer>仅供研究复盘，不构成投资建议。数据来自本地 SQLite 记录。</footer>
    """
    return _layout("home", "主页", f"更新时间 {_e(data['generated_at'])}", content, settings, notice)


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
          {level_checkbox("warning", "建议看：普通买卖信号、明显异动")}
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
    if route == "/settings/watchlist":
        return render_watchlist_settings(settings, notice)
    if route == "/settings/model":
        return render_model_settings(settings, notice)
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
                "/api.json",
                "/health",
                "/settings/watchlist",
                "/settings/model",
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
                "/settings/watchlist",
                "/settings/model",
                "/settings/features",
                "/settings/personalization",
                "/settings/factors",
            }:
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or "0")
            raw = self.rfile.read(length).decode("utf-8")
            params = parse_qs(raw, keep_blank_values=True)
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
