"""Small read-only local dashboard for StockWatch."""
from __future__ import annotations

import argparse
import html
import json
import os
import sqlite3
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from analysis.report import build_report
from utils.storage import Storage


def _rows(conn: sqlite3.Connection, sql: str, params: list | None = None) -> list[dict]:
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(sql, params or []).fetchall()]


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


def render_html(data: dict) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StockWatch Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f4;
      --text: #1f2723;
      --muted: #63706a;
      --line: #d9ded7;
      --panel: #ffffff;
      --green: #167a4a;
      --red: #b63d35;
      --orange: #9f6619;
      --blue: #27628f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    header {{
      padding: 24px clamp(16px, 4vw, 44px) 16px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1 {{ margin: 0; font-size: clamp(24px, 4vw, 36px); letter-spacing: 0; }}
    .sub {{ margin-top: 6px; color: var(--muted); }}
    main {{ padding: 20px clamp(16px, 4vw, 44px) 44px; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 10px;
      margin-bottom: 22px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      min-height: 76px;
    }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 8px; font-size: 20px; font-weight: 650; }}
    section {{ margin-top: 22px; }}
    h2 {{ margin: 0 0 10px; font-size: 17px; letter-spacing: 0; }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; background: #fbfcfa; }}
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
  </style>
</head>
<body>
  <header>
    <h1>StockWatch Dashboard</h1>
    <div class="sub">更新时间 {_e(data['generated_at'])}</div>
  </header>
  <main>
    <div class="metrics">{_summary_cards(data)}</div>
    {_report_table(data['report'])}
    {_decisions_table(data['decisions'])}
    {_positions_table(data['positions'])}
    {_alerts_table(data['price_alerts'])}
    {_runs_table(data['runs'])}
    <footer>仅供研究复盘，不构成投资建议。数据来自本地 SQLite 记录。</footer>
  </main>
</body>
</html>"""


def make_handler(storage: Storage):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in {"/", "/api.json", "/health"}:
                self.send_error(404)
                return
            if self.path == "/health":
                body = b"ok"
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            data = load_dashboard_data(storage)
            if self.path == "/api.json":
                body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                body = render_html(data).encode("utf-8")
                content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

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
