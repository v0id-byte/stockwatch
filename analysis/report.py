"""Read-only decision quality report for stored StockWatch signals."""
from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from utils.storage import Storage


def _date_part(value: str) -> str:
    return str(value or "")[:10]


def _future_return(kline: list[dict], run_date: str, horizon: int) -> float | None:
    if not kline:
        return None
    start_idx = None
    for idx, row in enumerate(kline):
        # Use the NEXT trading day's close as the entry, not the decision day's:
        # midday/after-close runs are made before that day's close is final, so
        # ">= run_date" would price the entry at an unavailable bar (look-ahead).
        if str(row.get("trade_date", "")) > run_date:
            start_idx = idx
            break
    if start_idx is None or start_idx + horizon >= len(kline):
        return None
    start = float(kline[start_idx].get("close") or 0)
    end = float(kline[start_idx + horizon].get("close") or 0)
    if start <= 0 or end <= 0:
        return None
    return (end / start - 1) * 100


def _hit(action: str, forward_return: float) -> bool:
    if action == "BUY":
        return forward_return > 0
    if action == "SELL":
        return forward_return < 0
    return abs(forward_return) <= 2


def _bucket(confidence: float) -> str:
    if confidence >= 0.75:
        return ">=75%"
    if confidence >= 0.60:
        return "60%-75%"
    return "<60%"


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {"count": 0}
    returns = [row["forward_return"] for row in rows]
    hits = [row["hit"] for row in rows]
    return {
        "count": len(rows),
        "hit_rate": sum(1 for hit in hits if hit) / len(hits),
        "avg_return": statistics.fmean(returns),
        "median_return": statistics.median(returns),
    }


def build_report(storage: Storage, horizon: int = 5, limit: int = 1000) -> dict:
    with storage._conn() as conn:
        conn.row_factory = sqlite3.Row
        decisions = conn.execute("""
            SELECT id, run_id, run_ts, code, name, action, confidence, one_liner
            FROM decisions
            WHERE action IN ('BUY', 'SELL', 'HOLD')
            ORDER BY run_ts DESC
            LIMIT ?
        """, [limit]).fetchall()

    samples = []
    today = datetime.now().strftime("%Y-%m-%d")
    for row in decisions:
        item = dict(row)
        kline = storage.get_kline(item["code"], "2020-01-01", today)
        forward_return = _future_return(kline, _date_part(item["run_ts"]), horizon)
        if forward_return is None:
            continue
        action = item.get("action", "HOLD")
        confidence = float(item.get("confidence") or 0)
        samples.append({
            **item,
            "confidence": confidence,
            "forward_return": round(forward_return, 2),
            "hit": _hit(action, forward_return),
            "confidence_bucket": _bucket(confidence),
        })

    by_action = defaultdict(list)
    by_bucket = defaultdict(list)
    for row in samples:
        by_action[row["action"]].append(row)
        by_bucket[row["confidence_bucket"]].append(row)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "horizon": horizon,
        "scanned_decisions": len(decisions),
        "resolved_samples": len(samples),
        "overall": _summary(samples),
        "by_action": {key: _summary(value) for key, value in sorted(by_action.items())},
        "by_bucket": {key: _summary(value) for key, value in sorted(by_bucket.items())},
        "latest_samples": samples[:20],
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _fmt_return(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.2f}%"


def _section(title: str, rows: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", "", "| 分组 | 样本 | 命中率 | 平均收益 | 中位收益 |", "| --- | ---: | ---: | ---: | ---: |"]
    if not rows:
        lines.append("| 暂无 | 0 | - | - | - |")
        return lines
    for key, stats in rows.items():
        lines.append(
            f"| {key} | {stats.get('count', 0)} | {_fmt_pct(stats.get('hit_rate'))} | "
            f"{_fmt_return(stats.get('avg_return'))} | {_fmt_return(stats.get('median_return'))} |"
        )
    return lines


def render_markdown(report: dict) -> str:
    overall = report.get("overall", {})
    lines = [
        "# StockWatch 信号回测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 观察窗口：信号后 {report['horizon']} 个交易日",
        f"- 扫描决策：{report['scanned_decisions']} 条",
        f"- 可结算样本：{report['resolved_samples']} 条",
        f"- 总体命中率：{_fmt_pct(overall.get('hit_rate'))}",
        f"- 总体平均收益：{_fmt_return(overall.get('avg_return'))}",
        "",
        "> 命中率定义：BUY 后窗口收益为正、SELL 后窗口收益为负、HOLD 后窗口收益在 ±2% 内。该报告只用于研究复盘，不构成投资建议。",
        "",
    ]
    lines.extend(_section("按动作统计", report.get("by_action", {})))
    lines.append("")
    lines.extend(_section("按置信度统计", report.get("by_bucket", {})))
    lines.extend(["", "## 最近样本", ""])
    lines.append("| 日期 | 代码 | 名称 | 动作 | 置信度 | 窗口收益 | 命中 |")
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    for row in report.get("latest_samples", []):
        lines.append(
            f"| {_date_part(row['run_ts'])} | {row['code']} | {row.get('name') or ''} | "
            f"{row['action']} | {row['confidence']:.0%} | {_fmt_return(row['forward_return'])} | "
            f"{'是' if row['hit'] else '否'} |"
        )
    if not report.get("latest_samples"):
        lines.append("| 暂无 | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a read-only StockWatch signal report.")
    parser.add_argument("--horizon", type=int, default=5, help="Forward trading days, default: 5")
    parser.add_argument("--limit", type=int, default=1000, help="Max decisions to scan, default: 1000")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    args = parser.parse_args(argv)

    storage = Storage()
    report = build_report(storage, horizon=max(1, args.horizon), limit=max(1, args.limit))
    markdown = render_markdown(report)
    if args.output:
        path = Path(args.output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"报告已写入: {path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
