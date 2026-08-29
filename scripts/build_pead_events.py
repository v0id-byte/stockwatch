#!/usr/bin/env python3
"""Build structured signed earnings-drift events from AKShare snapshots."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import get_config


POSITIVE_TYPES = ("预增", "略增", "扭亏", "扭亏为盈", "减亏")
NEGATIVE_TYPES = ("预减", "略减", "首亏", "续亏", "增亏", "由盈转亏", "转亏")
NEUTRAL_TYPES = ("不确定", "预平")
TURNAROUND_TYPES = ("扭亏", "减亏", "首亏", "续亏", "增亏", "由盈转亏", "转亏")
PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")
NET_PROFIT_METRIC = "归属于上市公司股东的净利润"
DEDUCTED_METRIC = "扣除非经常性损益"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structured PEAD event cache.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-periods", type=int, default=0, help="Limit periods for smoke tests.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _quarter_periods(start_year: int, end_year: int) -> list[str]:
    today = date.today()
    periods = []
    for year in range(start_year, end_year + 1):
        for md in ("0331", "0630", "0930", "1231"):
            period_date = date(year, int(md[:2]), int(md[2:]))
            if period_date <= today:
                periods.append(f"{year}{md}")
    return periods


def _number(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "").replace("，", "")
    if not text or text.lower() in {"nan", "none", "--", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _percent_values(text: str) -> list[float]:
    return [float(item) for item in PCT_RE.findall(str(text or ""))]


def _conservative_magnitude_pct(raw_type: str, raw_text: str, fallback, sign: int,
                                is_turnaround: bool) -> tuple[float | None, dict]:
    details = {
        "magnitude_source": None,
        "magnitude_lower_pct": None,
        "magnitude_upper_pct": None,
        "magnitude_mid_pct": None,
        "magnitude_is_primary": False,
    }
    if is_turnaround:
        details["magnitude_source"] = "turnaround_direction_only"
        return None, details

    values = [abs(v) for v in _percent_values(raw_text)]
    if values:
        lower = min(values)
        upper = max(values)
        details.update({
            "magnitude_source": "raw_change_text_percent",
            "magnitude_lower_pct": lower,
            "magnitude_upper_pct": upper,
            "magnitude_mid_pct": (lower + upper) / 2,
            "magnitude_is_primary": True,
        })
        return min(lower, 500.0) if sign > 0 else min(upper, 500.0), details

    fallback_num = _number(fallback)
    if fallback_num is None:
        details["magnitude_source"] = "missing"
        return None, details

    magnitude = min(abs(fallback_num), 500.0)
    details.update({
        "magnitude_source": "structured_percent",
        "magnitude_lower_pct": abs(fallback_num),
        "magnitude_upper_pct": abs(fallback_num),
        "magnitude_mid_pct": abs(fallback_num),
        "magnitude_is_primary": True,
    })
    return magnitude, details


def _sign_from_preannounce_type(raw_type: str) -> tuple[int | None, str | None, bool]:
    text = str(raw_type or "")
    if any(term in text for term in NEUTRAL_TYPES):
        return None, None, False
    if "扭亏" in text:
        return 1, "positive", True
    if "由盈转亏" in text or "转亏" in text:
        return -1, "negative", True
    if "减亏" in text:
        return 1, "positive", True
    if any(term in text for term in NEGATIVE_TYPES):
        return -1, "negative", any(term in text for term in TURNAROUND_TYPES)
    if any(term in text for term in POSITIVE_TYPES):
        return 1, "positive", any(term in text for term in TURNAROUND_TYPES)
    return None, None, False


def _score(sign: int, magnitude_pct: float | None) -> float:
    if magnitude_pct is None:
        return float(sign)
    return float(sign) * math.log1p(float(magnitude_pct) / 100.0)


def _preannounce_rows(df: pd.DataFrame, period: str, fetched_at: str) -> list[dict]:
    if df.empty:
        return []
    rows = []
    for item in df.to_dict("records"):
        metric = str(item.get("预测指标") or "")
        if NET_PROFIT_METRIC not in metric or DEDUCTED_METRIC in metric:
            continue
        raw_type = str(item.get("预告类型") or "")
        sign, event_type, is_turnaround = _sign_from_preannounce_type(raw_type)
        if sign is None:
            continue
        raw_text = str(item.get("业绩变动") or "")
        magnitude, details = _conservative_magnitude_pct(
            raw_type,
            raw_text,
            item.get("业绩变动幅度"),
            sign,
            is_turnaround,
        )
        available_at = pd.to_datetime(item.get("公告日期"), errors="coerce")
        if pd.isna(available_at):
            continue
        rows.append({
            "source": "yjyg",
            "code": str(item.get("股票代码") or "").zfill(6),
            "name": item.get("股票简称"),
            "report_period": period,
            "available_at": available_at.date().isoformat(),
            "event_type": event_type,
            "metric": metric,
            "signed_score": _score(sign, magnitude),
            "sign": sign,
            "magnitude_pct": magnitude,
            "net_profit_value": _number(item.get("预测数值")),
            "last_year_value": _number(item.get("上年同期值")),
            "raw_type": raw_type,
            "raw_change_text": raw_text,
            "raw_reason": item.get("业绩变动原因"),
            "snapshot_fetched_at": fetched_at,
            "true_vintage": False,
            "vintage_note": "AKShare/Eastmoney current snapshot; true historical vintage not verified",
            "is_turnaround": bool(is_turnaround),
            "strong_positive": bool(sign > 0 and magnitude is not None and magnitude >= 50),
            "strong_negative": bool(sign < 0 and magnitude is not None and magnitude >= 50),
            **details,
        })
    return [row for row in rows if row["code"] and row["code"] != "000nan"]


def _express_rows(df: pd.DataFrame, period: str, fetched_at: str) -> list[dict]:
    if df.empty:
        return []
    rows = []
    for item in df.to_dict("records"):
        yoy = _number(item.get("净利润-同比增长"))
        if yoy is None or yoy == 0:
            continue
        net_profit = _number(item.get("净利润-净利润"))
        last_year = _number(item.get("净利润-去年同期"))
        sign = 1 if yoy > 0 else -1
        is_turnaround = (
            net_profit is not None
            and last_year is not None
            and (last_year <= 0 or net_profit * last_year < 0)
        )
        magnitude = None if is_turnaround else min(abs(yoy), 500.0)
        available_at = pd.to_datetime(item.get("公告日期"), errors="coerce")
        if pd.isna(available_at):
            continue
        rows.append({
            "source": "yjkb",
            "code": str(item.get("股票代码") or "").zfill(6),
            "name": item.get("股票简称"),
            "report_period": period,
            "available_at": available_at.date().isoformat(),
            "event_type": "positive" if sign > 0 else "negative",
            "metric": "归属于上市公司股东的净利润",
            "signed_score": _score(sign, magnitude),
            "sign": sign,
            "magnitude_pct": magnitude,
            "net_profit_value": net_profit,
            "last_year_value": last_year,
            "raw_type": "业绩快报",
            "raw_change_text": f"净利润同比增长={yoy}",
            "raw_reason": None,
            "snapshot_fetched_at": fetched_at,
            "true_vintage": False,
            "vintage_note": "AKShare/Eastmoney current snapshot; true historical vintage not verified",
            "is_turnaround": bool(is_turnaround),
            "strong_positive": bool(sign > 0 and magnitude is not None and magnitude >= 50),
            "strong_negative": bool(sign < 0 and magnitude is not None and magnitude >= 50),
            "magnitude_source": "structured_percent" if magnitude is not None else "turnaround_direction_only",
            "magnitude_lower_pct": abs(yoy),
            "magnitude_upper_pct": abs(yoy),
            "magnitude_mid_pct": abs(yoy),
            "magnitude_is_primary": bool(magnitude is not None),
        })
    return [row for row in rows if row["code"] and row["code"] != "000nan"]


def _fetch_period(source: str, period: str) -> pd.DataFrame:
    import akshare as ak

    if source == "yjyg":
        return ak.stock_yjyg_em(date=period)
    if source == "yjkb":
        return ak.stock_yjkb_em(date=period)
    raise ValueError(source)


def _load_existing(output: Path, force: bool) -> pd.DataFrame:
    if force or not output.exists():
        return pd.DataFrame()
    return pd.read_parquet(output)


def _write_events(output: Path, frame: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(output)


def _vintage_audit() -> dict:
    db_path = get_config().db_path
    if not Path(db_path).expanduser().exists():
        return {"true_vintage": False, "note": "local announcement DB missing"}
    query = """
        SELECT code, title, published_at
        FROM announcements
        WHERE title LIKE '%业绩预告%'
          AND (title LIKE '%修正%' OR title LIKE '%更正%')
        ORDER BY published_at DESC
        LIMIT 20
    """
    import sqlite3
    with sqlite3.connect(str(db_path)) as conn:
        rows = pd.read_sql_query(query, conn)
    return {
        "true_vintage": False,
        "best_effort_reconstructed_pit": True,
        "revision_examples": rows.to_dict("records"),
        "note": "AKShare structured endpoints are current snapshots; revision examples require manual vintage verification.",
    }


def _report(frame: pd.DataFrame, statuses: list[dict], vintage: dict) -> dict:
    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "codes": int(frame["code"].nunique()) if not frame.empty else 0,
        "sources": {},
        "statuses": statuses,
        "vintage_audit": vintage,
        "true_vintage": False,
        "best_effort_reconstructed_pit": True,
    }
    if not frame.empty:
        for source, group in frame.groupby("source"):
            out["sources"][source] = {
                "rows": int(len(group)),
                "periods": int(group["report_period"].nunique()),
                "positive": int((group["sign"] > 0).sum()),
                "negative": int((group["sign"] < 0).sum()),
                "with_magnitude": int(group["magnitude_pct"].notna().sum()),
                "strong_positive": int(group["strong_positive"].sum()),
                "strong_negative": int(group["strong_negative"].sum()),
                "turnaround": int(group["is_turnaround"].sum()),
            }
    return out


def main() -> None:
    args = _parse_args()
    root = _root()
    output = Path(args.output).expanduser() if args.output else root / "pead_events_structured.parquet"
    report_path = Path(args.report).expanduser() if args.report else root / "pead_events_structured_report.json"
    periods = _quarter_periods(args.start_year, args.end_year)
    if args.max_periods:
        periods = periods[:args.max_periods]

    existing = _load_existing(output, args.force)
    frames = [] if existing.empty else [existing]
    done = set()
    if not existing.empty:
        done = set(zip(existing["source"], existing["report_period"].astype(str)))

    statuses: list[dict] = []
    for period in periods:
        for source in ("yjyg", "yjkb"):
            if (source, period) in done:
                statuses.append({"source": source, "period": period, "status": "skipped_existing"})
                continue
            fetched_at = datetime.now().isoformat(timespec="seconds")
            try:
                raw = _fetch_period(source, period)
                rows = _preannounce_rows(raw, period, fetched_at) if source == "yjyg" else _express_rows(raw, period, fetched_at)
                if rows:
                    frames.append(pd.DataFrame(rows))
                    combined = pd.concat(frames, ignore_index=True)
                    _write_events(output, combined)
                statuses.append({
                    "source": source,
                    "period": period,
                    "status": "done",
                    "raw_rows": int(len(raw)),
                    "event_rows": int(len(rows)),
                })
                print(f"{source} {period}: raw={len(raw)} events={len(rows)}", flush=True)
            except Exception as exc:
                statuses.append({
                    "source": source,
                    "period": period,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                })
                print(f"{source} {period}: failed {type(exc).__name__}: {exc}", flush=True)
            if args.sleep > 0:
                time.sleep(args.sleep)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not frame.empty:
        frame = frame.sort_values(["available_at", "code", "source", "report_period"]).reset_index(drop=True)
        _write_events(output, frame)
    vintage = _vintage_audit()
    report = _report(frame, statuses, vintage)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"structured PEAD events saved: {output}, rows={len(frame)}")
    print(f"structured PEAD report saved: {report_path}")


if __name__ == "__main__":
    main()
