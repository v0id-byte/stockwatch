#!/usr/bin/env python3
"""Evaluate signed earnings-announcement events as a standalone PEAD gate."""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from config import get_config
from scripts.build_sentiment_features import _announcement_available_at, _to_feature_date


POSITIVE_TERMS = (
    "预增",
    "略增",
    "扭亏",
    "扭亏为盈",
    "增长",
    "增加",
    "上升",
)
NEGATIVE_TERMS = (
    "预减",
    "略减",
    "首亏",
    "续亏",
    "亏损",
    "转亏",
    "由盈转亏",
    "下降",
    "减少",
)
EVENT_TERMS = ("业绩预告", "业绩快报", "业绩预增", "业绩预减", "业绩扭亏", "经营业绩")
PCT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate signed PEAD-style CNINFO earnings events.")
    parser.add_argument("--target", default="", help="Target return column. Defaults to training label horizon.")
    parser.add_argument("--cost-bps", type=float, default=20.0, help="One-way long-only cost deducted from event return.")
    parser.add_argument("--events-output", default=None, help="Parsed PEAD event parquet path.")
    parser.add_argument("--output", default=None, help="JSON report path.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _target_col(root: Path, args: argparse.Namespace) -> str:
    if args.target:
        return args.target
    report_path = root / "training_set_report.json"
    label_horizon = 20
    if report_path.exists():
        label_horizon = int(json.loads(report_path.read_text()).get("label_horizon_days", 20))
    return f"forward_{label_horizon}d_return"


def _json_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _is_candidate_title(title: str) -> bool:
    text = str(title or "")
    if "业绩" not in text:
        return False
    return any(term in text for term in EVENT_TERMS) or any(
        term in text for term in (*POSITIVE_TERMS, *NEGATIVE_TERMS)
    )


def _parse_magnitude_pct(title: str) -> float | None:
    values = [abs(float(item)) for item in PCT_RE.findall(str(title or ""))]
    return max(values) if values else None


def _parse_pead_title(title: str) -> dict | None:
    text = str(title or "")
    if not _is_candidate_title(text):
        return None

    has_positive = any(term in text for term in POSITIVE_TERMS)
    has_negative = any(term in text for term in NEGATIVE_TERMS)
    if "扭亏" in text:
        has_positive = True
        has_negative = False
    if "由盈转亏" in text or "转亏" in text:
        has_negative = True
        has_positive = False
    if has_positive == has_negative:
        return None

    sign = 1 if has_positive else -1
    magnitude_pct = _parse_magnitude_pct(text)
    score = float(sign) if magnitude_pct is None else sign * magnitude_pct
    return {
        "direction": "positive" if sign > 0 else "negative",
        "sign": sign,
        "magnitude_pct": magnitude_pct,
        "signed_score": score,
    }


def _load_training(root: Path, target: str) -> pd.DataFrame:
    train = pd.read_parquet(root / "training_set.parquet", columns=["trade_date", "code", target])
    train["trade_date"] = pd.to_datetime(train["trade_date"]).dt.normalize()
    train["code"] = train["code"].astype(str).str.zfill(6)
    return train.dropna(subset=[target]).copy()


def _load_candidate_announcements(codes: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT source, announcement_id, code, title, published_at, url
        FROM announcements
        WHERE code IN ({placeholders})
          AND title LIKE '%业绩%'
          AND (
            title LIKE '%预告%'
            OR title LIKE '%快报%'
            OR title LIKE '%预增%'
            OR title LIKE '%预减%'
            OR title LIKE '%扭亏%'
            OR title LIKE '%增长%'
            OR title LIKE '%下降%'
            OR title LIKE '%亏损%'
          )
    """
    with sqlite3.connect(str(get_config().db_path)) as conn:
        return pd.read_sql_query(query, conn, params=codes)


def _build_events(announcements: pd.DataFrame, trade_dates: pd.Index) -> tuple[pd.DataFrame, dict]:
    rows = []
    candidate_rows = 0
    directionless_rows = 0
    for row in announcements.itertuples(index=False):
        title = str(row.title or "")
        if not _is_candidate_title(title):
            continue
        candidate_rows += 1
        parsed = _parse_pead_title(title)
        if parsed is None:
            directionless_rows += 1
            continue
        available_at = _announcement_available_at(row.published_at)
        feature_date = _to_feature_date(available_at, trade_dates)
        if feature_date is None:
            continue
        rows.append({
            "code": str(row.code).zfill(6),
            "feature_date": pd.Timestamp(feature_date).normalize(),
            "source": row.source,
            "announcement_id": row.announcement_id,
            "title": title[:200],
            "published_at": row.published_at,
            "url": row.url,
            **parsed,
        })

    events = pd.DataFrame(rows)
    stats = {
        "announcement_rows_scanned": int(len(announcements)),
        "candidate_rows": int(candidate_rows),
        "directionless_candidate_rows": int(directionless_rows),
    }
    if events.empty:
        return events, stats

    events["has_magnitude"] = events["magnitude_pct"].notna()
    stats.update({
        "parsed_events": int(len(events)),
        "parsed_with_magnitude": int(events["has_magnitude"].sum()),
        "positive_events": int((events["sign"] > 0).sum()),
        "negative_events": int((events["sign"] < 0).sum()),
    })
    return events.sort_values(["feature_date", "code", "announcement_id"]), stats


def _dedupe_code_date(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    scored = events.assign(abs_score=events["signed_score"].abs())
    idx = scored.sort_values(["code", "feature_date", "abs_score"]).groupby(
        ["code", "feature_date"], sort=False
    ).tail(1).index
    return events.loc[idx].sort_values(["feature_date", "code"]).reset_index(drop=True)


def _distribution(values: pd.Series) -> dict | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "count": int(len(values)),
        "mean": _json_float(values.mean()),
        "median": _json_float(values.median()),
        "std": _json_float(values.std()),
        "positive_rate": float((values > 0).mean()),
        "q10": _json_float(values.quantile(0.10)),
        "q25": _json_float(values.quantile(0.25)),
        "q75": _json_float(values.quantile(0.75)),
        "q90": _json_float(values.quantile(0.90)),
    }


def _study_subset(data: pd.DataFrame, target: str, cost: float) -> dict | None:
    if data.empty:
        return None
    data = data.copy()
    data["gross_excess_return"] = data[target] - data["universe_return"]
    data["net_return"] = data[target] - cost
    data["net_excess_return"] = data["gross_excess_return"] - cost
    by_year = {}
    for year, group in data.groupby(data["feature_date"].dt.year):
        by_year[str(int(year))] = {
            "rows": int(len(group)),
            "net_excess_return": _distribution(group["net_excess_return"]),
        }
    return {
        "rows": int(len(data)),
        "dates": int(data["feature_date"].nunique()),
        "codes": int(data["code"].nunique()),
        "gross_return": _distribution(data[target]),
        "gross_excess_return": _distribution(data["gross_excess_return"]),
        "net_return": _distribution(data["net_return"]),
        "net_excess_return": _distribution(data["net_excess_return"]),
        "by_year": by_year,
    }


def _event_study(events: pd.DataFrame, train: pd.DataFrame, target: str, cost: float) -> dict:
    if events.empty:
        return {}
    data = events.merge(
        train,
        left_on=["feature_date", "code"],
        right_on=["trade_date", "code"],
        how="inner",
    )
    universe = train.groupby("trade_date")[target].mean()
    data["universe_return"] = data["feature_date"].map(universe)

    return {
        "all_signed": _study_subset(data, target, cost),
        "positive": _study_subset(data[data["sign"] > 0], target, cost),
        "negative": _study_subset(data[data["sign"] < 0], target, cost),
        "positive_with_magnitude": _study_subset(
            data[(data["sign"] > 0) & data["magnitude_pct"].notna()],
            target,
            cost,
        ),
        "positive_magnitude_ge_50": _study_subset(
            data[(data["sign"] > 0) & (data["magnitude_pct"].fillna(0) >= 50)],
            target,
            cost,
        ),
        "negative_magnitude_ge_50": _study_subset(
            data[(data["sign"] < 0) & (data["magnitude_pct"].fillna(0) >= 50)],
            target,
            cost,
        ),
    }


def main() -> None:
    args = _parse_args()
    root = _root()
    target = _target_col(root, args)
    train = _load_training(root, target)
    trade_dates = pd.Index(sorted(train["trade_date"].drop_duplicates()))
    codes = sorted(train["code"].drop_duplicates())
    announcements = _load_candidate_announcements(codes)
    events, parse_stats = _build_events(announcements, trade_dates)
    code_date_events = _dedupe_code_date(events)
    cost = args.cost_bps / 10000.0

    events_output = Path(args.events_output).expanduser() if args.events_output else root / "pead_events.parquet"
    events_output.parent.mkdir(parents=True, exist_ok=True)
    code_date_events.to_parquet(events_output, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": target,
        "cost_bps": args.cost_bps,
        "events_path": str(events_output),
        "train_rows": int(len(train)),
        "train_dates": int(train["trade_date"].nunique()),
        "train_codes": int(train["code"].nunique()),
        "parse_stats": {
            **parse_stats,
            "code_date_events": int(len(code_date_events)),
        },
        "event_study": _event_study(code_date_events, train, target, cost),
    }

    output = Path(args.output).expanduser() if args.output else root / "pead_event_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PEAD events saved: {events_output}, rows={len(code_date_events)}")
    print(f"PEAD report saved: {output}")
    print(json.dumps(report["parse_stats"], ensure_ascii=False))
    for name, item in report["event_study"].items():
        if not item:
            continue
        net = item["net_excess_return"] or {}
        print(
            f"{name}: rows={item['rows']}, dates={item['dates']}, "
            f"net_excess_mean={net.get('mean')}, positive_rate={net.get('positive_rate')}"
        )


if __name__ == "__main__":
    main()
