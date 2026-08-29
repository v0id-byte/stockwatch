#!/usr/bin/env python3
"""Build PIT daily factors from the versioned announcement event library."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.backfill_announcement_documents import CATEGORY_PRIORITY, EXTRACTOR_VERSION
from scripts.build_sentiment_features import _base_frame, _load_grid, _rolling_sum, _to_feature_date


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Build daily PIT factors from extracted announcement events.")
    parser.add_argument("--library-db", default=str(root / "announcement_event_library.sqlite"))
    parser.add_argument("--output", default=str(root / "structured_event_features.parquet"))
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--codes", default="")
    parser.add_argument("--max-codes", type=int, default=0)
    return parser.parse_args()


def _load_events(path: Path, codes: list[str], trade_dates: pd.Index,
                 start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"event library missing: {path}")
    placeholders = ",".join("?" for _ in codes)
    query = f"""
        SELECT code, category, direction, event_status, magnitude_value,
               magnitude_percent, signed_score, novelty, confidence,
               published_at, available_at, extractor_version
        FROM events
        WHERE code IN ({placeholders})
          AND available_at >= ? AND available_at < ?
        ORDER BY available_at, code
    """
    warm_start = start - pd.Timedelta(days=40)
    with sqlite3.connect(path) as conn:
        events = pd.read_sql_query(
            query, conn,
            params=[*codes, warm_start.isoformat(sep=" "), (end + pd.Timedelta(days=2)).isoformat(sep=" ")],
        )
    if events.empty:
        return events
    events["code"] = events["code"].astype(str).str.zfill(6)
    events["feature_date"] = [
        _to_feature_date(value, trade_dates) for value in events["available_at"]
    ]
    return events.dropna(subset=["feature_date"])


def _add_event_features(frame: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    daily_columns = [
        "event_count", "event_signed_score", "event_abs_score", "event_positive",
        "event_negative", "event_novelty", "event_confidence", "event_magnitude_known",
    ]
    if events.empty:
        out = frame.copy()
        for window in (7, 20):
            for name in daily_columns:
                out[f"{name}_{window}d"] = 0.0
        for category in CATEGORY_PRIORITY:
            out[f"event_{category}_count_20d"] = 0.0
            out[f"event_{category}_score_20d"] = 0.0
        out["event_latest_age_days"] = float("nan")
        return out

    values = events.copy()
    values["event_count"] = 1.0
    values["event_signed_score"] = pd.to_numeric(values["signed_score"], errors="coerce").fillna(0.0)
    values["event_abs_score"] = values["event_signed_score"].abs()
    values["event_positive"] = (values["direction"] > 0).astype(float)
    values["event_negative"] = (values["direction"] < 0).astype(float)
    values["event_novelty"] = pd.to_numeric(values["novelty"], errors="coerce").fillna(0.0)
    values["event_confidence"] = pd.to_numeric(values["confidence"], errors="coerce").fillna(0.0)
    values["event_magnitude_known"] = (
        values["magnitude_value"].notna() | values["magnitude_percent"].notna()
    ).astype(float)
    for category in CATEGORY_PRIORITY:
        mask = values["category"].eq(category)
        values[f"event_{category}_count"] = mask.astype(float)
        values[f"event_{category}_score"] = values["event_signed_score"].where(mask, 0.0)
    aggregate_columns = [
        *daily_columns,
        *[f"event_{category}_{kind}" for category in CATEGORY_PRIORITY for kind in ("count", "score")],
    ]
    daily = values.groupby(["code", "feature_date"], as_index=False)[aggregate_columns].sum()
    daily = daily.rename(columns={"feature_date": "trade_date"})
    out = frame.merge(daily, on=["code", "trade_date"], how="left", validate="one_to_one")
    out[aggregate_columns] = out[aggregate_columns].fillna(0.0)
    for window in (7, 20):
        for name in daily_columns:
            out[f"{name}_{window}d"] = _rolling_sum(out, name, window).astype("float32")
    for category in CATEGORY_PRIORITY:
        for kind in ("count", "score"):
            name = f"event_{category}_{kind}"
            out[f"{name}_20d"] = _rolling_sum(out, name, 20).astype("float32")
    event_idx = out["trade_idx"].where(out["event_count"] > 0)
    last_idx = event_idx.groupby(out["code"], sort=False).ffill()
    age = out["trade_idx"] - last_idx
    out["event_latest_age_days"] = age.where(age <= 20).astype("float32")
    return out.drop(columns=aggregate_columns)


def main() -> None:
    args = _parse_args()
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    grid, trade_dates, codes, output_start, output_end = _load_grid(root, args)
    frame = _base_frame(grid, trade_dates)
    events = _load_events(
        Path(args.library_db).expanduser(), codes, trade_dates,
        pd.to_datetime(grid["trade_date"]).min(), output_end,
    )
    frame = _add_event_features(frame, events)
    frame = frame[(frame["trade_date"] >= output_start) & (frame["trade_date"] <= output_end)]
    frame = frame.drop(columns=["trade_idx"]).sort_values(["trade_date", "code"])
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    versions = sorted(events["extractor_version"].dropna().unique().tolist()) if not events.empty else []
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(frame)),
        "codes": int(frame["code"].nunique()),
        "events_used": int(len(events)),
        "event_codes": int(events["code"].nunique()) if not events.empty else 0,
        "date_start": str(output_start.date()),
        "date_end": str(output_end.date()),
        "extractor_versions": versions,
        "expected_extractor_version": EXTRACTOR_VERSION,
        "research_only": True,
        "incomplete_library_warning": (
            "Zero means no downloaded event, not necessarily no market event, until the full six-class backfill completes."
        ),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"structured event features saved: {output}, rows={len(frame)}, events={len(events)}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
