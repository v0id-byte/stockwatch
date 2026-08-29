#!/usr/bin/env python3
"""Build auditable expectation-revision and post-open reaction datasets."""
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

from analysis.pit_information import (
    attach_overnight_reaction,
    build_analyst_revision_features,
    build_earnings_revision_events,
)


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(
        description="Build PIT company-guidance, analyst-revision and overnight-reaction features."
    )
    parser.add_argument("--library-db", default=str(root / "announcement_event_library.sqlite"))
    parser.add_argument("--analyst-input", default=str(root / "analyst_forecasts.parquet"))
    parser.add_argument("--earnings-output", default=str(root / "pit_earnings_information.parquet"))
    parser.add_argument("--analyst-output", default=str(root / "pit_analyst_revisions.parquet"))
    parser.add_argument("--report", default=str(root / "pit_information_report.json"))
    parser.add_argument("--market-price", default=str(root / "market_sh000905.parquet"))
    return parser.parse_args()


def _load_earnings_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"announcement event library missing: {path}")
    query = """
        SELECT e.source, e.announcement_id, e.event_index, e.code, e.category,
               e.event_type, e.direction, e.event_status, e.signed_score,
               e.magnitude_value, e.magnitude_unit, e.magnitude_percent,
               e.evidence, e.published_at, e.available_at,
               e.document_sha256, e.extractor_version, d.title
        FROM events e
        JOIN documents d
          ON d.source = e.source AND d.announcement_id = e.announcement_id
        WHERE e.category = 'earnings'
        ORDER BY e.available_at, e.code
    """
    with sqlite3.connect(path) as conn:
        return pd.read_sql_query(query, conn)


def _load_raw_stock_prices(root: Path, codes: set[str]) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    missing = []
    for code in sorted(codes):
        path = root / "stocks" / f"{code}.parquet"
        if not path.exists():
            missing.append(code)
            continue
        frame = pd.read_parquet(path)
        required = {"trade_date", "raw_open", "raw_close"}
        if not required.issubset(frame.columns):
            missing.append(code)
            continue
        selected = frame[["trade_date", "raw_open", "raw_close"]].copy()
        selected["code"] = code
        frames.append(selected)
    if not frames:
        return pd.DataFrame(), missing
    return pd.concat(frames, ignore_index=True), missing


def _load_raw_market_prices(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if not {"trade_date", "raw_open", "raw_close"}.issubset(frame.columns):
        return None
    return frame[["trade_date", "raw_open", "raw_close"]].copy()


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def main() -> None:
    args = _parse_args()
    history_root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    earnings_output = Path(args.earnings_output).expanduser()
    analyst_output = Path(args.analyst_output).expanduser()
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")

    source_events = _load_earnings_events(Path(args.library_db).expanduser())
    earnings = build_earnings_revision_events(source_events)
    surprise_mask = (
        earnings.get("information_kind", pd.Series(index=earnings.index, dtype=str))
        .eq("earnings_surprise")
        & earnings.get(
            "expectation_is_strictly_prior",
            pd.Series(False, index=earnings.index),
        ).fillna(False)
    )

    raw_prices, missing_raw_codes = _load_raw_stock_prices(
        history_root,
        set(earnings["code"].dropna().astype(str)) if not earnings.empty else set(),
    )
    reaction_status = "built"
    if raw_prices.empty:
        reaction_status = "blocked_missing_raw_price_columns"
        earnings["reaction_trade_date"] = pd.NaT
        earnings["overnight_gap"] = float("nan")
        earnings["abnormal_overnight_gap"] = float("nan")
        earnings["reaction_available_at"] = pd.NaT
        earnings["same_open_execution_allowed"] = False
    else:
        market = _load_raw_market_prices(Path(args.market_price).expanduser())
        if market is None:
            reaction_status = "built_without_market_adjustment"
        earnings = attach_overnight_reaction(earnings, raw_prices, market)
    _write_frame(earnings, earnings_output)

    analyst_input = Path(args.analyst_input).expanduser()
    if analyst_input.exists():
        analyst = build_analyst_revision_features(pd.read_parquet(analyst_input))
        _write_frame(analyst, analyst_output)
        analyst_status = "built"
        analyst_verified = int(analyst["analyst_pit_verified"].sum()) if len(analyst) else 0
        analyst_verified_rate = float(analyst_verified / len(analyst)) if len(analyst) else 0.0
        analyst_code_count = int(analyst["code"].nunique()) if len(analyst) else 0
    else:
        analyst = pd.DataFrame()
        analyst_status = "blocked_missing_analyst_archive"
        analyst_verified = 0
        analyst_verified_rate = 0.0
        analyst_code_count = 0

    revision_count = int(earnings["information_revision_score"].notna().sum())
    surprise_count = int(surprise_mask.sum())
    raw_coverage = (
        float(1 - len(set(missing_raw_codes)) / max(earnings["code"].nunique(), 1))
        if not earnings.empty else 0.0
    )
    report = {
        "generated_at": generated_at,
        "earnings_output": str(earnings_output),
        "analyst_output": str(analyst_output) if analyst_status == "built" else None,
        "source_earnings_events": int(len(source_events)),
        "earnings_information_rows": int(len(earnings)),
        "information_revisions": revision_count,
        "strict_prior_earnings_surprises": surprise_count,
        "reaction_status": reaction_status,
        "raw_price_code_coverage": raw_coverage,
        "missing_raw_price_code_count": len(set(missing_raw_codes)),
        "analyst_status": analyst_status,
        "analyst_rows": int(len(analyst)),
        "analyst_pit_verified_rows": analyst_verified,
        "analyst_pit_verified_rate": analyst_verified_rate,
        "analyst_code_count": analyst_code_count,
        "research_ready_thresholds": {
            "strict_prior_earnings_surprises": 100,
            "raw_price_code_coverage": 0.95,
            "analyst_codes": 100,
            "analyst_pit_verified_rate": 0.95,
        },
        "research_ready": bool(
            surprise_count >= 100
            and reaction_status == "built"
            and raw_coverage >= 0.95
            and analyst_status == "built"
            and analyst_code_count >= 100
            and analyst_verified_rate >= 0.95
        ),
        "warnings": [
            "Company guidance revisions are not analyst-consensus surprises.",
            "Opening-gap features become available after the opening auction and cannot claim same-open execution.",
            "Missing analyst or raw-price history is not encoded as a zero feature.",
            "The current six-class announcement library is partial; absence of an event is not a verified zero.",
        ],
    }
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
