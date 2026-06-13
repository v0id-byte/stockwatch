#!/usr/bin/env python3
"""Inspect local raw CNINFO announcement timestamp distribution."""
from __future__ import annotations

import argparse
import sys
from datetime import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from utils.storage import Storage


CUTOFF_TIME = time(15, 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect local CNINFO announcement raw store.")
    parser.add_argument("--codes", default="", help="Comma-separated stock codes.")
    parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD.")
    parser.add_argument("--top-times", type=int, default=10, help="Number of exact timestamp buckets to print.")
    return parser.parse_args()


def _bound(value: str | None, end: bool = False) -> str | None:
    if not value:
        return None
    ts = pd.Timestamp(value).normalize()
    if end:
        ts += pd.Timedelta(days=1)
    return ts.strftime("%Y-%m-%d")


def main() -> None:
    args = _parse_args()
    codes = [code.strip().zfill(6) for code in args.codes.split(",") if code.strip()] or None
    rows = Storage().get_announcements(
        codes=codes,
        start=_bound(args.start),
        end=_bound(args.end, end=True),
    )
    if not rows:
        print("no announcements in local raw store for this filter")
        return

    frame = pd.DataFrame(rows)
    published_at = pd.to_datetime(frame["published_at"], errors="coerce")
    frame = frame[published_at.notna()].copy()
    frame["published_at"] = published_at[published_at.notna()]
    frame["clock"] = frame["published_at"].dt.time
    frame["clock_text"] = frame["published_at"].dt.strftime("%H:%M:%S")
    before_or_at_cutoff = frame["clock"].map(lambda value: value <= CUTOFF_TIME)
    midnight = frame["clock"].map(lambda value: value == time(0, 0))
    after_cutoff_or_midnight = (~before_or_at_cutoff) | midnight

    print(f"rows={len(frame)}, codes={frame['code'].nunique()}, date_min={frame['published_at'].min()}, date_max={frame['published_at'].max()}")
    print(f"midnight_timestamp_rate={midnight.mean():.2%}")
    print(f"before_or_at_15_rate={before_or_at_cutoff.mean():.2%}")
    print(f"after_15_or_date_only_rate={after_cutoff_or_midnight.mean():.2%}")
    print("top clock buckets:")
    for clock_text, count in frame["clock_text"].value_counts().head(args.top_times).items():
        print(f"  {clock_text}: {count}")


if __name__ == "__main__":
    main()
