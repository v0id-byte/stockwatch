#!/usr/bin/env python3
"""Aggregate cached LLM announcement scores into daily PIT-safe features.

Implements docs/llm_scoring_spec_v1.md §5 (available_at contract) and §6
(feature definitions).  Only events with ``available_trade_date <= trade_date``
ever contribute to a row.  Three count columns keep "no scored event"
distinguishable from "no announcement at all".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_VERSION = "ann_score_v1"
FEATURE_CUTOFF_HOUR = 18  # Asia/Shanghai, spec §5
HALF_LIFE_DAYS = 10
WINDOWS = (5, 20)

NEG_FAMILIES = {
    "减持": ("减持",),
    "问询处罚": ("问询监管", "处罚立案"),
    "预亏": ("业绩预告", "业绩快报"),
}


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=str(root / "announcement_llm_scores.sqlite"))
    parser.add_argument("--announcements-db", default=str(Path("~/.stockwatch/db.sqlite").expanduser()))
    parser.add_argument("--calendar", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--universe", default=str(root / "pit_universe_daily.parquet"))
    parser.add_argument("--output", default=str(root / "llm_event_features.parquet"))
    parser.add_argument("--tier", default="title")
    return parser.parse_args()


def _calendar(path: Path) -> pd.DatetimeIndex:
    dates = pd.read_parquet(path, columns=["trade_date"])["trade_date"]
    return pd.DatetimeIndex(pd.to_datetime(dates).dt.normalize().drop_duplicates().sort_values())


def _available_trade_date(published: pd.Series, quality: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    ts = pd.to_datetime(published, errors="coerce")
    out = []
    for when, q in zip(ts, quality):
        if pd.isna(when):
            out.append(pd.NaT)
            continue
        if q == "EXACT_TIMESTAMP" and when.time() <= pd.Timestamp(2000, 1, 1, FEATURE_CUTOFF_HOUR).time() \
                and when.normalize() in calendar and when.time() != pd.Timestamp(2000, 1, 1, 0, 0).time():
            out.append(when.normalize())
            continue
        pos = calendar.searchsorted(when.normalize(), side="right")
        out.append(calendar[pos] if pos < len(calendar) else pd.NaT)
    return pd.Series(out, index=published.index)


def _rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=1).sum()


def build(args: argparse.Namespace) -> dict:
    calendar = _calendar(Path(args.calendar).expanduser())
    universe = pd.read_parquet(args.universe, columns=["code"])
    codes = sorted(universe["code"].unique())

    cache = sqlite3.connect(args.cache)
    try:
        scores = pd.read_sql_query(
            "SELECT code, published_at, publication_time_quality, event_type, direction,"
            " severity, is_substantive FROM scores"
            " WHERE parse_status='ok' AND prompt_version=? AND tier=?",
            cache, params=(PROMPT_VERSION, args.tier),
        )
    finally:
        cache.close()
    scores["available_trade_date"] = _available_trade_date(
        scores["published_at"], scores["publication_time_quality"], calendar
    )
    scores = scores.dropna(subset=["available_trade_date"])

    ann = sqlite3.connect(args.announcements_db)
    try:
        counts = pd.read_sql_query(
            "SELECT code, title, published_at FROM announcements", ann)
    finally:
        ann.close()
    counts["code"] = counts["code"].astype(str).str.extract(r"(\d{6})")[0]
    counts = counts[counts["code"].isin(set(codes))].copy()
    from scripts.score_announcements_llm import PREFILTER_TERMS  # frozen vocabulary
    prefilter_pattern = "|".join(re.escape(t) for t in PREFILTER_TERMS)
    counts["prefilter_selected"] = counts["title"].str.contains(prefilter_pattern, regex=True, na=False)
    counts["available_trade_date"] = _available_trade_date(
        counts["published_at"], pd.Series("DATE_ONLY", index=counts.index), calendar
    )
    counts = counts.dropna(subset=["available_trade_date"])

    decay = np.exp(np.log(0.5) / HALF_LIFE_DAYS)
    frames = []
    scores_by_code = dict(tuple(scores.groupby("code")))
    counts_by_code = dict(tuple(counts.groupby("code")))
    for code in codes:
        base = pd.DataFrame(index=calendar)
        sub = scores_by_code.get(code)
        cnt = counts_by_code.get(code)
        daily = pd.DataFrame(0.0, index=calendar, columns=[
            "neg_sev", "pos_sev", "worst_dir", "substantive", "scored",
        ])
        if sub is not None:
            grouped = sub.groupby("available_trade_date")
            neg = grouped.apply(lambda g: float((np.minimum(g["direction"], 0) * g["severity"]).sum()))
            pos = grouped.apply(lambda g: float((np.maximum(g["direction"], 0) * g["severity"]).sum()))
            worst = grouped["direction"].min().astype(float)
            substantive = grouped["is_substantive"].sum().astype(float)
            scored = grouped.size().astype(float)
            daily.loc[neg.index, "neg_sev"] = neg
            daily.loc[pos.index, "pos_sev"] = pos
            daily.loc[worst.index, "worst_dir"] = worst
            daily.loc[substantive.index, "substantive"] = substantive
            daily.loc[scored.index, "scored"] = scored
        out = base
        for window in WINDOWS:
            out[f"llm_neg_sev_sum_{window}d"] = _rolling_sum(daily["neg_sev"], window)
            out[f"llm_pos_sev_sum_{window}d"] = _rolling_sum(daily["pos_sev"], window)
        out["llm_neg_sev_decay_20d"] = daily["neg_sev"].ewm(alpha=1 - decay, adjust=True).mean() * 20
        out["llm_pos_sev_decay_20d"] = daily["pos_sev"].ewm(alpha=1 - decay, adjust=True).mean() * 20
        out["llm_worst_direction_20d"] = daily["worst_dir"].rolling(20, min_periods=1).min()
        out["llm_substantive_count_20d"] = _rolling_sum(daily["substantive"], 20)
        out["llm_scored_count_20d"] = _rolling_sum(daily["scored"], 20)
        out["llm_any_event_20d"] = (out["llm_scored_count_20d"] > 0).astype(np.float32)
        for family, types in NEG_FAMILIES.items():
            fam_daily = pd.Series(0.0, index=calendar)
            if sub is not None:
                mask = sub["event_type"].isin(types) & (sub["direction"] < 0)
                fam = sub[mask].groupby("available_trade_date").size().astype(float)
                fam_daily.loc[fam.index] = fam
            out[f"llm_family_{family}_20d"] = _rolling_sum(fam_daily, 20)
        ann_daily = pd.Series(0.0, index=calendar)
        pre_daily = pd.Series(0.0, index=calendar)
        if cnt is not None:
            per_day = cnt.groupby("available_trade_date").size().astype(float)
            ann_daily.loc[per_day.index] = per_day
            pre_day = cnt[cnt["prefilter_selected"]].groupby("available_trade_date").size().astype(float)
            pre_daily.loc[pre_day.index] = pre_day
        out["announcement_count_20d"] = _rolling_sum(ann_daily, 20)
        out["prefilter_selected_count_20d"] = _rolling_sum(pre_daily, 20)
        out["code"] = code
        frames.append(out.reset_index(names="trade_date"))

    result = pd.concat(frames, ignore_index=True)
    feature_columns = [c for c in result.columns if c not in ("trade_date", "code")]
    result[feature_columns] = result[feature_columns].astype(np.float32)
    out_path = Path(args.output).expanduser()
    result.to_parquet(out_path, index=False)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "tier": args.tier,
        "scored_events_used": int(len(scores)),
        "announcement_rows_used": int(len(counts)),
        "codes": len(codes),
        "rows": int(len(result)),
        "features": feature_columns,
        "output": str(out_path),
    }
    Path(str(out_path) + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    report = build(args)
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
