#!/usr/bin/env python3
"""Synthetic point-in-time checks for sentiment/event feature construction."""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_sentiment_features import (
    _add_announcement_features,
    _add_news_features,
    _base_frame,
    _to_feature_date,
)


def _trade_dates() -> pd.Index:
    return pd.Index(pd.to_datetime([
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]))


def _grid(trade_dates: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": trade_dates,
        "code": ["000001"] * len(trade_dates),
    })


def _news_from_available(trade_dates: pd.Index, available_at: str) -> pd.DataFrame:
    feature_date = _to_feature_date(available_at, trade_dates)
    if feature_date is None:
        return pd.DataFrame(columns=["code", "feature_date", "score", "title"])
    return pd.DataFrame([{
        "code": "000001",
        "feature_date": feature_date,
        "score": 0.7,
        "title": "synthetic news",
    }])


def _ann_from_available(trade_dates: pd.Index, available_at: str) -> pd.DataFrame:
    feature_date = _to_feature_date(available_at, trade_dates)
    if feature_date is None:
        return pd.DataFrame(columns=["code", "feature_date", "title", "ann_type", "source"])
    return pd.DataFrame([{
        "code": "000001",
        "feature_date": feature_date,
        "title": "合成公告",
        "ann_type": "other",
        "source": "synthetic",
    }])


def _first_true_date(frame: pd.DataFrame, col: str) -> str:
    rows = frame[frame[col] > 0]
    assert not rows.empty, f"{col} never turns on"
    return str(pd.to_datetime(rows.iloc[0]["trade_date"]).date())


def _test_cutoff_mapping() -> None:
    dates = _trade_dates()
    assert _to_feature_date("2024-01-03 14:59:59", dates) == pd.Timestamp("2024-01-03")
    assert _to_feature_date("2024-01-03 15:00:00", dates) == pd.Timestamp("2024-01-03")
    assert _to_feature_date("2024-01-03 15:00:01", dates) == pd.Timestamp("2024-01-04")
    assert _to_feature_date("2024-01-06 10:00:00", dates) == pd.Timestamp("2024-01-08")
    assert _to_feature_date("2024-01-09 15:00:01", dates) is None


def _test_news_shift() -> None:
    dates = _trade_dates()
    frame = _base_frame(_grid(dates), dates)
    news = _news_from_available(dates, "2024-01-03 16:00:00")
    shifted = _news_from_available(dates, "2024-01-04 16:00:00")

    out = _add_news_features(frame.copy(), news)
    shifted_out = _add_news_features(frame.copy(), shifted)
    assert _first_true_date(out, "has_news_7d") == "2024-01-04"
    assert _first_true_date(shifted_out, "has_news_7d") == "2024-01-05"
    assert pd.isna(out.loc[out["trade_date"] == pd.Timestamp("2024-01-03"), "news_score_1d"].iloc[0])
    assert shifted_out.loc[
        shifted_out["trade_date"] == pd.Timestamp("2024-01-04"), "has_news_7d"
    ].iloc[0] == 0


def _test_announcement_shift() -> None:
    dates = _trade_dates()
    frame = _base_frame(_grid(dates), dates)
    ann = _ann_from_available(dates, "2024-01-03 16:00:00")
    shifted_ann = _ann_from_available(dates, "2024-01-04 16:00:00")

    out = _add_announcement_features(frame.copy(), ann)
    shifted_out = _add_announcement_features(frame.copy(), shifted_ann)
    assert _first_true_date(out, "has_ann_7d") == "2024-01-04"
    assert _first_true_date(shifted_out, "has_ann_7d") == "2024-01-05"
    assert out.loc[out["trade_date"] == pd.Timestamp("2024-01-03"), "ann_count_7d"].iloc[0] == 0
    assert shifted_out.loc[
        shifted_out["trade_date"] == pd.Timestamp("2024-01-04"), "ann_count_7d"
    ].iloc[0] == 0

    shifted_by_one_day = pd.Timestamp("2024-01-03 16:00:00") + timedelta(days=1)
    assert _to_feature_date(shifted_by_one_day, dates) == pd.Timestamp("2024-01-05")


def main() -> None:
    _test_cutoff_mapping()
    _test_news_shift()
    _test_announcement_shift()
    print("PIT sentiment checks passed")


if __name__ == "__main__":
    main()
