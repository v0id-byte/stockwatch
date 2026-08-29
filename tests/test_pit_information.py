import math

import pandas as pd
import pytest

from analysis.pit_information import (
    attach_overnight_reaction,
    build_analyst_revision_features,
    build_earnings_revision_events,
    infer_report_period,
)


def test_infer_report_period_distinguishes_quarters():
    assert infer_report_period("2024年第一季度报告") == "20240331"
    assert infer_report_period("2024年半年度业绩预告") == "20240630"
    assert infer_report_period("2024年第三季度报告") == "20240930"
    assert infer_report_period("2024年度业绩快报") == "20241231"
    assert infer_report_period("没有期间") is None


def test_earnings_surprise_uses_only_prior_same_period_forecast():
    events = pd.DataFrame([
        {"code": "1", "available_at": "2024-01-10 16:00", "event_status": "forecast",
         "signed_score": 0.4, "report_period": "20231231", "category": "earnings",
         "document_sha256": "forecast-a"},
        {"code": "1", "available_at": "2024-02-01 16:00", "event_status": "forecast",
         "signed_score": 0.7, "report_period": "20231231", "category": "earnings",
         "document_sha256": "forecast-b"},
        {"code": "1", "available_at": "2024-03-20 16:00", "event_status": "reported",
         "signed_score": 0.2, "report_period": "20231231", "category": "earnings",
         "document_sha256": "reported"},
        {"code": "1", "available_at": "2024-03-21 16:00", "event_status": "reported",
         "signed_score": 1.0, "report_period": "20240331", "category": "earnings",
         "document_sha256": "no-forecast"},
    ])

    result = build_earnings_revision_events(events)
    revision = result[result["information_kind"].eq("forecast_revision")].iloc[0]
    surprise = result[
        result["information_kind"].eq("earnings_surprise")
        & result["report_period"].eq("20231231")
    ].iloc[0]
    no_expectation = result[
        result["information_kind"].eq("earnings_surprise")
        & result["report_period"].eq("20240331")
    ].iloc[0]

    assert math.isclose(revision["information_revision_score"], 0.3)
    assert math.isclose(surprise["expected_score_before"], 0.7)
    assert math.isclose(surprise["information_revision_score"], -0.5)
    assert surprise["expectation_document_sha256"] == "forecast-b"
    assert bool(surprise["expectation_is_strictly_prior"])
    assert pd.isna(no_expectation["information_revision_score"])
    assert not bool(no_expectation["has_prior_expectation"])


def test_analyst_revision_is_publication_ordered_and_not_claimed_pit_by_default():
    reports = pd.DataFrame([
        {"code": "1", "published_at": "2024-01-01", "target_period": "2024",
         "forecast_eps": 1.0},
        {"code": "1", "published_at": "2024-01-10", "target_period": "2024",
         "forecast_eps": 2.0},
        {"code": "1", "published_at": "2024-01-20", "target_period": "2024",
         "forecast_eps": 3.0, "pit_verified": True},
    ])

    result = build_analyst_revision_features(reports)

    assert pd.isna(result.iloc[0]["analyst_revision_eps"])
    assert math.isclose(result.iloc[1]["analyst_consensus_eps_before"], 1.0)
    assert math.isclose(result.iloc[1]["analyst_consensus_eps"], 1.5)
    assert math.isclose(result.iloc[1]["analyst_revision_eps"], 0.5)
    assert result.iloc[2]["analyst_report_count_180d"] == 3
    assert not bool(result.iloc[0]["analyst_pit_verified"])
    assert bool(result.iloc[2]["analyst_pit_verified"])


def test_analyst_consensus_batches_reports_with_same_information_time():
    reports = pd.DataFrame([
        {"code": "1", "published_at": "2024-01-01", "target_period": "2024",
         "forecast_eps": 1.0, "pit_verified": True},
        {"code": "1", "published_at": "2024-01-10", "target_period": "2024",
         "forecast_eps": 2.0, "pit_verified": True},
        {"code": "1", "published_at": "2024-01-10", "target_period": "2024",
         "forecast_eps": 4.0, "pit_verified": True},
    ])

    result = build_analyst_revision_features(reports)

    assert len(result) == 2
    assert result.iloc[1]["analyst_reports_at_timestamp"] == 2
    assert result.iloc[1]["analyst_report_count_before"] == 1
    assert result.iloc[1]["analyst_report_count_180d"] == 3
    assert math.isclose(result.iloc[1]["analyst_consensus_eps_before"], 1.0)
    assert math.isclose(result.iloc[1]["analyst_consensus_eps"], 2.0)
    assert math.isclose(result.iloc[1]["analyst_revision_eps"], 1.0)


def test_analyst_string_false_is_not_treated_as_verified():
    reports = pd.DataFrame([{
        "code": "1", "published_at": "2024-01-01", "target_period": "2024",
        "forecast_eps": 1.0, "pit_verified": "False",
    }])
    result = build_analyst_revision_features(reports)
    assert not bool(result.iloc[0]["analyst_pit_verified"])

    reports.loc[0, "pit_verified"] = "unknown"
    with pytest.raises(ValueError, match="invalid boolean"):
        build_analyst_revision_features(reports)


def test_analyst_rejects_impossible_availability_and_deduplicates_report_id():
    impossible = pd.DataFrame([{
        "code": "1", "published_at": "2024-01-02", "available_at": "2024-01-01",
        "target_period": "2024", "forecast_eps": 1.0,
    }])
    with pytest.raises(ValueError, match="before published_at"):
        build_analyst_revision_features(impossible)

    duplicate = pd.DataFrame([
        {"code": "1", "published_at": "2024-01-01", "available_at": "2024-01-01",
         "target_period": "2024", "forecast_eps": 1.0, "report_id": "same"},
        {"code": "1", "published_at": "2024-01-01", "available_at": "2024-01-01",
         "target_period": "2024", "forecast_eps": 1.0, "report_id": "same"},
    ])
    result = build_analyst_revision_features(duplicate)
    assert len(result) == 1
    assert result.iloc[0]["analyst_reports_at_timestamp"] == 1


def test_overnight_reaction_is_available_after_open_and_market_adjusted():
    events = pd.DataFrame([
        {"code": "1", "available_at": "2024-01-02 16:00:00"},
    ])
    prices = pd.DataFrame([
        {"code": "1", "trade_date": "2024-01-02", "raw_open": 10.0,
         "raw_close": 10.0, "volume_shares": 1000},
        {"code": "1", "trade_date": "2024-01-03", "raw_open": 11.0,
         "raw_close": 11.2, "volume_shares": 1000},
    ])
    market = pd.DataFrame([
        {"trade_date": "2024-01-02", "raw_open": 100.0, "raw_close": 100.0},
        {"trade_date": "2024-01-03", "raw_open": 101.0, "raw_close": 102.0},
    ])

    result = attach_overnight_reaction(events, prices, market).iloc[0]

    assert result["reaction_trade_date"] == pd.Timestamp("2024-01-03")
    assert math.isclose(result["overnight_gap"], 0.10)
    assert math.isclose(result["market_overnight_gap"], 0.01)
    assert math.isclose(result["abnormal_overnight_gap"], 0.09)
    assert result["reaction_available_at"] == pd.Timestamp("2024-01-03 09:31")
    assert not bool(result["same_open_execution_allowed"])


def test_overnight_reaction_does_not_use_full_day_volume_as_of_0931():
    events = pd.DataFrame([{"code": "1", "available_at": "2024-01-02 16:00:00"}])
    prices = pd.DataFrame([
        {"code": "1", "trade_date": "2024-01-02", "raw_open": 10.0,
         "raw_close": 10.0, "volume_shares": 1000},
        {"code": "1", "trade_date": "2024-01-03", "raw_open": 11.0,
         "raw_close": 11.0, "volume_shares": 0, "is_limit_up": True},
    ])
    result = attach_overnight_reaction(events, prices).iloc[0]
    assert bool(result["reaction_tradable"])
    assert result["reaction_available_at"] == pd.Timestamp("2024-01-03 09:31")


def test_overnight_market_adjustment_uses_stock_previous_trade_date_after_suspension():
    events = pd.DataFrame([{"code": "1", "available_at": "2024-01-02 16:00:00"}])
    prices = pd.DataFrame([
        {"code": "1", "trade_date": "2024-01-02", "raw_open": 10.0, "raw_close": 10.0},
        {"code": "1", "trade_date": "2024-01-05", "raw_open": 11.0, "raw_close": 11.0},
    ])
    market = pd.DataFrame([
        {"trade_date": "2024-01-02", "raw_open": 100.0, "raw_close": 100.0},
        {"trade_date": "2024-01-03", "raw_open": 101.0, "raw_close": 101.0},
        {"trade_date": "2024-01-04", "raw_open": 102.0, "raw_close": 103.0},
        {"trade_date": "2024-01-05", "raw_open": 104.0, "raw_close": 104.0},
    ])

    result = attach_overnight_reaction(events, prices, market).iloc[0]

    assert result["reaction_trade_date"] == pd.Timestamp("2024-01-05")
    assert result["previous_trade_date"] == pd.Timestamp("2024-01-02")
    assert result["market_overnight_gap"] == pytest.approx(0.04)
    assert result["abnormal_overnight_gap"] == pytest.approx(0.06)
