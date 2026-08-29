import pandas as pd
import pytest

from scripts.build_pead_events import (
    _conservative_magnitude_pct,
    _preannounce_rows,
    _sign_from_preannounce_type,
)
from scripts.evaluate_pead_factor import _calendar_portfolio, _entry_date_after


def test_preannounce_rows_keep_chinese_columns():
    raw = pd.DataFrame([{
        "股票代码": "1",
        "股票简称": "测试股份",
        "预测指标": "归属于上市公司股东的净利润",
        "预告类型": "预增",
        "业绩变动": "预计增长50%至80%",
        "业绩变动幅度": None,
        "公告日期": "2024-01-31",
        "预测数值": "1000",
        "上年同期值": "600",
        "业绩变动原因": "主营业务增长",
    }])

    rows = _preannounce_rows(raw, "20231231", "2024-02-01T00:00:00")

    assert len(rows) == 1
    assert rows[0]["code"] == "000001"
    assert rows[0]["sign"] == 1
    assert rows[0]["magnitude_pct"] == 50.0
    assert rows[0]["magnitude_source"] == "raw_change_text_percent"


def test_negative_range_uses_conservative_worse_bound():
    magnitude, details = _conservative_magnitude_pct("预减", "预计下降50%至80%", None, -1, False)

    assert magnitude == 80.0
    assert details["magnitude_lower_pct"] == 50.0
    assert details["magnitude_upper_pct"] == 80.0


def test_turnaround_magnitude_is_direction_only():
    sign, event_type, is_turnaround = _sign_from_preannounce_type("扭亏为盈")
    magnitude, details = _conservative_magnitude_pct("扭亏为盈", "增长9999%", 9999, sign, is_turnaround)

    assert sign == 1
    assert event_type == "positive"
    assert is_turnaround is True
    assert magnitude is None
    assert details["magnitude_source"] == "turnaround_direction_only"
    assert details["magnitude_is_primary"] is False


def test_entry_date_after_weekend_uses_next_trading_day():
    trade_dates = pd.Index(pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]))

    entry = _entry_date_after(pd.Timestamp("2024-01-06"), trade_dates)

    assert entry == pd.Timestamp("2024-01-08")


def test_calendar_portfolio_uses_overlapping_holdings_and_entry_cost():
    dates = pd.Index(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]))
    date_to_idx = {date: idx for idx, date in enumerate(dates)}
    cohort = pd.DataFrame([{
        "code": "000001",
        "entry_date": pd.Timestamp("2024-01-02"),
        "signed_score": 1.0,
    }])
    stock_returns = pd.DataFrame([
        {"code": "000001", "trade_date": pd.Timestamp("2024-01-03"), "daily_return": 0.01},
        {"code": "000001", "trade_date": pd.Timestamp("2024-01-04"), "daily_return": 0.02},
    ])
    csi_daily = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-03"), "csi300_daily_return": 0.0},
        {"trade_date": pd.Timestamp("2024-01-04"), "csi300_daily_return": 0.0},
    ])
    universe_daily = pd.DataFrame([
        {"trade_date": pd.Timestamp("2024-01-03"), "universe_daily_return": 0.0},
        {"trade_date": pd.Timestamp("2024-01-04"), "universe_daily_return": 0.0},
    ])

    result = _calendar_portfolio(
        cohort,
        horizon=2,
        cost=0.002,
        dates=dates,
        date_to_idx=date_to_idx,
        stock_returns=stock_returns,
        csi_daily=csi_daily,
        universe_daily=universe_daily,
    )

    assert result["active_days"] == 2
    assert result["avg_active_names"] == 1.0
    assert result["avg_new_positions"] == 0.5
    assert result["portfolio_daily_return"]["mean"] == pytest.approx(0.014)
