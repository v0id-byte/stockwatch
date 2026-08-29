"""Tests for the strict frozen-OOS panel producer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.alpha158 import QLIB_ALPHA158_FEATURES
from analysis.oos_baselines import DataContractError
from scripts import build_frozen_oos_panel as builder


def test_market_fields_use_next_open_label_and_pmo_abnormal_turnover(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=260)
    stock = pd.DataFrame({
        "trade_date": dates,
        "raw_close": np.arange(100.5, 360.5),
        "adj_open": np.arange(100.0, 360.0),
        "adj_high": np.arange(101.0, 361.0),
        "adj_low": np.arange(99.0, 359.0),
        "adj_close": np.arange(100.5, 360.5),
        "adj_vwap": np.arange(100.25, 360.25),
        "adj_factor": 1.0,
        "volume_shares": 1_000_000.0,
        "turnover": np.arange(1.0, 261.0) / 10_000,
        "amihud_1d": np.arange(1.0, 261.0) / 1e10,
        "float_market_cap": 1e10,
    })
    stock.loc[0, "amihud_1d"] = np.nan
    benchmark = pd.DataFrame({
        "trade_date": dates,
        "benchmark_return": 0.001,
        "benchmark_close_return": np.linspace(0.0001, 0.002, len(dates)),
    })

    def fake_alpha(frame):
        out = pd.DataFrame(0.0, index=frame.index, columns=QLIB_ALPHA158_FEATURES)
        out.insert(0, "trade_date", frame["trade_date"].to_numpy())
        return out

    monkeypatch.setattr(builder, "compute_qlib_alpha158_frame", fake_alpha)
    result = builder.derive_stock_market_fields(stock, benchmark, "000001")
    first = result.iloc[0]
    assert first["execution_at"] == dates[1] + pd.Timedelta(hours=9, minutes=30)
    assert first["exit_at"] == dates[2] + pd.Timedelta(hours=9, minutes=30)
    assert first["next_open_return"] == pytest.approx(stock.loc[2, "adj_open"] / stock.loc[1, "adj_open"] - 1)
    row_240 = result[result["signal_date"] == dates[239]].iloc[0]
    expected = stock.loc[220:239, "turnover"].mean() / stock.loc[0:239, "turnover"].mean()
    assert row_240["abnormal_turnover_20_240"] == pytest.approx(expected)


def test_market_fields_use_shared_calendar_and_do_not_jump_over_suspension(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=4)
    observed_dates = dates.delete(1)
    stock = pd.DataFrame({
        "trade_date": observed_dates,
        "raw_close": [100.5, 102.5, 103.5],
        "adj_open": [100.0, 102.0, 103.0],
        "adj_high": [101.0, 103.0, 104.0],
        "adj_low": [99.0, 101.0, 102.0],
        "adj_close": [100.5, 102.5, 103.5],
        "adj_vwap": [100.2, 102.2, 103.2],
        "adj_factor": 1.0,
        "volume_shares": 1_000_000.0,
        "turnover": 0.01,
        "amihud_1d": [np.nan, 1e-10, 1e-10],
        "float_market_cap": 1e10,
    })
    benchmark = pd.DataFrame({
        "trade_date": dates,
        "raw_close": 100.0,
        "benchmark_return": 0.001,
        "benchmark_close_return": 0.001,
    })

    captured = {}

    def fake_alpha(frame):
        captured["input"] = frame.copy()
        out = pd.DataFrame(0.0, index=frame.index, columns=QLIB_ALPHA158_FEATURES)
        out.insert(0, "trade_date", frame["trade_date"].to_numpy())
        return out

    monkeypatch.setattr(builder, "compute_qlib_alpha158_frame", fake_alpha)
    result = builder.derive_stock_market_fields(stock, benchmark, "000001")
    first = result[result["signal_date"] == dates[0]].iloc[0]
    suspended = captured["input"].set_index("trade_date").loc[dates[1]]
    assert suspended[["open", "high", "low", "close", "vwap", "volume"]].isna().all()
    assert first["execution_at"] == dates[1] + pd.Timedelta(hours=9, minutes=30)
    assert first["exit_at"] == dates[2] + pd.Timedelta(hours=9, minutes=30)
    assert first["next_open_return"] == pytest.approx(102.0 / 100.5 - 1)


def test_alpha158_input_uses_qlib_adjusted_volume_convention(monkeypatch):
    dates = pd.bdate_range("2025-01-02", periods=4)
    stock = pd.DataFrame({
        "trade_date": dates,
        "raw_close": 100.0,
        "adj_open": 100.0,
        "adj_high": 101.0,
        "adj_low": 99.0,
        "adj_close": 100.0,
        "adj_vwap": 100.0,
        "adj_factor": [1.0, 1.0, 2.0, 2.0],
        "volume_shares": [100.0, 100.0, 200.0, 200.0],
        "turnover": 0.01,
        "amihud_1d": [np.nan, 1e-10, 1e-10, 1e-10],
        "float_market_cap": 1e10,
    })
    benchmark = pd.DataFrame({
        "trade_date": dates,
        "benchmark_return": 0.001,
        "benchmark_close_return": 0.001,
    })
    captured = {}

    def fake_alpha(frame):
        captured["input"] = frame.copy()
        out = pd.DataFrame(0.0, index=frame.index, columns=QLIB_ALPHA158_FEATURES)
        out.insert(0, "trade_date", frame["trade_date"].to_numpy())
        return out

    monkeypatch.setattr(builder, "compute_qlib_alpha158_frame", fake_alpha)
    builder.derive_stock_market_fields(stock, benchmark, "000001")
    assert captured["input"]["close"].tolist() == pytest.approx([1.0] * 4)
    assert captured["input"]["vwap"].tolist() == pytest.approx([1.0] * 4)
    assert captured["input"]["volume"].tolist() == pytest.approx([10_000.0] * 4)


def _strict_join_inputs():
    signal = pd.Timestamp("2025-01-02")
    execution = pd.Timestamp("2025-01-03 09:30")
    frame = pd.DataFrame({
        "trade_date": [signal],
        "signal_date": [signal],
        "execution_at": [execution],
        "exit_at": [pd.Timestamp("2025-01-06 09:30")],
        "feature_available_at": [pd.Timestamp("2025-01-02 15:00")],
        "code": ["000001"],
        "execution_price_observed": [True],
        "exit_price_observed": [True],
        "next_open_return": [0.01],
        "raw_close": [100.0],
    })
    flags = pd.DataFrame({
        "trade_date": [signal, execution.normalize(), pd.Timestamp("2025-01-06")],
        "code": ["000001", "000001", "000001"],
        "universe_member": [True, True, True],
        "is_listed": [True, True, True],
        "is_st": [False, False, False],
        "is_suspended": [False, False, False],
        "is_limit_up": [False, True, False],
        "is_limit_down": [False, False, False],
    })
    weights = pd.DataFrame({
        "trade_date": [signal],
        "code": ["000001"],
        "benchmark_weight": [1.0],
        "available_at": [pd.Timestamp("2025-01-01 12:00")],
    })
    sectors = pd.DataFrame({
        "code": ["000001"], "sector": ["银行"], "available_at": [pd.Timestamp("2024-12-01")],
    })
    fundamentals = pd.DataFrame({
        "code": ["000001"],
        "trailing_eps": [8.0],
        "vintage_verified": [True],
        "available_at": [pd.Timestamp("2025-01-02 18:00")],
    })
    return frame, flags, weights, sectors, fundamentals


def test_strict_join_uses_asof_availability_and_execution_day_limits():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    result = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    assert result.iloc[0]["sector"] == "银行"
    assert result.iloc[0]["earnings_to_price"] == pytest.approx(0.08)
    assert result.iloc[0]["feature_available_at"] == pd.Timestamp("2025-01-02 18:00")
    assert not bool(result.iloc[0]["buyable"])
    assert bool(result.iloc[0]["sellable"])


def test_earnings_to_price_revalues_latest_verified_ttm_eps_each_day():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    first = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    frame = frame.copy()
    frame["raw_close"] = 200.0
    second = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    assert first.iloc[0]["earnings_to_price"] == pytest.approx(0.08)
    assert second.iloc[0]["earnings_to_price"] == pytest.approx(0.04)


def test_unverified_ttm_eps_cannot_enter_b_feature():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    fundamentals["vintage_verified"] = False
    result = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    assert result["earnings_to_price"].isna().all()


def test_unavailable_pit_earnings_leave_b_feature_missing_without_blocking_a():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    fundamentals["available_at"] = pd.Timestamp("2025-01-03 10:00")
    result = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    assert result["earnings_to_price"].isna().all()


def test_strict_join_fails_instead_of_dropping_missing_signal_flag():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    flags = flags[flags["trade_date"] != frame.loc[0, "signal_date"]]
    with pytest.raises(DataContractError, match="universe coverage is incomplete"):
        builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)


def test_missing_historical_index_weights_blocks_only_c_baseline():
    frame, flags, _weights, sectors, fundamentals = _strict_join_inputs()
    result = builder.merge_pit_inputs(frame, flags, None, sectors, fundamentals)
    assert result["benchmark_weight"].isna().all()
    assert result.iloc[0]["earnings_to_price"] == pytest.approx(0.08)
    assert result.iloc[0]["sector"] == "银行"


def test_missing_sector_and_fundamentals_still_leave_a_panel_fields():
    frame, flags, _weights, _sectors, _fundamentals = _strict_join_inputs()
    result = builder.merge_pit_inputs(frame, flags, None, None, None)
    assert result["sector"].isna().all()
    assert result["earnings_to_price"].isna().all()
    assert result.iloc[0]["feature_available_at"] == pd.Timestamp("2025-01-02 15:00")


def test_delisting_without_exit_price_gets_conservative_terminal_loss():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    frame["exit_price_observed"] = False
    flags.loc[flags["trade_date"] == pd.Timestamp("2025-01-06"), "is_listed"] = False
    result = builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)
    assert result.iloc[0]["next_open_return"] == -1.0


def test_missing_price_cannot_be_marked_tradable_by_flags():
    frame, flags, weights, sectors, fundamentals = _strict_join_inputs()
    frame["execution_price_observed"] = False
    with pytest.raises(DataContractError, match="price history is missing"):
        builder.merge_pit_inputs(frame, flags, weights, sectors, fundamentals)


def test_pit_flag_loader_rejects_conflicting_trading_states(tmp_path):
    path = tmp_path / "pit.csv"
    pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-02"],
        "code": ["000001", "000001"],
        "index_code": ["000300", "000905"],
        "is_member": [True, True],
        "is_listed": [True, True],
        "is_st": [False, True],
        "is_suspended": [False, False],
        "is_limit_up": [False, False],
        "is_limit_down": [False, False],
    }).to_csv(path, index=False)
    with pytest.raises(DataContractError, match="conflicting PIT"):
        builder._load_pit_flags(path)


def test_historical_investable_member_cannot_be_omitted_from_stock_scope():
    flags = pd.DataFrame({
        "code": ["000001", "000002"],
        "universe_member": [True, True],
    })
    with pytest.raises(DataContractError, match="missing stock parquet"):
        builder._validate_investable_stock_scope(flags, {"000001"})
