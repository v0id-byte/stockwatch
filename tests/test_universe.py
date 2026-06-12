"""Tests for Universe trading-day detection."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from datetime import date
import pandas as pd


def _make_trade_dates_df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": pd.to_datetime(dates)})


class TestIsTradingDay:
    def _universe(self):
        from data.universe import Universe
        return Universe(storage=None)

    @patch("akshare.tool_trade_date_hist_sina")
    def test_known_trading_day(self, mock_ak):
        mock_ak.return_value = _make_trade_dates_df(["2026-06-13"])
        u = self._universe()
        with patch("data.universe.date") as mock_date:
            mock_date.today.return_value = date(2026, 6, 13)
            mock_date.fromisoformat = date.fromisoformat
            assert u.is_trading_day() is True

    @patch("akshare.tool_trade_date_hist_sina")
    def test_weekend_is_not_trading(self, mock_ak):
        mock_ak.return_value = _make_trade_dates_df(["2026-06-13"])
        u = self._universe()
        with patch("data.universe.date") as mock_date:
            mock_date.today.return_value = date(2026, 6, 14)  # Sunday
            mock_date.fromisoformat = date.fromisoformat
            assert u.is_trading_day() is False

    @patch("akshare.tool_trade_date_hist_sina", side_effect=Exception("network error"))
    def test_fallback_to_holidays_table(self, mock_ak):
        from data.universe import _HOLIDAYS_FALLBACK
        u = self._universe()
        with patch("data.universe._load_trade_dates_cache", return_value=set()):
            with patch("data.universe.date") as mock_date:
                holiday = next(iter(_HOLIDAYS_FALLBACK))
                mock_date.today.return_value = date.fromisoformat(holiday)
                mock_date.fromisoformat = date.fromisoformat
                # Weekend check comes before fallback table, so pick a weekday holiday
                weekday_holiday = next(
                    d for d in _HOLIDAYS_FALLBACK
                    if date.fromisoformat(d).weekday() < 5
                )
                mock_date.today.return_value = date.fromisoformat(weekday_holiday)
                assert u.is_trading_day() is False
