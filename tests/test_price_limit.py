from datetime import date

import pytest

from analysis.price_limit import board_of, limit_prices, price_limit_rule


SEASONED = 300  # trading_day_index far past any IPO window


def _prices(code, d, prev_close, *, is_st=False, idx=SEASONED):
    rule = price_limit_rule(code, d, is_st=is_st, trading_day_index=idx)
    return limit_prices(prev_close, rule), rule


class TestBoard:
    def test_boards(self):
        assert board_of("600519") == "main"
        assert board_of("000001") == "main"
        assert board_of("002415") == "main"
        assert board_of("300750") == "chinext"
        assert board_of("301236") == "chinext"
        assert board_of("688981") == "star"

    def test_unknown_board_fails_closed(self):
        with pytest.raises(ValueError):
            board_of("830799")  # Beijing exchange: unsupported on purpose


class TestGoldenCases:
    """Regime matrix with exchange half-up rounding, asserted on final prices."""

    def test_main_board_10pct(self):
        (up, down), rule = _prices("600519", date(2024, 3, 1), 10.00)
        assert (up, down) == (11.00, 9.00)
        assert rule.regime == "main_10pct"

    def test_main_board_st_5pct_half_up_rounding(self):
        # 4.50 * 1.05 = 4.725 -> 4.73 (half-up; banker's rounding would give 4.72)
        (up, down), rule = _prices("000662", date(2024, 3, 1), 4.50, is_st=True)
        assert (up, down) == (4.73, 4.28)
        assert rule.regime == "main_st_5pct"

    def test_chinext_pre_reform_10pct(self):
        (up, down), rule = _prices("300059", date(2020, 8, 21), 20.00)
        assert (up, down) == (22.00, 18.00)
        assert rule.regime == "chinext_prereform_10pct"

    def test_chinext_post_reform_20pct(self):
        (up, down), rule = _prices("300059", date(2020, 8, 24), 20.00)
        assert (up, down) == (24.00, 16.00)
        assert rule.regime == "chinext_20pct"

    def test_chinext_st_keeps_20pct_after_reform(self):
        (up, down), rule = _prices("300116", date(2024, 3, 1), 5.00, is_st=True)
        assert (up, down) == (6.00, 4.00)
        assert rule.regime == "chinext_20pct"

    def test_star_20pct(self):
        (up, down), rule = _prices("688981", date(2024, 3, 1), 50.00)
        assert (up, down) == (60.00, 40.00)
        assert rule.regime == "star_20pct"

    def test_main_ipo_day_pre_registration_44_36(self):
        (up, down), rule = _prices("601728", date(2021, 8, 20), 4.53, idx=0)
        # 4.53*1.44=6.5232 -> 6.52 ; 4.53*0.64=2.8992 -> 2.90
        assert (up, down) == (6.52, 2.90)
        assert rule.regime == "main_ipo_day_44_36"

    def test_main_first5_unlimited_after_registration_reform(self):
        for idx in range(5):
            (up, down), rule = _prices("601061", date(2023, 3, 1), 10.0, idx=idx)
            assert (up, down) == (None, None)
            assert not rule.limited
        (up, _), _ = _prices("601061", date(2023, 3, 8), 10.0, idx=5)
        assert up == 11.00

    def test_star_chinext_first5_unlimited(self):
        for code in ("688599", "301269"):
            (up, down), rule = _prices(code, date(2024, 3, 1), 30.0, idx=3)
            assert (up, down) == (None, None)

    def test_float_repr_trap(self):
        # 29.71 * 1.1 = 32.681000000000004 in binary floats; must still yield 32.68
        (up, _), _ = _prices("600000", date(2024, 3, 1), 29.71)
        assert up == 32.68

    def test_negative_index_rejected(self):
        with pytest.raises(ValueError):
            price_limit_rule("600000", date(2024, 3, 1), is_st=False, trading_day_index=-1)
