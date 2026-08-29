"""A-share daily price-limit rules as an auditable, standalone module.

The PIT universe builder must not hardcode limit ratios: the applicable rule
depends on board, ST status, listing age and regime dates.  Unknown boards
raise instead of guessing (fail closed).

Scope: Shanghai/Shenzhen listed A shares (main board, ChiNext, STAR).  The
Beijing exchange is intentionally unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal


# Regime effective dates (inclusive).
CHINEXT_REGISTRATION_REFORM = date(2020, 8, 24)   # ChiNext moved to +/-20%, 5 unlimited days
MAIN_BOARD_REGISTRATION_REFORM = date(2023, 2, 17)  # main board: 5 unlimited days for IPOs

_UNLIMITED = None


@dataclass(frozen=True)
class PriceLimit:
    """Limit ratios for one stock-day; ``None`` means no price limit applies."""

    up_pct: float | None
    down_pct: float | None
    regime: str

    @property
    def limited(self) -> bool:
        return self.up_pct is not None or self.down_pct is not None


def board_of(code: str) -> str:
    code = str(code).zfill(6)
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("300", "301", "302")):
        return "chinext"
    if code.startswith(("600", "601", "603", "605", "000", "001", "002", "003")):
        return "main"
    raise ValueError(f"unsupported board for code {code}; refusing to guess price limits")


def price_limit_rule(
    code: str,
    trade_date: date,
    *,
    is_st: bool,
    trading_day_index: int,
) -> PriceLimit:
    """Return the limit regime for one stock-day.

    ``trading_day_index`` is the 0-based position of ``trade_date`` in the
    stock's own trading history (0 = first listed trading day).
    """
    if trading_day_index < 0:
        raise ValueError("trading_day_index must be >= 0")
    board = board_of(code)

    if board == "star":
        if trading_day_index < 5:
            return PriceLimit(_UNLIMITED, _UNLIMITED, "star_first5_unlimited")
        return PriceLimit(0.20, 0.20, "star_20pct")

    if board == "chinext":
        if trade_date >= CHINEXT_REGISTRATION_REFORM:
            if trading_day_index < 5:
                return PriceLimit(_UNLIMITED, _UNLIMITED, "chinext_first5_unlimited")
            return PriceLimit(0.20, 0.20, "chinext_20pct")
        if trading_day_index == 0:
            return PriceLimit(0.44, 0.36, "chinext_prereform_ipo_day")
        if is_st:
            return PriceLimit(0.05, 0.05, "chinext_prereform_st_5pct")
        return PriceLimit(0.10, 0.10, "chinext_prereform_10pct")

    # main board
    if trade_date >= MAIN_BOARD_REGISTRATION_REFORM:
        if trading_day_index < 5:
            return PriceLimit(_UNLIMITED, _UNLIMITED, "main_first5_unlimited")
    elif trading_day_index == 0:
        return PriceLimit(0.44, 0.36, "main_ipo_day_44_36")
    if is_st:
        return PriceLimit(0.05, 0.05, "main_st_5pct")
    return PriceLimit(0.10, 0.10, "main_10pct")


def limit_prices(prev_close: float, limit: PriceLimit) -> tuple[float | None, float | None]:
    """Exchange rounding: half-up to 0.01 CNY on prev_close * (1 +/- pct)."""

    def _round(pct: float | None, sign: int) -> float | None:
        if pct is None:
            return None
        raw = Decimal(str(prev_close)) * (Decimal("1") + sign * Decimal(str(pct)))
        return float(raw.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    return _round(limit.up_pct, 1), _round(limit.down_pct, -1)
