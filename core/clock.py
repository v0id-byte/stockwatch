"""Market-time helpers.

StockWatch follows the mainland China trading calendar.  Host-local wall time
is therefore never a valid source for market dates or scheduled slots.
"""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo


MARKET_TZ_NAME = "Asia/Shanghai"
MARKET_TZ = ZoneInfo(MARKET_TZ_NAME)


def market_now(value: datetime | None = None) -> datetime:
    """Return an aware datetime in the market timezone.

    Naive values are interpreted as market wall time.  This keeps existing
    tests and explicit scheduler probes readable while production calls always
    use an aware clock.
    """
    if value is None:
        return datetime.now(MARKET_TZ)
    if value.tzinfo is None:
        return value.replace(tzinfo=MARKET_TZ)
    return value.astimezone(MARKET_TZ)


def market_today(value: datetime | None = None) -> date:
    return market_now(value).date()


def market_naive_now(value: datetime | None = None) -> datetime:
    """Shanghai wall time for compatibility with legacy naive DB timestamps."""
    return market_now(value).replace(tzinfo=None)
