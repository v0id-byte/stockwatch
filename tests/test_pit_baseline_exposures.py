import pandas as pd
import pytest

from scripts.build_pit_baseline_exposures import (
    build_sector_events,
    build_trailing_eps_events,
)


def test_sector_events_reject_current_map_and_keep_historical_effective_time():
    current = pd.DataFrame([{
        "code": "000001", "sector": "银行", "start_date": "2024-01-01",
        "industry_code": "801780", "sector_kind": "sw_current_component",
    }])
    with pytest.raises(ValueError, match="not historical PIT"):
        build_sector_events(current)

    historical = current.assign(sector_kind="sw_historical_effective")
    result = build_sector_events(historical)
    assert result.iloc[0]["available_at"] == pd.Timestamp("2024-01-01 15:00:01")


def test_ttm_eps_requires_verified_vintages_and_uses_only_prior_periods():
    rows = []
    for period, available, eps in [
        ("20230331", "2023-04-20 15:00:01", 0.2),
        ("20231231", "2024-03-20 15:00:01", 1.0),
        ("20240331", "2024-04-20 15:00:01", 0.3),
    ]:
        rows.append({
            "code": "000001", "available_at": available, "report_period": period,
            "eps": eps, "vintage_verified": True, "source_row_sha256": period,
            "extraction_version": "test",
        })
    result = build_trailing_eps_events(pd.DataFrame(rows))
    values = result.set_index("report_period")["trailing_eps"].to_dict()
    assert values["20231231"] == pytest.approx(1.0)
    assert values["20240331"] == pytest.approx(1.1)
    assert "20230331" not in values

    unverified = pd.DataFrame(rows).assign(vintage_verified=False)
    with pytest.raises(ValueError, match="no vintage-verified"):
        build_trailing_eps_events(unverified)
