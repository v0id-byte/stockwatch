"""Point-in-time information features that keep expectations and reactions separate.

The helpers in this module deliberately do not turn missing history into zero.
Callers must decide whether the returned coverage is sufficient for research.
"""
from __future__ import annotations

import re
from datetime import time


_REPORT_PERIOD_RE = re.compile(
    r"(?P<year>(?:19|20)\d{2})\s*年\s*"
    r"(?P<period>第一季度|一季度|半年度|中期|第三季度|三季度|年度|年)?"
)


def _require_columns(frame, required: set[str], name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {', '.join(missing)}")


def _strict_boolean(series, name: str):
    """Parse audited boolean fields without treating the string ``False`` as true."""
    import pandas as pd

    mapping = {
        True: True, False: False, 1: True, 0: False,
        "true": True, "false": False, "1": True, "0": False,
        "yes": True, "no": False,
    }
    normalized = series.map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower()
    )
    invalid = ~normalized.isin(mapping)
    if bool(invalid.any()):
        examples = pd.Series(series[invalid]).drop_duplicates().head(5).tolist()
        raise ValueError(f"{name} contains invalid boolean values: {examples}")
    return normalized.map(mapping).astype(bool)


def infer_report_period(text: str, available_at=None) -> str | None:
    """Infer a fiscal period from an announcement title/evidence string."""
    match = _REPORT_PERIOD_RE.search(str(text or ""))
    if not match:
        return None
    suffix = {
        "第一季度": "0331",
        "一季度": "0331",
        "半年度": "0630",
        "中期": "0630",
        "第三季度": "0930",
        "三季度": "0930",
        "年度": "1231",
        "年": "1231",
        None: "1231",
    }[match.group("period")]
    return f"{match.group('year')}{suffix}"


def build_earnings_revision_events(events):
    """Build forecast revisions and report-vs-forecast surprise proxies.

    A surprise is emitted only when an earlier forecast for the same stock and
    fiscal period exists. This is a company-guidance expectation proxy, not an
    analyst-consensus surprise.
    """
    import numpy as np
    import pandas as pd

    _require_columns(
        events,
        {"code", "available_at", "event_status", "signed_score"},
        "earnings events",
    )
    data = events.copy()
    if "category" in data:
        data = data[data["category"].eq("earnings")]
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["available_at"] = pd.to_datetime(data["available_at"], errors="coerce")
    data["signed_score"] = pd.to_numeric(data["signed_score"], errors="coerce")
    if "report_period" not in data:
        title = data.get("title", pd.Series("", index=data.index)).fillna("")
        evidence = data.get("evidence", pd.Series("", index=data.index)).fillna("")
        data["report_period"] = [
            infer_report_period(f"{left} {right}", available)
            for left, right, available in zip(title, evidence, data["available_at"])
        ]
    data["report_period"] = data["report_period"].astype("string")
    data["event_status"] = data["event_status"].astype(str).str.lower()
    data = data.dropna(
        subset=["code", "available_at", "signed_score", "report_period"]
    ).sort_values(["code", "report_period", "available_at"], kind="stable")

    rows = []
    for (_code, _period), group in data.groupby(
        ["code", "report_period"], sort=False
    ):
        prior_forecast = None
        for _, row in group.iterrows():
            status = row["event_status"]
            if status == "forecast":
                previous = prior_forecast
                prior_forecast = row
                kind = "forecast_initial" if previous is None else "forecast_revision"
                expected = np.nan if previous is None else float(previous["signed_score"])
                revision = np.nan if previous is None else float(row["signed_score"] - expected)
                expected_at = pd.NaT if previous is None else previous["available_at"]
                expected_hash = None if previous is None else previous.get("document_sha256")
            elif status in {"reported", "actual"}:
                previous = prior_forecast
                kind = "earnings_surprise"
                expected = np.nan if previous is None else float(previous["signed_score"])
                revision = np.nan if previous is None else float(row["signed_score"] - expected)
                expected_at = pd.NaT if previous is None else previous["available_at"]
                expected_hash = None if previous is None else previous.get("document_sha256")
            else:
                continue

            item = row.to_dict()
            item.update({
                "information_kind": kind,
                "expected_score_before": expected,
                "expectation_available_at": expected_at,
                "expectation_document_sha256": expected_hash,
                "information_revision_score": revision,
                "has_prior_expectation": previous is not None,
                "expectation_is_strictly_prior": bool(
                    previous is not None
                    and previous["available_at"] < row["available_at"]
                ),
            })
            if previous is not None and previous["available_at"] >= row["available_at"]:
                item["information_revision_score"] = np.nan
                item["has_prior_expectation"] = False
                item["expectation_is_strictly_prior"] = False
            rows.append(item)

    if not rows:
        return pd.DataFrame(columns=[
            *data.columns,
            "information_kind",
            "expected_score_before",
            "expectation_available_at",
            "expectation_document_sha256",
            "information_revision_score",
            "has_prior_expectation",
            "expectation_is_strictly_prior",
        ])
    return pd.DataFrame(rows).sort_values(["available_at", "code"], kind="stable")


def build_analyst_revision_features(reports, lookback_days: int = 180):
    """Compute publication-time analyst consensus revisions without future rows."""
    import numpy as np
    import pandas as pd

    _require_columns(
        reports,
        {"code", "published_at", "target_period", "forecast_eps"},
        "analyst reports",
    )
    data = reports.copy()
    data["code"] = data["code"].astype(str).str.zfill(6)
    data["published_at"] = pd.to_datetime(data["published_at"], errors="coerce")
    information_column = "available_at" if "available_at" in data else "published_at"
    data["_information_time"] = pd.to_datetime(
        data[information_column], errors="coerce"
    )
    if data["published_at"].isna().any() or data["_information_time"].isna().any():
        raise ValueError("analyst reports contain invalid published_at/available_at timestamps")
    if information_column == "available_at":
        impossible = data["_information_time"] < data["published_at"]
        if impossible.any():
            raise ValueError(
                f"analyst reports contain {int(impossible.sum())} available_at values before published_at"
            )
    data["forecast_eps"] = pd.to_numeric(data["forecast_eps"], errors="coerce")
    if "pit_verified" not in data:
        data["pit_verified"] = False
    data["pit_verified"] = _strict_boolean(
        data["pit_verified"].fillna(False), "analyst reports.pit_verified"
    )
    if "report_id" in data:
        identified = data["report_id"].notna() & data["report_id"].astype(str).str.strip().ne("")
        with_id = data.loc[identified].drop_duplicates(
            ["code", "target_period", "report_id"], keep="last"
        )
        without_id = data.loc[~identified]
        data = pd.concat([with_id, without_id], ignore_index=True)
    fallback_dedup = [
        column for column in (
            "code", "target_period", "_information_time", "institution", "title", "forecast_eps"
        ) if column in data
    ]
    data = data.drop_duplicates(fallback_dedup, keep="last")
    data = data.dropna(
        subset=["_information_time", "target_period", "forecast_eps"]
    ).sort_values(["code", "target_period", "_information_time"], kind="stable")

    rows = []
    lookback = pd.Timedelta(days=lookback_days)
    for (_code, _target), group in data.groupby(
        ["code", "target_period"], sort=False
    ):
        history = []
        for information_time, batch in group.groupby("_information_time", sort=True):
            cutoff = information_time - lookback
            history = [item for item in history if item[0] >= cutoff]
            before_values = [item[1] for item in history]
            batch_values = batch["forecast_eps"].astype(float).tolist()
            after_values = [*before_values, *batch_values]
            before = float(np.median(before_values)) if before_values else np.nan
            after = float(np.median(after_values))
            revision = after - before if before_values else np.nan
            item = batch.iloc[-1].to_dict()
            item.update({
                "analyst_consensus_eps_before": before,
                "analyst_consensus_eps": after,
                "analyst_revision_eps": revision,
                "analyst_revision_pct": (
                    revision / abs(before)
                    if before_values and before != 0
                    else np.nan
                ),
                "analyst_report_count_before": len(before_values),
                "analyst_report_count_180d": len(after_values),
                "analyst_reports_at_timestamp": len(batch_values),
                "analyst_pit_verified": bool(batch["pit_verified"].all()),
            })
            rows.append(item)
            history.extend((information_time, value) for value in batch_values)
    return pd.DataFrame(rows).drop(columns=["_information_time"], errors="ignore")


def attach_overnight_reaction(
    events,
    prices,
    market_prices=None,
    *,
    open_cutoff: time = time(9, 25),
):
    """Attach the first observable raw-price opening gap after an event.

    The reaction is marked available after that opening auction, so it must not
    be used to claim execution at the same open.
    """
    import numpy as np
    import pandas as pd

    _require_columns(events, {"code", "available_at"}, "events")
    _require_columns(
        prices,
        {"code", "trade_date", "raw_open", "raw_close"},
        "raw prices",
    )
    event_data = events.copy()
    event_data["code"] = event_data["code"].astype(str).str.zfill(6)
    event_data["available_at"] = pd.to_datetime(
        event_data["available_at"], errors="coerce"
    )
    px = prices.copy()
    px["code"] = px["code"].astype(str).str.zfill(6)
    px["trade_date"] = pd.to_datetime(px["trade_date"], errors="coerce").dt.normalize()
    px["raw_open"] = pd.to_numeric(px["raw_open"], errors="coerce")
    px["raw_close"] = pd.to_numeric(px["raw_close"], errors="coerce")
    px = px.dropna(subset=["trade_date"]).sort_values(["code", "trade_date"])
    px["previous_trade_date"] = px.groupby("code", sort=False)["trade_date"].shift(1)
    px["previous_raw_close"] = px.groupby("code", sort=False)["raw_close"].shift(1)
    px["overnight_gap"] = px["raw_open"] / px["previous_raw_close"] - 1.0
    # At 09:31 only the opening print and the previous close are observable.
    # Full-day volume and daily limit flags would leak information from later in
    # the session, so they are deliberately excluded from this timestamped
    # reaction feature.
    px["reaction_tradable"] = (
        np.isfinite(px["raw_open"]) & np.isfinite(px["previous_raw_close"])
    )

    calendars = {
        code: group["trade_date"].to_numpy(dtype="datetime64[ns]")
        for code, group in px.groupby("code", sort=False)
    }

    def entry_date(row):
        available = row["available_at"]
        dates = calendars.get(row["code"])
        if pd.isna(available) or dates is None or not len(dates):
            return pd.NaT
        day = np.datetime64(available.normalize(), "ns")
        side = "left" if available.time() <= open_cutoff else "right"
        index = int(np.searchsorted(dates, day, side=side))
        return pd.NaT if index >= len(dates) else pd.Timestamp(dates[index])

    event_data["reaction_trade_date"] = event_data.apply(entry_date, axis=1)
    attach_cols = [
        "code", "trade_date", "overnight_gap", "reaction_tradable",
        "raw_open", "previous_raw_close", "previous_trade_date",
    ]
    out = event_data.merge(
        px[attach_cols],
        left_on=["code", "reaction_trade_date"],
        right_on=["code", "trade_date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["trade_date"])

    if market_prices is not None:
        _require_columns(
            market_prices,
            {"trade_date", "raw_open", "raw_close"},
            "raw market prices",
        )
        market = market_prices.copy()
        market["trade_date"] = pd.to_datetime(
            market["trade_date"], errors="coerce"
        ).dt.normalize()
        market["raw_open"] = pd.to_numeric(market["raw_open"], errors="coerce")
        market["raw_close"] = pd.to_numeric(market["raw_close"], errors="coerce")
        market = market.dropna(subset=["trade_date"]).sort_values("trade_date")
        market = market.drop_duplicates("trade_date", keep="last")
        reaction_market = market[["trade_date", "raw_open"]].rename(
            columns={"raw_open": "market_reaction_raw_open"}
        )
        previous_market = market[["trade_date", "raw_close"]].rename(
            columns={"trade_date": "previous_trade_date", "raw_close": "market_previous_raw_close"}
        )
        out = out.merge(
            reaction_market,
            left_on="reaction_trade_date",
            right_on="trade_date",
            how="left",
            validate="many_to_one",
        ).drop(columns=["trade_date"])
        out = out.merge(
            previous_market,
            on="previous_trade_date",
            how="left",
            validate="many_to_one",
        )
        out["market_overnight_gap"] = (
            out["market_reaction_raw_open"] / out["market_previous_raw_close"] - 1.0
        )
        out["abnormal_overnight_gap"] = (
            out["overnight_gap"] - out["market_overnight_gap"]
        )
    else:
        out["market_overnight_gap"] = np.nan
        out["abnormal_overnight_gap"] = np.nan

    out["reaction_available_at"] = (
        out["reaction_trade_date"] + pd.Timedelta(hours=9, minutes=31)
    )
    out["same_open_execution_allowed"] = False
    return out
