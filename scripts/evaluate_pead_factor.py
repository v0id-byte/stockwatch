#!/usr/bin/env python3
"""Evaluate structured signed earnings-drift events with calendar-time portfolios."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate structured PEAD factor.")
    parser.add_argument("--events-path", default=None)
    parser.add_argument("--target-horizons", default="5,20,60")
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--top-n", default="20,50,100")
    parser.add_argument("--benchmark", default="csi300")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-stock-files", type=int, default=0, help="Limit stock files for tests only.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _json_float(value) -> float | None:
    return None if pd.isna(value) else float(value)


def _distribution(values: pd.Series) -> dict | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "count": int(len(values)),
        "mean": _json_float(values.mean()),
        "median": _json_float(values.median()),
        "std": _json_float(values.std()),
        "positive_rate": float((values > 0).mean()),
        "q10": _json_float(values.quantile(0.10)),
        "q25": _json_float(values.quantile(0.25)),
        "q75": _json_float(values.quantile(0.75)),
        "q90": _json_float(values.quantile(0.90)),
    }


def _newey_west_tstat(values: pd.Series, lag: int) -> dict | None:
    x = pd.to_numeric(values, errors="coerce").dropna()
    n = len(x)
    if n < 3:
        return None
    demeaned = x - x.mean()
    lag = max(0, min(int(lag), n - 1))
    gamma0 = float((demeaned * demeaned).sum() / n)
    var = gamma0
    for k in range(1, lag + 1):
        cov = float((demeaned.iloc[k:].to_numpy() * demeaned.iloc[:-k].to_numpy()).sum() / n)
        weight = 1.0 - k / (lag + 1)
        var += 2.0 * weight * cov
    if var <= 0:
        return {"n": int(n), "mean": _json_float(x.mean()), "lag": lag, "t": None}
    se = math.sqrt(var / n)
    return {
        "n": int(n),
        "mean": _json_float(x.mean()),
        "lag": lag,
        "se": float(se),
        "t": float(x.mean() / se) if se else None,
    }


def _max_drawdown(returns: pd.Series) -> float | None:
    returns = pd.to_numeric(returns, errors="coerce").dropna()
    if returns.empty:
        return None
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return _json_float(drawdown.min())


def _load_training(root: Path, horizons: list[int]) -> pd.DataFrame:
    columns = ["trade_date", "code", *[f"forward_{h}d_return" for h in horizons]]
    train = pd.read_parquet(root / "training_set.parquet", columns=columns)
    train["trade_date"] = pd.to_datetime(train["trade_date"]).dt.normalize()
    train["code"] = train["code"].astype(str).str.zfill(6)
    return train


def _load_events(path: Path) -> pd.DataFrame:
    events = pd.read_parquet(path)
    events["code"] = events["code"].astype(str).str.zfill(6)
    events["available_at"] = pd.to_datetime(events["available_at"], errors="coerce").dt.normalize()
    events["signed_score"] = pd.to_numeric(events["signed_score"], errors="coerce")
    events["magnitude_pct"] = pd.to_numeric(events["magnitude_pct"], errors="coerce")
    events["sign"] = pd.to_numeric(events["sign"], errors="coerce")
    for col in ["strong_positive", "strong_negative", "is_turnaround", "magnitude_is_primary"]:
        if col not in events.columns:
            events[col] = False
        events[col] = events[col].fillna(False).astype(bool)
    return events.dropna(subset=["available_at", "code", "signed_score", "sign"]).copy()


def _entry_date_after(disclosure_date: pd.Timestamp, trade_dates: pd.Index) -> pd.Timestamp | None:
    if pd.isna(disclosure_date):
        return None
    idx = int(trade_dates.searchsorted(pd.Timestamp(disclosure_date).normalize(), side="right"))
    if idx >= len(trade_dates):
        return None
    return pd.Timestamp(trade_dates[idx]).normalize()


def _prepare_events(events: pd.DataFrame, trade_dates: pd.Index) -> pd.DataFrame:
    out = events.copy()
    out["entry_date"] = out["available_at"].map(lambda value: _entry_date_after(value, trade_dates))
    out = out.dropna(subset=["entry_date"]).copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"]).dt.normalize()
    source_rank = {"yjyg": 1, "yjkb": 2}
    out["source_rank"] = out["source"].map(source_rank).fillna(0)
    out["abs_score"] = out["signed_score"].abs()
    out = out.sort_values(["code", "report_period", "source", "entry_date", "abs_score"])
    out = out.drop_duplicates(["code", "report_period", "source", "entry_date"], keep="last")
    out = out.sort_values(["code", "report_period", "entry_date", "source_rank", "abs_score"])
    out = out.drop_duplicates(["code", "report_period", "entry_date"], keep="last")
    out = out.sort_values(["code", "entry_date", "abs_score"])
    out = out.drop_duplicates(["code", "entry_date"], keep="last")
    out = out.sort_values(["entry_date", "code"]).reset_index(drop=True)
    out["has_primary_magnitude"] = out["magnitude_pct"].notna() & out["magnitude_is_primary"]
    return out


def _load_market(root: Path, horizons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = root / "market_sh000300.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    market = pd.read_parquet(path)
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    market = market.sort_values("trade_date").drop_duplicates("trade_date")
    market["csi300_daily_return"] = pd.to_numeric(market["close"], errors="coerce").pct_change()
    forward = market[["trade_date"]].copy()
    close = pd.to_numeric(market["close"], errors="coerce")
    for horizon in horizons:
        forward[f"csi300_forward_{horizon}d_return"] = close.shift(-horizon) / close - 1
    return market[["trade_date", "csi300_daily_return"]], forward


def _load_stock_returns(root: Path, codes: set[str], min_date: pd.Timestamp, max_date: pd.Timestamp,
                        max_files: int = 0) -> pd.DataFrame:
    stock_dir = root / "stocks"
    frames = []
    selected = sorted(codes)
    if max_files:
        selected = selected[:max_files]
    for code in selected:
        path = stock_dir / f"{code}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        if df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.normalize()
        df = df[(df["trade_date"] >= min_date) & (df["trade_date"] <= max_date)].copy()
        if df.empty:
            continue
        df["code"] = code
        df = df.sort_values("trade_date")
        df["prev_close"] = pd.to_numeric(df["close"], errors="coerce").shift(1)
        df["daily_return"] = pd.to_numeric(df["close"], errors="coerce").pct_change()
        frames.append(df[["code", "trade_date", "open", "high", "low", "close", "prev_close", "volume", "daily_return"]])
    if not frames:
        return pd.DataFrame(columns=["code", "trade_date", "daily_return"])
    return pd.concat(frames, ignore_index=True)


def _limit_rate(code: str) -> float:
    text = str(code)
    if text.startswith(("300", "301", "688")):
        return 0.20
    return 0.10


def _attach_tradability(events: pd.DataFrame, stock_returns: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    if stock_returns.empty:
        out["entry_suspended_or_missing"] = True
        out["entry_open_limit_up"] = pd.NA
        return out
    day = stock_returns[["code", "trade_date", "open", "low", "prev_close", "volume"]].rename(
        columns={"trade_date": "entry_date"}
    )
    out = out.merge(day, on=["code", "entry_date"], how="left")
    out["entry_suspended_or_missing"] = out["open"].isna()
    rates = out["code"].map(_limit_rate)
    threshold = out["prev_close"] * (1 + rates - 0.002)
    out["entry_open_limit_up"] = (
        out["open"].notna()
        & out["prev_close"].notna()
        & (out["open"] >= threshold)
        & (out["low"] >= threshold)
    )
    return out


def _diag_for_subset(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0}
    return {
        "rows": int(len(frame)),
        "entry_suspended_or_missing": int(frame["entry_suspended_or_missing"].fillna(False).sum()),
        "entry_open_limit_up": int(frame["entry_open_limit_up"].fillna(False).sum()),
        "entry_suspended_or_missing_rate": float(frame["entry_suspended_or_missing"].fillna(False).mean()),
        "entry_open_limit_up_rate": float(frame["entry_open_limit_up"].fillna(False).mean()),
    }


def _tradability_report(events: pd.DataFrame, stock_returns: pd.DataFrame) -> dict:
    source = "local_stock_parquet" if not stock_returns.empty else "missing_local_stock_parquet"
    return {
        "limit_filter_applied": False,
        "tradability_source": source,
        "note": "Diagnostics only; missing entry bars include out-of-local-universe codes and are not necessarily suspensions.",
        "all": _diag_for_subset(events),
        "positive": _diag_for_subset(events[events["sign"] > 0]),
        "strong_positive": _diag_for_subset(events[events["strong_positive"]]),
    }


def _event_panel(events: pd.DataFrame, train: pd.DataFrame, market_forward: pd.DataFrame,
                 horizons: list[int]) -> pd.DataFrame:
    data = events.merge(
        train,
        left_on=["entry_date", "code"],
        right_on=["trade_date", "code"],
        how="left",
    )
    data = data.drop(columns=["trade_date"])
    data = data.merge(market_forward, left_on="entry_date", right_on="trade_date", how="left")
    if "trade_date" in data.columns:
        data = data.drop(columns=["trade_date"])
    for horizon in horizons:
        target = f"forward_{horizon}d_return"
        data[f"universe_forward_{horizon}d_return"] = data["entry_date"].map(
            train.groupby("trade_date")[target].mean()
        )
    return data


def _study_subset(data: pd.DataFrame, horizon: int, cost: float) -> dict | None:
    target = f"forward_{horizon}d_return"
    csi = f"csi300_forward_{horizon}d_return"
    uni = f"universe_forward_{horizon}d_return"
    subset = data.dropna(subset=[target]).copy()
    if subset.empty:
        return None
    subset["net_return"] = subset[target] - cost
    subset["net_excess_universe"] = subset[target] - subset[uni] - cost
    subset["net_excess_csi300"] = subset[target] - subset[csi] - cost
    by_year = {}
    for year, group in subset.groupby(subset["entry_date"].dt.year):
        by_year[str(int(year))] = {
            "rows": int(len(group)),
            "net_excess_universe": _distribution(group["net_excess_universe"]),
            "net_excess_csi300": _distribution(group["net_excess_csi300"]),
        }
    return {
        "rows": int(len(subset)),
        "dates": int(subset["entry_date"].nunique()),
        "codes": int(subset["code"].nunique()),
        "net_return": _distribution(subset["net_return"]),
        "net_excess_universe": _distribution(subset["net_excess_universe"]),
        "net_excess_csi300": _distribution(subset["net_excess_csi300"]),
        "by_year": by_year,
    }


def _cohorts(data: pd.DataFrame) -> dict[str, pd.Series]:
    positive = data["sign"] > 0
    positive_primary = positive & data["has_primary_magnitude"]
    threshold = data.loc[positive_primary, "signed_score"].quantile(0.8) if positive_primary.any() else np.nan
    return {
        "all_positive": positive,
        "strong_positive": data["strong_positive"],
        "positive_primary_magnitude": positive_primary,
        "positive_top_score_q80": positive_primary & (data["signed_score"] >= threshold if pd.notna(threshold) else False),
        "positive_turnaround": positive & data["is_turnaround"],
        "all_negative": data["sign"] < 0,
        "strong_negative": data["strong_negative"],
        "yjyg": data["source"].eq("yjyg"),
        "yjkb": data["source"].eq("yjkb"),
    }


def _event_study(data: pd.DataFrame, horizons: list[int], cost: float) -> dict:
    out = {}
    cohorts = _cohorts(data)
    for name, mask in cohorts.items():
        item = {}
        for horizon in horizons:
            item[str(horizon)] = _study_subset(data[mask], horizon, cost)
        out[name] = item
    return out


def _sparse_ic(data: pd.DataFrame, horizons: list[int]) -> dict:
    out = {}
    for horizon in horizons:
        target = f"forward_{horizon}d_return"
        rows = []
        for entry_date, group in data.dropna(subset=["signed_score", target]).groupby("entry_date"):
            if len(group) < 20 or group["signed_score"].nunique() <= 1:
                continue
            ic = group["signed_score"].corr(group[target], method="spearman")
            if pd.notna(ic):
                rows.append({"entry_date": str(pd.Timestamp(entry_date).date()), "ic": float(ic), "n": int(len(group))})
        values = pd.Series([row["ic"] for row in rows])
        out[str(horizon)] = {
            "count": int(len(rows)),
            "mean": _json_float(values.mean()) if len(values) else None,
            "positive_rate": float((values > 0).mean()) if len(values) else None,
            "newey_west": _newey_west_tstat(values, horizon - 1) if len(values) else None,
            "worst5": sorted(rows, key=lambda row: row["ic"])[:5],
            "best5": sorted(rows, key=lambda row: row["ic"], reverse=True)[:5],
        }
    return out


def _entry_baskets(data: pd.DataFrame, horizons: list[int], top_ns: list[int], cost: float) -> dict:
    out = {}
    positive = data[(data["sign"] > 0) & data["has_primary_magnitude"]].copy()
    for horizon in horizons:
        target = f"forward_{horizon}d_return"
        csi = f"csi300_forward_{horizon}d_return"
        uni = f"universe_forward_{horizon}d_return"
        by_n = {}
        for top_n in top_ns:
            rows = []
            for entry_date, group in positive.dropna(subset=[target]).groupby("entry_date"):
                picks = group.sort_values("signed_score", ascending=False).head(top_n)
                if picks.empty:
                    continue
                rows.append({
                    "entry_date": entry_date,
                    "names": len(picks),
                    "net_return": picks[target].mean() - cost,
                    "net_excess_universe": picks[target].mean() - picks[uni].mean() - cost,
                    "net_excess_csi300": picks[target].mean() - picks[csi].mean() - cost,
                })
            frame = pd.DataFrame(rows)
            by_n[str(top_n)] = {
                "entry_dates": int(len(frame)),
                "avg_names": _json_float(frame["names"].mean()) if not frame.empty else None,
                "net_return": _distribution(frame["net_return"]) if not frame.empty else None,
                "net_excess_universe": _distribution(frame["net_excess_universe"]) if not frame.empty else None,
                "net_excess_csi300": _distribution(frame["net_excess_csi300"]) if not frame.empty else None,
            }
        out[str(horizon)] = by_n
    return out


def _calendar_index(train: pd.DataFrame) -> tuple[pd.Index, dict[pd.Timestamp, int]]:
    dates = pd.Index(sorted(train["trade_date"].drop_duplicates()))
    return dates, {pd.Timestamp(date): idx for idx, date in enumerate(dates)}


def _calendar_portfolio(cohort: pd.DataFrame, horizon: int, cost: float, dates: pd.Index,
                        date_to_idx: dict[pd.Timestamp, int], stock_returns: pd.DataFrame,
                        csi_daily: pd.DataFrame, universe_daily: pd.DataFrame) -> dict | None:
    if cohort.empty or stock_returns.empty:
        return None
    ret_lookup = stock_returns.set_index(["code", "trade_date"])["daily_return"]
    records = []
    for row in cohort.itertuples(index=False):
        entry_idx = date_to_idx.get(pd.Timestamp(row.entry_date))
        if entry_idx is None:
            continue
        end_idx = min(entry_idx + horizon, len(dates) - 1)
        for day_idx in range(entry_idx + 1, end_idx + 1):
            day = pd.Timestamp(dates[day_idx])
            key = (row.code, day)
            if key not in ret_lookup:
                continue
            records.append({
                "trade_date": day,
                "code": row.code,
                "daily_return": ret_lookup.loc[key],
                "entry_cost": cost if day_idx == entry_idx + 1 else 0.0,
                "entry_idx": entry_idx,
                "abs_score": abs(float(row.signed_score)),
            })
    if not records:
        return None
    active = pd.DataFrame(records).dropna(subset=["daily_return"])
    active = active.sort_values(["trade_date", "code", "entry_idx", "abs_score"])
    active = active.drop_duplicates(["trade_date", "code"], keep="last")
    active["net_daily_return"] = active["daily_return"] - active["entry_cost"]
    daily = active.groupby("trade_date").agg(
        portfolio_return=("net_daily_return", "mean"),
        active_names=("code", "nunique"),
        new_positions=("entry_cost", lambda values: int((values > 0).sum())),
    ).reset_index()
    daily = daily.merge(csi_daily, on="trade_date", how="left")
    daily = daily.merge(universe_daily, on="trade_date", how="left")
    daily["excess_csi300"] = daily["portfolio_return"] - daily["csi300_daily_return"]
    daily["excess_universe"] = daily["portfolio_return"] - daily["universe_daily_return"]
    return {
        "active_days": int(len(daily)),
        "avg_active_names": _json_float(daily["active_names"].mean()),
        "avg_new_positions": _json_float(daily["new_positions"].mean()),
        "portfolio_daily_return": _distribution(daily["portfolio_return"]),
        "excess_csi300": _distribution(daily["excess_csi300"]),
        "excess_universe": _distribution(daily["excess_universe"]),
        "newey_west_excess_csi300": _newey_west_tstat(daily["excess_csi300"], horizon - 1),
        "newey_west_excess_universe": _newey_west_tstat(daily["excess_universe"], horizon - 1),
        "max_drawdown": _max_drawdown(daily["portfolio_return"]),
    }


def _calendar_portfolios(events: pd.DataFrame, horizons: list[int], cost: float, train: pd.DataFrame,
                         stock_returns: pd.DataFrame, csi_daily: pd.DataFrame,
                         universe_daily: pd.DataFrame) -> dict:
    dates, date_to_idx = _calendar_index(train)
    cohorts = _cohorts(events)
    selected = {
        "all_positive": cohorts["all_positive"],
        "strong_positive": cohorts["strong_positive"],
        "positive_primary_magnitude": cohorts["positive_primary_magnitude"],
        "positive_turnaround": cohorts["positive_turnaround"],
    }
    out = {}
    for name, mask in selected.items():
        item = {}
        for horizon in horizons:
            item[str(horizon)] = _calendar_portfolio(
                events[mask],
                horizon,
                cost,
                dates,
                date_to_idx,
                stock_returns,
                csi_daily,
                universe_daily,
            )
        out[name] = item
    return out


def _universe_daily_returns(stock_returns: pd.DataFrame) -> pd.DataFrame:
    if stock_returns.empty:
        return pd.DataFrame(columns=["trade_date", "universe_daily_return"])
    return stock_returns.groupby("trade_date")["daily_return"].mean().reset_index().rename(
        columns={"daily_return": "universe_daily_return"}
    )


def main() -> None:
    args = _parse_args()
    root = _root()
    horizons = _int_list(args.target_horizons)
    top_ns = _int_list(args.top_n)
    cost = args.cost_bps / 10000.0
    events_path = Path(args.events_path).expanduser() if args.events_path else root / "pead_events_structured.parquet"
    output = Path(args.output).expanduser() if args.output else root / "pead_factor_report.json"

    train = _load_training(root, horizons)
    trade_dates = pd.Index(sorted(train["trade_date"].drop_duplicates()))
    events = _prepare_events(_load_events(events_path), trade_dates)
    min_date = min(events["entry_date"].min(), train["trade_date"].min())
    max_date = max(train["trade_date"].max(), events["entry_date"].max() + pd.Timedelta(days=max(horizons) * 2))
    event_stock_returns = _load_stock_returns(root, set(events["code"]), min_date, max_date, args.max_stock_files)
    events = _attach_tradability(events, event_stock_returns)

    csi_daily, market_forward = _load_market(root, horizons)
    all_stock_returns = _load_stock_returns(root, set(train["code"].drop_duplicates()), train["trade_date"].min(), train["trade_date"].max(), args.max_stock_files)
    universe_daily = _universe_daily_returns(all_stock_returns)
    panel = _event_panel(events, train, market_forward, horizons)
    first_target = f"forward_{horizons[0]}d_return"
    return_available = panel[first_target].notna() if first_target in panel else pd.Series(False, index=panel.index)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "events_path": str(events_path),
        "benchmark": args.benchmark,
        "benchmark_flavor": "CSI300 price index",
        "cost_bps_round_trip": args.cost_bps,
        "target_horizons": horizons,
        "true_vintage": bool(events["true_vintage"].fillna(False).all()) if "true_vintage" in events else False,
        "best_effort_reconstructed_pit": True,
        "limit_filter_applied": False,
        "return_convention": "forward_Nd_return is assumed close-to-close from entry_date rows in training_set.parquet",
        "parse_coverage": {
            "events_input": int(len(events)),
            "codes": int(events["code"].nunique()),
            "sources": {str(key): int(value) for key, value in events["source"].value_counts().items()},
            "with_primary_magnitude": int(events["has_primary_magnitude"].sum()),
            "positive": int((events["sign"] > 0).sum()),
            "strong_positive": int(events["strong_positive"].sum()),
            "negative": int((events["sign"] < 0).sum()),
            "strong_negative": int(events["strong_negative"].sum()),
            "turnaround": int(events["is_turnaround"].sum()),
        },
        "return_coverage": {
            "events_with_any_forward_return": int(return_available.sum()),
            "events_without_forward_return": int((~return_available).sum()),
            "forward_return_available_rate": float(return_available.mean()) if len(return_available) else None,
            "note": "Primary performance metrics use only events that can be joined to training_set forward returns.",
        },
        "missing_return_diagnostics": {
            str(h): {
                "missing_event_returns": int(panel[f"forward_{h}d_return"].isna().sum()),
                "available_event_returns": int(panel[f"forward_{h}d_return"].notna().sum()),
            }
            for h in horizons
        },
        "tradability_diagnostics": _tradability_report(events, event_stock_returns),
        "calendar_time_portfolio": _calendar_portfolios(panel, horizons, cost, train, event_stock_returns, csi_daily, universe_daily),
        "event_study": _event_study(panel, horizons, cost),
        "entry_date_baskets": _entry_baskets(panel, horizons, top_ns, cost),
        "sparse_ic_auxiliary": _sparse_ic(panel, horizons),
        "notes": [
            "AKShare/Eastmoney structured rows are treated as current snapshots unless a separate vintage audit proves otherwise.",
            "Limit-up/suspension diagnostics are reported but not excluded from primary results in this run.",
            "CSI300 benchmark uses price index, not total return; alpha versus total return would be lower.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"PEAD factor report saved: {output}")
    for cohort in ("all_positive", "strong_positive", "positive_primary_magnitude", "positive_turnaround"):
        item = report["calendar_time_portfolio"].get(cohort, {})
        for horizon in horizons:
            stats = item.get(str(horizon)) or {}
            nw = stats.get("newey_west_excess_csi300") or {}
            ex = stats.get("excess_csi300") or {}
            print(
                f"{cohort} h={horizon}: active_days={stats.get('active_days')} "
                f"excess_csi300_mean={ex.get('mean')} nw_t={nw.get('t')}"
            )


if __name__ == "__main__":
    main()
