#!/usr/bin/env python3
"""Build the one strict PIT panel consumed by frozen OOS baseline research.

The builder joins normalized per-stock history, exact Qlib Alpha158, PIT
universe/tradeability, CSI500 weights, PIT industry, and PIT earnings-to-price.
It always writes an audit report.  Missing inputs or incomplete PIT coverage
produce ``BLOCKED`` and no evaluable panel instead of guessed zero exposures.
"""
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

from analysis.alpha158 import QLIB_ALPHA158_FEATURES, compute_qlib_alpha158_frame
from analysis.oos_baselines import DataContractError


STOCK_COLUMNS = (
    "trade_date",
    "raw_close",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_vwap",
    "adj_factor",
    "volume_shares",
    "turnover",
    "amihud_1d",
    "float_market_cap",
)
PIT_COLUMNS = (
    "trade_date",
    "code",
    "index_code",
    "is_member",
    "is_listed",
    "is_st",
    "is_suspended",
    "is_limit_up",
    "is_limit_down",
)
WEIGHT_COLUMNS = ("trade_date", "code", "index_code", "benchmark_weight", "available_at")
SECTOR_COLUMNS = ("code", "sector", "available_at")
FUNDAMENTAL_COLUMNS = ("code", "trailing_eps", "available_at", "vintage_verified")


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Build strict PIT frozen-OOS research panel.")
    parser.add_argument("--root", default=str(root))
    parser.add_argument("--pit-flags", default=str(root / "pit_universe_daily.parquet"))
    parser.add_argument("--benchmark", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--benchmark-weights", default=str(root / "csi500_weights_pit.parquet"))
    parser.add_argument("--sector-events", default=str(root / "sector_pit.parquet"))
    parser.add_argument("--fundamental-events", default=str(root / "fundamental_pit.parquet"))
    parser.add_argument("--output", default=str(root / "oos_baseline_panel.parquet"))
    parser.add_argument("--report", default=str(root / "oos_baseline_panel.report.json"))
    parser.add_argument("--max-codes", type=int, default=0, help="Smoke-only partial build; never evaluable.")
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataContractError(f"required PIT input is missing: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataContractError(f"{source} missing columns: {missing}")


def _normalize_code(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.strip().str.extract(r"(\d{1,6})(?:\.0)?$", expand=False)
    return extracted.map(lambda value: str(value).zfill(6) if pd.notna(value) and str(value) else "")


def _normalize_boolean(frame: pd.DataFrame, columns: tuple[str, ...], source: str) -> pd.DataFrame:
    out = frame.copy()
    for name in columns:
        if out[name].isna().any():
            raise DataContractError(f"{source}.{name} contains unknown values")
        values = set(out[name].dropna().unique().tolist())
        if not values.issubset({0, 1, True, False}):
            raise DataContractError(f"{source}.{name} must be boolean")
        out[name] = out[name].astype(bool)
    return out


def _load_pit_flags(path: Path) -> pd.DataFrame:
    raw = _read_table(path)
    _require(raw, PIT_COLUMNS, str(path))
    raw = raw[list(PIT_COLUMNS)].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["code"] = _normalize_code(raw["code"])
    raw = _normalize_boolean(
        raw,
        ("is_member", "is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down"),
        str(path),
    )
    if raw[["trade_date", "code"]].isna().any().any() or raw["code"].isin({"", "000000"}).any():
        raise DataContractError("PIT flags contain invalid keys")
    state_columns = ("is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down")
    conflicts = raw.groupby(["trade_date", "code"])[list(state_columns)].nunique(dropna=False).max(axis=1) > 1
    if conflicts.any():
        examples = [
            {"trade_date": str(pd.Timestamp(date).date()), "code": code}
            for date, code in conflicts[conflicts].index[:5]
        ]
        raise DataContractError(f"conflicting PIT trading-state flags across index rows: {examples}")
    grouped = raw.groupby(["trade_date", "code"], as_index=False).agg({
        "is_member": "max",
        "is_listed": "max",
        "is_st": "max",
        "is_suspended": "max",
        "is_limit_up": "max",
        "is_limit_down": "max",
    })
    grouped["universe_member"] = grouped["is_member"] & grouped["is_listed"] & ~grouped["is_st"]
    return grouped.sort_values(["trade_date", "code"])


def _load_benchmark(path: Path) -> pd.DataFrame:
    raw = _read_table(path)
    open_col = "raw_open" if "raw_open" in raw.columns else "open"
    close_col = "raw_close" if "raw_close" in raw.columns else "close"
    _require(raw, ("trade_date", open_col, close_col), str(path))
    out = raw[["trade_date", open_col, close_col]].copy().sort_values("trade_date")
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["benchmark_open"] = pd.to_numeric(out[open_col], errors="coerce")
    out["benchmark_close"] = pd.to_numeric(out[close_col], errors="coerce")
    out["benchmark_return"] = out["benchmark_open"].shift(-2) / out["benchmark_open"].shift(-1) - 1
    out["benchmark_close_return"] = out["benchmark_close"].pct_change(fill_method=None)
    if out[["trade_date", "benchmark_open", "benchmark_close"]].isna().any().any():
        raise DataContractError("benchmark history contains invalid dates or prices")
    return out[["trade_date", "benchmark_return", "benchmark_close_return"]]


def _load_weights(path: Path) -> pd.DataFrame:
    raw = _read_table(path)
    _require(raw, WEIGHT_COLUMNS, str(path))
    raw = raw[list(WEIGHT_COLUMNS)].copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"], errors="coerce").dt.normalize()
    raw["available_at"] = pd.to_datetime(raw["available_at"], errors="coerce")
    raw["code"] = _normalize_code(raw["code"])
    raw["benchmark_weight"] = pd.to_numeric(raw["benchmark_weight"], errors="coerce")
    index = raw["index_code"].astype(str).str.lower().str.replace("sh", "", regex=False)
    raw = raw[index.str.extract(r"(\d{6})", expand=False) == "000905"]
    if (
        raw.empty
        or raw[["trade_date", "available_at", "benchmark_weight"]].isna().any().any()
        or raw["code"].isin({"", "000000"}).any()
    ):
        raise DataContractError("CSI500 PIT weights are empty or invalid")
    if (raw["benchmark_weight"] <= 0).any():
        raise DataContractError("CSI500 weight rows must be positive constituents only")
    if raw.duplicated(["trade_date", "code"]).any():
        raise DataContractError("CSI500 PIT weights contain duplicate date/code rows")
    sums = raw.groupby("trade_date")["benchmark_weight"].sum()
    if ((sums < 0.995) | (sums > 1.005)).any():
        raise DataContractError("CSI500 PIT weights do not sum to one by date")
    return raw.drop(columns="index_code").sort_values(["trade_date", "code"])


def _load_events(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    raw = _read_table(path)
    _require(raw, columns, str(path))
    raw = raw[list(columns)].copy()
    raw["code"] = _normalize_code(raw["code"])
    raw["available_at"] = pd.to_datetime(raw["available_at"], errors="coerce")
    if raw[["code", "available_at"]].isna().any().any() or raw["code"].isin({"", "000000"}).any():
        raise DataContractError(f"{path} contains invalid PIT event keys")
    if raw.duplicated(["code", "available_at"]).any():
        raise DataContractError(f"{path} contains duplicate code/available_at rows")
    if "trailing_eps" in raw:
        raw["trailing_eps"] = pd.to_numeric(raw["trailing_eps"], errors="coerce")
        raw = _normalize_boolean(raw, ("vintage_verified",), str(path))
    return raw.sort_values(["available_at", "code"])


def _asof_event(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    value: str,
    extra_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    event = events[["code", "available_at", value, *extra_columns]].rename(
        columns={"available_at": f"{value}_available_at"},
    )
    left = frame.sort_values(["execution_at", "code"])
    right = event.sort_values([f"{value}_available_at", "code"])
    return pd.merge_asof(
        left,
        right,
        left_on="execution_at",
        right_on=f"{value}_available_at",
        by="code",
        direction="backward",
        allow_exact_matches=False,
    ).sort_values(["signal_date", "code"])


def derive_stock_market_fields(stock: pd.DataFrame, benchmark: pd.DataFrame, code: str) -> pd.DataFrame:
    """Derive exact Alpha158 inputs, next-open label, PMO proxy, and risk fields."""

    _require(stock, STOCK_COLUMNS, f"stock:{code}")
    data = stock[list(STOCK_COLUMNS)].copy().sort_values("trade_date")
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce").dt.normalize()
    for name in STOCK_COLUMNS[1:]:
        data[name] = pd.to_numeric(data[name], errors="coerce")
    if data["trade_date"].isna().any() or data.duplicated("trade_date").any():
        raise DataContractError(f"stock:{code} contains invalid or duplicate dates")
    observed_required = [name for name in STOCK_COLUMNS[1:] if name != "amihud_1d"]
    if data[observed_required].isna().any().any():
        raise DataContractError(f"stock:{code} has missing normalized market fields")
    amihud = data["amihud_1d"].dropna()
    missing_amihud = int(data["amihud_1d"].isna().sum())
    allowed_missing = max(1, int(math.ceil(len(data) * 0.10)))
    if (
        missing_amihud > allowed_missing
        or not np.isfinite(amihud.to_numpy(dtype=float)).all()
    ):
        raise DataContractError(f"stock:{code} has invalid or insufficient Amihud observations")

    if (data["adj_factor"] <= 0).any():
        raise DataContractError(f"stock:{code} has non-positive adjustment factors")
    first_hfq_close = float(data["adj_close"].iloc[0])
    if not np.isfinite(first_hfq_close) or first_hfq_close <= 0:
        raise DataContractError(f"stock:{code} has invalid first HFQ close")
    # Match Qlib's collector convention: normalized adjusted price and
    # inverse-factor adjusted volume.  The common first-close multiplier on
    # volume preserves Qlib's price-volume operator scale convention.
    alpha_observed = pd.DataFrame({
        "trade_date": data["trade_date"],
        "open": data["adj_open"] / first_hfq_close,
        "high": data["adj_high"] / first_hfq_close,
        "low": data["adj_low"] / first_hfq_close,
        "close": data["adj_close"] / first_hfq_close,
        "vwap": data["adj_vwap"] / first_hfq_close,
        "volume": data["volume_shares"] / data["adj_factor"] * first_hfq_close,
    })
    calendar = pd.Index(pd.to_datetime(benchmark["trade_date"]).dt.normalize().unique()).sort_values()
    calendar = calendar[calendar >= data["trade_date"].min()]
    off_calendar = set(data["trade_date"]) - set(calendar)
    if off_calendar:
        raise DataContractError(f"stock:{code} contains dates outside the benchmark trading calendar")
    alpha_calendar = alpha_observed.set_index("trade_date").reindex(calendar)
    alpha_calendar.index.name = "trade_date"
    alpha = compute_qlib_alpha158_frame(alpha_calendar.reset_index()).reset_index(drop=True)
    if "trade_date" in alpha.columns:
        alpha = alpha.drop(columns="trade_date")
    data = data.set_index("trade_date").reindex(calendar)
    data.index.name = "trade_date"
    data = pd.concat([data.reset_index(), alpha[list(QLIB_ALPHA158_FEATURES)]], axis=1).set_index("trade_date")
    data["price_observed"] = data["adj_open"].notna()
    data["volume_shares"] = data["volume_shares"].fillna(0.0)
    data["turnover"] = data["turnover"].fillna(0.0)
    data["float_market_cap"] = data["float_market_cap"].ffill()
    data["raw_close"] = data["raw_close"].ffill()
    valuation_close = data["adj_close"].ffill()
    valuation_open = data["adj_open"].where(data["price_observed"], valuation_close)
    data = data.reset_index()

    data["signal_date"] = data["trade_date"]
    data["execution_at"] = data["trade_date"].shift(-1) + pd.Timedelta(hours=9, minutes=30)
    data["exit_at"] = data["trade_date"].shift(-2) + pd.Timedelta(hours=9, minutes=30)
    data["execution_price_observed"] = data["price_observed"].shift(-1).fillna(False).astype(bool)
    data["exit_price_observed"] = data["price_observed"].shift(-2).fillna(False).astype(bool)
    data["feature_available_at"] = data["trade_date"] + pd.Timedelta(hours=15)
    data["next_open_return"] = valuation_open.shift(-2).to_numpy() / valuation_open.shift(-1).to_numpy() - 1
    data["turnover_20d"] = data["turnover"].rolling(20, min_periods=20).mean()
    data["abnormal_turnover_20_240"] = data["turnover_20d"] / data["turnover"].rolling(240, min_periods=240).mean()
    data["amihud_20d"] = data["amihud_1d"].rolling(20, min_periods=20).mean()
    stock_return = valuation_close.pct_change(fill_method=None).to_numpy()
    stock_return = pd.Series(stock_return, index=data.index)
    data["volatility_20d"] = stock_return.rolling(20, min_periods=20).std() * math.sqrt(252)
    data["log_market_cap"] = np.log(data["float_market_cap"].where(data["float_market_cap"] > 0))
    data = data.merge(benchmark, on="trade_date", how="left", validate="one_to_one")
    market_return = data["benchmark_close_return"]
    data["beta"] = stock_return.rolling(60, min_periods=60).cov(market_return) / market_return.rolling(60, min_periods=60).var()
    data["code"] = code
    return data.dropna(subset=["execution_at", "exit_at", "next_open_return", "benchmark_return"])


def merge_pit_inputs(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    weights: pd.DataFrame | None,
    sectors: pd.DataFrame | None,
    fundamentals: pd.DataFrame | None,
) -> pd.DataFrame:
    """Join exact-date execution flags and as-of exposures without imputation."""

    signal_flags = flags[["trade_date", "code", "universe_member"]]
    out = frame.merge(
        signal_flags,
        left_on=["signal_date", "code"],
        right_on=["trade_date", "code"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_signal_flag"),
    ).drop(columns="trade_date_signal_flag")
    if out["universe_member"].isna().any():
        missing = out.loc[out["universe_member"].isna(), ["signal_date", "code"]].head(5).to_dict("records")
        raise DataContractError(f"signal-date PIT universe coverage is incomplete: {missing}")
    out["universe_member"] = out["universe_member"].astype(bool)
    execution_flags = flags[[
        "trade_date", "code", "is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down",
    ]].rename(columns={"trade_date": "execution_date"})
    out["execution_date"] = out["execution_at"].dt.normalize()
    out = out.merge(execution_flags, on=["execution_date", "code"], how="left", validate="one_to_one")
    if out[["is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down"]].isna().any().any():
        raise DataContractError("execution-day PIT tradeability flags are incomplete")
    stale_price = ~out["execution_price_observed"] & out["is_listed"] & ~out["is_suspended"]
    if stale_price.any():
        examples = out.loc[stale_price, ["execution_date", "code"]].head(5).to_dict("records")
        raise DataContractError(f"price history is missing while PIT flags say tradable: {examples}")
    out["buyable"] = (
        out["execution_price_observed"]
        & out["is_listed"]
        & ~out["is_st"]
        & ~out["is_suspended"]
        & ~out["is_limit_up"]
    )
    out["sellable"] = (
        out["execution_price_observed"]
        & out["is_listed"]
        & ~out["is_suspended"]
        & ~out["is_limit_down"]
    )
    exit_flags = flags[["trade_date", "code", "is_listed"]].rename(columns={
        "trade_date": "exit_date",
        "is_listed": "exit_is_listed",
    })
    out["exit_date"] = out["exit_at"].dt.normalize()
    out = out.merge(exit_flags, on=["exit_date", "code"], how="left", validate="one_to_one")
    if out["exit_is_listed"].isna().any():
        raise DataContractError("exit-day PIT listing flags are incomplete")
    terminal = out["is_listed"] & ~out["exit_is_listed"] & ~out["exit_price_observed"]
    out.loc[terminal, "next_open_return"] = -1.0

    if weights is None:
        out["benchmark_weight"] = np.nan
        out["benchmark_weight_available_at"] = pd.NaT
    else:
        weight = weights.rename(columns={"available_at": "benchmark_weight_available_at"})
        out = out.merge(
            weight,
            left_on=["signal_date", "code"],
            right_on=["trade_date", "code"],
            how="left",
            validate="one_to_one",
        )
        out["benchmark_weight"] = out["benchmark_weight"].fillna(0.0)
        out = out.drop(columns=[name for name in ("trade_date_y", "trade_date") if name in out.columns])
    if sectors is None:
        out["sector"] = None
        out["sector_available_at"] = pd.NaT
    else:
        out = _asof_event(out, sectors, "sector")
    if fundamentals is None:
        out["trailing_eps"] = np.nan
        out["vintage_verified"] = False
        out["trailing_eps_available_at"] = pd.NaT
    else:
        out = _asof_event(out, fundamentals, "trailing_eps", ("vintage_verified",))
        verified = out["vintage_verified"].fillna(False).astype(bool)
        out.loc[~verified, "trailing_eps"] = np.nan
    out["earnings_to_price"] = pd.to_numeric(out["trailing_eps"], errors="coerce") / out["raw_close"]
    if weights is not None and out.loc[out["benchmark_weight"] > 0, "benchmark_weight_available_at"].isna().any():
        raise DataContractError("CSI500 constituent weight availability timestamp is missing")
    availability = out[[
        "feature_available_at", "sector_available_at", "trailing_eps_available_at",
        "benchmark_weight_available_at",
    ]]
    out["feature_available_at"] = availability.max(axis=1)
    if (out["feature_available_at"] >= out["execution_at"]).any():
        raise DataContractError("an exposure was not public before next-open execution")
    return out


def _output_columns() -> list[str]:
    return [
        "signal_date", "execution_at", "exit_at", "feature_available_at", "code",
        "next_open_return", "buyable", "sellable", "universe_member", "benchmark_return",
        "benchmark_weight", "sector", "log_market_cap", "beta", "volatility_20d",
        "earnings_to_price", "turnover_20d", "abnormal_turnover_20_240", "amihud_20d",
        *QLIB_ALPHA158_FEATURES,
    ]


def _validate_investable_stock_scope(flags: pd.DataFrame, stock_codes: set[str]) -> None:
    investable_codes = set(flags.loc[flags["universe_member"], "code"].astype(str))
    missing = sorted(investable_codes - stock_codes)
    if missing:
        raise DataContractError(
            "historical PIT universe members are missing stock parquet history: "
            f"{missing[:10]}"
        )


def build_panel(args: argparse.Namespace) -> dict:
    root = Path(args.root).expanduser()
    stock_dir = root / "stocks"
    if not stock_dir.exists():
        raise DataContractError(f"normalized stock directory is missing: {stock_dir}")
    flags = _load_pit_flags(Path(args.pit_flags).expanduser())
    benchmark = _load_benchmark(Path(args.benchmark).expanduser())
    weight_path = Path(args.benchmark_weights).expanduser()
    weight_error = None
    weights = None
    if weight_path.exists():
        try:
            weights = _load_weights(weight_path)
        except DataContractError as exc:
            weight_error = str(exc)
    else:
        weight_error = f"historical CSI500 PIT weights missing: {weight_path}"
    sector_path = Path(args.sector_events).expanduser()
    sector_error = None
    sectors = None
    if sector_path.exists():
        try:
            sectors = _load_events(sector_path, SECTOR_COLUMNS)
        except DataContractError as exc:
            sector_error = str(exc)
    else:
        sector_error = f"PIT sector events missing: {sector_path}"
    fundamental_path = Path(args.fundamental_events).expanduser()
    fundamental_error = None
    fundamentals = None
    if fundamental_path.exists():
        try:
            fundamentals = _load_events(fundamental_path, FUNDAMENTAL_COLUMNS)
        except DataContractError as exc:
            fundamental_error = str(exc)
    else:
        fundamental_error = f"verified PIT trailing EPS missing: {fundamental_path}"
    stock_paths = sorted(stock_dir.glob("*.parquet"))
    if not stock_paths:
        raise DataContractError("no normalized stock parquet files")
    stock_codes = {path.stem.zfill(6) for path in stock_paths}
    _validate_investable_stock_scope(flags, stock_codes)
    if weights is not None:
        weight_codes = set(weights["code"].unique())
        missing_weight_history = sorted(weight_codes - stock_codes)
        if missing_weight_history:
            weight_error = f"CSI500 constituents missing normalized stock history: {missing_weight_history[:10]}"
            weights = None
    partial = bool(args.max_codes)
    if args.max_codes:
        stock_paths = stock_paths[: args.max_codes]

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - train dependency
        raise RuntimeError("panel build requires pyarrow from requirements-train.txt") from exc

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    writer = None
    rows = 0
    processed = []
    weight_mass: dict[pd.Timestamp, float] = {}
    written_dates: set[pd.Timestamp] = set()
    sector_ready = sectors is not None
    fundamental_ready = fundamentals is not None
    try:
        for path in stock_paths:
            code = path.stem.zfill(6)
            market = derive_stock_market_fields(pd.read_parquet(path), benchmark, code)
            panel = merge_pit_inputs(market, flags, weights, sectors, fundamentals)
            required = panel["universe_member"]
            if panel.loc[required, "sector"].isna().any():
                sector_ready = False
                sector_error = "PIT sector coverage is incomplete for the investable universe"
            if panel.loc[required, "earnings_to_price"].isna().any():
                fundamental_ready = False
                fundamental_error = "verified PIT trailing-EPS coverage is incomplete for the investable universe"
            panel = panel[_output_columns()].copy()
            if panel.empty:
                continue
            written_dates.update(pd.to_datetime(panel["signal_date"]).dt.normalize().unique())
            if weights is not None:
                for date, value in panel.groupby("signal_date")["benchmark_weight"].sum().items():
                    weight_mass[pd.Timestamp(date)] = weight_mass.get(pd.Timestamp(date), 0.0) + float(value)
            table = pa.Table.from_pandas(panel, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(panel)
            processed.append(code)
    finally:
        if writer is not None:
            writer.close()
    if writer is None or rows == 0:
        temp.unlink(missing_ok=True)
        raise DataContractError("strict joins produced no panel rows")
    coverage_start = max(pd.Timestamp(flags["trade_date"].min()), pd.Timestamp(benchmark["trade_date"].min()))
    expected_dates = set(
        pd.to_datetime(
            benchmark.loc[
                benchmark["benchmark_return"].notna() & (benchmark["trade_date"] >= coverage_start),
                "trade_date",
            ]
        ).dt.normalize().unique()
    )
    missing_dates = sorted(expected_dates - written_dates)
    if missing_dates:
        temp.unlink(missing_ok=True)
        raise DataContractError(
            "panel does not cover the complete benchmark/fold calendar: "
            f"{[str(pd.Timestamp(date).date()) for date in missing_dates[:10]]}"
        )
    if weights is not None and not partial:
        invalid_dates = [str(date.date()) for date, value in weight_mass.items() if not 0.995 <= value <= 1.005]
        if invalid_dates:
            weight_error = f"written CSI500 weight mass is incomplete on dates: {invalid_dates[:10]}"
            weights = None
    temp.replace(output)
    b_ready = fundamental_ready and not partial
    c_ready = weights is not None and sector_ready and not partial
    if partial:
        status = "PARTIAL_NOT_EVALUABLE"
    else:
        ready = "A" + ("B" if b_ready else "") + ("C" if c_ready else "")
        blocked = "" + ("B" if not b_ready else "") + ("C" if not c_ready else "")
        status = f"{ready}_READY" + (f"_{blocked}_BLOCKED" if blocked else "_FOR_OOS_EVALUATION")
    return {
        "status": status,
        "output": str(output),
        "rows": int(rows),
        "codes": int(len(processed)),
        "partial_smoke_build": partial,
        "baseline_readiness": {
            "A_qlib_alpha158": "READY" if not partial else "PARTIAL",
            "B_ch3_ch4_style": "READY" if b_ready else "BLOCKED",
            "B_block_reason": fundamental_error,
            "C_csi500_index_enhancement": "READY" if c_ready else "BLOCKED",
            "C_block_reason": "; ".join(reason for reason in (weight_error, sector_error) if reason) or None,
        },
        "columns": _output_columns(),
        "semantics": {
            "prices": "Qlib convention: HFQ OHLC/VWAP divided by first HFQ close",
            "alpha158_volume": "raw shares / adj_factor * first HFQ close",
            "liquidity": "raw turnover and raw-yuan Amihud inputs",
            "label": "signal after close; next open to following open return",
            "suspension": "shared benchmark calendar, prior-close mark, and execution blocked by PIT flags",
            "delisting": "listed-to-unlisted terminal transition without an observed exit is marked -100%",
            "pmo_proxy": "mean turnover last 20 sessions / mean turnover last 240 sessions",
            "pit": "membership/tradeability exact-date; sector and E/P as-of available_at",
        },
    }


def main() -> None:
    args = _parse_args()
    report_path = Path(args.report).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "stocks": str(Path(args.root).expanduser() / "stocks"),
            "pit_flags": args.pit_flags,
            "benchmark": args.benchmark,
            "benchmark_weights": args.benchmark_weights,
            "sector_events": args.sector_events,
            "fundamental_events": args.fundamental_events,
        },
        "fail_closed": True,
    }
    try:
        required_paths = [
            Path(args.pit_flags).expanduser(),
            Path(args.benchmark).expanduser(),
        ]
        missing_inputs = [str(path) for path in required_paths if not path.exists()]
        if missing_inputs:
            raise DataContractError(f"required PIT inputs are missing: {missing_inputs}")
        result = build_panel(args)
    except Exception as exc:
        report = {**base, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"strict panel blocked; audit report saved: {report_path}", file=sys.stderr)
        raise
    report = {**base, **result}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"strict panel saved: {result['output']}")
    print(f"audit report saved: {report_path}")


if __name__ == "__main__":
    main()
