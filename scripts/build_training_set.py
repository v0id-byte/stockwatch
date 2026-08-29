#!/usr/bin/env python3
"""Build the legacy technical training set on an auditable PIT foundation."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ALPHA158_FEATURES, WINDOWS, compute_alpha158_frame
from analysis.propagation import PROPAGATION_FEATURES, add_propagation_features

WARMUP = max(WINDOWS)  # 最长滚动窗口，窗口未填满的行丢弃
DEFAULT_HORIZONS = (5, 20, 60)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


FUNDAMENTAL_FEATURES = ["ocf_to_eps"]
MARKET_DATA_SCHEMA_VERSION = 2
HISTORY_REQUIRED_COLUMNS = {
    "trade_date", "open", "high", "low", "close",
    "raw_open", "raw_high", "raw_low", "raw_close",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor",
    "volume", "volume_shares", "amount", "turnover", "vwap", "adj_vwap", "amihud_1d",
    "data_source", "market_data_schema_version", "return_adjustment",
}
PIT_REQUIRED_COLUMNS = {
    "trade_date", "code", "index_code", "is_member", "is_listed", "is_st",
    "is_suspended", "is_limit_up", "is_limit_down",
}


def _training_stock_paths(root: Path, stock_dir: Path) -> tuple[list[Path], dict]:
    paths = sorted(stock_dir.glob("*.parquet"))
    manifest_path = root / "history_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("history_manifest.json is required; legacy stock directories are not PIT-auditable")
    try:
        manifest = __import__("json").loads(manifest_path.read_text())
    except Exception as exc:
        raise RuntimeError("history_manifest.json is unreadable") from exc
    failed = int(manifest.get("failed") or 0)
    codes = {str(code).zfill(6) for code in manifest.get("codes") or []}
    excluded_failed = {
        str(code).zfill(6) for code in manifest.get("excluded_failed_codes") or []
    }
    if failed:
        raise RuntimeError(
            f"history manifest is incomplete: {failed} stock downloads failed; "
            "rerun scripts/bootstrap_history.py before rebuilding training_set.parquet"
        )
    if not codes:
        raise RuntimeError("history manifest has no completed stock codes")
    selected = [path for path in paths if path.stem in codes]
    missing_paths = sorted(codes - {path.stem for path in selected})
    if missing_paths:
        raise RuntimeError(f"history manifest references {len(missing_paths)} missing parquet files")
    return selected, {
        "source": "history_manifest.codes",
        "membership_kind": str(manifest.get("constituent_membership_kind") or "unknown"),
        "manifest_code_count": len(codes),
        "excluded_failed_code_count": len(excluded_failed),
        "path_count": len(selected),
        "ignored_stale_path_count": len(paths) - len(selected),
        "market_data_schema": manifest.get("market_data_schema") or {},
        "pit_universe_manifest": manifest.get("pit_universe") or {},
    }


def _validate_training_foundation(root: Path, universe_meta: dict) -> dict:
    schema = universe_meta.get("market_data_schema") or {}
    if int(schema.get("version") or 0) != MARKET_DATA_SCHEMA_VERSION:
        raise RuntimeError(
            "history schema v2 is required (raw OHLC + separate HFQ return OHLC + explicit units); "
            "rerun scripts/bootstrap_history.py"
        )
    if universe_meta.get("membership_kind") != "point_in_time_daily":
        raise RuntimeError(
            "PIT universe gate failed: current constituent snapshots cannot be used as historical membership; "
            "provide <history>/pit_universe_daily.parquet and rerun bootstrap_history.py"
        )
    pit_path = root / "pit_universe_daily.parquet"
    if not pit_path.exists():
        raise RuntimeError(f"PIT universe/status file is missing: {pit_path}")
    expected_hash = str(
        (universe_meta.get("pit_universe_manifest") or {}).get("sha256") or ""
    )
    if not expected_hash:
        raise RuntimeError("history manifest does not pin the PIT universe SHA-256")
    actual_hash = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "PIT universe hash differs from history_manifest.json; rerun "
            "scripts/bootstrap_history.py before training"
        )
    return {
        "status": "PASS",
        "history_schema_version": MARKET_DATA_SCHEMA_VERSION,
        "membership_kind": universe_meta["membership_kind"],
        "pit_universe_path": str(pit_path),
        "pit_universe_sha256": actual_hash,
    }


def _factor_input(kline, code: str):
    missing = HISTORY_REQUIRED_COLUMNS - set(kline.columns)
    if missing:
        raise RuntimeError(f"{code} history schema v2 columns missing: {sorted(missing)}")
    out = kline.copy()
    for column in ("open", "high", "low", "close"):
        out[column] = out[f"adj_{column}"]
    # Legacy custom300 derives VWAP as amount / volume.  Scale the temporary
    # factor-frame amount so its implied VWAP matches the HFQ OHLC basis.
    out["amount"] = out["adj_vwap"] * out["volume_shares"]
    return out


def _apply_pit_eligibility(data, root: Path):
    """Apply daily membership/tradability flags; missing rows fail closed."""
    import pandas as pd

    path = root / "pit_universe_daily.parquet"
    pit = pd.read_parquet(path)
    missing = PIT_REQUIRED_COLUMNS - set(pit.columns)
    if missing:
        raise RuntimeError(f"PIT universe/status columns missing: {sorted(missing)}")
    pit = pit[list(PIT_REQUIRED_COLUMNS)].copy()
    pit["trade_date"] = pd.to_datetime(pit["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pit["code"] = pit["code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    pit["index_code"] = pit["index_code"].astype(str).str.extract(r"(\d{6})", expand=False).fillna("")
    flag_columns = [
        "is_member", "is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down",
    ]
    for column in flag_columns:
        valid = pit[column].isin([True, False, 1, 0])
        if not bool(valid.all()):
            raise RuntimeError(f"PIT flag {column} contains null/non-boolean values")
        pit[column] = pit[column].astype(bool)
    status_nunique = pit.groupby(["trade_date", "code"])[flag_columns].nunique(dropna=False)
    if bool(status_nunique.gt(1).any().any()):
        raise RuntimeError("PIT status conflicts across index rows for the same trade_date/code")
    pit = pit.groupby(["trade_date", "code"], as_index=False).agg({
        "index_code": lambda values: ",".join(sorted(set(values))),
        "is_member": "any",
        "is_listed": "first",
        "is_st": "first",
        "is_suspended": "first",
        "is_limit_up": "first",
        "is_limit_down": "first",
    })
    panel = data.copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = panel.merge(pit, on=["trade_date", "code"], how="left", validate="many_to_one", indicator=True)
    missing_rows = merged["_merge"].ne("both")
    if bool(missing_rows.any()):
        examples = merged.loc[missing_rows, ["trade_date", "code"]].head(5).to_dict("records")
        raise RuntimeError(
            f"PIT universe/status coverage gate failed: {int(missing_rows.sum())} sample rows missing; "
            f"examples={examples}"
        )
    eligible = (
        merged["is_member"] & merged["is_listed"] & ~merged["is_st"]
        & ~merged["is_suspended"] & ~merged["is_limit_up"] & ~merged["is_limit_down"]
    )
    report = {
        "status": "PASS",
        "rows_before": int(len(merged)),
        "rows_eligible": int(eligible.sum()),
        "rows_excluded": int((~eligible).sum()),
        "excluded_st": int(merged["is_st"].sum()),
        "excluded_suspended": int(merged["is_suspended"].sum()),
        "excluded_limit_up": int(merged["is_limit_up"].sum()),
        "excluded_limit_down": int(merged["is_limit_down"].sum()),
        "coverage": 1.0,
        "entry_rule": "member & listed & !ST & !suspended & !limit_up & !limit_down",
    }
    return merged.loc[eligible].drop(columns=["_merge"]).reset_index(drop=True), report


def _merge_fundamental(data, root):
    """Merge only independently vintage-verified fundamental rows.

    A row for trade date D only sees reports announced strictly BEFORE D
    (allow_exact_matches=False). Current vendor snapshots are not historical
    vintages and therefore cannot silently enable the legacy feature."""
    import pandas as pd

    path = root / "fundamental_features.parquet"
    cols = FUNDAMENTAL_FEATURES
    if not path.exists():
        for c in cols:
            data[c] = 0.0
        return data, False
    fund = pd.read_parquet(path)
    required = {"code", "available_at", "vintage_verified", *cols}
    if not required.issubset(fund.columns):
        for c in cols:
            data[c] = 0.0
        return data, False
    fund = fund[fund["vintage_verified"].eq(True)][["code", "available_at", *cols]]
    if fund.empty:
        for c in cols:
            data[c] = 0.0
        return data, False
    fund["code"] = fund["code"].astype(str).str.zfill(6)
    fund["available_at"] = pd.to_datetime(fund["available_at"]).astype("datetime64[ns]")
    fund = fund.dropna(subset=["available_at"]).sort_values("available_at")
    d = data.copy()
    d["_td"] = pd.to_datetime(d["trade_date"]).astype("datetime64[ns]")
    d["code"] = d["code"].astype(str).str.zfill(6)
    d = d.sort_values("_td")
    merged = pd.merge_asof(d, fund, left_on="_td", right_on="available_at",
                           by="code", direction="backward", allow_exact_matches=False)
    stale = (merged["_td"] - merged["available_at"]).dt.days > 400
    for c in cols:
        merged.loc[stale, c] = 0.0
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0.0)
    return merged.drop(columns=["_td", "available_at"]), True


def _forward_return(close, horizon: int):
    return close.shift(-horizon) / close - 1


def _forward_drawdown(close, horizon: int):
    """Worst future close/entry drawdown over the next horizon days."""
    future_min = close.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    return future_min / close - 1


def _require_legacy_training_opt_in() -> None:
    """Keep the old close-to-close research label out of normal workflows."""
    if not _env_bool("STOCKWATCH_ALLOW_LEGACY_UNEXECUTABLE_LABELS", False):
        raise RuntimeError(
            "legacy training_set labels are close-to-close and not executable at the "
            "signal timestamp; use scripts/build_frozen_oos_panel.py and "
            "scripts/evaluate_frozen_oos_baselines.py instead. Set "
            "STOCKWATCH_ALLOW_LEGACY_UNEXECUTABLE_LABELS=true only to reproduce "
            "historical research artifacts."
        )


def main():
    _require_legacy_training_opt_in()
    import pandas as pd

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = lambda x, **_: x

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    stock_dir = root / "stocks"
    market_path = root / "market_sh000300.parquet"
    if not stock_dir.exists() or not market_path.exists():
        raise RuntimeError("历史数据缺失，请先运行 scripts/bootstrap_history.py")

    market = pd.read_parquet(market_path)
    label_horizon = _env_int("STOCKWATCH_LABEL_HORIZON_DAYS", 20)
    drawdown_penalty = float(os.getenv("STOCKWATCH_DRAWDOWN_PENALTY", "0.5"))
    enable_propagation = _env_bool("STOCKWATCH_ENABLE_PROPAGATION_FEATURES", True)
    if label_horizon not in DEFAULT_HORIZONS:
        horizons = tuple(sorted({*DEFAULT_HORIZONS, label_horizon}))
    else:
        horizons = DEFAULT_HORIZONS

    frames = []
    skipped = []
    stock_paths, universe_meta = _training_stock_paths(root, stock_dir)
    foundation_audit = _validate_training_foundation(root, universe_meta)
    for path in tqdm(stock_paths, desc="factors"):
        code = path.stem
        kline = pd.read_parquet(path)
        min_rows = WARMUP + max(horizons) + 10
        if len(kline) < min_rows:
            skipped.append({"code": code, "rows": len(kline), "reason": f"<{min_rows}"})
            continue
        factor_kline = _factor_input(kline, code)
        factors = compute_alpha158_frame(factor_kline, market)
        factors["code"] = code
        close = pd.to_numeric(kline["adj_close"], errors="coerce")
        factors["raw_close"] = pd.to_numeric(kline["raw_close"], errors="coerce").values
        factors["adj_close"] = close.values
        factors["volume"] = pd.to_numeric(kline["volume_shares"], errors="coerce").values
        factors["amount"] = pd.to_numeric(kline["amount"], errors="coerce").values
        factors["turnover"] = pd.to_numeric(kline["turnover"], errors="coerce").values
        factors["vwap"] = pd.to_numeric(kline["vwap"], errors="coerce").values
        factors["amihud_1d"] = pd.to_numeric(kline["amihud_1d"], errors="coerce").values
        for horizon in horizons:
            factors[f"forward_{horizon}d_return"] = _forward_return(close, horizon).values
        drawdown_col = f"forward_{label_horizon}d_drawdown"
        return_col = f"forward_{label_horizon}d_return"
        factors[drawdown_col] = _forward_drawdown(close, label_horizon).values
        factors["label_score"] = factors[return_col] + drawdown_penalty * factors[drawdown_col]
        factors = factors.iloc[WARMUP:].reset_index(drop=True)
        frames.append(factors)

    if not frames:
        raise RuntimeError("没有可用训练样本")
    data = pd.concat(frames, ignore_index=True)
    data, pit_report = _apply_pit_eligibility(data, root)
    return_col = f"forward_{label_horizon}d_return"
    data = data.dropna(subset=[return_col, "label_score"])
    data["trade_date"] = data["trade_date"].astype(str)
    feature_names = list(ALPHA158_FEATURES)
    if enable_propagation:
        data = add_propagation_features(data)
        feature_names.extend(PROPAGATION_FEATURES)
    else:
        for name in PROPAGATION_FEATURES:
            data[name] = 0.0
    data, fundamental_enabled = _merge_fundamental(data, root)
    feature_names.extend(FUNDAMENTAL_FEATURES)
    data["label"] = data.groupby("trade_date")["label_score"].transform(
        lambda values: (values.rank(method="first", pct=True) * 10).clip(0, 9).astype(int)
    )
    meta_cols = [
        "trade_date", "code", "label", "label_score",
        "index_code", "is_member", "is_listed", "is_st", "is_suspended",
        "is_limit_up", "is_limit_down", "raw_close", "adj_close", "volume",
        "amount", "turnover", "vwap", "amihud_1d",
        *[f"forward_{horizon}d_return" for horizon in horizons],
        f"forward_{label_horizon}d_drawdown",
    ]
    keep = [*meta_cols, *feature_names]
    out = data[keep].copy()
    out[feature_names] = out[feature_names].replace(
        [float("inf"), float("-inf")], 0,
    ).fillna(0)
    output = root / "training_set.parquet"
    out.to_parquet(output, index=False)
    per_date = out.groupby("trade_date")["code"].nunique()
    report = {
        "rows": len(out),
        "codes": int(out["code"].nunique()),
        "features": len(ALPHA158_FEATURES),
        "propagation_features_enabled": enable_propagation,
        "propagation_features": PROPAGATION_FEATURES if enable_propagation else [],
        "fundamental_enabled": fundamental_enabled,
        "fundamental_features": FUNDAMENTAL_FEATURES if fundamental_enabled else [],
        "total_features": len(feature_names),
        "label_horizon_days": label_horizon,
        "drawdown_penalty": drawdown_penalty,
        "date_start": str(out["trade_date"].min()),
        "date_end": str(out["trade_date"].max()),
        "per_date_min_codes": int(per_date.min()),
        "per_date_median_codes": float(per_date.median()),
        "per_date_max_codes": int(per_date.max()),
        "universe_membership": universe_meta,
        "foundation_audit": foundation_audit,
        "pit_eligibility": pit_report,
        "price_semantics": {
            "feature_and_label_ohlc": "AKShare hfq adj_* columns",
            "observed_price": "unadjusted raw_* columns",
            "volume": "shares",
            "amount": "CNY",
            "turnover": "decimal",
            "vwap": "CNY per share",
            "amihud_1d": "abs(hfq return) / amount_CNY; unscaled",
        },
        "skipped": skipped[:50],
        "skipped_count": len(skipped),
    }
    (root / "training_set_report.json").write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2))
    print(f"training set saved: {output}, rows={len(out)}")
    print(f"training report saved: {root / 'training_set_report.json'}")


if __name__ == "__main__":
    main()
