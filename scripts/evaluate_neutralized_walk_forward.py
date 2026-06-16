#!/usr/bin/env python3
"""Walk-forward diagnosis for raw vs neutralized return labels.

This is a research script. It does not write model files and does not change the
production LightGBM health gate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from analysis.factors import ALPHA158_FEATURES, ROBUST_FEATURES
from analysis.propagation import PROPAGATION_FEATURES
from analysis.regime import is_bull_trend


STABLE_FEATURE_PREFIXES = (
    "ILLIQ", "BETA", "RSV", "DD", "RET", "ROC", "RELV", "STD",
    "RSQR", "CORR", "VMA", "WVMA", "TURN", "VOLZ", "MOM", "SHARPE",
)
DEFAULT_STYLE_EXPOSURES = (
    "BETA20", "BETA60", "STD20", "STD60",
    "ILLIQ20", "ILLIQ60", "TURN120", "VOLZ120",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw vs neutralized labels with walk-forward OOS.")
    parser.add_argument("--feature-set", choices=["robust", "stable", "all"], default="robust")
    parser.add_argument("--target", default="", help="Forward return column. Defaults to training horizon.")
    parser.add_argument("--style-exposures", default=",".join(DEFAULT_STYLE_EXPOSURES))
    parser.add_argument("--sector-map", default="", help="Optional CSV/parquet with code,sector columns.")
    parser.add_argument("--market-cap", default="", help="Optional CSV/parquet with code,trade_date,market_cap columns.")
    parser.add_argument("--min-train-months", type=int, default=12)
    parser.add_argument("--fold-months", type=int, default=3)
    parser.add_argument("--val-months", type=int, default=3)
    parser.add_argument("--max-folds", type=int, default=0, help="0 means all folds.")
    parser.add_argument("--num-boost-round", type=int, default=240)
    parser.add_argument("--early-stopping", type=int, default=30)
    parser.add_argument("--num-threads", type=int, default=2, help="LightGBM worker threads.")
    parser.add_argument("--winsor", type=float, default=0.01, help="Per-date target winsorization tail.")
    parser.add_argument("--min-per-date", type=int, default=30)
    parser.add_argument("--output", default="", help="JSON report path.")
    return parser.parse_args()


def _root() -> Path:
    return Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()


def _training_horizon(root: Path) -> int:
    report_path = root / "training_set_report.json"
    if not report_path.exists():
        return 20
    return int(json.loads(report_path.read_text()).get("label_horizon_days", 20))


def _feature_names(feature_set: str, columns: set[str]) -> list[str]:
    if feature_set == "all":
        names = [*ALPHA158_FEATURES, *PROPAGATION_FEATURES]
    elif feature_set == "stable":
        names = [name for name in ALPHA158_FEATURES if name.startswith(STABLE_FEATURE_PREFIXES)]
        names.extend(PROPAGATION_FEATURES)
    else:
        names = list(ROBUST_FEATURES)
    return [name for name in names if name in columns]


def _parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("读取 parquet schema 需要 pyarrow，请先安装 pyarrow") from exc
    return list(pq.ParquetFile(path).schema.names)


def _ordered_existing(columns: list[str], available: set[str]) -> list[str]:
    seen = set()
    out = []
    for col in columns:
        if col in available and col not in seen:
            seen.add(col)
            out.append(col)
    return out


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_sector_map(path_text: str) -> tuple[dict[str, str], str]:
    if path_text:
        path = Path(path_text).expanduser()
        raw = _read_table(path)
        if "code" not in raw.columns or "sector" not in raw.columns:
            raise RuntimeError(f"sector map must contain code,sector columns: {path}")
        mapping = dict(zip(raw["code"].astype(str).str.zfill(6), raw["sector"].astype(str)))
        return mapping, str(path)

    try:
        from utils.storage import Storage

        mapping = Storage().get_all_stock_sectors(max_age_days=3650)
        return mapping, "sqlite:stock_sector_map"
    except Exception:
        return {}, "none"


def _pick_market_cap_col(columns: list[str]) -> str:
    candidates = ["market_cap", "total_mv", "circ_mv", "total_market_cap", "市值", "总市值", "流通市值"]
    return next((col for col in candidates if col in columns), "")


def _merge_market_cap(data: pd.DataFrame, path_text: str) -> tuple[pd.DataFrame, dict]:
    if not path_text:
        return data, {"enabled": False, "source": "none", "reason": "no market-cap file configured"}
    path = Path(path_text).expanduser()
    raw = _read_table(path)
    cap_col = _pick_market_cap_col(list(raw.columns))
    if "code" not in raw.columns or not cap_col:
        raise RuntimeError(f"market-cap table must contain code and a market cap column: {path}")

    caps = raw.copy()
    caps["code"] = caps["code"].astype(str).str.zfill(6)
    caps["market_cap"] = pd.to_numeric(caps[cap_col], errors="coerce")
    caps = caps.dropna(subset=["market_cap"])
    if "trade_date" in caps.columns:
        caps["trade_date"] = pd.to_datetime(caps["trade_date"])
        out = data.merge(caps[["trade_date", "code", "market_cap"]], on=["trade_date", "code"], how="left")
        source_kind = "point_in_time"
    else:
        latest = caps.sort_values("code").drop_duplicates("code", keep="last")
        out = data.merge(latest[["code", "market_cap"]], on="code", how="left")
        source_kind = "static_by_code"
    out["log_market_cap"] = np.log(out["market_cap"].where(out["market_cap"] > 0))
    coverage = float(out["log_market_cap"].notna().mean())
    return out, {"enabled": coverage > 0, "source": str(path), "kind": source_kind, "coverage": coverage}


def _rank_normalize_features(data: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    out = data.copy()
    grouped = out.groupby("trade_date", sort=False)
    for name in feature_names:
        out[name] = grouped[name].rank(pct=True) - 0.5
    return out


def _winsorize(values: pd.Series, tail: float) -> pd.Series:
    y = pd.to_numeric(values, errors="coerce")
    if tail <= 0 or y.notna().sum() < 10:
        return y
    lo = y.quantile(tail)
    hi = y.quantile(1 - tail)
    return y.clip(lo, hi)


def _standardize(values: pd.Series) -> pd.Series | None:
    x = pd.to_numeric(values, errors="coerce")
    std = x.std()
    if pd.isna(std) or std == 0:
        return None
    return (x - x.mean()) / std


def _neutralize_one_day(group: pd.DataFrame, target_col: str, numeric_exposures: list[str],
                        sector_col: str | None, winsor_tail: float) -> pd.Series:
    y = _winsorize(group[target_col], winsor_tail)
    valid = y.notna()
    if valid.sum() <= 5:
        return pd.Series(np.nan, index=group.index)

    parts = [pd.Series(1.0, index=group.index, name="intercept")]
    for col in numeric_exposures:
        if col not in group.columns:
            continue
        x = _standardize(group[col])
        if x is not None:
            parts.append(x.rename(col))
    if sector_col and sector_col in group.columns:
        sectors = group[sector_col].fillna("UNKNOWN").astype(str)
        if sectors.nunique() > 1:
            dummies = pd.get_dummies(sectors, prefix="sector", drop_first=True, dtype=float)
            parts.append(dummies)

    x_frame = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)
    valid = valid & x_frame.notna().all(axis=1)
    if valid.sum() <= x_frame.shape[1] + 5:
        residual = y - y[valid].mean()
        return residual.reindex(group.index)

    x = x_frame.loc[valid].to_numpy(dtype=float)
    target = y.loc[valid].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    fitted = x_frame.to_numpy(dtype=float) @ beta
    residual = pd.Series(y.to_numpy(dtype=float) - fitted, index=group.index)
    residual.loc[~valid] = np.nan
    return residual


def neutralize_by_date(data: pd.DataFrame, target_col: str, numeric_exposures: list[str],
                       sector_col: str | None = None, winsor_tail: float = 0.01) -> pd.Series:
    residuals = []
    for _date, group in data.groupby("trade_date", sort=False):
        residuals.append(_neutralize_one_day(group, target_col, numeric_exposures, sector_col, winsor_tail))
    return pd.concat(residuals).sort_index()


def _rank_label_by_date(data: pd.DataFrame, value_col: str) -> pd.Series:
    return data.groupby("trade_date", sort=False)[value_col].transform(
        lambda values: (values.rank(method="first", pct=True) * 10).clip(0, 9)
    )


def make_walk_forward_folds(dates: pd.Index, horizon_days: int, min_train_months: int,
                            fold_months: int, val_months: int, max_folds: int = 0) -> list[dict]:
    dates = pd.Index(pd.to_datetime(sorted(dates)).unique())
    first_start = dates[0] + pd.DateOffset(months=min_train_months + val_months)
    min_val_start = dates[0] + pd.DateOffset(months=min_train_months)
    raw_starts = pd.date_range(first_start, dates[-1], freq=pd.DateOffset(months=fold_months))
    folds = []
    for raw_start in raw_starts:
        test_start_idx = int(dates.searchsorted(raw_start, side="left"))
        if test_start_idx >= len(dates):
            continue
        raw_end = dates[test_start_idx] + pd.DateOffset(months=fold_months)
        test_end_idx = int(dates.searchsorted(raw_end, side="left"))
        if test_end_idx <= test_start_idx:
            continue

        purge_start_idx = max(0, test_start_idx - max(0, horizon_days))
        val_start_raw = dates[purge_start_idx] - pd.DateOffset(months=val_months)
        val_start_idx = int(dates.searchsorted(val_start_raw, side="left"))
        if val_start_idx <= 0 or purge_start_idx <= val_start_idx:
            continue
        if dates[val_start_idx] < min_val_start:
            continue
        folds.append({
            "train_end_exclusive": dates[val_start_idx],
            "val_start": dates[val_start_idx],
            "val_end_exclusive": dates[purge_start_idx],
            "purge_start": dates[purge_start_idx],
            "test_start": dates[test_start_idx],
            "test_end_exclusive": dates[test_end_idx] if test_end_idx < len(dates) else dates[-1] + pd.Timedelta(days=1),
            "purge_trading_days": int(test_start_idx - purge_start_idx),
        })
        if max_folds and len(folds) >= max_folds:
            break
    return folds


def _ic_stats(data: pd.DataFrame, pred_col: str, target_col: str, min_per_date: int) -> dict | None:
    rows = []
    for date, group in data.dropna(subset=[pred_col, target_col]).groupby("trade_date", sort=False):
        if len(group) < min_per_date or group[pred_col].nunique() <= 1 or group[target_col].nunique() <= 1:
            continue
        ic = group[pred_col].corr(group[target_col], method="spearman")
        if pd.notna(ic):
            rows.append({"date": str(pd.Timestamp(date).date()), "ic": float(ic), "n": int(len(group))})
    if not rows:
        return None
    values = pd.Series([row["ic"] for row in rows])
    std = values.std()
    return {
        "mean": float(values.mean()),
        "std": float(std) if pd.notna(std) else None,
        "icir": float(values.mean() / std) if pd.notna(std) and std else None,
        "median": float(values.median()),
        "positive_rate": float((values > 0).mean()),
        "count": int(len(values)),
    }


def _decile_spread(data: pd.DataFrame, pred_col: str, target_col: str, min_per_date: int) -> dict | None:
    rows = []
    for _date, group in data.dropna(subset=[pred_col, target_col]).groupby("trade_date", sort=False):
        if len(group) < min_per_date or group[pred_col].nunique() <= 1:
            continue
        ranks = group[pred_col].rank(method="first", pct=True)
        deciles = (ranks * 10).clip(upper=9).astype(int)
        means = group.assign(decile=deciles).groupby("decile")[target_col].mean()
        rows.append({decile: means.get(decile) for decile in range(10)})
    if not rows:
        return None
    result = pd.DataFrame(rows)
    return {
        "spread_9_minus_0": float(result[9].mean() - result[0].mean()),
        "decile_0": float(result[0].mean()),
        "decile_9": float(result[9].mean()),
        "count": int(len(result)),
    }


def _non_overlapping(data: pd.DataFrame, step: int) -> pd.DataFrame:
    if step <= 1:
        return data
    dates = list(data["trade_date"].drop_duplicates())
    keep = set(dates[::step])
    return data[data["trade_date"].isin(keep)]


def _model_metrics(data: pd.DataFrame, pred_col: str, target_cols: list[str],
                   min_per_date: int, non_overlap_step: int) -> dict:
    out = {}
    for target_col in target_cols:
        out[target_col] = {
            "ic": _ic_stats(data, pred_col, target_col, min_per_date),
            "decile": _decile_spread(data, pred_col, target_col, min_per_date),
        }
    sampled = _non_overlapping(data, non_overlap_step)
    out["non_overlapping"] = {}
    for target_col in target_cols:
        out["non_overlapping"][target_col] = {
            "ic": _ic_stats(sampled, pred_col, target_col, min_per_date),
            "decile": _decile_spread(sampled, pred_col, target_col, min_per_date),
        }
    return out


def _metric_value(block: dict, model_key: str, target_col: str,
                  metric_kind: str = "ic", sampled: bool = False) -> float | None:
    model_block = block.get(model_key) or {}
    if sampled:
        target_block = (model_block.get("non_overlapping") or {}).get(target_col) or {}
    else:
        target_block = model_block.get(target_col) or {}
    metric = target_block.get(metric_kind) or {}
    key = "mean" if metric_kind == "ic" else "spread_9_minus_0"
    value = metric.get(key)
    return float(value) if value is not None else None


def _fold_stability(by_fold: dict, target_cols: list[str]) -> dict:
    out = {}
    for model_key in ("raw_label_model", "neutral_label_model"):
        out[model_key] = {}
        for target_col in target_cols:
            values = [
                _metric_value(block, model_key, target_col, "ic")
                for block in by_fold.values()
            ]
            values = [value for value in values if value is not None]
            if not values:
                out[model_key][target_col] = None
                continue
            series = pd.Series(values, dtype="float64")
            std = series.std()
            out[model_key][target_col] = {
                "fold_mean_ic": float(series.mean()),
                "fold_ic_std": float(std) if pd.notna(std) else None,
                "fold_icir": float(series.mean() / std) if pd.notna(std) and std else None,
                "positive_fold_rate": float((series > 0).mean()),
                "fold_count": int(len(series)),
                "fold_ics": [float(value) for value in values],
            }
    return out


def _regime_by_date(root: Path, dates: pd.Series) -> pd.Series:
    path = root / "market_sh000300.parquet"
    if not path.exists():
        return pd.Series("unknown", index=dates.index)
    market = pd.read_parquet(path, columns=["trade_date", "close"]).sort_values("trade_date")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["regime"] = np.where(is_bull_trend(market["close"]).fillna(True), "bull", "bear")
    mapping = dict(zip(market["trade_date"], market["regime"]))
    return dates.map(mapping).fillna("unknown")


def _train_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   features: list[str], label_col: str, args: argparse.Namespace) -> np.ndarray:
    import lightgbm as lgb

    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "max_depth": 4,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
        "num_threads": max(1, int(args.num_threads)),
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(train[features], label=train[label_col], feature_name=features)
    val_set = lgb.Dataset(val[features], label=val[label_col], feature_name=features, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
    )
    return model.predict(test[features], num_iteration=model.best_iteration)


def _run_walk_forward(data: pd.DataFrame, features: list[str], folds: list[dict],
                      args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for idx, fold in enumerate(folds, start=1):
        train = data[data["trade_date"] < fold["val_start"]]
        val = data[(data["trade_date"] >= fold["val_start"]) & (data["trade_date"] < fold["val_end_exclusive"])]
        test = data[(data["trade_date"] >= fold["test_start"]) & (data["trade_date"] < fold["test_end_exclusive"])]
        if train.empty or val.empty or test.empty:
            continue
        print(
            f"fold {idx}/{len(folds)} train<{fold['train_end_exclusive'].date()} "
            f"val={fold['val_start'].date()}~{fold['val_end_exclusive'].date()} "
            f"test={fold['test_start'].date()}~{fold['test_end_exclusive'].date()} rows={len(test)}",
            flush=True,
        )
        out = test[["trade_date", "code", args.target, "neutral_return"]].copy()
        out["fold"] = idx
        out["pred_raw_label"] = _train_predict(train, val, test, features, "raw_label", args)
        out["pred_neutral_label"] = _train_predict(train, val, test, features, "neutral_label", args)
        rows.append(out)
    if not rows:
        raise RuntimeError("walk-forward produced no test predictions")
    return pd.concat(rows, ignore_index=True)


def _summarize_predictions(preds: pd.DataFrame, root: Path, args: argparse.Namespace,
                           non_overlap_step: int) -> dict:
    preds = preds.copy()
    preds["regime"] = _regime_by_date(root, preds["trade_date"])
    target_cols = [args.target, "neutral_return"]
    report = {
        "raw_label_model": _model_metrics(
            preds, "pred_raw_label", target_cols, args.min_per_date, non_overlap_step,
        ),
        "neutral_label_model": _model_metrics(
            preds, "pred_neutral_label", target_cols, args.min_per_date, non_overlap_step,
        ),
        "by_year": {},
        "by_regime": {},
        "by_fold": {},
    }
    for fold, group in preds.groupby("fold"):
        report["by_fold"][str(int(fold))] = {
            "rows": int(len(group)),
            "dates": int(group["trade_date"].nunique()),
            "date_start": str(group["trade_date"].min().date()),
            "date_end": str(group["trade_date"].max().date()),
            "raw_label_model": _model_metrics(
                group, "pred_raw_label", target_cols, args.min_per_date, non_overlap_step,
            ),
            "neutral_label_model": _model_metrics(
                group, "pred_neutral_label", target_cols, args.min_per_date, non_overlap_step,
            ),
        }
    for year, group in preds.groupby(preds["trade_date"].dt.year):
        report["by_year"][str(int(year))] = {
            "raw_label_model": _model_metrics(
                group, "pred_raw_label", target_cols, args.min_per_date, non_overlap_step,
            ),
            "neutral_label_model": _model_metrics(
                group, "pred_neutral_label", target_cols, args.min_per_date, non_overlap_step,
            ),
        }
    for regime, group in preds.groupby("regime"):
        report["by_regime"][str(regime)] = {
            "rows": int(len(group)),
            "dates": int(group["trade_date"].nunique()),
            "raw_label_model": _model_metrics(
                group, "pred_raw_label", target_cols, args.min_per_date, non_overlap_step,
            ),
            "neutral_label_model": _model_metrics(
                group, "pred_neutral_label", target_cols, args.min_per_date, non_overlap_step,
            ),
        }
    report["fold_stability"] = _fold_stability(report["by_fold"], target_cols)
    return report


def _leakage_audit(folds: list[dict], horizon: int, feature_names: list[str],
                   neutralization: dict) -> dict:
    purge_values = [int(fold["purge_trading_days"]) for fold in folds]
    return {
        "status": "PASS" if purge_values and min(purge_values) >= horizon else "FAIL",
        "checks": {
            "neutralization_beta_scope": {
                "status": "PASS",
                "detail": "neutralize_by_date groups rows by trade_date and runs a separate cross-sectional OLS for each day.",
            },
            "raw_and_neutral_models_share_folds": {
                "status": "PASS",
                "detail": "Each walk-forward fold trains raw_label and neutral_label models on the same train/val/test rows.",
            },
            "purge_gap": {
                "status": "PASS" if purge_values and min(purge_values) >= horizon else "FAIL",
                "required_trading_days": int(horizon),
                "min_observed_trading_days": int(min(purge_values)) if purge_values else None,
            },
            "feature_normalization_scope": {
                "status": "PASS",
                "detail": "rank_normalize_features ranks each feature within trade_date only; no pooled time-series statistics are used.",
                "feature_count": int(len(feature_names)),
            },
        },
        "limitations": {
            "full_industry_size_neutral": bool(
                neutralization["sector_enabled"]
                and neutralization["market_cap"].get("enabled")
                and neutralization["market_cap"].get("kind") == "point_in_time"
            ),
            "sector_enabled": neutralization["sector_enabled"],
            "market_cap_enabled": bool(neutralization["market_cap"].get("enabled")),
            "market_cap_kind": neutralization["market_cap"].get("kind"),
            "note": "PASS means the current diagnostic mechanics are PIT-safe; it does not mean the residual alpha survives missing industry/size controls.",
        },
    }


def main() -> None:
    args = _parse_args()
    root = _root()
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")
    horizon = _training_horizon(root)
    if not args.target:
        args.target = f"forward_{horizon}d_return"

    available_columns = set(_parquet_columns(data_path))
    if args.target not in available_columns:
        raise RuntimeError(f"target column not found: {args.target}")

    style_exposures = [col.strip() for col in args.style_exposures.split(",") if col.strip()]
    features = _feature_names(args.feature_set, available_columns)
    if not features:
        raise RuntimeError("feature set is empty")
    read_columns = _ordered_existing(
        ["trade_date", "code", args.target, *features, *style_exposures],
        available_columns,
    )
    data = pd.read_parquet(data_path, columns=read_columns).sort_values(["trade_date", "code"]).reset_index(drop=True)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data = _rank_normalize_features(data, features)

    sector_map, sector_source = _load_sector_map(args.sector_map)
    if sector_map:
        data["sector"] = data["code"].map(sector_map)
    data, cap_meta = _merge_market_cap(data, args.market_cap)

    numeric_exposures = [col for col in style_exposures if col in data.columns]
    if "log_market_cap" in data.columns and data["log_market_cap"].notna().any():
        numeric_exposures.append("log_market_cap")
    sector_col = "sector" if "sector" in data.columns and data["sector"].notna().any() else None
    data["neutral_return"] = neutralize_by_date(
        data,
        args.target,
        numeric_exposures=numeric_exposures,
        sector_col=sector_col,
        winsor_tail=args.winsor,
    )
    data["raw_return_winsor"] = data.groupby("trade_date", sort=False)[args.target].transform(
        lambda values: _winsorize(values, args.winsor)
    )
    data["raw_label"] = _rank_label_by_date(data, "raw_return_winsor")
    data["neutral_label"] = _rank_label_by_date(data, "neutral_return")
    needed = [args.target, "neutral_return", "raw_label", "neutral_label", *features]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).reset_index(drop=True)

    dates = pd.Index(data["trade_date"].drop_duplicates())
    folds = make_walk_forward_folds(
        dates,
        horizon_days=horizon,
        min_train_months=args.min_train_months,
        fold_months=args.fold_months,
        val_months=args.val_months,
        max_folds=args.max_folds,
    )
    if not folds:
        raise RuntimeError("no walk-forward folds; relax min-train-months/fold settings")
    preds = _run_walk_forward(data, features, folds, args)
    metrics = _summarize_predictions(preds, root, args, non_overlap_step=horizon)
    neutralization_meta = {
        "numeric_exposures": numeric_exposures,
        "sector_enabled": bool(sector_col),
        "sector_source": sector_source,
        "sector_coverage": float(data["sector"].notna().mean()) if "sector" in data.columns else 0.0,
        "market_cap": cap_meta,
        "winsor_tail": args.winsor,
        "note": "If sector or market_cap coverage is 0, this is style-neutral only, not full industry/size neutral.",
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "training_set": str(data_path),
        "rows_used": int(len(data)),
        "codes": int(data["code"].nunique()),
        "date_start": str(data["trade_date"].min().date()),
        "date_end": str(data["trade_date"].max().date()),
        "feature_set": args.feature_set,
        "feature_count": len(features),
        "target": args.target,
        "neutralization": neutralization_meta,
        "leakage_audit": _leakage_audit(folds, horizon, features, neutralization_meta),
        "folds": [
            {key: (str(value.date()) if hasattr(value, "date") else value) for key, value in fold.items()}
            for fold in folds
        ],
        "pred_rows": int(len(preds)),
        "pred_dates": int(preds["trade_date"].nunique()),
        "non_overlap_step": horizon,
        "metrics": metrics,
    }
    output = Path(args.output).expanduser() if args.output else root / "neutralized_walk_forward_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"neutralized walk-forward report saved: {output}")
    for model_key in ("raw_label_model", "neutral_label_model"):
        raw_ic = (((metrics.get(model_key) or {}).get(args.target) or {}).get("ic") or {}).get("mean")
        neutral_ic = (((metrics.get(model_key) or {}).get("neutral_return") or {}).get("ic") or {}).get("mean")
        raw_spread = (((metrics.get(model_key) or {}).get(args.target) or {}).get("decile") or {}).get("spread_9_minus_0")
        neutral_spread = (((metrics.get(model_key) or {}).get("neutral_return") or {}).get("decile") or {}).get("spread_9_minus_0")
        print(
            f"{model_key}: raw_return_IC={raw_ic}, neutral_return_IC={neutral_ic}, "
            f"raw_spread={raw_spread}, neutral_spread={neutral_spread}"
        )


if __name__ == "__main__":
    main()
