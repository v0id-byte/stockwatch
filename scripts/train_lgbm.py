#!/usr/bin/env python3
"""Train LightGBM LambdaRank model from the offline training set."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ALPHA158_FEATURES, ROBUST_FEATURES, BEAR_FEATURES, BULL_FEATURES
from analysis.propagation import PROPAGATION_FEATURES
from analysis.regime import is_bull_trend

STABLE_FEATURE_PREFIXES = (
    "ILLIQ", "BETA", "RSV", "DD", "RET", "ROC", "RELV", "STD",
    "RSQR", "CORR", "VMA", "WVMA", "TURN", "VOLZ", "MOM", "SHARPE",
)


def _split_by_date_with_purge(df, label_horizon: int):
    import pandas as pd

    max_date = df["trade_date"].max()
    raw_test_start = max_date - pd.DateOffset(months=6)
    raw_val_start = raw_test_start - pd.DateOffset(months=6)
    dates = pd.Index(sorted(df["trade_date"].drop_duplicates()))
    val_idx = int(dates.searchsorted(raw_val_start, side="left"))
    test_idx = int(dates.searchsorted(raw_test_start, side="left"))
    if val_idx >= len(dates) or test_idx >= len(dates):
        raise RuntimeError("训练/验证/测试切分为空，请检查历史数据跨度")

    purge_days = max(0, label_horizon)
    train_end_idx = max(0, val_idx - purge_days)
    val_end_idx = max(val_idx, test_idx - purge_days)
    train_end = dates[train_end_idx]
    val_start = dates[val_idx]
    val_end = dates[val_end_idx]
    test_start = dates[test_idx]

    train = df[df["trade_date"] < train_end]
    val = df[(df["trade_date"] >= val_start) & (df["trade_date"] < val_end)]
    test = df[df["trade_date"] >= test_start]
    split = {
        "raw_val_start": str(raw_val_start.date()),
        "raw_test_start": str(raw_test_start.date()),
        "val_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "train_end_exclusive": str(train_end.date()),
        "val_end_exclusive": str(val_end.date()),
        "purge_trading_days": purge_days,
    }
    return train, val, test, split


def _regime() -> str:
    value = os.getenv("STOCKWATCH_LGBM_REGIME", "all").strip().lower()
    return value if value in {"all", "bull", "bear"} else "all"


def _regime_feature_names(regime: str, available_columns: set[str]) -> tuple[str, list[str]]:
    source = BEAR_FEATURES if regime == "bear" else BULL_FEATURES
    return regime, [name for name in source if name in available_columns]


def _filter_to_regime(df, history_dir, regime: str):
    """Keep only the trade dates whose CSI300 trend matches the regime (point-in-time)."""
    import pandas as pd

    csi_path = history_dir / "market_sh000300.parquet"
    if not csi_path.exists():
        raise RuntimeError(f"按 regime 训练需要基准指数: {csi_path}")
    csi = pd.read_parquet(csi_path)[["trade_date", "close"]].copy()
    csi["trade_date"] = pd.to_datetime(csi["trade_date"])
    csi = csi.sort_values("trade_date")
    csi["bull"] = is_bull_trend(csi["close"].reset_index(drop=True)).to_numpy()
    bull_by_date = dict(zip(csi["trade_date"], csi["bull"]))
    want_bull = regime == "bull"
    mask = df["trade_date"].map(bull_by_date)
    kept = df[mask == want_bull]
    return kept


def _split_by_position_with_purge(df, label_horizon: int, frac_train=0.70, frac_val=0.15):
    """Fraction-of-dates split with purge. Used for regime models, whose matching
    trade dates are sparse/discontinuous at the recent end so a calendar 'last 6m'
    split would leave validation/test nearly empty."""
    import pandas as pd

    dates = pd.Index(sorted(df["trade_date"].drop_duplicates()))
    n = len(dates)
    purge = max(0, label_horizon)
    train_cut = max(1, int(n * frac_train))
    val_cut = max(train_cut + 1, int(n * (frac_train + frac_val)))
    if val_cut >= n:
        raise RuntimeError("regime 样本日期过少，无法切分 train/val/test")
    train_end = dates[max(0, train_cut - purge)]
    val_start = dates[train_cut]
    val_end = dates[max(train_cut + 1, val_cut - purge)]
    test_start = dates[val_cut]
    train = df[df["trade_date"] < train_end]
    val = df[(df["trade_date"] >= val_start) & (df["trade_date"] < val_end)]
    test = df[df["trade_date"] >= test_start]
    split = {
        "split_kind": "position_fraction",
        "frac_train": frac_train,
        "frac_val": frac_val,
        "val_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "train_end_exclusive": str(train_end.date()),
        "val_end_exclusive": str(val_end.date()),
        "purge_trading_days": purge,
    }
    return train, val, test, split


def _feature_names(available_columns: set[str]) -> tuple[str, list[str]]:
    feature_set = os.getenv("STOCKWATCH_LGBM_FEATURE_SET", "robust").strip().lower()
    if feature_set == "all":
        names = [*ALPHA158_FEATURES, *PROPAGATION_FEATURES]
        return "all", [name for name in names if name in available_columns]
    if feature_set == "stable":
        names = [name for name in ALPHA158_FEATURES if name.startswith(STABLE_FEATURE_PREFIXES)]
        names.extend(PROPAGATION_FEATURES)
        return "stable", [name for name in names if name in available_columns]
    if feature_set != "robust":
        print(f"unknown STOCKWATCH_LGBM_FEATURE_SET={feature_set}, fallback to robust")
    # robust: sign-stable cross-sectional factors only (see analysis/factors.ROBUST_FEATURES)
    return "robust", [name for name in ROBUST_FEATURES if name in available_columns]


def _cross_sectional_rank_normalize(df, feature_names):
    """Rank-normalize each feature to [-0.5, 0.5] WITHIN each trade date.

    This is the single most important fix vs the previous model: it removes
    time-varying factor scale/level so a tree threshold means the same thing
    ("top of today's cross-section") in every regime, which is what lets the
    cross-sectional signal generalize out-of-sample.
    """
    grouped = df.groupby("trade_date", sort=False)
    for name in feature_names:
        df[name] = grouped[name].rank(pct=True) - 0.5
    return df


def _mean_ndcg(df, pred, k: int) -> float | None:
    try:
        from sklearn.metrics import ndcg_score
        import numpy as np
    except Exception:
        return None
    scores = []
    offset = 0
    for _date, group in df.groupby("trade_date", sort=False):
        n = len(group)
        if n <= 1:
            offset += n
            continue
        y_true = group["label"].to_numpy().reshape(1, -1)
        y_score = np.asarray(pred[offset:offset+n]).reshape(1, -1)
        scores.append(float(ndcg_score(y_true, y_score, k=min(k, n))))
        offset += n
    return sum(scores) / len(scores) if scores else None


def _spearman_ic_stats(df, pred, target_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", target_col]].copy()
    tmp["pred"] = pred
    rows = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) <= 2 or group["pred"].nunique() <= 1 or group[target_col].nunique() <= 1:
            continue
        value = group["pred"].corr(group[target_col], method="spearman")
        if pd.notna(value):
            rows.append({"date": str(pd.Timestamp(_date).date()), "ic": float(value)})
    if not rows:
        return None
    values = pd.Series([row["ic"] for row in rows]).dropna()
    std = values.std()
    return {
        "mean": float(values.mean()),
        "std": float(std) if pd.notna(std) else None,
        "icir": float(values.mean() / std) if pd.notna(std) and std else None,
        "median": float(values.median()),
        "positive_rate": float((values > 0).mean()),
        "count": int(len(values)),
        "worst5": sorted(rows, key=lambda row: row["ic"])[:5],
        "best5": sorted(rows, key=lambda row: row["ic"], reverse=True)[:5],
        "by_date": rows,
    }


def _mean_topk_return(df, pred, k: int, return_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", return_col]].copy()
    tmp["pred"] = pred
    rows = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) < k:
            continue
        top = group.sort_values("pred", ascending=False).head(k)
        rows.append({
            "top_return": top[return_col].mean(),
            "universe_return": group[return_col].mean(),
        })
    if not rows:
        return None
    result = pd.DataFrame(rows)
    excess = result["top_return"] - result["universe_return"]
    return {
        "top_return": float(result["top_return"].mean()),
        "universe_return": float(result["universe_return"].mean()),
        "excess_return": float(excess.mean()),
        "excess_std": float(excess.std()),
        "excess_positive_rate": float((excess > 0).mean()),
        "period_count": int(len(result)),
    }


def _decile_returns(df, pred, return_col: str) -> dict | None:
    import pandas as pd

    tmp = df[["trade_date", return_col]].copy()
    tmp["pred"] = pred
    rows = []
    universe = []
    for _date, group in tmp.groupby("trade_date", sort=False):
        if len(group) < 10 or group["pred"].nunique() <= 1:
            continue
        ranks = group["pred"].rank(method="first", pct=True)
        deciles = (ranks * 10).clip(upper=9).astype(int)
        means = group.assign(decile=deciles).groupby("decile")[return_col].mean()
        rows.append({decile: means.get(decile) for decile in range(10)})
        universe.append(group[return_col].mean())
    if not rows:
        return None
    result = pd.DataFrame(rows)
    universe_return = float(pd.Series(universe).mean())
    deciles = [
        {
            "decile": int(decile),
            "mean_return": float(result[decile].mean()),
            "excess_return": float(result[decile].mean() - universe_return),
        }
        for decile in range(10)
    ]
    return {
        "note": "decile 0 is the lowest model score; decile 9 is the highest model score",
        "universe_return": universe_return,
        "deciles": deciles,
        "spread_9_minus_0": float(result[9].mean() - result[0].mean()),
    }


def _evaluate_split_core(df, pred, return_col: str) -> dict:
    label_ic_stats = _spearman_ic_stats(df, pred, "label_score")
    return_ic_stats = _spearman_ic_stats(df, pred, return_col)
    return {
        "ndcg5": _mean_ndcg(df, pred, 5),
        "ndcg10": _mean_ndcg(df, pred, 10),
        "spearman_ic": label_ic_stats["mean"] if label_ic_stats else None,
        "spearman_ic_target": "label_score",
        "spearman_ic_stats": label_ic_stats,
        "return_spearman_ic": return_ic_stats["mean"] if return_ic_stats else None,
        "return_spearman_ic_stats": return_ic_stats,
        "top5": _mean_topk_return(df, pred, 5, return_col),
        "top10": _mean_topk_return(df, pred, 10, return_col),
        "top30": _mean_topk_return(df, pred, 30, return_col),
        "top50": _mean_topk_return(df, pred, 50, return_col),
        "decile_returns": _decile_returns(df, pred, return_col),
    }


def _non_overlapping_sample(df, pred, step: int):
    import numpy as np

    if step <= 1:
        return df, pred
    dates = list(df["trade_date"].drop_duplicates())
    keep_dates = set(dates[::step])
    mask = df["trade_date"].isin(keep_dates).to_numpy()
    return df[mask], np.asarray(pred)[mask]


def _per_year_ic(df, pred, return_col: str) -> dict:
    """Per-calendar-year cross-sectional IC — the honest stability picture that one
    train/val/test split (which lands on a single recent regime) can hide."""
    import pandas as pd

    tmp = df[["trade_date", return_col]].copy()
    tmp["pred"] = pred
    out = {}
    for year, gy in tmp.groupby(tmp["trade_date"].dt.year):
        ics = []
        for _date, g in gy.groupby("trade_date"):
            if len(g) <= 2 or g["pred"].nunique() <= 1 or g[return_col].nunique() <= 1:
                continue
            value = g["pred"].corr(g[return_col], method="spearman")
            if pd.notna(value):
                ics.append(float(value))
        if ics:
            s = pd.Series(ics)
            std = s.std()
            out[str(int(year))] = {
                "mean_ic": float(s.mean()),
                "icir": float(s.mean() / std) if std else None,
                "positive_rate": float((s > 0).mean()),
                "days": int(len(s)),
            }
    return out


def _evaluate_split(df, pred, return_col: str, label_horizon: int) -> dict:
    metrics = _evaluate_split_core(df, pred, return_col)
    sampled_df, sampled_pred = _non_overlapping_sample(df, pred, label_horizon)
    metrics["non_overlapping"] = (
        _evaluate_split_core(sampled_df, sampled_pred, return_col)
        if len(sampled_df) else None
    )
    return metrics


def _summary(metrics: dict) -> dict:
    return {
        "ndcg5": metrics["ndcg5"],
        "ndcg10": metrics["ndcg10"],
        "spearman_ic": metrics["spearman_ic"],
        "return_spearman_ic": metrics["return_spearman_ic"],
        "top5_excess": metrics["top5"]["excess_return"] if metrics["top5"] else None,
        "top30_excess": metrics["top30"]["excess_return"] if metrics["top30"] else None,
        "decile_spread_9_minus_0": (
            metrics["decile_returns"]["spread_9_minus_0"]
            if metrics["decile_returns"] else None
        ),
    }


def main():
    import lightgbm as lgb
    import pandas as pd

    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    data_path = root / "training_set.parquet"
    if not data_path.exists():
        raise RuntimeError("训练集缺失，请先运行 scripts/build_training_set.py")
    df = pd.read_parquet(data_path).sort_values(["trade_date", "code"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    report_path = root / "training_set_report.json"
    training_report = {}
    if report_path.exists():
        training_report = json.loads(report_path.read_text())
    label_horizon = int(training_report.get("label_horizon_days", 20))
    return_col = f"forward_{label_horizon}d_return"
    if return_col not in df.columns:
        return_col = "forward_5d_return"

    regime = _regime()
    if regime == "all":
        feature_set, feature_names = _feature_names(set(df.columns))
    else:
        feature_set, feature_names = _regime_feature_names(regime, set(df.columns))
    if not feature_names:
        raise RuntimeError("训练特征为空，请检查 STOCKWATCH_LGBM_FEATURE_SET / STOCKWATCH_LGBM_REGIME")
    df = _cross_sectional_rank_normalize(df, feature_names)
    if regime != "all":
        rows_before = len(df)
        df = _filter_to_regime(df, root, regime)
        print(f"regime={regime}: 保留 {len(df)}/{rows_before} 行（{regime} 行情交易日）")
        if df.empty:
            raise RuntimeError(f"regime={regime} 过滤后无样本")

    if regime == "all":
        train, val, test, split = _split_by_date_with_purge(df, label_horizon)
    else:
        train, val, test, split = _split_by_position_with_purge(df, label_horizon)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("训练/验证/测试切分为空，请检查历史数据跨度")

    # Regression on the cross-sectional rank label directly optimizes rank
    # correlation (IC). This replaces the previous lambdarank setup, which paired
    # with unstable risk factors produced a NEGATIVE out-of-sample IC.
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
    }
    train_set = lgb.Dataset(train[feature_names], label=train["label"], feature_name=feature_names)
    val_set = lgb.Dataset(val[feature_names], label=val["label"], feature_name=feature_names, reference=train_set)
    model = lgb.train(
        params,
        train_set,
        num_boost_round=600,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)],
    )

    val_pred = model.predict(val[feature_names], num_iteration=model.best_iteration)
    test_pred = model.predict(test[feature_names], num_iteration=model.best_iteration)
    validation_metrics = _evaluate_split(val, val_pred, return_col, label_horizon)
    test_metrics = _evaluate_split(test, test_pred, return_col, label_horizon)
    full_pred = model.predict(df[feature_names], num_iteration=model.best_iteration)
    per_year_ic = _per_year_ic(df, full_pred, return_col)
    out_dir = ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if regime == "all" else f"_{regime}"
    model_path = out_dir / f"lgbm{suffix}.txt"
    model.save_model(str(model_path))
    importance = sorted(
        zip(feature_names, model.feature_importance(importance_type="gain")),
        key=lambda item: item[1],
        reverse=True,
    )
    meta = {
        "trained_at": datetime.now().isoformat(),
        "features": feature_names,
        "feature_set": feature_set,
        "regime": regime,
        "feature_normalization": "cross_sectional_rank_pct_centered",
        "objective": params["objective"],
        "feature_count": len(feature_names),
        "available_feature_count": len(ALPHA158_FEATURES),
        "propagation_feature_count": len([name for name in PROPAGATION_FEATURES if name in feature_names]),
        "propagation_features": [name for name in PROPAGATION_FEATURES if name in feature_names],
        "label_horizon_days": label_horizon,
        "return_col": return_col,
        "ndcg5": test_metrics["ndcg5"],
        "ndcg10": test_metrics["ndcg10"],
        "spearman_ic": test_metrics["spearman_ic"],
        "top5": test_metrics["top5"],
        "top10": test_metrics["top10"],
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "per_year_ic": per_year_ic,
        "per_year_ic_note": (
            "逐年横截面 IC（含样本内年份）。train 截止 "
            + split["train_end_exclusive"]
            + "，test 起始 " + split["test_start"]
            + "；其后年份为样本外。IC 逐年为正说明排序方向稳定。"
        ),
        "best_iteration": model.best_iteration,
        "split": split,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_codes": int(train["code"].nunique()),
        "val_codes": int(val["code"].nunique()),
        "test_codes": int(test["code"].nunique()),
        "date_start": str(df["trade_date"].min().date()),
        "date_end": str(df["trade_date"].max().date()),
        "training_report": training_report,
        "feature_importance_top30": [
            {"feature": name, "gain": float(gain)}
            for name, gain in importance[:30]
        ],
    }
    with open(out_dir / f"lgbm{suffix}_meta.json", "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"model saved: {model_path}")
    print(f"validation={_summary(validation_metrics)}")
    print(f"test={_summary(test_metrics)}")


if __name__ == "__main__":
    main()
