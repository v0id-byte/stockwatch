#!/usr/bin/env python3
"""Build Alpha158 training set and forward-return labels."""
from __future__ import annotations

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


def _merge_fundamental(data, root):
    """Strict point-in-time as-of merge of fundamental features by announcement date.

    A row for trade date D only sees reports announced strictly BEFORE D
    (allow_exact_matches=False); reports older than ~400 days are treated as stale
    and reset to neutral (0). Missing -> 0 (neutral after rank-normalization)."""
    import pandas as pd

    path = root / "fundamental_features.parquet"
    cols = FUNDAMENTAL_FEATURES
    if not path.exists():
        for c in cols:
            data[c] = 0.0
        return data, False
    fund = pd.read_parquet(path, columns=["code", "available_at", *cols])
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


def main():
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
    for path in tqdm(sorted(stock_dir.glob("*.parquet")), desc="factors"):
        code = path.stem
        kline = pd.read_parquet(path)
        min_rows = WARMUP + max(horizons) + 10
        if len(kline) < min_rows:
            skipped.append({"code": code, "rows": len(kline), "reason": f"<{min_rows}"})
            continue
        factors = compute_alpha158_frame(kline, market)
        factors["code"] = code
        close = pd.to_numeric(kline["close"], errors="coerce")
        factors["close"] = close.values
        factors["volume"] = pd.to_numeric(kline["volume"], errors="coerce").values
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
        *[f"forward_{horizon}d_return" for horizon in horizons],
        f"forward_{label_horizon}d_drawdown",
    ]
    keep = [*meta_cols, *feature_names]
    out = data[keep].replace([float("inf"), float("-inf")], 0).fillna(0)
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
        "skipped": skipped[:50],
        "skipped_count": len(skipped),
    }
    (root / "training_set_report.json").write_text(__import__("json").dumps(report, ensure_ascii=False, indent=2))
    print(f"training set saved: {output}, rows={len(out)}")
    print(f"training report saved: {root / 'training_set_report.json'}")


if __name__ == "__main__":
    main()
