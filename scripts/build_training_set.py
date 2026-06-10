#!/usr/bin/env python3
"""Build Alpha158 training set and forward-return labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ALPHA158_FEATURES, WINDOWS, compute_alpha158_frame

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
    data["label"] = data.groupby("trade_date")["label_score"].transform(
        lambda values: (values.rank(method="first", pct=True) * 10).clip(0, 9).astype(int)
    )
    meta_cols = [
        "trade_date", "code", "label", "label_score",
        *[f"forward_{horizon}d_return" for horizon in horizons],
        f"forward_{label_horizon}d_drawdown",
    ]
    keep = [*meta_cols, *ALPHA158_FEATURES]
    out = data[keep].replace([float("inf"), float("-inf")], 0).fillna(0)
    output = root / "training_set.parquet"
    out.to_parquet(output, index=False)
    per_date = out.groupby("trade_date")["code"].nunique()
    report = {
        "rows": len(out),
        "codes": int(out["code"].nunique()),
        "features": len(ALPHA158_FEATURES),
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
