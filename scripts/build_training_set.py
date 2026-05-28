#!/usr/bin/env python3
"""Build Alpha158 training set and forward-return labels."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ALPHA158_FEATURES, compute_alpha158_frame


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
    frames = []
    for path in tqdm(sorted(stock_dir.glob("*.parquet")), desc="factors"):
        code = path.stem
        kline = pd.read_parquet(path)
        if len(kline) < 80:
            continue
        factors = compute_alpha158_frame(kline, market)
        factors["code"] = code
        factors["close"] = pd.to_numeric(kline["close"], errors="coerce").values
        factors["forward_5d_return"] = factors["close"].shift(-5) / factors["close"] - 1
        frames.append(factors)

    if not frames:
        raise RuntimeError("没有可用训练样本")
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=["forward_5d_return"])
    data["trade_date"] = data["trade_date"].astype(str)
    data["label"] = data.groupby("trade_date")["forward_5d_return"].transform(
        lambda values: (values.rank(method="first", pct=True) * 10).clip(0, 9).astype(int)
    )
    keep = ["trade_date", "code", "label", "forward_5d_return", *ALPHA158_FEATURES]
    out = data[keep].replace([float("inf"), float("-inf")], 0).fillna(0)
    output = root / "training_set.parquet"
    out.to_parquet(output, index=False)
    print(f"training set saved: {output}, rows={len(out)}")


if __name__ == "__main__":
    main()
