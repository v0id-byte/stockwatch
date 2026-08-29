#!/usr/bin/env python3
"""Build causal numeric OHLCV sequence features without rendering chart images."""
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
import pyarrow as pa
import pyarrow.parquet as pq


LONG_LAGS = 60
SHORT_LAGS = 20
LONG_CHANNELS = ("RET", "RELRET")
SHORT_CHANNELS = ("GAP", "RANGE", "CLOSEPOS", "VOLZ")
SEQUENCE_FEATURES = [
    *[f"SEQ_{channel}_{lag:02d}" for channel in LONG_CHANNELS for lag in range(LONG_LAGS)],
    *[f"SEQ_{channel}_{lag:02d}" for channel in SHORT_CHANNELS for lag in range(SHORT_LAGS)],
]


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Build Alpha360-like numeric sequence features.")
    parser.add_argument("--training-set", default=str(root / "training_set.parquet"))
    parser.add_argument("--stock-dir", default=str(root / "stocks"))
    parser.add_argument("--market", default=str(root / "market_sh000300.parquet"))
    parser.add_argument("--output", default=str(root / "sequence_features.parquet"))
    parser.add_argument("--max-codes", type=int, default=0)
    return parser.parse_args()


def _numeric_sequence_frame(kline: pd.DataFrame, market_returns: pd.Series | None = None) -> pd.DataFrame:
    frame = kline.copy().sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    for name in ("open", "high", "low", "close", "volume"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    close = frame["close"].replace(0, np.nan)
    previous_close = close.shift(1)
    returns = close.pct_change(fill_method=None)
    gap = frame["open"] / previous_close - 1.0
    price_range = frame["high"] / frame["low"].replace(0, np.nan) - 1.0
    spread = (frame["high"] - frame["low"]).replace(0, np.nan)
    close_position = (close - frame["low"]) / spread - 0.5
    log_volume = np.log1p(frame["volume"].clip(lower=0))
    volume_mean = log_volume.rolling(60, min_periods=20).mean()
    volume_std = log_volume.rolling(60, min_periods=20).std().replace(0, np.nan)
    volume_z = (log_volume - volume_mean) / volume_std
    if market_returns is None:
        relative_return = returns
    else:
        aligned_market = market_returns.reindex(pd.Index(frame["trade_date"])).to_numpy()
        relative_return = returns - pd.Series(aligned_market, index=frame.index)
    channels = {
        "RET": returns,
        "RELRET": relative_return,
        "GAP": gap,
        "RANGE": price_range,
        "CLOSEPOS": close_position,
        "VOLZ": volume_z,
    }
    data: dict[str, object] = {"trade_date": frame["trade_date"]}
    for channel in LONG_CHANNELS:
        for lag in range(LONG_LAGS):
            data[f"SEQ_{channel}_{lag:02d}"] = channels[channel].shift(lag).astype("float32")
    for channel in SHORT_CHANNELS:
        for lag in range(SHORT_LAGS):
            data[f"SEQ_{channel}_{lag:02d}"] = channels[channel].shift(lag).astype("float32")
    return pd.DataFrame(data)


def main() -> None:
    args = _parse_args()
    training_path = Path(args.training_set).expanduser()
    stock_dir = Path(args.stock_dir).expanduser()
    market_path = Path(args.market).expanduser()
    output = Path(args.output).expanduser()
    if not training_path.exists() or not stock_dir.exists() or not market_path.exists():
        raise RuntimeError("training set, stock directory, and market history are required")
    keys = pd.read_parquet(training_path, columns=["trade_date", "code"])
    keys["trade_date"] = pd.to_datetime(keys["trade_date"]).dt.normalize()
    keys["code"] = keys["code"].astype(str).str.zfill(6)
    codes = sorted(keys["code"].unique())
    if args.max_codes > 0:
        codes = codes[:args.max_codes]
    date_start = keys["trade_date"].min()
    date_end = keys["trade_date"].max()
    market = pd.read_parquet(market_path, columns=["trade_date", "close"]).sort_values("trade_date")
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    market_close = pd.to_numeric(market["close"], errors="coerce")
    market_returns = pd.Series(market_close.pct_change(fill_method=None).to_numpy(), index=market["trade_date"])
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    writer: pq.ParquetWriter | None = None
    row_count = 0
    code_count = 0
    try:
        for index, code in enumerate(codes, start=1):
            path = stock_dir / f"{code}.parquet"
            if not path.exists():
                continue
            raw = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close", "volume"])
            features = _numeric_sequence_frame(raw, market_returns)
            features.insert(1, "code", code)
            features = features[
                (features["trade_date"] >= date_start) & (features["trade_date"] <= date_end)
            ]
            features = features.dropna(subset=SEQUENCE_FEATURES)
            if features.empty:
                continue
            features["trade_date"] = features["trade_date"].dt.strftime("%Y-%m-%d")
            table = pa.Table.from_pandas(features[["trade_date", "code", *SEQUENCE_FEATURES]], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(features)
            code_count += 1
            if index % 100 == 0 or index == len(codes):
                print(f"sequence features codes={index}/{len(codes)} rows={row_count}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("no sequence features produced")
    tmp.replace(output)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(row_count),
        "codes": int(code_count),
        "features": len(SEQUENCE_FEATURES),
        "long_lags": LONG_LAGS,
        "short_lags": SHORT_LAGS,
        "channels": [*LONG_CHANNELS, *SHORT_CHANNELS],
        "date_start": str(date_start.date()),
        "date_end": str(date_end.date()),
        "causal": True,
        "representation": "numeric normalized OHLCV sequence; no chart rendering",
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"sequence features saved: {output}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
