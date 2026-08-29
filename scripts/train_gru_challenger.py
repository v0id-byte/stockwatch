#!/usr/bin/env python3
"""GRU sequence challenger (exploratory) on the same folds, labels and gates.

Input per stock-day: a [60 timesteps x 6 channels] causal window built with the
exact channel definitions of build_sequence_features (RET / RELRET at 60 lags;
GAP / RANGE / CLOSEPOS / VOLZ at 20 lags, zero-padded beyond), on the adjusted
price basis.  Target: the same executable 20d cross-sectional rank label as the
LightGBM candidate.  Evaluation: identical purged expanding folds and the same
seven-gate summarize_baseline — predictions are emitted in the
fit_lightgbm_oos output schema so the simulator code is shared, not imitated.

Promotion (pre-registered, AND logic — never OR):
  seven gates all PASS
  AND retro mean IC >= LightGBM candidate + 0.01
  AND net excess / drawdown / turnover not worse than preset bands
  AND RPi ops budget acceptable (latency / artifact size / RAM)
Otherwise this run is archived as an exploratory record.

GPU: requires the llama-server on the training box to be stopped first
(coordinated by the user); falls back to CPU for --max-codes smokes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.oos_baselines import (  # noqa: E402
    COMMON_COLUMNS,
    DataContractError,
    TradingCosts,
    equal_weight_builder,
    make_expanding_folds,
    simulate_portfolio,
    summarize_baseline,
    topk_dropout_builder,
    validate_pit_panel,
)
from scripts.build_sequence_features import (  # noqa: E402
    LONG_CHANNELS,
    LONG_LAGS,
    SHORT_CHANNELS,
    SHORT_LAGS,
    _numeric_sequence_frame,
)
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

CHANNELS = (*LONG_CHANNELS, *SHORT_CHANNELS)
SEED = 20260829


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "development_panel.parquet"))
    parser.add_argument("--stocks-dir", default=str(root / "stocks"))
    parser.add_argument("--market", default=str(root / "market_sh000300.parquet"))
    parser.add_argument("--output", default=str(root / "gru_challenger_report.json"))
    parser.add_argument("--retrospective-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=22)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--drop-n", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-codes", type=int, default=0, help="CPU smoke only")
    return parser.parse_args()


def _build_sequences(args, panel_keys: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """[N, 60, 6] float32 tensor aligned to (signal_date, code) rows."""
    # Same construction as build_sequence_features.main: index close returns.
    market = pd.read_parquet(args.market, columns=["trade_date", "close"]).sort_values("trade_date")
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.normalize()
    market_returns = pd.Series(
        pd.to_numeric(market["close"], errors="coerce").pct_change(fill_method=None).to_numpy(),
        index=pd.Index(market["trade_date"]),
    )

    codes = sorted(panel_keys["code"].unique())
    if args.max_codes:
        codes = codes[: args.max_codes]
    frames = []
    for index, code in enumerate(codes):
        path = Path(args.stocks_dir) / f"{code}.parquet"
        kline = pd.read_parquet(path, columns=["trade_date", "open", "high", "low", "close", "volume"])
        seq = _numeric_sequence_frame(kline, market_returns)
        seq["code"] = code
        frames.append(seq)
        if (index + 1) % 100 == 0:
            print(f"sequences {index + 1}/{len(codes)}", flush=True)
    seq_all = pd.concat(frames, ignore_index=True)
    seq_all["trade_date"] = pd.to_datetime(seq_all["trade_date"])
    merged = panel_keys.merge(
        seq_all, left_on=["signal_date", "code"], right_on=["trade_date", "code"],
        how="inner", validate="one_to_one",
    )
    tensor = np.zeros((len(merged), LONG_LAGS, len(CHANNELS)), dtype=np.float32)
    for channel_index, channel in enumerate(CHANNELS):
        lags = LONG_LAGS if channel in LONG_CHANNELS else SHORT_LAGS
        cols = [f"SEQ_{channel}_{lag:02d}" for lag in range(lags)]
        block = merged[cols].to_numpy(dtype=np.float32)
        # lag 0 = today; put oldest first so the GRU reads time forward.
        tensor[:, LONG_LAGS - lags:, channel_index] = block[:, ::-1]
    keys = merged[["signal_date", "code"]].reset_index(drop=True)
    return np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0), keys


def _train_fold(X, y, train_pos, val_pos, args, device):
    """train_pos/val_pos index rows of X (sequence order); y is aligned to X."""
    import torch
    from torch import nn

    torch.manual_seed(SEED)

    class GRUHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(len(CHANNELS), args.hidden, num_layers=args.layers, batch_first=True)
            self.head = nn.Linear(args.hidden, 1)

        def forward(self, batch):
            output, _ = self.gru(batch)
            return self.head(output[:, -1]).squeeze(-1)

    model = GRUHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_pos), generator=torch.Generator().manual_seed(SEED + epoch))
        order = torch.as_tensor(train_pos)[perm]
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx].to(device)), y_t[idx].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_losses = []
            for start in range(0, len(val_pos), args.batch_size):
                idx = torch.as_tensor(val_pos[start:start + args.batch_size])
                val_losses.append(loss_fn(model(X_t[idx].to(device)), y_t[idx].to(device)).item() * len(idx))
            val_loss = sum(val_losses) / max(len(val_pos), 1)
        if val_loss < best_val - 1e-6:
            best_val, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    import torch

    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    # Column projection: the GRU consumes raw sequences, not tabular features,
    # so only the PIT contract columns and labels are loaded.
    columns = [*COMMON_COLUMNS, "next_open_return_20d", "label_end_date_20d"]
    panel = pd.read_parquet(args.panel, columns=columns)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"])
    if (panel["signal_date"] >= pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("panel contains lockbox rows; refusing to train")
    panel = validate_pit_panel(panel, [])
    labeled = panel["next_open_return_20d"].notna()
    panel.loc[labeled, "target_rank"] = (
        panel[labeled].groupby("signal_date")["next_open_return_20d"].rank(pct=True)
    )

    keys = panel[["signal_date", "code"]]
    X, seq_keys = _build_sequences(args, keys)
    key_index = pd.MultiIndex.from_frame(seq_keys)
    panel_index = pd.MultiIndex.from_frame(keys)
    position = pd.Series(np.arange(len(seq_keys)), index=key_index)
    panel["seq_pos"] = position.reindex(panel_index).to_numpy()

    folds = make_expanding_folds(
        panel["signal_date"],
        min_train_days=args.min_train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
    )
    ends = panel[["signal_date", "label_end_date_20d"]].dropna()
    ends["label_end_date_20d"] = pd.to_datetime(ends["label_end_date_20d"])
    for fold_id, fold in enumerate(folds, start=1):
        train_max = ends.loc[ends["signal_date"] < fold["train_end_exclusive"], "label_end_date_20d"].max()
        val_max = ends.loc[
            (ends["signal_date"] >= fold["validation_start"])
            & (ends["signal_date"] < fold["validation_end_exclusive"]),
            "label_end_date_20d",
        ].max()
        if pd.notna(train_max) and train_max >= pd.Timestamp(fold["validation_start"]):
            raise DataContractError(f"fold {fold_id} violates the train->val purge invariant")
        if pd.notna(val_max) and val_max >= pd.Timestamp(fold["test_start"]):
            raise DataContractError(f"fold {fold_id} violates the val->test purge invariant")

    # y_seq is aligned to X's row order; masks below live on panel rows and are
    # translated to X positions via seq_pos before touching the tensor.
    y_panel = panel["target_rank"].to_numpy(dtype=np.float32)
    y_seq = np.full(len(X), np.nan, dtype=np.float32)
    has_pos = panel["seq_pos"].notna().to_numpy()
    y_seq[panel.loc[has_pos, "seq_pos"].to_numpy(dtype=int)] = y_panel[has_pos]
    has_seq = panel["universe_member"].to_numpy() & has_pos
    trainable = has_seq & ~np.isnan(y_panel)
    outputs = []
    for fold_id, fold in enumerate(folds, start=1):
        dates = panel["signal_date"]
        train_mask = trainable & (dates < fold["train_end_exclusive"]).to_numpy()
        val_mask = trainable & ((dates >= fold["validation_start"]) & (dates < fold["validation_end_exclusive"])).to_numpy()
        test_mask = ((dates >= fold["test_start"]) & (dates < fold["test_end_exclusive"])).to_numpy()
        model = _train_fold(
            X, y_seq,
            panel.loc[train_mask, "seq_pos"].to_numpy(dtype=int),
            panel.loc[val_mask, "seq_pos"].to_numpy(dtype=int),
            args, device,
        )
        out = panel[test_mask].copy()
        out["score"] = np.nan
        scoreable = test_mask & has_seq  # score members with sequences, labeled or not
        if scoreable.any():
            positions = panel.loc[scoreable, "seq_pos"].to_numpy(dtype=int)
            with torch.no_grad():
                scores = []
                for start in range(0, len(positions), args.batch_size):
                    chunk = torch.from_numpy(X[positions[start:start + args.batch_size]]).to(device)
                    scores.append(model(chunk).cpu().numpy())
            out.loc[panel.index[scoreable], "score"] = np.concatenate(scores)
        out["fold"] = fold_id
        outputs.append(out)
        print(f"fold {fold_id}/{len(folds)} done", flush=True)

    predictions = pd.concat(outputs, ignore_index=True).sort_values(["signal_date", "code"])
    costs = TradingCosts(commission_bps=3.0, stamp_duty_bps=5.0, slippage_bps=5.0)
    portfolio = simulate_portfolio(predictions, topk_dropout_builder(args.top_k, args.drop_n), costs=costs)
    universe = simulate_portfolio(predictions, equal_weight_builder(), costs=costs)
    summary = summarize_baseline(
        predictions, portfolio, universe,
        frozen_start=args.retrospective_start,
        min_frozen_days=126, min_portfolio_cagr=0.05,
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "candidate": "gru_challenger",
        "exploratory": True,
        "device": device,
        "architecture": {"channels": list(CHANNELS), "timesteps": LONG_LAGS,
                         "hidden": args.hidden, "layers": args.layers},
        "promotion_rule": "seven gates AND retro IC >= lgbm+0.01 AND secondaries within bands AND ops budget",
        "result": summary,
    }
    Path(args.output).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": summary.get("status"), "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
