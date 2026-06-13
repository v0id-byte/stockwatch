#!/usr/bin/env python3
"""Honest, reproducible A-share strategy backtest for the StockWatch quant signal.

Unlike the per-split metrics in lgbm_meta.json, this reports the WHOLE picture so a
reader can judge the signal without being misled by one lucky/unlucky window:

  - per-calendar-year cross-sectional IC (does the ranking work each year?)
  - a long-only top-k portfolio rebalanced on NON-OVERLAPPING horizons (no
    overlapping-window inflation), net of a round-trip transaction cost
  - benchmarked against CSI300 (buy & hold) and a risk-free rate, with
    CAGR / annualized vol / Sharpe / max drawdown / per-period win rate
  - the score-decile forward-return profile (monotonic == real ranking power)

Scores come from the trained LightGBM model by default (cross-sectionally
rank-normalized features, matching training). Pass --signal composite to instead
score an equal-weight z-score blend of the robust factors, so the headline edge
can be reproduced WITHOUT the model file.

This is research only. It assumes close-to-close fills at each rebalance, does not
model limit-up/down unfillable opens, capacity or borrow, and past performance does
not imply future returns.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.factors import ROBUST_FEATURES, BEAR_FEATURES, BULL_FEATURES
from analysis.regime import is_bull_trend


def _winsor_z(s):
    import pandas as pd
    s = s.clip(s.quantile(0.01), s.quantile(0.99))
    sd = s.std()
    if not sd or pd.isna(sd):
        return s * 0.0
    return (s - s.mean()) / sd


# Sign for the equal-weight composite: +1 if high factor value predicts high forward
# return, -1 otherwise (reversal/turnover factors). Trees learn this themselves; the
# composite needs it explicit.
_COMPOSITE_SIGN = {
    "RET20": -1, "RET30": -1, "RET60": -1, "RELV20": -1, "RELV30": -1, "RELV60": -1,
    "ROC20": +1, "ROC30": +1, "ROC60": +1, "QTLD20": +1, "QTLD30": +1, "QTLD60": +1,
    "QTLU60": +1, "QTLU120": +1, "MA20": +1, "MA30": +1, "MA60": +1, "IMIN30": +1,
    "TURN120": -1, "TURN250": -1, "VOLZ120": -1, "VOLZ250": -1, "AMTMA60": +1,
    "AMTMA120": +1, "AMTMA250": +1, "VMA120": +1, "VMA250": +1, "VSUMN60": +1,
    "VSUMN120": +1, "CORD60": -1, "RESI10": -1, "WVMA20": -1,
}


def _load(history_dir: Path, target: str):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    data_path = history_dir / "training_set.parquet"
    if not data_path.exists():
        raise SystemExit(f"训练集缺失: {data_path}（先运行 build_training_set.py）")
    feats = sorted(set(ROBUST_FEATURES) | set(BEAR_FEATURES) | set(BULL_FEATURES))
    avail = set(pq.ParquetFile(data_path).schema.names)
    feats = [f for f in feats if f in avail]
    cols = ["trade_date", "code", target] + feats
    df = pd.read_parquet(data_path, columns=cols)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").reset_index(drop=True)

    csi_path = history_dir / "market_sh000300.parquet"
    csi_fwd = {}
    if csi_path.exists():
        csi = pd.read_parquet(csi_path)[["trade_date", "close"]].copy()
        csi["trade_date"] = pd.to_datetime(csi["trade_date"])
        csi = csi.sort_values("trade_date").reset_index(drop=True)
        csi["fwd"] = csi["close"].shift(-20) / csi["close"] - 1
        csi_fwd = dict(zip(csi["trade_date"], csi["fwd"]))
        csi["bull"] = is_bull_trend(csi["close"]).to_numpy()
        df["bull"] = df["trade_date"].map(dict(zip(csi["trade_date"], csi["bull"])))
    else:
        df["bull"] = True
    return df, feats, csi_fwd


def _predict(df, model_path: Path, fallback_feats):
    """Cross-sectionally rank-normalize the model's features per date (matching
    training) and return the model's score for every row."""
    import lightgbm as lgb
    import json
    if not model_path.exists():
        raise SystemExit(f"模型缺失: {model_path}（先运行 train_lgbm.py，或用 --signal composite）")
    booster = lgb.Booster(model_file=str(model_path))
    meta_path = model_path.parent / model_path.name.replace(".txt", "_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model_feats = meta.get("features", fallback_feats)
    norm = df[["trade_date"]].copy()
    g = df.groupby("trade_date", sort=False)
    for f in model_feats:
        norm[f] = (g[f].rank(pct=True) - 0.5) if f in df.columns else 0.0
    return booster.predict(norm[model_feats])


def _score(df, feats, signal: str, model_path: Path):
    import pandas as pd

    if signal == "composite":
        grouped = df.groupby("trade_date", sort=False)
        parts = [grouped[f].transform(_winsor_z) * _COMPOSITE_SIGN.get(f, 1) for f in feats]
        df["score"] = pd.concat(parts, axis=1).mean(axis=1)
        return df, "composite (equal-weight z-score of robust factors)"

    if signal == "regime":
        # Asymmetric: the bear specialization beats the universal model out-of-sample,
        # but a bull-specialized model did NOT (bull alpha is too thin), so bull/normal
        # days use the universal model. bear-trend days use lgbm_bear.
        models_dir = model_path.parent
        univ_score = _predict(df, models_dir / "lgbm.txt", feats)
        bear_score = _predict(df, models_dir / "lgbm_bear.txt", feats)
        df["score_univ"] = univ_score
        df["score"] = pd.Series(univ_score, index=df.index).where(
            df["bull"].fillna(True), pd.Series(bear_score, index=df.index))
        return df, "regime-aware (universal on bull days, lgbm_bear on bear days)"

    df["score"] = _predict(df, model_path, feats)
    return df, f"LightGBM model ({model_path.name})"


def _per_year_ic(df, target):
    import pandas as pd
    print("\n=== 逐年横截面 IC（score vs %s） ===" % target)
    print("  %-6s %9s %7s %8s %6s" % ("year", "meanIC", "ICIR", "posrate", "ndays"))
    for y, gy in df.groupby(df["trade_date"].dt.year):
        ics = []
        for _, g in gy.groupby("trade_date"):
            if len(g) < 30 or g["score"].nunique() <= 1:
                continue
            ic = g["score"].corr(g[target], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
        s = pd.Series(ics)
        if len(s):
            icir = s.mean() / s.std() if s.std() else 0
            print("  %-6d %+9.4f %+7.2f %8.2f %6d" % (y, s.mean(), icir, (s > 0).mean(), len(s)))


def _backtest(df, target, csi_fwd, hold, topk, cost, rf_annual, test_start):
    import numpy as np
    import pandas as pd

    rf_per = rf_annual * hold / 252
    dates = np.sort(df["trade_date"].unique())

    def run(period, label):
        rb = list(period)[::hold]
        port, uni, bench = [], [], []
        prev = set()
        dec_rows = []
        for d in rb:
            g = df[df["trade_date"] == d]
            if len(g) < topk * 2 or g["score"].nunique() <= 1:
                continue
            top = g.sort_values("score", ascending=False).head(topk)
            sel = set(top["code"])
            turn = 1.0 if not prev else 1 - len(prev & sel) / topk
            port.append(top[target].mean() - turn * cost)
            uni.append(g[target].mean())
            bench.append(csi_fwd.get(pd.Timestamp(d), np.nan))
            prev = sel
            dq = (g["score"].rank(pct=True) * 10).clip(upper=9).astype(int)
            dec_rows.append(g.assign(dec=dq).groupby("dec")[target].mean())
        if not port:
            print("\n=== %s: 样本不足 ===" % label)
            return
        port = np.array(port); uni = np.array(uni)
        bench = pd.Series(bench).fillna(pd.Series(uni)).to_numpy()
        n = len(port); pa = 252 / hold
        cagr = lambda r: np.prod(1 + r) ** (pa / n) - 1
        c = np.cumprod(1 + port); peak = np.maximum.accumulate(c)
        maxdd = ((c - peak) / peak).min()
        sharpe = (port - rf_per).mean() / port.std() * np.sqrt(pa) if port.std() else 0
        dec = pd.DataFrame(dec_rows).mean()
        print("\n=== %s  (%d 个非重叠 %d 日持有期, top%d, 扣 %.2f%% 双边成本) ===" % (
            label, n, hold, topk, cost * 100))
        print("  策略    : CAGR=%+.2f%%  年化波动=%.2f%%  Sharpe=%+.2f  最大回撤=%+.2f%%" % (
            cagr(port) * 100, port.std() * np.sqrt(pa) * 100, sharpe, maxdd * 100))
        print("  CSI300  : CAGR=%+.2f%%   |   等权universe: CAGR=%+.2f%%   |   无风险: %.1f%%/年" % (
            cagr(bench) * 100, cagr(uni) * 100, rf_annual * 100))
        print("  跑赢无风险的持有期占比: %.0f%%   跑赢CSI300: %.0f%%   跑赢universe: %.0f%%" % (
            (port > rf_per).mean() * 100, (port > bench).mean() * 100, (port > uni).mean() * 100))
        print("  分位前向收益 0(低分)->9(高分): " + " ".join("%+.3f" % dec.get(i, float("nan")) for i in range(10)))
        print("  decile 9-0 spread = %+.4f" % (dec.get(9) - dec.get(0)))

    ts = pd.Timestamp(test_start)
    run([d for d in dates if d < ts], "TRAIN / 样本内 (< %s)" % test_start)
    run([d for d in dates if d >= ts], "TEST / 样本外 (>= %s)" % test_start)


def _per_regime_compare(df, target):
    """Show regime-aware vs universal IC separately on bull-trend and bear-trend days —
    the apples-to-apples evidence for whether the bull/bear split actually helps."""
    import pandas as pd

    def ic_on(mask, col):
        ics = []
        for _, g in df[mask].groupby("trade_date"):
            if len(g) < 30 or g[col].nunique() <= 1:
                continue
            v = g[col].corr(g[target], method="spearman")
            if pd.notna(v):
                ics.append(v)
        s = pd.Series(ics)
        return (s.mean(), s.mean() / s.std() if len(s) and s.std() else 0, len(s))

    print("\n=== regime 分段 IC：regime-aware vs 通用模型 ===")
    print("  %-10s %20s %20s" % ("行情", "通用 IC(ICIR)", "regime-aware IC(ICIR)"))
    for label, mask in [("熊市(bear)", df["bull"] == False), ("牛市(bull)", df["bull"] == True)]:  # noqa: E712
        um, ui, _ = ic_on(mask, "score_univ")
        rm, ri, n = ic_on(mask, "score")
        print("  %-10s   %+.4f (%+.2f)        %+.4f (%+.2f)   [%d 日]" % (label, um, ui, rm, ri, n))


def main(argv=None):
    p = argparse.ArgumentParser(description="StockWatch 量化信号回测（诚实、可复现）")
    p.add_argument("--signal", choices=["model", "composite", "regime"], default="model")
    p.add_argument("--history-dir", default=os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history"))
    p.add_argument("--model-path", default=str(ROOT / "models" / "lgbm.txt"))
    p.add_argument("--target", default="forward_20d_return")
    p.add_argument("--hold", type=int, default=20, help="每个持有期的交易日数（=横截面非重叠采样步长）")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--cost", type=float, default=0.0030, help="双边换手成本，默认 0.30%%")
    p.add_argument("--rf", type=float, default=0.02, help="无风险年化，默认 2%%")
    p.add_argument("--test-start", default="2025-01-01", help="样本外起始日")
    args = p.parse_args(argv)

    history_dir = Path(args.history_dir).expanduser()
    df, feats, csi_fwd = _load(history_dir, args.target)
    df, desc = _score(df, feats, args.signal, Path(args.model_path).expanduser())
    print("信号: %s   |   特征数: %d   |   样本: %d 行, %s ~ %s" % (
        desc, len(feats), len(df), df["trade_date"].min().date(), df["trade_date"].max().date()))
    _per_year_ic(df, args.target)
    if args.signal == "regime" and "score_univ" in df.columns:
        _per_regime_compare(df, args.target)
    _backtest(df, args.target, csi_fwd, args.hold, args.topk, args.cost, args.rf, args.test_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
