"""Lead-lag propagation features for related-stock follow-through.

The module keeps the first version intentionally small: it looks for stocks
that historically reacted one trading day after today's leaders, then exposes
that relationship as model features and LLM-ready context.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


PROPAGATION_FEATURES = [
    "prop_leader_ret_1d",
    "prop_leader_volz_20d",
    "prop_relation_corr_1d",
    "prop_relation_weight",
    "prop_underreaction_1d",
    "prop_leader_count",
    "prop_score",
]

DEFAULT_LEADER_RETURN = 0.04
DEFAULT_LEADER_VOLZ = 1.0
DEFAULT_UNDERREACTION_CEIL = 0.025
DEFAULT_MIN_CORR = 0.15
DEFAULT_LOOKBACK = 60
DEFAULT_MAX_LEADERS = 8


def _as_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        return num if np.isfinite(num) else default
    except (TypeError, ValueError):
        return default


def _empty_feature_row() -> dict[str, float]:
    return {name: 0.0 for name in PROPAGATION_FEATURES}


def _pivot_panel(df: pd.DataFrame, value: str) -> pd.DataFrame:
    panel = df.pivot_table(index="trade_date", columns="code", values=value, aggfunc="last")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index().replace([np.inf, -np.inf], np.nan)


def _volume_zscore(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    mean = volume.rolling(window, min_periods=max(5, window // 2)).mean()
    std = volume.rolling(window, min_periods=max(5, window // 2)).std()
    return ((volume - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _lag_corr_from_returns(leader_ret: pd.Series, candidate_ret: pd.Series,
                           lookback: int = DEFAULT_LOOKBACK) -> float:
    aligned = pd.concat([
        leader_ret.shift(1).rename("leader"),
        candidate_ret.rename("candidate"),
    ], axis=1).dropna()
    if len(aligned) < max(20, lookback // 3):
        return 0.0
    recent = aligned.tail(lookback)
    if recent["leader"].std() == 0 or recent["candidate"].std() == 0:
        return 0.0
    corr = recent["leader"].corr(recent["candidate"])
    return max(0.0, _as_float(corr))


def _corrwith_leader(candidate_hist: pd.DataFrame, leader_hist: pd.Series,
                     columns: list[str], min_periods: int) -> pd.Series:
    """Fast column-wise corr(candidate_hist[col], leader_hist)."""
    y = leader_hist.to_numpy(dtype="float64")
    x = candidate_hist[columns].to_numpy(dtype="float64")
    y_finite = np.isfinite(y)
    mask = np.isfinite(x) & y_finite[:, None]
    counts = mask.sum(axis=0)
    valid = counts >= min_periods
    result = np.zeros(len(columns), dtype="float64")
    if not valid.any():
        return pd.Series(result, index=columns)

    x_clean = np.where(mask, x, 0.0)
    y_matrix = np.where(mask, y[:, None], 0.0)
    x_mean = np.divide(
        x_clean.sum(axis=0),
        counts,
        out=np.zeros(len(columns), dtype="float64"),
        where=counts > 0,
    )
    y_mean = np.divide(
        y_matrix.sum(axis=0),
        counts,
        out=np.zeros(len(columns), dtype="float64"),
        where=counts > 0,
    )
    x_centered = np.where(mask, x - x_mean, 0.0)
    y_centered = np.where(mask, y[:, None] - y_mean, 0.0)
    cov = (x_centered * y_centered).sum(axis=0)
    x_var = (x_centered * x_centered).sum(axis=0)
    y_var = (y_centered * y_centered).sum(axis=0)
    denom = np.sqrt(x_var * y_var)
    corr = np.divide(
        cov,
        denom,
        out=np.zeros(len(columns), dtype="float64"),
        where=(denom > 0) & valid,
    )
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    result[valid] = np.clip(corr[valid], 0.0, None)
    return pd.Series(result, index=columns)


def _lag_corr_from_frames(leader_df: pd.DataFrame, candidate_df: pd.DataFrame,
                          lookback: int = DEFAULT_LOOKBACK) -> float:
    if leader_df.empty or candidate_df.empty:
        return 0.0
    leader = leader_df[["trade_date", "close"]].copy()
    candidate = candidate_df[["trade_date", "close"]].copy()
    leader["trade_date"] = pd.to_datetime(leader["trade_date"])
    candidate["trade_date"] = pd.to_datetime(candidate["trade_date"])
    leader = leader.sort_values("trade_date").set_index("trade_date")["close"].astype(float).pct_change()
    candidate = candidate.sort_values("trade_date").set_index("trade_date")["close"].astype(float).pct_change()
    return _lag_corr_from_returns(leader, candidate, lookback)


def _latest_volume_z(kline: list[dict], window: int = 20) -> float:
    if len(kline) < max(5, window // 2):
        return 0.0
    volume = pd.Series([_as_float(row.get("volume")) for row in kline], dtype="float64")
    recent = volume.tail(window)
    std = recent.std()
    if not std:
        return 0.0
    return _as_float((recent.iloc[-1] - recent.mean()) / std)


def detect_leaders_from_quotes(quotes: dict[str, dict],
                               threshold: float = DEFAULT_LEADER_RETURN,
                               max_leaders: int = DEFAULT_MAX_LEADERS) -> list[str]:
    """Return today's strongest visible leaders from realtime quotes."""
    rows = []
    for code, quote in quotes.items():
        pct = _as_float(quote.get("pct_change")) / 100.0
        if pct >= threshold:
            rows.append((code, pct))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [code for code, _pct in rows[:max_leaders]]


def compute_propagation_feature_frame(df: pd.DataFrame,
                                      leader_return_threshold: float = DEFAULT_LEADER_RETURN,
                                      leader_volz_threshold: float = DEFAULT_LEADER_VOLZ,
                                      underreaction_ceiling: float = DEFAULT_UNDERREACTION_CEIL,
                                      min_corr: float = DEFAULT_MIN_CORR,
                                      lookback: int = DEFAULT_LOOKBACK,
                                      max_leaders: int = DEFAULT_MAX_LEADERS) -> pd.DataFrame:
    """Compute point-in-time propagation features for a long OHLCV panel.

    Input columns: trade_date, code, close, volume. Each row's features use the
    row date's visible returns/volume plus lag correlations estimated from
    rolling history up to that date.
    """
    required = {"trade_date", "code", "close", "volume"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["trade_date", "code", *PROPAGATION_FEATURES])

    panel = df[["trade_date", "code", "close", "volume"]].copy()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["close"] = pd.to_numeric(panel["close"], errors="coerce")
    panel["volume"] = pd.to_numeric(panel["volume"], errors="coerce")
    close = _pivot_panel(panel, "close")
    volume = _pivot_panel(panel, "volume")
    ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan)
    volz = _volume_zscore(volume)
    leader_mask = (ret1 >= leader_return_threshold) & (volz >= leader_volz_threshold)

    rows = []
    codes = list(close.columns)
    zero = pd.Series(0.0, index=codes)
    for pos, trade_date in enumerate(close.index):
        if pos < 2:
            for code in codes:
                rows.append({"trade_date": trade_date, "code": code, **_empty_feature_row()})
            continue

        leaders = ret1.loc[trade_date][leader_mask.loc[trade_date].fillna(False)]
        leaders = leaders.sort_values(ascending=False).head(max_leaders)
        if leaders.empty:
            for code in codes:
                rows.append({"trade_date": trade_date, "code": code, **_empty_feature_row()})
            continue

        best_score = zero.copy()
        best_ret = zero.copy()
        best_volz = zero.copy()
        best_corr = zero.copy()
        leader_count = pd.Series(0.0, index=codes)
        hist_start = max(0, pos - lookback - 1)

        for leader_code, leader_ret_today in leaders.items():
            leader_hist = ret1[leader_code].iloc[hist_start:pos].shift(1)
            candidate_hist = ret1.iloc[hist_start:pos]
            min_periods = max(20, lookback // 3)
            aligned = candidate_hist.loc[leader_hist.notna()]
            leader_aligned = leader_hist.loc[leader_hist.notna()]
            if len(aligned) < min_periods:
                continue
            corr = _corrwith_leader(aligned, leader_aligned, codes, min_periods)
            corr.loc[leader_code] = 0.0
            leader_count += (corr >= min_corr).astype(float)
            volz_today = _as_float(volz.at[trade_date, leader_code])
            raw_score = corr * max(0.0, _as_float(leader_ret_today)) * (1 + max(0.0, volz_today))
            update = raw_score > best_score
            best_score[update] = raw_score[update]
            best_ret[update] = _as_float(leader_ret_today)
            best_volz[update] = volz_today
            best_corr[update] = corr[update]

        candidate_ret = ret1.loc[trade_date].fillna(0.0)
        underreaction = (best_ret - candidate_ret).clip(lower=0.0)
        not_chased = (candidate_ret <= underreaction_ceiling).astype(float)
        prop_score = best_score * (1 + underreaction.clip(upper=0.12)) * not_chased

        for code in codes:
            rows.append({
                "trade_date": trade_date,
                "code": code,
                "prop_leader_ret_1d": _as_float(best_ret.get(code)),
                "prop_leader_volz_20d": _as_float(best_volz.get(code)),
                "prop_relation_corr_1d": _as_float(best_corr.get(code)),
                "prop_relation_weight": _as_float(best_score.get(code)),
                "prop_underreaction_1d": _as_float(underreaction.get(code)),
                "prop_leader_count": _as_float(leader_count.get(code)),
                "prop_score": _as_float(prop_score.get(code)),
            })

    out = pd.DataFrame(rows)
    out["trade_date"] = out["trade_date"].dt.strftime("%Y-%m-%d")
    return out.replace([np.inf, -np.inf], 0).fillna(0.0)


def add_propagation_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return df with propagation features left-joined by date/code."""
    features = compute_propagation_feature_frame(df, **kwargs)
    if features.empty:
        out = df.copy()
        for name in PROPAGATION_FEATURES:
            out[name] = 0.0
        return out
    out = df.copy()
    out["trade_date"] = out["trade_date"].astype(str)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out = out.merge(features, on=["trade_date", "code"], how="left")
    for name in PROPAGATION_FEATURES:
        out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
    return out


def compute_latest_propagation_features(codes: list[str], quotes: dict[str, dict],
                                        kline_by_code: dict[str, list[dict]],
                                        leader_return_threshold: float = DEFAULT_LEADER_RETURN,
                                        min_corr: float = DEFAULT_MIN_CORR,
                                        lookback: int = DEFAULT_LOOKBACK,
                                        max_leaders: int = DEFAULT_MAX_LEADERS
                                        ) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    """Compute today's propagation features and compact explanation strings."""
    leaders = detect_leaders_from_quotes(quotes, leader_return_threshold, max_leaders)
    feature_map = {code: _empty_feature_row() for code in codes}
    contexts = {code: "" for code in codes}
    if not leaders:
        return feature_map, contexts

    leader_frames = {
        code: pd.DataFrame(kline_by_code.get(code, []))
        for code in leaders
        if kline_by_code.get(code)
    }
    if not leader_frames:
        return feature_map, contexts

    for code in codes:
        candidate_frame = pd.DataFrame(kline_by_code.get(code, []))
        if candidate_frame.empty:
            continue
        best = _empty_feature_row()
        best_leader = ""
        leader_count = 0
        candidate_ret = _as_float(quotes.get(code, {}).get("pct_change")) / 100.0
        for leader_code, leader_frame in leader_frames.items():
            if leader_code == code:
                continue
            corr = _lag_corr_from_frames(leader_frame, candidate_frame, lookback)
            if corr >= min_corr:
                leader_count += 1
            leader_ret = _as_float(quotes.get(leader_code, {}).get("pct_change")) / 100.0
            leader_volz = _latest_volume_z(kline_by_code.get(leader_code, []))
            relation_weight = corr * max(0.0, leader_ret) * (1 + max(0.0, leader_volz))
            underreaction = max(0.0, leader_ret - candidate_ret)
            chased_penalty = 0.35 if candidate_ret > DEFAULT_UNDERREACTION_CEIL else 1.0
            score = relation_weight * (1 + min(0.12, underreaction)) * chased_penalty
            if score > best["prop_score"]:
                best = {
                    "prop_leader_ret_1d": leader_ret,
                    "prop_leader_volz_20d": leader_volz,
                    "prop_relation_corr_1d": corr,
                    "prop_relation_weight": relation_weight,
                    "prop_underreaction_1d": underreaction,
                    "prop_leader_count": 0.0,
                    "prop_score": score,
                }
                best_leader = leader_code
        best["prop_leader_count"] = float(leader_count)
        feature_map[code] = best
        if best_leader and best["prop_score"] > 0:
            leader_name = quotes.get(best_leader, {}).get("name") or best_leader
            contexts[code] = (
                "关联补涨观察: "
                f"领涨参考 {leader_name}({best_leader}) {best['prop_leader_ret_1d']:+.1%}; "
                f"历史1日滞后相关 {best['prop_relation_corr_1d']:.2f}; "
                f"候选今日 {candidate_ret:+.1%}; "
                f"传播分 {best['prop_score']:.4f}"
            )
    return feature_map, contexts


def _read_history_frame(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path, columns=["trade_date", "close", "volume"])
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna(subset=["trade_date", "close"]).sort_values("trade_date")


def find_related_candidates_from_history(stock_dir: Path, leaders: list[str],
                                         leader_returns: dict[str, float],
                                         existing_codes: set[str],
                                         max_candidates: int = 10,
                                         min_corr: float = DEFAULT_MIN_CORR,
                                         lookback: int = 120) -> list[dict]:
    """Rank extra candidates from local historical parquet files."""
    stock_dir = Path(stock_dir).expanduser()
    if not stock_dir.exists() or not leaders:
        return []

    leader_frames = {
        leader: _read_history_frame(stock_dir / f"{leader}.parquet")
        for leader in leaders
    }
    leader_frames = {code: df for code, df in leader_frames.items() if not df.empty}
    if not leader_frames:
        return []

    rows = []
    existing = {str(code).zfill(6) for code in existing_codes}
    for path in stock_dir.glob("*.parquet"):
        code = path.stem.zfill(6)
        if code in existing or code in leader_frames:
            continue
        candidate = _read_history_frame(path)
        if len(candidate) < max(80, lookback // 2):
            continue
        best = {"score": 0.0, "leader": "", "corr": 0.0, "underreaction": 0.0}
        candidate_ret = candidate["close"].pct_change().tail(1).iloc[0]
        candidate_ret = _as_float(candidate_ret)
        for leader_code, leader_df in leader_frames.items():
            corr = _lag_corr_from_frames(leader_df, candidate, lookback)
            if corr < min_corr:
                continue
            leader_ret = _as_float(leader_returns.get(leader_code))
            underreaction = max(0.0, leader_ret - candidate_ret)
            score = corr * max(0.0, leader_ret) * (1 + min(0.12, underreaction))
            if score > best["score"]:
                best = {
                    "score": score,
                    "leader": leader_code,
                    "corr": corr,
                    "underreaction": underreaction,
                }
        if best["score"] > 0:
            rows.append({"code": code, **best})

    rows.sort(key=lambda row: row["score"], reverse=True)
    result = rows[:max_candidates]
    if result:
        logger.info(
            "关联补涨候选: "
            + ", ".join(f"{row['code']}<-{row['leader']}:{row['score']:.4f}" for row in result[:8])
        )
    return result
