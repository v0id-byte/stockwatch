"""Frozen out-of-sample baselines and common long-only portfolio accounting.

This module is research-only.  It intentionally consumes a strict, already
point-in-time panel instead of guessing missing trading flags or exposures.
Missing data raises :class:`DataContractError`; it is never converted into a
zero signal or an apparently valid backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


class DataContractError(RuntimeError):
    """The research panel cannot support a point-in-time backtest."""


COMMON_COLUMNS = (
    "signal_date",
    "execution_at",
    "exit_at",
    "feature_available_at",
    "code",
    "next_open_return",
    "buyable",
    "sellable",
    "universe_member",
    "benchmark_return",
)

INDEX_COLUMNS = (
    "benchmark_weight",
    "sector",
    "log_market_cap",
    "beta",
    "volatility_20d",
)


@dataclass(frozen=True)
class TradingCosts:
    """Proportional A-share execution cost assumptions."""

    commission_bps: float = 3.0
    stamp_duty_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def buy_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000.0

    @property
    def sell_rate(self) -> float:
        return (self.commission_bps + self.stamp_duty_bps + self.slippage_bps) / 10_000.0


@dataclass(frozen=True)
class EnhancementConstraints:
    """Daily long-only constraints using a diagonal-volatility TE proxy."""

    max_tracking_error: float = 0.065
    max_one_way_turnover: float = 0.20
    max_active_weight: float = 0.01
    max_stock_weight: float = 0.05
    exposure_tolerance: float = 1e-8


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise DataContractError(f"{context} missing required columns: {missing}")


def validate_pit_panel(
    frame: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    require_index_columns: bool = False,
) -> pd.DataFrame:
    """Normalize and validate the strict signal-date/next-open panel."""

    feature_columns = list(feature_columns)
    required = [*COMMON_COLUMNS, *feature_columns]
    if require_index_columns:
        required.extend(INDEX_COLUMNS)
    _require_columns(frame, required, "OOS panel")
    if frame.empty:
        raise DataContractError("OOS panel is empty")

    out = frame.copy()
    for name in ("signal_date", "execution_at", "exit_at", "feature_available_at"):
        out[name] = pd.to_datetime(out[name], errors="coerce")
    out["signal_date"] = out["signal_date"].dt.normalize()
    extracted_code = out["code"].astype(str).str.strip().str.extract(r"(\d{1,6})(?:\.0)?$", expand=False)
    out["code"] = extracted_code.map(lambda value: str(value).zfill(6) if pd.notna(value) and str(value) else "")

    if out[["signal_date", "execution_at", "exit_at", "feature_available_at"]].isna().any().any():
        raise DataContractError("PIT timestamps contain null or invalid values")
    if (out["feature_available_at"] >= out["execution_at"]).any():
        raise DataContractError("feature_available_at must be strictly before next-open execution_at")
    if (out["execution_at"] >= out["exit_at"]).any():
        raise DataContractError("exit_at must be after execution_at")
    execution_counts = out.groupby("signal_date")["execution_at"].nunique(dropna=False)
    exit_counts = out.groupby("signal_date")["exit_at"].nunique(dropna=False)
    if (execution_counts != 1).any() or (exit_counts != 1).any():
        raise DataContractError("each signal date must share one market-calendar execution and exit timestamp")
    if out.duplicated(["signal_date", "code"]).any():
        raise DataContractError("signal_date/code keys are not unique")
    if out["code"].isin({"", "000000"}).any():
        raise DataContractError("panel contains invalid stock codes")

    for name in ("buyable", "sellable", "universe_member"):
        if out[name].isna().any():
            raise DataContractError(f"{name} contains unknown values")
        if not pd.api.types.is_bool_dtype(out[name]):
            values = set(out[name].dropna().unique().tolist())
            if not values.issubset({0, 1, True, False}):
                raise DataContractError(f"{name} must be boolean")
            out[name] = out[name].astype(bool)

    numeric = ["next_open_return", "benchmark_return"]
    if require_index_columns:
        numeric.append("benchmark_weight")
    for name in numeric:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    if not np.isfinite(out[numeric].to_numpy(dtype=float)).all():
        raise DataContractError("required returns or exposures contain non-finite values")
    for name in feature_columns:
        out[name] = pd.to_numeric(out[name], errors="coerce")
        finite = np.isfinite(out[name].dropna().to_numpy(dtype=float)).all()
        coverage = float(out[name].notna().mean())
        if not finite or coverage < 0.50:
            raise DataContractError(f"feature {name} has invalid values or coverage below 50%")
    if (out["next_open_return"] < -1).any():
        raise DataContractError("next_open_return is below a total loss")

    benchmark_counts = out.groupby("signal_date")["benchmark_return"].nunique(dropna=False)
    if (benchmark_counts != 1).any():
        raise DataContractError("benchmark_return must be unique within each signal date")
    if require_index_columns:
        for name in ("log_market_cap", "beta", "volatility_20d"):
            out[name] = pd.to_numeric(out[name], errors="coerce")
            finite = np.isfinite(out[name].dropna().to_numpy(dtype=float)).all()
            if not finite or float(out[name].notna().mean()) < 0.50:
                raise DataContractError(f"index exposure {name} is invalid or below 50% coverage")
        members = out[out["benchmark_weight"] > 0]
        if members.empty:
            raise DataContractError("no positive CSI500 benchmark weights")
        sums = members.groupby("signal_date")["benchmark_weight"].sum()
        if ((sums < 0.995) | (sums > 1.005)).any():
            raise DataContractError("CSI500 benchmark weights must sum to one on every date")
        if out.loc[out["benchmark_weight"] > 0, "sector"].isna().any():
            raise DataContractError("PIT sector is missing for CSI500 constituents")
        if (out["volatility_20d"].dropna() < 0).any():
            raise DataContractError("volatility_20d must be non-negative")
    return out.sort_values(["signal_date", "code"]).reset_index(drop=True)


def make_expanding_folds(
    dates: Iterable[pd.Timestamp],
    *,
    min_train_days: int = 504,
    validation_days: int = 63,
    test_days: int = 63,
    purge_days: int = 2,
) -> list[dict]:
    """Create expanding train/validation/test folds with a trading-day purge."""

    ordered = pd.Index(pd.to_datetime(sorted(set(dates))).normalize().unique())
    if min(min_train_days, validation_days, test_days) <= 0 or purge_days < 1:
        raise ValueError("fold lengths must be positive and purge_days >= 1")
    first_test = min_train_days + purge_days + validation_days + purge_days
    folds = []
    for test_start in range(first_test, len(ordered), test_days):
        test_end = min(len(ordered), test_start + test_days)
        val_end = test_start - purge_days
        val_start = val_end - validation_days
        train_end = val_start - purge_days
        if train_end < min_train_days:
            continue
        folds.append({
            "train_start": ordered[0],
            "train_end_exclusive": ordered[train_end],
            "validation_start": ordered[val_start],
            "validation_end_exclusive": ordered[val_end],
            "test_start": ordered[test_start],
            "test_end_exclusive": ordered[test_end] if test_end < len(ordered) else ordered[-1] + pd.Timedelta(days=1),
            "purge_days": int(purge_days),
        })
    return folds


def fit_lightgbm_oos(
    panel: pd.DataFrame,
    features: list[str],
    target: str,
    folds: list[dict],
    *,
    seed: int = 20260716,
    num_threads: int = 4,
) -> pd.DataFrame:
    """Fit one fixed LightGBM recipe per expanding fold and return test scores."""

    _require_columns(panel, ["signal_date", "code", target, *features], "model panel")
    if not features:
        raise DataContractError("model feature list is empty")
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - exercised in train environment
        raise RuntimeError("LightGBM baseline requires requirements-train.txt") from exc

    params = {
        "objective": "regression_l2",
        "metric": "l2",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(num_threads),
    }
    outputs = []
    for fold_id, fold in enumerate(folds, start=1):
        train = panel[panel["signal_date"] < fold["train_end_exclusive"]]
        valid = panel[
            (panel["signal_date"] >= fold["validation_start"])
            & (panel["signal_date"] < fold["validation_end_exclusive"])
        ]
        test = panel[
            (panel["signal_date"] >= fold["test_start"])
            & (panel["signal_date"] < fold["test_end_exclusive"])
        ]
        # Non-members remain in the test frame so an existing position can be
        # liquidated after it leaves the PIT universe.  They must not influence
        # model fitting or candidate ranking.
        train = train[train["universe_member"]]
        valid = valid[valid["universe_member"]]
        if train.empty or valid.empty or test.empty:
            raise DataContractError(f"fold {fold_id} has empty preregistered train/validation/test rows")
        train = train.dropna(subset=[target, *features])
        valid = valid.dropna(subset=[target, *features])
        test_model = test.dropna(subset=[target, *features])
        if train.empty or valid.empty or test_model.empty:
            raise DataContractError(f"fold {fold_id} lacks complete model rows after feature/target checks")
        train_set = lgb.Dataset(train[features], label=train[target], feature_name=features)
        valid_set = lgb.Dataset(valid[features], label=valid[target], reference=train_set, feature_name=features)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[valid_set],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        out = test.copy()
        out["score"] = np.nan
        out.loc[test_model.index, "score"] = model.predict(
            test_model[features], num_iteration=model.best_iteration,
        )
        out["fold"] = fold_id
        outputs.append(out)
    if not outputs:
        raise DataContractError("walk-forward produced no test predictions")
    if len(outputs) != len(folds):
        raise DataContractError("walk-forward did not cover every preregistered fold")
    return pd.concat(outputs, ignore_index=True).sort_values(["signal_date", "code"]).reset_index(drop=True)


def assign_size_bucket(frame: pd.DataFrame, bottom_fraction: float = 0.30) -> pd.Series:
    """Mark the daily smallest market-cap bucket without using future data."""

    if not 0 < bottom_fraction < 1:
        raise ValueError("bottom_fraction must be between zero and one")
    _require_columns(frame, ["signal_date", "log_market_cap"], "size bucket")
    result = pd.Series("not_in_universe", index=frame.index, dtype="object")
    eligible = frame["universe_member"]
    pct = frame.loc[eligible].groupby("signal_date", sort=False)["log_market_cap"].rank(
        method="first", pct=True,
    )
    result.loc[pct.index] = np.where(pct <= bottom_fraction, "micro_bottom_30", "large_top_70")
    return result


def _topk_dropout_target(
    group: pd.DataFrame,
    current: pd.Series,
    *,
    top_k: int,
    drop_n: int,
) -> pd.Series:
    candidates = group[group["universe_member"] & group["score"].notna()].sort_values(
        ["score", "code"], ascending=[False, True],
    )
    pred_score = candidates.set_index("code")["score"]
    held = list(current[current > 1e-12].index)
    if not held:
        selected = list(pred_score.index[:top_k])
        if not selected:
            raise DataContractError("TopkDropout has no PIT-universe candidates")
        return pd.Series(1.0 / len(selected), index=selected, dtype="float64")

    last = pred_score.reindex(held).sort_values(ascending=False, na_position="last").index
    new_candidates = pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index
    today_count = max(0, drop_n + top_k - len(last))
    today = pd.Index(new_candidates[:today_count])
    combined = pred_score.reindex(last.union(today)).sort_values(ascending=False, na_position="last").index
    bottom = combined[-drop_n:]
    sell = pd.Index(last[last.isin(bottom)])
    buy_count = max(0, len(sell) + top_k - len(last))
    buy = pd.Index(today[:buy_count])
    survivors = [code for code in held if code not in set(sell)]
    if not survivors and buy.empty:
        raise DataContractError("TopkDropout has no selectable names")
    target = current.reindex(survivors, fill_value=0.0).copy()
    available_cash = max(0.0, 1.0 - float(current.sum())) + float(current.reindex(sell, fill_value=0.0).sum())
    if len(buy):
        allocation = available_cash / len(buy)
        target = pd.concat([target, pd.Series(allocation, index=buy, dtype="float64")])
    return target.groupby(level=0).sum()


def topk_dropout_builder(top_k: int = 50, drop_n: int = 5) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    if top_k <= 0 or drop_n <= 0 or drop_n > top_k:
        raise ValueError("require top_k > 0 and 0 < drop_n <= top_k")

    def build(group: pd.DataFrame, current: pd.Series) -> pd.Series:
        return _topk_dropout_target(group, current, top_k=top_k, drop_n=drop_n)

    return build


def equal_weight_builder(group_filter: str = "universe_member") -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    def build(group: pd.DataFrame, _current: pd.Series) -> pd.Series:
        eligible = group[group[group_filter]].copy()
        if eligible.empty:
            raise DataContractError(f"no names in equal-weight {group_filter} universe")
        return pd.Series(1.0 / len(eligible), index=eligible["code"], dtype="float64")

    return build


def _residualize_signal(group: pd.DataFrame) -> np.ndarray:
    sectors = pd.get_dummies(group["sector"].astype(str), dtype=float)
    numeric = group[["log_market_cap", "beta"]].to_numpy(dtype=float)
    numeric = (numeric - numeric.mean(axis=0)) / np.where(numeric.std(axis=0) > 0, numeric.std(axis=0), 1.0)
    design = np.column_stack([np.ones(len(group)), sectors.to_numpy(dtype=float), numeric])
    score = group["score"].to_numpy(dtype=float)
    score = (score - score.mean()) / (score.std() or 1.0)
    fitted = design @ np.linalg.lstsq(design, score, rcond=None)[0]
    return score - fitted


def _one_way_turnover(weights: pd.Series, current: pd.Series) -> float:
    union = weights.index.union(current.index)
    return float((weights.reindex(union, fill_value=0) - current.reindex(union, fill_value=0)).abs().sum() / 2)


def _constraint_scale(
    benchmark: np.ndarray,
    direction: np.ndarray,
    annual_vol: np.ndarray,
    constraints: EnhancementConstraints,
) -> float:
    caps = [1.0]
    active_abs = np.abs(direction)
    if active_abs.max(initial=0) > 0:
        caps.append(constraints.max_active_weight / active_abs.max())
    negative = direction < 0
    if negative.any():
        caps.append(float(np.min(benchmark[negative] / -direction[negative])))
    positive = direction > 0
    if positive.any():
        headroom = constraints.max_stock_weight - benchmark[positive]
        if (headroom < -constraints.exposure_tolerance).any():
            raise DataContractError("benchmark constituent already exceeds max_stock_weight")
        caps.append(float(np.min(np.maximum(0.0, headroom) / direction[positive])))
    diagonal_te_proxy = float(np.sqrt(np.sum(np.square(direction * annual_vol))))
    if diagonal_te_proxy > 0:
        caps.append(constraints.max_tracking_error / diagonal_te_proxy)
    return max(0.0, min(caps))


def index_enhancement_builder(
    constraints: EnhancementConstraints = EnhancementConstraints(),
) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    """Build benchmark-relative, industry/size/beta neutral long-only weights."""

    def build(group: pd.DataFrame, current: pd.Series) -> pd.Series:
        _require_columns(group, ["score", *INDEX_COLUMNS], "index enhancement date")
        members = group[group["benchmark_weight"] > 0].copy().sort_values("code")
        if members.empty or members[["score", *INDEX_COLUMNS]].isna().any().any():
            raise DataContractError("incomplete CSI500 score, PIT weights, or exposures")
        benchmark = members["benchmark_weight"].to_numpy(dtype=float)
        benchmark = benchmark / benchmark.sum()
        if (benchmark > constraints.max_stock_weight + constraints.exposure_tolerance).any():
            raise DataContractError("benchmark constituent exceeds max_stock_weight; no grandfathering is enabled")
        residual = _residualize_signal(members)
        gross = float(np.abs(residual).sum())
        if gross <= 1e-12:
            return pd.Series(benchmark, index=members["code"], dtype="float64")
        direction = residual / gross
        scale_cap = _constraint_scale(
            benchmark,
            direction,
            members["volatility_20d"].to_numpy(dtype=float),
            constraints,
        )
        benchmark_weights = pd.Series(benchmark, index=members["code"], dtype="float64")
        if not current.empty and _one_way_turnover(benchmark_weights, current) > constraints.max_one_way_turnover + 1e-12:
            raise DataContractError("benchmark rebalance alone exceeds max_one_way_turnover")

        def weights_at(scale: float) -> pd.Series:
            return pd.Series(benchmark + scale * direction, index=members["code"], dtype="float64")

        if not current.empty and _one_way_turnover(weights_at(scale_cap), current) > constraints.max_one_way_turnover:
            low, high = 0.0, scale_cap
            for _ in range(60):
                middle = (low + high) / 2
                if _one_way_turnover(weights_at(middle), current) <= constraints.max_one_way_turnover:
                    low = middle
                else:
                    high = middle
            scale_cap = low
        weights = weights_at(scale_cap)
        active = weights.to_numpy() - benchmark
        if (
            abs(weights.sum() - 1) > constraints.exposure_tolerance
            or (weights < -constraints.exposure_tolerance).any()
            or (weights > constraints.max_stock_weight + constraints.exposure_tolerance).any()
        ):
            raise DataContractError("index-enhancement weight projection violated long-only/full-investment")
        sector_active = pd.Series(active, index=members["sector"]).groupby(level=0).sum().abs().max()
        size_active = abs(float(active @ members["log_market_cap"].to_numpy(dtype=float)))
        beta_active = abs(float(active @ members["beta"].to_numpy(dtype=float)))
        if max(float(sector_active), size_active, beta_active) > constraints.exposure_tolerance:
            raise DataContractError("index-enhancement exposure projection exceeded tolerance")
        return weights.clip(lower=0)

    return build


def _execute_target(
    group: pd.DataFrame,
    current: pd.Series,
    desired: pd.Series,
) -> tuple[pd.Series, float, float]:
    rows = group.set_index("code")
    missing_held = [code for code in current[current > 1e-12].index if code not in rows.index]
    if missing_held:
        raise DataContractError(f"held names disappeared from panel: {missing_held[:5]}")
    union = current.index.union(desired.index)
    previous = current.reindex(union, fill_value=0.0)
    target = desired.reindex(union, fill_value=0.0).clip(lower=0.0)
    if target.sum() > 1 + 1e-8:
        raise DataContractError("target stock weights exceed 100%")
    buyable = rows["buyable"].reindex(union, fill_value=False)
    sellable = rows["sellable"].reindex(union, fill_value=False)
    increases = target > previous
    reductions = target < previous
    target.loc[increases & ~buyable] = previous.loc[increases & ~buyable]
    target.loc[reductions & ~sellable] = previous.loc[reductions & ~sellable]

    increases = target > previous
    fixed_total = float(target.loc[~increases].sum())
    increase_total = float(target.loc[increases].sum())
    if fixed_total + increase_total > 1 + 1e-12:
        room = max(0.0, 1.0 - fixed_total)
        if increase_total:
            target.loc[increases] *= room / increase_total
    delta = target - previous
    return target[target > 1e-14], float(delta.clip(lower=0).sum()), float((-delta.clip(upper=0)).sum())


def _fund_trades(
    target: pd.Series,
    current: pd.Series,
    costs: TradingCosts,
) -> tuple[pd.Series, float, float, float, float]:
    """Scale purchases so commissions/slippage never create negative cash."""

    union = target.index.union(current.index)
    previous = current.reindex(union, fill_value=0.0)
    funded = target.reindex(union, fill_value=0.0)
    delta = funded - previous
    sells = float((-delta.clip(upper=0)).sum())
    buys = float(delta.clip(lower=0).sum())
    cash_before = 1.0 - float(previous.sum())
    if cash_before < -1e-10:
        raise DataContractError("previous portfolio has negative cash or leverage")
    affordable_buys = max(0.0, cash_before + sells * (1 - costs.sell_rate)) / (1 + costs.buy_rate)
    if buys > affordable_buys + 1e-14:
        increase = delta > 0
        scale = affordable_buys / buys if buys else 0.0
        funded.loc[increase] = previous.loc[increase] + delta.loc[increase] * scale
        delta = funded - previous
        buys = float(delta.clip(lower=0).sum())
        sells = float((-delta.clip(upper=0)).sum())
    trading_cost = buys * costs.buy_rate + sells * costs.sell_rate
    cash_after = cash_before + sells - buys - trading_cost
    if cash_after < -1e-10:
        raise DataContractError("execution costs created negative cash")
    return funded[funded > 1e-14], buys, sells, trading_cost, max(0.0, cash_after)


def simulate_portfolio(
    predictions: pd.DataFrame,
    target_builder: Callable[[pd.DataFrame, pd.Series], pd.Series],
    *,
    costs: TradingCosts = TradingCosts(),
) -> pd.DataFrame:
    """Execute daily next-open targets with suspension/limit and cost accounting."""

    _require_columns(
        predictions,
        ["signal_date", "code", "next_open_return", "buyable", "sellable", "benchmark_return", "fold"],
        "portfolio predictions",
    )
    current = pd.Series(dtype="float64")
    records = []
    for date, group in predictions.sort_values(["signal_date", "code"]).groupby("signal_date", sort=True):
        desired = target_builder(group, current)
        weights, _buys, _sells = _execute_target(group, current, desired)
        weights, buys, sells, trading_cost, cash_after_trade = _fund_trades(weights, current, costs)
        returns = group.set_index("code")["next_open_return"].reindex(weights.index)
        if returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
            raise DataContractError("held positions have missing next-open returns")
        gross_return = float(weights @ returns)
        benchmark = float(group["benchmark_return"].iloc[0])
        end_values = weights * (1 + returns)
        nav = float(end_values.sum() + cash_after_trade)
        if nav <= 0:
            raise DataContractError("portfolio NAV became non-positive")
        net_return = nav - 1.0
        current = (end_values / nav).replace([np.inf, -np.inf], np.nan).dropna()
        ending_cash_weight = float(cash_after_trade / nav)
        if current.sum() > 1 + 1e-8 or ending_cash_weight < -1e-12:
            raise DataContractError("post-return portfolio accounting created leverage")
        records.append({
            "signal_date": pd.Timestamp(date),
            "fold": int(group["fold"].iloc[0]),
            "gross_return": gross_return,
            "net_return": net_return,
            "benchmark_return": benchmark,
            "excess_vs_benchmark": net_return - benchmark,
            "buy_notional": buys,
            "sell_notional": sells,
            "one_way_turnover": (buys + sells) / 2,
            "trading_cost": trading_cost,
            "holding_count": int(len(weights)),
            "cash_weight": float(cash_after_trade),
            "ending_cash_weight": ending_cash_weight,
        })
    if not records:
        raise DataContractError("portfolio simulation produced no dates")
    return pd.DataFrame(records)


def _return_stats(values: pd.Series) -> dict | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    wealth = pd.concat([
        pd.Series([1.0], dtype="float64"),
        (1 + clean).cumprod().reset_index(drop=True),
    ], ignore_index=True)
    std = clean.std()
    cagr = float(wealth.iloc[-1] ** (252 / len(clean)) - 1) if wealth.iloc[-1] > 0 else None
    return {
        "count": int(len(clean)),
        "mean_daily": float(clean.mean()),
        "cagr": cagr,
        "annualized_volatility": float(std * np.sqrt(252)) if pd.notna(std) else None,
        "sharpe_zero_rate": float(clean.mean() / std * np.sqrt(252)) if pd.notna(std) and std else None,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
        "positive_rate": float((clean > 0).mean()),
    }


def factor_diagnostics(predictions: pd.DataFrame, min_names: int = 30) -> dict | None:
    rows = []
    eligible = predictions[predictions["universe_member"]]
    for date, group in eligible.dropna(subset=["score", "next_open_return"]).groupby("signal_date"):
        if len(group) < min_names or group["score"].nunique() <= 1:
            continue
        ordered = group.copy()
        ordered["decile"] = pd.qcut(ordered["score"].rank(method="first"), 10, labels=False)
        means = ordered.groupby("decile")["next_open_return"].mean()
        rows.append({
            "signal_date": pd.Timestamp(date),
            "fold": int(group["fold"].iloc[0]),
            "ic": float(group["score"].corr(group["next_open_return"], method="spearman")),
            "decile_spread": float(means.loc[9] - means.loc[0]),
        })
    if not rows:
        return None
    daily = pd.DataFrame(rows)
    by_fold = daily.groupby("fold").agg(ic=("ic", "mean"), decile_spread=("decile_spread", "mean"))
    return {
        "date_count": int(len(daily)),
        "mean_ic": float(daily["ic"].mean()),
        "icir": float(daily["ic"].mean() / daily["ic"].std()) if daily["ic"].std() else None,
        "ic_positive_rate": float((daily["ic"] > 0).mean()),
        "mean_decile_spread": float(daily["decile_spread"].mean()),
        "positive_fold_rate": float(((by_fold["ic"] > 0) & (by_fold["decile_spread"] > 0)).mean()),
        "folds": {
            str(int(index)): {"mean_ic": float(row["ic"]), "mean_decile_spread": float(row["decile_spread"])}
            for index, row in by_fold.iterrows()
        },
    }


def summarize_baseline(
    predictions: pd.DataFrame,
    portfolio: pd.DataFrame,
    universe_portfolio: pd.DataFrame,
    *,
    frozen_start: str,
    min_frozen_days: int = 126,
    min_portfolio_cagr: float = 0.05,
    tracking_error_limit: float | None = None,
) -> dict:
    """Apply one acceptance gate to all three baseline families."""

    start = pd.Timestamp(frozen_start)
    pred = predictions[predictions["signal_date"] >= start]
    port = portfolio[portfolio["signal_date"] >= start].copy()
    universe = universe_portfolio[universe_portfolio["signal_date"] >= start][["signal_date", "net_return"]]
    port = port.merge(universe, on="signal_date", how="inner", suffixes=("", "_universe"), validate="one_to_one")
    if port.empty:
        return {"status": "INSUFFICIENT_DATA", "reason": "no frozen OOS overlap"}
    port["excess_vs_universe"] = port["net_return"] - port["net_return_universe"]
    diag = factor_diagnostics(pred)
    active = port["excess_vs_benchmark"]
    active_std = active.std()
    tracking_error = float(active_std * np.sqrt(252)) if pd.notna(active_std) else None
    fold_excess = port.groupby("fold")["excess_vs_benchmark"].mean()
    annualized_excess_benchmark = float(active.mean() * 252)
    annualized_excess_universe = float(port["excess_vs_universe"].mean() * 252)
    portfolio_stats = _return_stats(port["net_return"])
    gates = {
        "enough_frozen_days": int(len(port)) >= min_frozen_days,
        "portfolio_cagr_at_least_target": bool(
            portfolio_stats["cagr"] is not None
            and portfolio_stats["cagr"] >= min_portfolio_cagr
        ),
        "positive_ic": bool(diag and diag["mean_ic"] > 0),
        "positive_decile_spread": bool(diag and diag["mean_decile_spread"] > 0),
        "positive_excess_vs_csi500": annualized_excess_benchmark > 0,
        "positive_excess_vs_tradable_universe": annualized_excess_universe > 0,
        "fold_stability": bool(len(fold_excess) and (fold_excess > 0).mean() >= 0.60),
    }
    if tracking_error_limit is not None:
        gates["tracking_error_within_limit"] = bool(
            tracking_error is not None and tracking_error <= tracking_error_limit
        )
    return {
        "status": "PASS" if all(gates.values()) else "REJECTED",
        "frozen_start": str(start.date()),
        "factor": diag,
        "minimum_portfolio_cagr": min_portfolio_cagr,
        "portfolio": portfolio_stats,
        "csi500": _return_stats(port["benchmark_return"]),
        "tradable_equal_weight_universe": _return_stats(port["net_return_universe"]),
        "excess_vs_csi500": _return_stats(port["excess_vs_benchmark"]),
        "excess_vs_tradable_universe": _return_stats(port["excess_vs_universe"]),
        "annualized_excess_vs_csi500": annualized_excess_benchmark,
        "annualized_excess_vs_tradable_universe": annualized_excess_universe,
        "realized_tracking_error": tracking_error,
        "information_ratio": float(active.mean() / active_std * np.sqrt(252)) if pd.notna(active_std) and active_std else None,
        "positive_fold_rate": float((fold_excess > 0).mean()) if len(fold_excess) else None,
        "average_one_way_turnover": float(port["one_way_turnover"].mean()),
        "total_trading_cost": float(port["trading_cost"].sum()),
        "acceptance": gates,
    }
