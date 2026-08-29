#!/usr/bin/env python3
"""Evaluate three distinct, frozen-time A-share OOS baselines.

Input is a strict PIT panel produced by the history/data pipeline.  The script
does not infer missing tradeability, index membership, sectors, or accounting
fields.  It writes research diagnostics only and never changes production
models.
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

import pandas as pd

from analysis.alpha158 import QLIB_ALPHA158_FEATURES
from analysis.oos_baselines import (
    DataContractError,
    EnhancementConstraints,
    TradingCosts,
    assign_size_bucket,
    equal_weight_builder,
    fit_lightgbm_oos,
    index_enhancement_builder,
    make_expanding_folds,
    simulate_portfolio,
    summarize_baseline,
    topk_dropout_builder,
    validate_pit_panel,
)


CH3_STYLE_FEATURES = ["beta", "log_market_cap", "earnings_to_price"]
# Liu-Stambaugh-Yuan's fourth China factor is PMO, formed from abnormal
# turnover.  This is a predictive style baseline, not a literal factor-return
# portfolio replication, so its name deliberately includes "style".
CH4_STYLE_FEATURES = [*CH3_STYLE_FEATURES, "abnormal_turnover_20_240"]
CH4_WITH_ILLIQUIDITY_FEATURES = [*CH4_STYLE_FEATURES, "amihud_20d"]


def _parse_args() -> argparse.Namespace:
    history = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Frozen OOS Alpha158, CH3/CH4, and CSI500 baselines.")
    parser.add_argument("--panel", default=str(history / "oos_baseline_panel.parquet"))
    parser.add_argument("--output", default=str(history / "frozen_oos_baselines_report.json"))
    parser.add_argument("--frozen-start", default="2025-01-01")
    parser.add_argument("--min-train-days", type=int, default=504)
    parser.add_argument("--validation-days", type=int, default=63)
    parser.add_argument("--test-days", type=int, default=63)
    parser.add_argument("--purge-days", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--drop-n", type=int, default=5)
    parser.add_argument("--min-frozen-days", type=int, default=126)
    parser.add_argument("--min-portfolio-cagr", type=float, default=0.05)
    parser.add_argument("--commission-bps", type=float, default=3.0)
    parser.add_argument("--stamp-duty-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--max-tracking-error", type=float, default=0.065)
    parser.add_argument("--max-turnover", type=float, default=0.20)
    parser.add_argument("--max-active-weight", type=float, default=0.01)
    parser.add_argument("--max-stock-weight", type=float, default=0.05)
    parser.add_argument("--num-threads", type=int, default=4)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataContractError(f"strict PIT baseline panel is missing: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _serialize_fold(fold: dict) -> dict:
    return {
        key: str(value.date()) if isinstance(value, pd.Timestamp) else value
        for key, value in fold.items()
    }


def _evaluate_topk(
    predictions: pd.DataFrame,
    *,
    args: argparse.Namespace,
    costs: TradingCosts,
    eligibility: str = "",
) -> dict:
    model_predictions = predictions
    if eligibility:
        diagnostic_predictions = predictions[predictions["size_bucket"] == eligibility]

        def target(group: pd.DataFrame, current: pd.Series) -> pd.Series:
            eligible = group[group["size_bucket"] == eligibility]
            return topk_dropout_builder(args.top_k, args.drop_n)(eligible, current)
    else:
        diagnostic_predictions = predictions
        target = topk_dropout_builder(args.top_k, args.drop_n)
    portfolio = simulate_portfolio(model_predictions, target, costs=costs)
    universe = simulate_portfolio(model_predictions, equal_weight_builder(), costs=costs)
    return summarize_baseline(
        diagnostic_predictions,
        portfolio,
        universe,
        frozen_start=args.frozen_start,
        min_frozen_days=args.min_frozen_days,
        min_portfolio_cagr=args.min_portfolio_cagr,
    )


def _evaluate_index_enhancement(
    predictions: pd.DataFrame,
    *,
    args: argparse.Namespace,
    costs: TradingCosts,
    constraints: EnhancementConstraints,
) -> dict:
    portfolio = simulate_portfolio(
        predictions,
        index_enhancement_builder(constraints),
        costs=costs,
    )
    universe = simulate_portfolio(predictions, equal_weight_builder(), costs=costs)
    return summarize_baseline(
        predictions[predictions["benchmark_weight"] > 0],
        portfolio,
        universe,
        frozen_start=args.frozen_start,
        min_frozen_days=args.min_frozen_days,
        min_portfolio_cagr=args.min_portfolio_cagr,
        tracking_error_limit=args.max_tracking_error,
    )


def main() -> None:
    args = _parse_args()
    panel_path = Path(args.panel).expanduser()
    output_path = Path(args.output).expanduser()
    panel = validate_pit_panel(_read_table(panel_path), QLIB_ALPHA158_FEATURES)
    panel["size_bucket"] = assign_size_bucket(panel, 0.30)
    panel["csi500_residual_return"] = panel["next_open_return"] - panel["benchmark_return"]
    b_panel = None
    b_block_reason = None
    try:
        b_panel = validate_pit_panel(panel, CH4_WITH_ILLIQUIDITY_FEATURES)
    except DataContractError as exc:
        b_block_reason = str(exc)
    csi500_panel = None
    csi500_block_reason = None
    try:
        csi500_panel = validate_pit_panel(
            panel, QLIB_ALPHA158_FEATURES, require_index_columns=True,
        )
    except DataContractError as exc:
        csi500_block_reason = str(exc)
    folds = make_expanding_folds(
        panel["signal_date"],
        min_train_days=args.min_train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        purge_days=args.purge_days,
    )
    if not folds:
        raise DataContractError("not enough history for the frozen expanding folds")

    costs = TradingCosts(
        commission_bps=args.commission_bps,
        stamp_duty_bps=args.stamp_duty_bps,
        slippage_bps=args.slippage_bps,
    )
    constraints = EnhancementConstraints(
        max_tracking_error=args.max_tracking_error,
        max_one_way_turnover=args.max_turnover,
        max_active_weight=args.max_active_weight,
        max_stock_weight=args.max_stock_weight,
    )

    alpha158 = fit_lightgbm_oos(
        panel,
        list(QLIB_ALPHA158_FEATURES),
        "next_open_return",
        folds,
        num_threads=args.num_threads,
    )
    baselines = {
        "A_qlib_alpha158_exact_next_open_topk_dropout": _evaluate_topk(alpha158, args=args, costs=costs),
    }
    if b_panel is None:
        for name in (
            "B1_ch3_large_top_70",
            "B2_ch4_style_pmo_large_top_70",
            "B3_ch4_style_plus_amihud_large_top_70",
            "B4_ch4_style_micro_bottom_30_separate",
        ):
            baselines[name] = {
                "status": "BLOCKED",
                "reason": b_block_reason,
                "fail_closed": True,
            }
    else:
        ch3 = fit_lightgbm_oos(
            b_panel, CH3_STYLE_FEATURES, "next_open_return", folds,
            num_threads=args.num_threads,
        )
        ch4 = fit_lightgbm_oos(
            b_panel, CH4_STYLE_FEATURES, "next_open_return", folds,
            num_threads=args.num_threads,
        )
        ch4_illiquidity = fit_lightgbm_oos(
            b_panel, CH4_WITH_ILLIQUIDITY_FEATURES, "next_open_return", folds,
            num_threads=args.num_threads,
        )
        baselines.update({
            "B1_ch3_large_top_70": _evaluate_topk(
                ch3, args=args, costs=costs, eligibility="large_top_70",
            ),
            "B2_ch4_style_pmo_large_top_70": _evaluate_topk(
                ch4, args=args, costs=costs, eligibility="large_top_70",
            ),
            "B3_ch4_style_plus_amihud_large_top_70": _evaluate_topk(
                ch4_illiquidity, args=args, costs=costs, eligibility="large_top_70",
            ),
            "B4_ch4_style_micro_bottom_30_separate": _evaluate_topk(
                ch4, args=args, costs=costs, eligibility="micro_bottom_30",
            ),
        })
    if csi500_panel is None:
        baselines["C_csi500_residual_index_enhancement"] = {
            "status": "BLOCKED",
            "reason": csi500_block_reason,
            "fail_closed": True,
            "note": "A/B results remain evaluable; current/static CSI500 weights are not substituted.",
        }
    else:
        residual = fit_lightgbm_oos(
            csi500_panel,
            list(QLIB_ALPHA158_FEATURES),
            "csi500_residual_return",
            folds,
            num_threads=args.num_threads,
        )
        baselines["C_csi500_residual_index_enhancement"] = _evaluate_index_enhancement(
            residual, args=args, costs=costs, constraints=constraints,
        )
    pass_count = sum(item.get("status") == "PASS" for item in baselines.values())
    any_pass = pass_count > 0
    any_blocked = any(item.get("status") == "BLOCKED" for item in baselines.values())
    status = (
        "REQUIRES_PROSPECTIVE_LOCKED_TEST" if any_pass
        else "REJECTED_WITH_BLOCKED_BASELINE" if any_blocked
        else "REJECTED"
    )
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "research_only": True,
        "panel": str(panel_path),
        "data_contract": {
            "label": "signal after close; execute next open; exit following open",
            "feature_timing": "feature_available_at < execution_at",
            "tradeability": "unknown suspension/limit flags fail closed",
            "universe": "point-in-time membership only",
            "small_cap_treatment": "daily bottom 30% market cap excluded from B1/B2/B3 and reported separately in B4",
        },
        "folds": [_serialize_fold(fold) for fold in folds],
        "costs_bps": {
            "commission": costs.commission_bps,
            "stamp_duty_sell_only": costs.stamp_duty_bps,
            "slippage_each_side": costs.slippage_bps,
        },
        "index_constraints": {
            "max_ex_ante_diagonal_te_proxy": constraints.max_tracking_error,
            "te_proxy_note": "diagonal stock volatility only; not a full covariance risk model",
            "max_one_way_turnover": constraints.max_one_way_turnover,
            "max_active_weight": constraints.max_active_weight,
            "max_stock_weight": constraints.max_stock_weight,
            "industry_size_beta_active_tolerance": constraints.exposure_tolerance,
        },
        "features": {
            "A_qlib_alpha158_exact": list(QLIB_ALPHA158_FEATURES),
            "B_ch3_style": CH3_STYLE_FEATURES,
            "B_ch4_style_pmo": CH4_STYLE_FEATURES,
            "B_ch4_style_plus_amihud": CH4_WITH_ILLIQUIDITY_FEATURES,
            "C_residual": list(QLIB_ALPHA158_FEATURES),
        },
        "baselines": baselines,
        "historical_individual_pass_count": pass_count,
        "multiple_testing_policy": (
            "No overall PASS is inferred from the best of several baselines. A candidate must be frozen "
            "before a new prospective paper-trading period; otherwise data-mining/FDR remains unresolved."
        ),
        "acceptance_gate": (
            "frozen days >= threshold; positive next-open IC and decile spread; positive net annualized "
            "portfolio CAGR >= configured target (default 5%); positive net annualized excess versus "
            "CSI500 and same-panel tradable equal-weight universe; >=60% positive folds; "
            "CSI500 enhancement also respects realized tracking-error limit"
        ),
        "deployment": "No baseline is production-enabled by this report.",
        "limitations": [
            "A/B readiness is independent from C; absent historical CSI500 weights block only C.",
            "A current constituent or current-weight snapshot is never backfilled into historical dates.",
            "The frozen start is retrospective OOS already visible to researchers, not a new live paper-trading period.",
            "The ex-ante optimizer limit is a diagonal-volatility TE proxy; realized tracking error is reported separately.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"frozen OOS baseline report saved: {output_path}")
    print(f"status={status}")


if __name__ == "__main__":
    main()
