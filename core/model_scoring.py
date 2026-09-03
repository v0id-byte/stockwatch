"""Nightly batch model scoring over the CSI500 reference universe.

Production contract (offline/online parity by construction):

    raw features -> live CSI500 reference universe (full pool)
      -> cross_sectional_rank_pct_centered (inside LgbmRanker.predict_batch)
      -> LightGBM -> full-pool scores persisted to ``model_scores``
      -> runner/bot LOOK UP scores; they never compute model features inline.

Parity is guaranteed by reusing the exact training data path: the bootstrap
download (paired raw/HFQ schema-v2 bars) and ``derive_stock_market_fields`` /
``compute_alpha158_frame`` — never the app's qfq quote endpoints, whose units
and adjustment basis differ from training.

``score_trade_date`` takes the universe snapshot as an explicit argument so a
historical replay passes that day's PIT membership while production passes
the current official constituents.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402
from core.clock import market_now, market_today  # noqa: E402

HISTORY_DAYS = 560  # 250-day windows + warmup + holidays margin


def _universe_sha(codes: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(codes)).encode("ascii")).hexdigest()


def fetch_reference_universe() -> list[str]:
    """Current official CSI500 constituents (matches the training universe)."""
    import os

    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    from scripts.bootstrap_history import _index_constituents

    import akshare as ak

    codes, source = _index_constituents(ak, "000905")
    if len(codes) != 500:
        raise RuntimeError(f"reference universe has {len(codes)} members from {source}, expected 500")
    return sorted(codes)


def _download_history(codes: list[str], *, end: str | None = None) -> dict[str, "pd.DataFrame"]:
    import akshare as ak
    import pandas as pd

    from scripts.bootstrap_history import _download_one

    end = end or market_today().strftime("%Y%m%d")
    start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=int(HISTORY_DAYS * 1.6))).strftime("%Y%m%d")
    out: dict[str, pd.DataFrame] = {}
    for index, code in enumerate(codes):
        frame, source = _download_one(ak, pd, code, start, end, max_retries=2)
        if frame is None:
            logger.warning(f"scoring: history unavailable for {code}; it will score as missing")
            continue
        out[code] = frame
        if (index + 1) % 100 == 0:
            logger.info(f"scoring history {index + 1}/{len(codes)}")
    return out


def _download_benchmarks(end: str | None = None) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """(csi500 benchmark frame for derive, csi300 market frame for custom factors)."""
    import pandas as pd

    from scripts.bootstrap_history import _download_index_baostock
    from scripts.build_frozen_oos_panel import _load_benchmark

    end = end or market_today().strftime("%Y%m%d")
    start = (datetime.strptime(end, "%Y%m%d") - timedelta(days=int(HISTORY_DAYS * 1.6))).strftime("%Y%m%d")
    import tempfile

    frames = {}
    for index_code in ("sh000905", "sh000300"):
        raw = None
        try:
            import akshare as ak

            symbol = {"sh000905": "csi000905", "sh000300": "sh000300"}[index_code]
            raw = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start, end_date=end)
        except Exception:
            raw = None
        if raw is None or len(raw) == 0:
            raw = _download_index_baostock(pd, index_code, start, end)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"scoring: benchmark download failed for {index_code}")
        frame = raw.rename(columns={"日期": "trade_date", "date": "trade_date",
                                    "开盘": "open", "收盘": "close",
                                    "最高": "high", "最低": "low",
                                    "成交量": "volume", "成交额": "amount"})
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frames[index_code] = frame.sort_values("trade_date").reset_index(drop=True)

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
        bench = frames["sh000905"].copy()
        bench["raw_open"] = bench["open"]
        bench["raw_close"] = bench["close"]
        bench.to_parquet(handle.name, index=False)
        benchmark = _load_benchmark(Path(handle.name))
    return benchmark, frames["sh000300"]


def compute_deploy_features(
    klines: dict[str, "pd.DataFrame"],
    benchmark: "pd.DataFrame",
    market_300: "pd.DataFrame",
    trade_date: "pd.Timestamp",
) -> dict[str, dict[str, float]]:
    """Exact training-path features for one signal date, keyed by code."""
    import pandas as pd

    from analysis.factors import compute_alpha158_frame
    from scripts.build_frozen_oos_panel import derive_stock_market_fields
    from scripts.build_training_panel_v2 import CUSTOM_FEATURES
    from scripts.build_training_set import _factor_input
    from analysis.alpha158 import QLIB_ALPHA158_FEATURES

    features_by_code: dict[str, dict[str, float]] = {}
    for code, bars in klines.items():
        try:
            derived = derive_stock_market_fields(bars, benchmark, code, keep_unlabeled=True)
            row = derived[pd.to_datetime(derived["signal_date"]) == trade_date]
            if row.empty:
                continue
            values = {name: float(row.iloc[-1][name]) for name in QLIB_ALPHA158_FEATURES
                      if pd.notna(row.iloc[-1][name])}
            custom = compute_alpha158_frame(_factor_input(bars, code), market_300)
            custom["trade_date"] = pd.to_datetime(custom["trade_date"])
            crow = custom[custom["trade_date"] == trade_date]
            if not crow.empty:
                for name in CUSTOM_FEATURES:
                    value = crow.iloc[-1].get(name)
                    if pd.notna(value):
                        values[name] = float(value)
            features_by_code[code] = values
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"scoring: feature build failed for {code}: {exc}")
    return features_by_code


def score_trade_date(
    trade_date,
    universe_snapshot: list[str],
    *,
    klines: dict[str, "pd.DataFrame"],
    benchmark: "pd.DataFrame",
    market_300: "pd.DataFrame",
    alpha_model_path: Path | None,
    risk_model_path: Path | None,
) -> list[dict]:
    """Pure scoring for one date over an explicit universe snapshot."""
    import pandas as pd

    from analysis.lgbm import LgbmRanker

    trade_date = pd.Timestamp(trade_date).normalize()
    scoped = {c: klines[c] for c in universe_snapshot if c in klines}
    features = compute_deploy_features(scoped, benchmark, market_300, trade_date)
    if len(features) < len(universe_snapshot) * 0.9:
        raise RuntimeError(
            f"scoring coverage {len(features)}/{len(universe_snapshot)} below 90%; refusing to rank"
        )
    rows: list[dict] = []
    scored_at = market_now().isoformat(timespec="seconds")
    universe_sha = _universe_sha(universe_snapshot)

    def _apply(model_path: Path | None, kind: str) -> tuple[dict, str | None, str | None]:
        if not model_path or not Path(model_path).exists():
            return {}, None, None
        ranker = LgbmRanker(Path(model_path))
        if ranker.model is None:
            logger.warning(f"scoring: {kind} model unavailable: {ranker.disabled_context}")
            return {}, None, None
        scores = ranker.predict_batch(features)
        version = f"{Path(model_path).stem}:{ranker.meta.get('trained_at', 'unknown')}"
        contract = ranker.meta.get("feature_contract_version")
        return scores, version, contract

    alpha_scores, alpha_version, contract = _apply(alpha_model_path, "alpha")
    risk_scores, risk_version, risk_contract = _apply(risk_model_path, "risk")
    contract = contract or risk_contract
    for code in universe_snapshot:
        rows.append({
            "trade_date": str(trade_date.date()),
            "code": code,
            "alpha_model_version": alpha_version,
            "alpha_score": alpha_scores.get(code),
            "risk_model_version": risk_version,
            "risk_score": risk_scores.get(code),
            "feature_contract_version": contract,
            "reference_universe_sha256": universe_sha,
            "scored_at": scored_at,
        })
    return rows


def run_nightly_scoring(storage, cfg) -> dict:
    """Fetch universe + history, score the latest completed trade date, persist."""
    import pandas as pd

    universe = fetch_reference_universe()
    benchmark_frame, market_300 = _download_benchmarks()
    klines = _download_history(universe)
    latest = max(pd.to_datetime(frame["trade_date"]).max() for frame in klines.values())
    alpha_path = cfg.resolve_lgbm_model_path(cfg.market_regime) if cfg.enable_lgbm else None
    risk_path = cfg.risk_model_path if cfg.enable_risk_model else None
    rows = score_trade_date(
        latest, universe,
        klines=klines, benchmark=benchmark_frame, market_300=market_300,
        alpha_model_path=alpha_path, risk_model_path=risk_path,
    )
    storage.upsert_model_scores(rows)
    scored = sum(1 for row in rows if row["alpha_score"] is not None or row["risk_score"] is not None)
    logger.info(f"model scoring: {scored}/{len(rows)} scored for {rows[0]['trade_date']}")
    return {"trade_date": rows[0]["trade_date"], "universe": len(universe), "scored": scored}


WORST_DECILE = 0.10


def _worst_decile_codes(rows: dict[str, dict]) -> set[str]:
    """Codes in the riskiest decile of the pool (lowest risk_score = riskiest)."""
    valid = {code: row["risk_score"] for code, row in rows.items()
             if row.get("risk_score") is not None}
    if len(valid) <= 1:
        return set()
    ordered = sorted(valid.items(), key=lambda item: item[1])
    denom = max(1, len(ordered) - 1)
    return {code for rank, (code, _s) in enumerate(ordered) if rank / denom <= WORST_DECILE}


def push_risk_alerts(storage, cfg) -> dict:
    """Warn on watchlist stocks ENTERING the pool's riskiest decile.

    Deduped against the previous scored date so a stock that stays risky
    does not alert every evening ("宁可漏，不要烦").
    """
    dates = storage.get_model_score_dates(limit=2)
    if not dates:
        return {"alerts": 0, "reason": "no scores"}
    latest_rows = storage.get_model_scores_for_date(dates[0])
    worst_now = _worst_decile_codes(latest_rows)
    worst_prev = (
        _worst_decile_codes(storage.get_model_scores_for_date(dates[1]))
        if len(dates) > 1 else set()
    )
    watch = [c for c in cfg.watchlist if c in worst_now and c not in worst_prev]
    if not watch:
        return {"alerts": 0, "trade_date": dates[0]}
    if cfg.notify_channel != "feishu":
        return {"alerts": len(watch), "trade_date": dates[0], "sent": False}

    from analysis.lgbm import format_risk_context
    from push.feishu import FeishuClient, render_text_card

    pool = {code: row.get("risk_score") for code, row in latest_rows.items()}
    contexts = format_risk_context(pool)
    lines = [f"【{code}】{contexts.get(code, '')}" for code in watch]
    lines.append("")
    lines.append(f"依据 {dates[0]} 收盘后全池回撤风险打分；预警 = 新进入全池最高风险 10% 分位。")
    card = render_text_card("⚠️ 回撤风险预警", lines, template="red")
    sent = FeishuClient().send_message(card)
    logger.info(f"risk alerts: {watch} sent={sent}")
    return {"alerts": len(watch), "codes": watch, "trade_date": dates[0], "sent": bool(sent)}
