#!/usr/bin/env python3
"""One-shot LOCKBOX evaluation. Criteria: docs/lockbox_go_no_go_v1.md (frozen).

Refuses to run without --unlock-lockbox.  Verifies every pinned artifact
sha256 before touching lockbox data, evaluates the single registered
candidate (lgbm_v2_risk) against the pre-frozen GO conditions, and writes
lockbox_one_shot_report.json.  The lockbox may be opened exactly once; this
script performs the whole procedure so nothing else ever has to read the file.

GO (all three, frozen in the doc — do not edit after viewing results):
  1. mean daily Spearman IC (score vs forward_drawdown_20d) > 0
  2. share of positive-IC days >= 0.60
  3. worst-decile enrichment < 0 (bottom-scored 10% has worse drawdowns)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.alpha158 import QLIB_ALPHA158_FEATURES  # noqa: E402
from analysis.oos_baselines import DataContractError, validate_pit_panel  # noqa: E402
from scripts.split_development_lockbox import LOCKBOX_START  # noqa: E402

TARGET = "forward_drawdown_20d"
BAD_TAIL_THRESHOLD = -0.15
EXCLUDE_FRACTION = 0.10
BOOTSTRAP_BLOCK = 5
BOOTSTRAP_DRAWS = 2000
SEED = 20260612  # lockbox start date; fixed forever

# Pinned in docs/lockbox_go_no_go_v1.md — any mismatch aborts the run.
PINNED_SHA256 = {
    "lockbox_panel.parquet": "6828a07fd86a4f186ab1eb2686f762f5f02e03d510e97553e6779b4b0a16918a",
    "development_panel.parquet": "631afc81122f73ecdb764cefa8990d9c58b0ae0aad968a164ab8b6207c58404d",
    "training_panel_v2.parquet": "b9bbda1549d2e699f92bd0867741e7242f1a719270488fcce7e61d7f1b2ece72",
    "models/lgbm_v2_risk.txt": "c2be0acc0bac38d55062aba404999a8c6af8b501c4e439871600366ff6a77a2e",
    "models/lgbm_v2_risk_meta.json": "832ad2a940f8595f7279d6baf119b7e8f055f17d63ebe99db4ca2e5527ae262d",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_scores(panel: pd.DataFrame, features: list[str], booster: lgb.Booster) -> pd.DataFrame:
    """Training-parity per-day member rank normalization, then booster scores."""
    member = panel["universe_member"].to_numpy()
    ranked = (
        panel.loc[member].groupby("signal_date", sort=False)[features].rank(pct=True) - 0.5
    ).astype(np.float32)
    normalized = pd.DataFrame(
        np.zeros((len(panel), len(features)), dtype=np.float32),
        index=panel.index, columns=features,
    )
    normalized.loc[member] = ranked.to_numpy()
    normalized = normalized.fillna(0.0)
    out = panel[["signal_date", "code", "universe_member", TARGET]].copy()
    out["score"] = np.nan
    out.loc[member, "score"] = booster.predict(normalized.loc[member].to_numpy())
    return out


def _daily_ic(frame: pd.DataFrame) -> pd.Series:
    def one(group: pd.DataFrame) -> float:
        if len(group) <= 2:
            return np.nan
        return group["score"].corr(group[TARGET], method="spearman")

    return frame.groupby("signal_date").apply(one).dropna().sort_index()


def _block_bootstrap_ci(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(values)
    blocks = max(1, n // BOOTSTRAP_BLOCK)
    means = []
    for _ in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, max(1, n - BOOTSTRAP_BLOCK + 1), size=blocks)
        sample = np.concatenate([values[s:s + BOOTSTRAP_BLOCK] for s in starts])[:n]
        means.append(float(sample.mean()))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _psi(dev_scores: np.ndarray, lockbox_scores: np.ndarray) -> float:
    edges = np.quantile(dev_scores, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf
    dev_pct = np.histogram(dev_scores, bins=edges)[0] / len(dev_scores)
    lb_pct = np.histogram(lockbox_scores, bins=edges)[0] / len(lockbox_scores)
    dev_pct = np.clip(dev_pct, 1e-6, None)
    lb_pct = np.clip(lb_pct, 1e-6, None)
    return float(np.sum((lb_pct - dev_pct) * np.log(lb_pct / dev_pct)))


def main() -> None:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unlock-lockbox", action="store_true")
    parser.add_argument("--output", default=str(root / "lockbox_one_shot_report.json"))
    args = parser.parse_args()
    if not args.unlock_lockbox:
        raise SystemExit(
            "LOCKBOX is sealed. This script may run exactly once, with --unlock-lockbox, "
            "after docs/lockbox_go_no_go_v1.md is committed. See that document first."
        )

    paths = {
        "lockbox_panel.parquet": root / "lockbox_panel.parquet",
        "development_panel.parquet": root / "development_panel.parquet",
        "training_panel_v2.parquet": root / "training_panel_v2.parquet",
        "models/lgbm_v2_risk.txt": ROOT / "models" / "lgbm_v2_risk.txt",
        "models/lgbm_v2_risk_meta.json": ROOT / "models" / "lgbm_v2_risk_meta.json",
    }
    receipts = {}
    for name, path in paths.items():
        actual = _sha256(path)
        if actual != PINNED_SHA256[name]:
            raise DataContractError(f"sha256 mismatch for {name}: {actual}")
        receipts[name] = actual
    print("artifact pins verified", flush=True)

    meta = json.loads(paths["models/lgbm_v2_risk_meta.json"].read_text())
    features = list(meta["features"])
    booster = lgb.Booster(model_file=str(paths["models/lgbm_v2_risk.txt"]))

    columns = ["signal_date", "code", "universe_member", TARGET, *features]
    lockbox = pd.read_parquet(paths["lockbox_panel.parquet"], columns=columns)
    lockbox["signal_date"] = pd.to_datetime(lockbox["signal_date"])
    if (lockbox["signal_date"] < pd.Timestamp(LOCKBOX_START)).any():
        raise DataContractError("lockbox panel contains development rows")
    lockbox = validate_pit_panel(lockbox, QLIB_ALPHA158_FEATURES)

    scored = _normalized_scores(lockbox, features, booster)
    labeled = scored[scored["universe_member"] & scored[TARGET].notna() & scored["score"].notna()]
    ic = _daily_ic(labeled)
    ic_values = ic.to_numpy(dtype=float)
    mean_ic = float(ic_values.mean())
    positive_rate = float((ic_values > 0).mean())
    icir = float(mean_ic / ic_values.std(ddof=1)) if len(ic_values) > 1 and ic_values.std(ddof=1) > 0 else None

    def bottom_mask(group: pd.DataFrame) -> pd.Series:
        cutoff = group["score"].quantile(EXCLUDE_FRACTION)
        return group["score"] <= cutoff

    bottom = labeled.groupby("signal_date", group_keys=False).apply(bottom_mask)
    expected_bottom = float(labeled.loc[bottom.to_numpy(), TARGET].mean())
    expected_universe = float(labeled[TARGET].mean())
    enrichment = expected_bottom - expected_universe
    bad_tail_bottom = float((labeled.loc[bottom.to_numpy(), TARGET] <= BAD_TAIL_THRESHOLD).mean())
    bad_tail_base = float((labeled[TARGET] <= BAD_TAIL_THRESHOLD).mean())
    bad_tail_lift = float(bad_tail_bottom / bad_tail_base) if bad_tail_base > 0 else None

    # Secondary (report-only): drift of the score distribution vs development.
    dev_columns = ["signal_date", "code", "universe_member", TARGET, *features]
    development = pd.read_parquet(paths["development_panel.parquet"], columns=dev_columns)
    development["signal_date"] = pd.to_datetime(development["signal_date"])
    dev_scored = _normalized_scores(development, features, booster)
    psi = _psi(
        dev_scored.loc[dev_scored["universe_member"], "score"].dropna().to_numpy(),
        scored.loc[scored["universe_member"], "score"].dropna().to_numpy(),
    )

    from scipy import stats  # local import; report-only statistic
    t_stat, p_two = stats.ttest_1samp(ic_values, 0.0)
    p_one_sided = float(p_two / 2) if t_stat > 0 else float(1 - p_two / 2)

    go_conditions = {
        "mean_ic_positive": mean_ic > 0,
        "positive_rate_at_least_060": positive_rate >= 0.60,
        "worst_decile_enrichment_negative": enrichment < 0,
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "criteria_document": "docs/lockbox_go_no_go_v1.md",
        "artifact_sha256_receipts": receipts,
        "candidate": "lgbm_v2_risk",
        "lockbox_rows": int(len(lockbox)),
        "labeled_member_rows": int(len(labeled)),
        "labeled_days": int(len(ic_values)),
        "label_coverage_note": "labels need t+21 open; unlabeled tail is expected",
        "primary": {
            "mean_daily_spearman_ic": mean_ic,
            "positive_ic_day_rate": positive_rate,
            "worst_decile_enrichment": enrichment,
            "expected_drawdown_bottom_decile": expected_bottom,
            "expected_drawdown_universe": expected_universe,
        },
        "go_conditions": go_conditions,
        "decision": "GO" if all(go_conditions.values()) else "NO_GO",
        "secondary_report_only": {
            "icir": icir,
            "bad_tail_lift": bad_tail_lift,
            "bad_tail_precision_bottom_decile": bad_tail_bottom,
            "bad_tail_base_rate": bad_tail_base,
            "one_sided_t_p_value": p_one_sided,
            "block_bootstrap_ci95": _block_bootstrap_ci(ic_values),
            "score_psi_vs_development": psi,
            "daily_ic_series": {str(k.date()): float(v) for k, v in ic.items()},
        },
    }
    Path(args.output).expanduser().write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], **report["primary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
