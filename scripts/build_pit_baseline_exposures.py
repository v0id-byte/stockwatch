#!/usr/bin/env python3
"""Build strict PIT sector and trailing-EPS events for frozen OOS baselines.

Current-component sectors and today's restated fundamental snapshots are
reported as BLOCKED and are never relabeled as historical observations.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


EXTRACTOR_VERSION = "pit_baseline_exposures_v1"


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Build strict PIT baseline exposures.")
    parser.add_argument("--sector-input", default=str(root / "sector_map_sw.parquet"))
    parser.add_argument("--fundamental-input", default=str(root / "fundamental_features.parquet"))
    parser.add_argument("--sector-output", default=str(root / "sector_pit.parquet"))
    parser.add_argument("--fundamental-output", default=str(root / "fundamental_pit.parquet"))
    parser.add_argument("--report", default=str(root / "pit_baseline_exposures_report.json"))
    return parser.parse_args()


def _normalize_code(values: pd.Series) -> pd.Series:
    extracted = values.astype(str).str.extract(r"(?<!\d)(\d{6})(?!\d)", expand=False)
    return extracted


def build_sector_events(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"code", "sector", "sector_kind"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sector source missing columns: {sorted(missing)}")
    kinds = set(frame["sector_kind"].dropna().astype(str))
    if kinds != {"sw_historical_effective"}:
        raise ValueError(
            "sector source is not historical PIT; current component maps cannot be backfilled"
        )
    if "start_date" not in frame:
        raise ValueError("historical sector source is missing start_date")
    if "industry_code" not in frame:
        frame = frame.assign(industry_code=None)
    out = frame[["code", "sector", "start_date", "industry_code", "sector_kind"]].copy()
    out["code"] = _normalize_code(out["code"])
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce").dt.normalize()
    if out[["code", "sector", "start_date"]].isna().any().any():
        raise ValueError("sector source contains invalid PIT keys")
    out["available_at"] = out["start_date"] + pd.Timedelta(hours=15, seconds=1)
    out["extractor_version"] = EXTRACTOR_VERSION
    out = out.drop_duplicates(["code", "available_at"], keep="last")
    return out.sort_values(["available_at", "code"]).reset_index(drop=True)


def _period_parts(value: object) -> tuple[int, str] | None:
    match = re.fullmatch(r"((?:19|20)\d{2})(0331|0630|0930|1231)", str(value))
    return (int(match.group(1)), match.group(2)) if match else None


def build_trailing_eps_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert vintage-verified cumulative YTD EPS reports to PIT TTM EPS."""
    required = {
        "code", "available_at", "report_period", "eps", "vintage_verified",
        "source_row_sha256", "extraction_version",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fundamental source missing columns: {sorted(missing)}")
    verified = frame[frame["vintage_verified"].eq(True)].copy()
    if verified.empty:
        raise ValueError(
            "no vintage-verified fundamentals; current vendor snapshots cannot be used as PIT"
        )
    verified["code"] = _normalize_code(verified["code"])
    verified["available_at"] = pd.to_datetime(verified["available_at"], errors="coerce")
    verified["eps"] = pd.to_numeric(verified["eps"], errors="coerce")
    verified["_period"] = verified["report_period"].map(_period_parts)
    verified = verified.dropna(subset=["code", "available_at", "eps", "_period"])
    if verified.empty:
        raise ValueError("vintage-verified fundamentals contain no valid EPS rows")
    verified = verified.sort_values(["code", "available_at", "report_period"], kind="stable")

    rows: list[dict] = []
    for code, group in verified.groupby("code", sort=False):
        known: dict[str, tuple[float, pd.Timestamp]] = {}
        for _, row in group.iterrows():
            year, suffix = row["_period"]
            period = f"{year}{suffix}"
            eps = float(row["eps"])
            available_at = row["available_at"]
            trailing_eps = None
            lineage = [period]
            if suffix == "1231":
                trailing_eps = eps
            else:
                prior_annual = f"{year - 1}1231"
                prior_same = f"{year - 1}{suffix}"
                if prior_annual in known and prior_same in known:
                    trailing_eps = eps + known[prior_annual][0] - known[prior_same][0]
                    lineage.extend([prior_annual, prior_same])
            known[period] = (eps, available_at)
            if trailing_eps is None:
                continue
            item = row.drop(labels=["_period"]).to_dict()
            item.update({
                "code": code,
                "trailing_eps": float(trailing_eps),
                "ttm_lineage": ",".join(lineage),
                "ttm_extractor_version": EXTRACTOR_VERSION,
            })
            rows.append(item)
    if not rows:
        raise ValueError("verified EPS history is insufficient to construct any TTM observations")
    out = pd.DataFrame(rows)
    if out.duplicated(["code", "available_at"]).any():
        raise ValueError("fundamental vintages contain ambiguous code/available_at rows")
    return out.sort_values(["available_at", "code"]).reset_index(drop=True)


def _component(source: Path, output: Path, builder) -> dict:
    if not source.exists():
        return {"status": "BLOCKED", "reason": f"missing source: {source}"}
    try:
        result = builder(pd.read_parquet(source))
    except Exception as exc:
        output.unlink(missing_ok=True)
        return {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return {
        "status": "READY",
        "output": str(output),
        "rows": int(len(result)),
        "codes": int(result["code"].nunique()),
        "available_start": str(result["available_at"].min()),
        "available_end": str(result["available_at"].max()),
    }


def main() -> None:
    args = _parse_args()
    sector = _component(
        Path(args.sector_input).expanduser(),
        Path(args.sector_output).expanduser(),
        build_sector_events,
    )
    fundamental = _component(
        Path(args.fundamental_input).expanduser(),
        Path(args.fundamental_output).expanduser(),
        build_trailing_eps_events,
    )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "READY" if sector["status"] == fundamental["status"] == "READY" else "BLOCKED",
        "sector": sector,
        "fundamental": fundamental,
        "fail_closed": True,
        "notes": [
            "Static current sector membership is never used as historical classification.",
            "Only vintage_verified=True reports may generate TTM EPS.",
            "Earnings-to-price must be calculated later with each signal day's raw close.",
        ],
    }
    path = Path(args.report).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "READY":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
