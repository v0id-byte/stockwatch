#!/usr/bin/env python3
"""Build market-cap and sector exposures for neutralized diagnostics."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily market-cap and sector exposure parquet files.")
    parser.add_argument("--root", default=os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history"))
    parser.add_argument("--training-set", default="", help="Defaults to <root>/training_set.parquet.")
    parser.add_argument("--max-codes", type=int, default=0, help="0 means all training-set codes.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Polite delay between share-structure fetches.")
    parser.add_argument("--refresh", action="store_true", help="Refetch cached raw share-structure files.")
    parser.add_argument("--min-pit-coverage", type=float, default=0.95,
                        help="Fail the exposure gate below this date/code market-cap coverage.")
    parser.add_argument("--skip-market-cap", action="store_true")
    parser.add_argument("--sector-mode", choices=["auto", "historical", "current", "none"], default="auto")
    parser.add_argument("--output-market-cap", default="", help="Defaults to <root>/market_cap_daily.parquet.")
    parser.add_argument("--output-sector", default="", help="Defaults to <root>/sector_map_sw.parquet.")
    return parser.parse_args()


def _normalize_code(values: pd.Series) -> pd.Series:
    return values.astype(str).str.extract(r"(\d{6})", expand=False).fillna("").str.zfill(6)


def _ak_symbol(code: str) -> str:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("4", "8")):
        return f"{code}.BJ"
    return f"{code}.SZ"


def _load_training_scope(path: Path, max_codes: int) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    data = pd.read_parquet(path, columns=["trade_date", "code"])
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    codes = sorted(data["code"].astype(str).str.zfill(6).unique().tolist())
    if max_codes:
        codes = codes[:max_codes]
    return codes, data["trade_date"].min(), data["trade_date"].max()


def _clean_share_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["share_date", "total_share", "float_a_share"])
    total_col = "总股本"
    float_col = "已上市流通A股" if "已上市流通A股" in raw.columns else "已流通股份"
    if "变更日期" not in raw.columns or total_col not in raw.columns or float_col not in raw.columns:
        return pd.DataFrame(columns=["share_date", "total_share", "float_a_share"])
    out = raw[["变更日期", total_col, float_col]].copy()
    out.columns = ["share_date", "total_share", "float_a_share"]
    out["share_date"] = pd.to_datetime(out["share_date"], errors="coerce").astype("datetime64[ns]")
    out["total_share"] = pd.to_numeric(out["total_share"], errors="coerce")
    out["float_a_share"] = pd.to_numeric(out["float_a_share"], errors="coerce")
    out = out.dropna(subset=["share_date", "total_share"]).sort_values("share_date")
    out["float_a_share"] = out["float_a_share"].where(out["float_a_share"].notna(), out["total_share"])
    return out.drop_duplicates("share_date", keep="last").reset_index(drop=True)


def _share_cache_path(cache_dir: Path, code: str) -> Path:
    return cache_dir / f"{code}.parquet"


def _load_or_fetch_shares(cache_dir: Path, code: str, refresh: bool) -> tuple[pd.DataFrame, str]:
    path = _share_cache_path(cache_dir, code)
    if path.exists() and not refresh:
        shares = pd.read_parquet(path)
        if "share_date" in shares.columns:
            shares["share_date"] = pd.to_datetime(shares["share_date"], errors="coerce").astype("datetime64[ns]")
        return shares, "cache"
    import akshare as ak

    raw = ak.stock_zh_a_gbjg_em(symbol=_ak_symbol(code))
    shares = _clean_share_history(raw)
    if not shares.empty:
        shares.to_parquet(path, index=False)
    return shares, "fetch"


def _market_cap_for_code(root: Path, cache_dir: Path, code: str, start: pd.Timestamp,
                         end: pd.Timestamp, refresh: bool) -> tuple[pd.DataFrame | None, str]:
    stock_path = root / "stocks" / f"{code}.parquet"
    if not stock_path.exists():
        return None, "missing stock history"
    try:
        shares, source = _load_or_fetch_shares(cache_dir, code, refresh)
    except Exception as exc:
        return None, f"share history error: {type(exc).__name__}"
    if shares.empty:
        return None, "missing total-share history"
    shares = shares.copy()
    shares["share_date"] = pd.to_datetime(
        shares["share_date"], errors="coerce",
    ).astype("datetime64[ns]")
    prices = pd.read_parquet(stock_path, columns=["trade_date", "raw_close"])
    prices = prices.rename(columns={"raw_close": "close"})
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).astype("datetime64[ns]")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices[(prices["trade_date"] >= start) & (prices["trade_date"] <= end)]
    prices = prices.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    if prices.empty:
        return None, "empty price history"
    merged = pd.merge_asof(
        prices,
        shares,
        left_on="trade_date",
        right_on="share_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.dropna(subset=["total_share"])
    if merged.empty:
        return None, "no as-of total-share coverage"
    merged["code"] = code
    merged["market_cap"] = merged["close"] * merged["total_share"]
    merged["float_market_cap"] = merged["close"] * merged["float_a_share"]
    merged["market_cap_source"] = f"{source}:raw_close_x_total_share"
    return merged[[
        "trade_date", "code", "market_cap", "float_market_cap",
        "total_share", "float_a_share", "share_date", "market_cap_source",
    ]], "ok"


def _build_market_cap(root: Path, training_set: Path, output: Path, max_codes: int,
                      sleep_seconds: float, refresh: bool, min_coverage: float) -> dict:
    codes, start, end = _load_training_scope(training_set, max_codes)
    cache_dir = root / "neutralization_exposures" / "share_history"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    skipped = []
    statuses = Counter()
    for idx, code in enumerate(codes, start=1):
        try:
            frame, status = _market_cap_for_code(root, cache_dir, code, start, end, refresh)
            if frame is None:
                skipped.append({"code": code, "reason": status})
            else:
                frames.append(frame)
            statuses[status] += 1
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append({"code": code, "reason": reason})
            statuses[reason] += 1
        if idx % 50 == 0 or idx == len(codes):
            common = ", ".join(f"{key}={value}" for key, value in statuses.most_common(4))
            print(f"market-cap {idx}/{len(codes)} ok={len(frames)} skipped={len(skipped)} {common}", flush=True)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    if not frames:
        raise RuntimeError("no market-cap exposures were built")
    data = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    scope = pd.read_parquet(training_set, columns=["trade_date", "code"])
    scope["trade_date"] = pd.to_datetime(scope["trade_date"]).astype("datetime64[ns]")
    scope["code"] = scope["code"].astype(str).str.zfill(6)
    scope = scope[scope["code"].isin(codes)].drop_duplicates(["trade_date", "code"])
    covered = scope.merge(
        data[["trade_date", "code", "market_cap"]].drop_duplicates(["trade_date", "code"]),
        on=["trade_date", "code"], how="left", validate="one_to_one",
    )["market_cap"].notna()
    coverage = float(covered.mean()) if len(covered) else 0.0
    return {
        "output": str(output),
        "rows": int(len(data)),
        "codes": int(data["code"].nunique()),
        "date_start": str(data["trade_date"].min().date()),
        "date_end": str(data["trade_date"].max().date()),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
        "status_counts": dict(statuses),
        "kind": "point_in_time",
        "price_basis": "unadjusted raw close in CNY/share",
        "share_basis": "total_share effective as of trade_date",
        "coverage": coverage,
        "minimum_coverage": min_coverage,
        "gate": "PASS" if coverage >= min_coverage else "FAIL",
        "note": "market_cap is always raw close * total shares; float market cap is never relabeled as total market cap.",
    }


def _fetch_sw_historical() -> pd.DataFrame:
    import akshare as ak

    try:
        return ak.stock_industry_clf_hist_sw()
    except Exception:
        import requests
        import urllib3

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        url = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
        session = requests.Session()
        session.trust_env = False
        response = session.get(url, timeout=60, verify=False)
        response.raise_for_status()
        return pd.read_excel(io.BytesIO(response.content), dtype={"股票代码": "str", "行业代码": "str"})


def _clean_sw_historical(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["code", "start_date", "sector", "industry_code", "sector_kind"])
    code_col = "symbol" if "symbol" in raw.columns else "股票代码"
    date_col = "start_date" if "start_date" in raw.columns else "计入日期"
    industry_col = "行业名称" if "行业名称" in raw.columns else "industry_name"
    industry_code_col = "industry_code" if "industry_code" in raw.columns else "行业代码"
    if code_col not in raw.columns or date_col not in raw.columns or industry_col not in raw.columns:
        return pd.DataFrame(columns=["code", "start_date", "sector", "industry_code", "sector_kind"])
    out = raw[[code_col, date_col, industry_col, industry_code_col]].copy()
    out.columns = ["code", "start_date", "sector", "industry_code"]
    out["code"] = _normalize_code(out["code"])
    out["start_date"] = pd.to_datetime(out["start_date"], errors="coerce")
    out = out[(out["code"] != "") & out["start_date"].notna() & out["sector"].notna()]
    out["sector_kind"] = "sw_historical_effective"
    return out.drop_duplicates(["code", "start_date"], keep="last").sort_values(["code", "start_date"])


def _fetch_sw_current() -> pd.DataFrame:
    import akshare as ak

    industries = ak.sw_index_first_info()
    rows = []
    for _, industry in industries.iterrows():
        industry_code = str(industry["行业代码"]).split(".")[0]
        sector = str(industry["行业名称"])
        try:
            cons = ak.index_component_sw(symbol=industry_code)
        except Exception as exc:
            print(f"sector current skip {industry_code} {sector}: {exc}", flush=True)
            continue
        code_col = "证券代码"
        if code_col not in cons.columns:
            continue
        for code in _normalize_code(cons[code_col]).tolist():
            if code:
                rows.append({
                    "code": code,
                    "sector": sector,
                    "industry_code": industry_code,
                    "sector_kind": "sw_current_component",
                })
    return pd.DataFrame(rows).drop_duplicates("code", keep="last")


def _sector_coverage(training_set: Path, sectors: pd.DataFrame, kind: str) -> float:
    scope = pd.read_parquet(training_set, columns=["trade_date", "code"])
    scope["trade_date"] = pd.to_datetime(scope["trade_date"]).astype("datetime64[ns]")
    scope["code"] = scope["code"].astype(str).str.zfill(6)
    scope = scope.drop_duplicates(["trade_date", "code"])
    if kind == "point_in_time":
        right = sectors[["code", "start_date", "sector"]].copy()
        right["start_date"] = pd.to_datetime(right["start_date"]).astype("datetime64[ns]")
        right = right.sort_values(["start_date", "code"])
        merged = pd.merge_asof(
            scope.sort_values(["trade_date", "code"]), right,
            left_on="trade_date", right_on="start_date", by="code",
            direction="backward", allow_exact_matches=True,
        )
        return float(merged["sector"].notna().mean()) if len(merged) else 0.0
    mapped = scope["code"].isin(set(sectors["code"].astype(str).str.zfill(6)))
    return float(mapped.mean()) if len(mapped) else 0.0


def _build_sector(output: Path, mode: str, training_set: Path | None = None,
                  min_coverage: float = 0.95) -> dict:
    if mode == "none":
        return {"enabled": False, "reason": "sector-mode=none"}
    source = ""
    kind = ""
    try:
        if mode in {"auto", "historical"}:
            data = _clean_sw_historical(_fetch_sw_historical())
            if data.empty:
                raise RuntimeError("empty Shenwan historical table")
            source = "akshare:stock_industry_clf_hist_sw"
            kind = "point_in_time"
        else:
            raise RuntimeError("historical mode skipped")
    except Exception:
        if mode in {"auto", "historical"}:
            raise
        data = _fetch_sw_current()
        source = "akshare:sw_index_first_info/index_component_sw"
        kind = "static_current"
    if data.empty:
        raise RuntimeError("no sector exposures were built")
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    coverage = _sector_coverage(training_set, data, kind) if training_set else None
    gate_pass = kind == "point_in_time" and (coverage is None or coverage >= min_coverage)
    return {
        "output": str(output),
        "source": source,
        "kind": kind,
        "coverage": coverage,
        "minimum_coverage": min_coverage,
        "gate": "PASS" if gate_pass else "FAIL",
        "rows": int(len(data)),
        "codes": int(data["code"].nunique()),
        "note": "Current-component sectors are not historical PIT classifications." if kind != "point_in_time" else "",
    }


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser()
    training_set = Path(args.training_set).expanduser() if args.training_set else root / "training_set.parquet"
    report = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    if not args.skip_market_cap:
        output = Path(args.output_market_cap).expanduser() if args.output_market_cap else root / "market_cap_daily.parquet"
        report["market_cap"] = _build_market_cap(
            root, training_set, output, args.max_codes, args.sleep, args.refresh,
            args.min_pit_coverage,
        )
    if args.sector_mode != "none":
        output = Path(args.output_sector).expanduser() if args.output_sector else root / "sector_map_sw.parquet"
        report["sector"] = _build_sector(
            output, args.sector_mode, training_set, args.min_pit_coverage,
        )
    report_path = root / "neutralization_exposures_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"exposure report saved: {report_path}")
    failed_gates = [
        name for name in ("market_cap", "sector")
        if name in report and report[name].get("gate") == "FAIL"
    ]
    if failed_gates:
        raise RuntimeError(f"PIT exposure coverage gate failed: {', '.join(failed_gates)}")


if __name__ == "__main__":
    main()
