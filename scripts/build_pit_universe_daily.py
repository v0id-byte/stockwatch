#!/usr/bin/env python3
"""Build pit_universe_daily.parquet as an explicit True/False grid, fail closed.

Two phases:

* ``--phase a`` — membership + listing + ST from official/dated sources.  The
  suspension and limit columns are provisional ``False`` (bootstrap only uses
  member & listed rows to pick its download scope, never these flags).
* ``--phase b`` — after ``bootstrap_history.py`` has refreshed schema-v2 raw
  prices, recompute ``is_suspended`` and ``is_limit_up/down`` from observed
  bars, announcement evidence and ``analysis.price_limit``.

Three-state discipline: a missing bar with no independent suspension evidence
is UNKNOWN.  UNKNOWN member-days above the gate threshold fail the build;
below it they are resolved to ``is_suspended=True`` — a conservative
*exclusion*, never a silent claim of fact — and each one is itemized in the
report (``missing_bar_unknown``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.price_limit import limit_prices, price_limit_rule  # noqa: E402

EXTRACTOR_VERSION = "pit_universe_daily_v1"
INDEX_CODE = "000905"
UNKNOWN_MEMBER_DAY_GATE = 0.005  # >0.5% unresolved member-days fails the build

ST_NAME_RE = re.compile(r"\*?ST|退")
ST_APPLY_RE = re.compile(r"实施(?:退市风险警示|其他风险警示)|被实施.{0,4}风险警示")
ST_REVOKE_RE = re.compile(r"撤销(?:退市风险警示|其他风险警示)")
SUSPEND_TITLE_RE = re.compile(r"停牌")


class PITBuildError(RuntimeError):
    pass


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("a", "b"), required=True)
    parser.add_argument("--history-dir", default=str(root))
    parser.add_argument("--membership", default=str(root / "csi500_membership_pit.parquet"))
    parser.add_argument("--calendar", default=str(root / "market_sh000905.parquet"))
    parser.add_argument("--announcements-db", default=str(Path("~/.stockwatch/db.sqlite").expanduser()))
    parser.add_argument("--output", default=str(root / "pit_universe_daily.parquet"))
    return parser.parse_args()


def _normalize_code(raw) -> str:
    match = re.search(r"(\d{6})", str(raw))
    return match.group(1) if match else ""


def _read_calendar(path: Path) -> pd.DatetimeIndex:
    frame = pd.read_parquet(path, columns=["trade_date"])
    dates = pd.to_datetime(frame["trade_date"]).dt.normalize().drop_duplicates().sort_values()
    if dates.empty:
        raise PITBuildError("calendar is empty")
    return pd.DatetimeIndex(dates)


def _next_trade_date(calendar: pd.DatetimeIndex, when: pd.Timestamp) -> pd.Timestamp | None:
    pos = calendar.searchsorted(when.normalize(), side="right")
    return calendar[pos] if pos < len(calendar) else None


# ---------------------------------------------------------------------------
# Listing / delisting


def _listing_table(ak) -> pd.DataFrame:
    """listing (and where known, delisting) dates for every code we may meet."""
    rows: list[dict] = []
    sh = ak.stock_info_sh_name_code(symbol="主板A股")
    for _, row in sh.iterrows():
        rows.append({
            "code": _normalize_code(row["证券代码"]),
            "list_date": row["上市日期"],
            "delist_date": None,
            "current_name": str(row["证券简称"]),
            "source": "sh_listed",
        })
    kcb = ak.stock_info_sh_name_code(symbol="科创板")
    for _, row in kcb.iterrows():
        rows.append({
            "code": _normalize_code(row["证券代码"]),
            "list_date": row["上市日期"],
            "delist_date": None,
            "current_name": str(row["证券简称"]),
            "source": "sh_star_listed",
        })
    sz = ak.stock_info_sz_name_code(symbol="A股列表")
    for _, row in sz.iterrows():
        rows.append({
            "code": _normalize_code(row["A股代码"]),
            "list_date": row["A股上市日期"],
            "delist_date": None,
            "current_name": str(row["A股简称"]),
            "source": "sz_listed",
        })
    sh_delist = ak.stock_info_sh_delist()
    for _, row in sh_delist.iterrows():
        rows.append({
            "code": _normalize_code(row["公司代码"]),
            "list_date": row["上市日期"],
            "delist_date": row["暂停上市日期"],
            "current_name": str(row["公司简称"]),
            "source": "sh_delisted",
        })
    sz_delist = ak.stock_info_sz_delist(symbol="终止上市公司")
    for _, row in sz_delist.iterrows():
        rows.append({
            "code": _normalize_code(row["证券代码"]),
            "list_date": row["上市日期"],
            "delist_date": row["终止上市日期"],
            "current_name": str(row["证券简称"]),
            "source": "sz_delisted",
        })
    frame = pd.DataFrame(rows)
    frame = frame[frame["code"] != ""]
    frame["list_date"] = pd.to_datetime(frame["list_date"], errors="coerce")
    frame["delist_date"] = pd.to_datetime(frame["delist_date"], errors="coerce")
    # A code can appear in both listed and delisted tables (relisting); keep the
    # delisted row's delist_date but the earliest listing date observed.
    agg = frame.groupby("code").agg(
        list_date=("list_date", "min"),
        delist_date=("delist_date", "max"),
        current_name=("current_name", "last"),
    )
    return agg


# ---------------------------------------------------------------------------
# ST intervals


def _sz_name_intervals(ak, scope: set[str]) -> dict[str, list[tuple[pd.Timestamp, str]]]:
    """(effective_date, name) change points per SZ code, from the dated official list."""
    changes = ak.stock_info_sz_change_name(symbol="简称变更")
    out: dict[str, list[tuple[pd.Timestamp, str]]] = {}
    for _, row in changes.iterrows():
        code = _normalize_code(row["证券代码"])
        if code not in scope:
            continue
        when = pd.to_datetime(row["变更日期"], errors="coerce")
        if pd.isna(when):
            continue
        points = out.setdefault(code, [])
        before = str(row["变更前简称"]).strip()
        after = str(row["变更后简称"]).strip()
        if not points:
            points.append((pd.Timestamp.min, before))
        points.append((when, after))
    for points in out.values():
        points.sort(key=lambda item: item[0])
    return out


def _announcement_st_events(db_path: Path, scope: set[str]) -> pd.DataFrame:
    """Dated ST apply/revoke transitions mined from local announcement titles."""
    if not db_path.exists():
        raise PITBuildError(f"announcements db missing: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            "SELECT code, title, published_at FROM announcements "
            "WHERE title LIKE '%风险警示%'",
            conn,
        )
    finally:
        conn.close()
    frame["code"] = frame["code"].map(_normalize_code)
    frame = frame[frame["code"].isin(scope)].copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    frame = frame.dropna(subset=["published_at"])
    events = []
    for _, row in frame.iterrows():
        title = str(row["title"])
        applies = bool(ST_APPLY_RE.search(title))
        revokes = bool(ST_REVOKE_RE.search(title))
        if not applies and not revokes:
            continue
        events.append({
            "code": row["code"],
            "published_at": row["published_at"],
            "applies": applies,
            "revokes": revokes,
            "title": title,
        })
    return pd.DataFrame(events)


def _em_st_events(code: str) -> list[tuple[pd.Timestamp, bool, bool]]:
    """Dated ST apply/revoke transitions from the EastMoney announcement API."""
    import requests

    try:
        response = requests.get(
            "https://np-anotice-stock.eastmoney.com/api/security/ann",
            params={"sr": -1, "page_size": 100, "page_index": 1, "ann_type": "A",
                    "stock_list": code},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
        )
        items = response.json()["data"]["list"]
    except Exception:  # noqa: BLE001
        return []
    events = []
    for item in items:
        title = str(item.get("title") or "")
        applies = bool(ST_APPLY_RE.search(title))
        revokes = bool(ST_REVOKE_RE.search(title))
        if not applies and not revokes:
            continue
        when = pd.to_datetime(item.get("notice_date"), errors="coerce")
        if pd.notna(when):
            events.append((when, applies, revokes))
    return sorted(events)


def _st_flag_frame(
    ak,
    calendar: pd.DatetimeIndex,
    scope: set[str],
    listing: pd.DataFrame,
    db_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Daily is_st per scope code; SZ from dated renames, SH from announcements."""
    sz_scope = {c for c in scope if not c.startswith(("6",))}
    sh_scope = scope - sz_scope
    flags = pd.DataFrame(False, index=calendar, columns=sorted(scope))
    meta: dict = {"sz_codes": len(sz_scope), "sh_codes": len(sh_scope)}

    intervals = _sz_name_intervals(ak, sz_scope)
    sz_missing = sorted(sz_scope - set(intervals))
    for code, points in intervals.items():
        dates = [p[0] for p in points] + [pd.Timestamp.max]
        for (start, name), end in zip(points, dates[1:]):
            if ST_NAME_RE.search(name):
                mask = (calendar >= start) & (calendar < end)
                flags.loc[mask, code] = True
    # An SZ code absent from the rename list has never been renamed: its current
    # name is its only name; ST iff the current name says so.
    for code in sz_missing:
        name = str(listing["current_name"].get(code, ""))
        if ST_NAME_RE.search(name):
            flags[code] = True
    meta["sz_codes_without_rename_history"] = len(sz_missing)

    events = _announcement_st_events(db_path, sh_scope)
    meta["sh_st_events"] = int(len(events))
    for code, group in (events.groupby("code") if not events.empty else []):
        group = group.sort_values("published_at")
        state_start: pd.Timestamp | None = None
        for _, ev in group.iterrows():
            effective = _next_trade_date(calendar, ev["published_at"])
            if effective is None:
                continue
            if ev["revokes"] and state_start is None:
                # revoke without a seen apply: ST since before our window
                flags.loc[(calendar >= calendar[0]) & (calendar < effective), code] = True
            if ev["applies"] and not ev["revokes"]:
                state_start = effective
            elif ev["revokes"]:
                if state_start is not None:
                    flags.loc[(calendar >= state_start) & (calendar < effective), code] = True
                state_start = None
        if state_start is not None:
            flags.loc[calendar >= state_start, code] = True

    # Cross-check final state against current names (both exchanges).
    mismatches = []
    for code in sorted(scope):
        name = str(listing["current_name"].get(code, ""))
        if not name or name == "nan":
            continue
        expected = bool(ST_NAME_RE.search(name))
        actual = bool(flags[code].iloc[-1])
        delisted = pd.notna(listing["delist_date"].get(code))
        if expected != actual and not delisted:
            mismatches.append({"code": code, "current_name": name, "flag": actual})

    # Local announcement coverage ends before the calendar tail; patch the few
    # mismatched codes from the EastMoney announcement API (dated, per code).
    patched = []
    for item in list(mismatches):
        events = _em_st_events(item["code"])
        applied = False
        for published_at, applies, revokes in events:
            effective = _next_trade_date(calendar, published_at)
            if effective is None:
                continue
            if applies and not revokes:
                flags.loc[calendar >= effective, item["code"]] = True
                applied = True
            elif revokes:
                flags.loc[calendar >= effective, item["code"]] = False
                applied = True
        if applied and bool(flags[item["code"]].iloc[-1]) == bool(ST_NAME_RE.search(item["current_name"])):
            patched.append({**item, "patched_events": len(events)})
            mismatches.remove(item)
    meta["em_patched"] = patched
    meta["final_state_mismatches"] = mismatches
    return flags, meta


# ---------------------------------------------------------------------------
# Phase B: suspension and limit flags from observed bars


def _suspension_evidence(db_path: Path, scope: set[str]) -> pd.DataFrame:
    conn = sqlite3.connect(str(db_path))
    try:
        frame = pd.read_sql_query(
            "SELECT code, title, published_at FROM announcements "
            "WHERE title LIKE '%停牌%'",
            conn,
        )
    finally:
        conn.close()
    frame["code"] = frame["code"].map(_normalize_code)
    frame = frame[frame["code"].isin(scope)].copy()
    frame["published_at"] = pd.to_datetime(frame["published_at"], errors="coerce")
    return frame.dropna(subset=["published_at"])


def _phase_b_flags(
    history_dir: Path,
    calendar: pd.DatetimeIndex,
    grid: pd.DataFrame,
    listing: pd.DataFrame,
    db_path: Path,
) -> tuple[pd.DataFrame, dict]:
    """Fill is_suspended / is_limit_up / is_limit_down from schema-v2 bars."""
    evidence = _suspension_evidence(db_path, set(grid["code"].unique()))
    evidence_by_code = {code: g["published_at"].to_numpy() for code, g in evidence.groupby("code")}
    meta = {
        "suspected_volume0_days": 0,
        "missing_bar_with_evidence": 0,
        "missing_bar_unknown": [],
        "limit_price_mismatch": [],
        "missing_raw_close_rows": 0,
        "stocks_missing_parquet": [],
    }
    frames = []
    for code, sub in grid.groupby("code"):
        sub = sub.sort_values("trade_date").copy()
        path = history_dir / "stocks" / f"{code}.parquet"
        if not path.exists():
            meta["stocks_missing_parquet"].append(code)
            frames.append(sub)
            continue
        bars = pd.read_parquet(path, columns=["trade_date", "raw_close", "volume_shares"])
        bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.normalize()
        bars = bars.set_index("trade_date").sort_index()
        listed_mask = sub["is_listed"].to_numpy()
        merged = sub.merge(bars.reset_index(), on="trade_date", how="left")
        has_bar = merged["raw_close"].notna() | merged["volume_shares"].notna()
        vol0 = has_bar & merged["volume_shares"].fillna(0).eq(0)
        meta["suspected_volume0_days"] += int((vol0 & listed_mask).sum())
        missing = ~has_bar.to_numpy() & listed_mask
        suspended = vol0.to_numpy() & listed_mask
        ev = evidence_by_code.get(code)
        for pos in missing.nonzero()[0]:
            day = merged["trade_date"].iloc[pos]
            if ev is not None and ((ev <= day.to_datetime64()) & (ev >= (day - pd.Timedelta(days=45)).to_datetime64())).any():
                suspended[pos] = True
                meta["missing_bar_with_evidence"] += 1
            else:
                # UNKNOWN: excluded conservatively, surfaced in the report
                suspended[pos] = True
                meta["missing_bar_unknown"].append({"code": code, "date": str(day.date())})
        merged["is_suspended"] = suspended

        # trading-day index: exact when the observed history starts at listing
        list_date = listing["list_date"].get(code, pd.NaT)
        exact_age = pd.notna(list_date) and not bars.empty and abs((bars.index[0] - list_date).days) <= 10
        bar_pos = {d: i for i, d in enumerate(bars.index)}
        prev_close = bars["raw_close"].shift(1)
        up_flags, down_flags = [], []
        for _, row in merged.iterrows():
            day = row["trade_date"]
            if not row["is_listed"] or row["is_suspended"] or day not in bar_pos or pd.isna(row["raw_close"]):
                if row["is_listed"] and not row["is_suspended"] and day in bar_pos and pd.isna(row["raw_close"]):
                    meta["missing_raw_close_rows"] += 1
                up_flags.append(False)
                down_flags.append(False)
                continue
            prev = prev_close.get(day)
            if pd.isna(prev):
                up_flags.append(False)
                down_flags.append(False)
                continue
            idx = bar_pos[day] if exact_age else bar_pos[day] + 1000
            rule = price_limit_rule(code, day.date(), is_st=bool(row["is_st"]), trading_day_index=idx)
            up_price, down_price = limit_prices(float(prev), rule)
            close = float(row["raw_close"])
            up_flags.append(up_price is not None and close >= up_price - 1e-9)
            down_flags.append(down_price is not None and close <= down_price + 1e-9)
        merged["is_limit_up"] = up_flags
        merged["is_limit_down"] = down_flags
        frames.append(merged[list(sub.columns) + ["is_suspended", "is_limit_up", "is_limit_down"]
                             if "is_suspended" not in sub.columns else list(sub.columns)])
    out = pd.concat(frames, ignore_index=True)
    return out, meta


# ---------------------------------------------------------------------------


def build(args: argparse.Namespace) -> dict:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    import akshare as ak

    history_dir = Path(args.history_dir).expanduser()
    calendar = _read_calendar(Path(args.calendar).expanduser())
    membership = pd.read_parquet(Path(args.membership).expanduser())
    membership["trade_date"] = pd.to_datetime(membership["trade_date"]).dt.normalize()
    membership["code"] = membership["code"].map(_normalize_code)
    coverage = pd.DatetimeIndex(sorted(membership["trade_date"].unique()))
    calendar = calendar[calendar <= coverage[-1]]
    # Days before the reconstructed membership window exist only so early price
    # history can warm up rolling features: is_member stays False there and the
    # panel's universe_member gate keeps those rows out of training/portfolios.
    pre_membership_days = int((calendar < coverage[0]).sum())
    covered = calendar[calendar >= coverage[0]]
    missing_days = set(covered) - set(coverage)
    if missing_days:
        raise PITBuildError(f"membership does not cover calendar days: {sorted(missing_days)[:3]}")
    scope = set(membership["code"].unique())

    listing = _listing_table(ak)
    unknown_listing = sorted(scope - set(listing.index))
    if unknown_listing:
        raise PITBuildError(
            f"{len(unknown_listing)} scope codes have no listing/delisting record: {unknown_listing[:10]}"
        )

    st_flags, st_meta = _st_flag_frame(ak, calendar, scope, listing, Path(args.announcements_db))

    grid = pd.MultiIndex.from_product([calendar, sorted(scope)], names=["trade_date", "code"]).to_frame(index=False)
    member_keys = membership[["trade_date", "code"]].assign(is_member=True)
    grid = grid.merge(member_keys, on=["trade_date", "code"], how="left")
    grid["is_member"] = grid["is_member"].fillna(False).astype(bool)
    grid["index_code"] = INDEX_CODE

    list_dates = listing["list_date"].reindex(grid["code"]).to_numpy()
    delist_dates = listing["delist_date"].reindex(grid["code"]).to_numpy()
    days = grid["trade_date"].to_numpy()
    listed = (days >= list_dates) & ~(days >= delist_dates)
    grid["is_listed"] = pd.Series(listed).fillna(False).to_numpy()

    st_long = st_flags.stack().rename("is_st").reset_index()
    st_long.columns = ["trade_date", "code", "is_st"]
    grid = grid.merge(st_long, on=["trade_date", "code"], how="left")
    grid["is_st"] = grid["is_st"].fillna(False).astype(bool)

    member_not_listed = grid[grid["is_member"] & ~grid["is_listed"]]
    st_members = grid[grid["is_member"] & grid["is_st"]]

    report: dict = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "extractor_version": EXTRACTOR_VERSION,
        "phase": args.phase.upper(),
        "coverage_start": str(calendar[0].date()),
        "coverage_end": str(calendar[-1].date()),
        "trade_dates": int(len(calendar)),
        "pre_membership_warmup_days": pre_membership_days,
        "scope_codes": len(scope),
        "grid_rows": int(len(grid)),
        "member_rows": int(grid["is_member"].sum()),
        "member_not_listed_rows": int(len(member_not_listed)),
        "member_not_listed_sample": member_not_listed[["trade_date", "code"]].head(10).astype(str).to_dict("records"),
        "st_member_rows": int(len(st_members)),
        "st_member_sample": st_members[["trade_date", "code"]].head(10).astype(str).to_dict("records"),
        "st_meta": st_meta,
    }

    if args.phase == "a":
        grid["is_suspended"] = False
        grid["is_limit_up"] = False
        grid["is_limit_down"] = False
        report["suspension_limit_status"] = "PROVISIONAL_FALSE_PHASE_A"
    else:
        grid, phase_b_meta = _phase_b_flags(history_dir, calendar, grid, listing, Path(args.announcements_db))
        member_days = int(grid["is_member"].sum())
        unknown = [x for x in phase_b_meta["missing_bar_unknown"]]
        member_unknown = sum(
            1 for x in unknown
            if ((grid["code"] == x["code"]) & (grid["trade_date"] == pd.Timestamp(x["date"])) & grid["is_member"]).any()
        )
        phase_b_meta["member_unknown_days"] = member_unknown
        phase_b_meta["member_unknown_ratio"] = member_unknown / max(member_days, 1)
        report["phase_b"] = {
            **{k: v for k, v in phase_b_meta.items() if k != "missing_bar_unknown"},
            "missing_bar_unknown_count": len(unknown),
            "missing_bar_unknown_sample": unknown[:20],
        }
        if phase_b_meta["member_unknown_ratio"] > UNKNOWN_MEMBER_DAY_GATE:
            raise PITBuildError(
                f"unresolved member-day gaps {member_unknown}/{member_days} exceed the "
                f"{UNKNOWN_MEMBER_DAY_GATE:.1%} gate; refusing to write a polluted universe"
            )
        if phase_b_meta["stocks_missing_parquet"]:
            raise PITBuildError(
                f"scope stocks missing price history: {phase_b_meta['stocks_missing_parquet'][:10]}"
            )
        report["suspension_limit_status"] = "FINAL_PHASE_B"

    grid["build_phase"] = report["suspension_limit_status"]
    for column in ("is_member", "is_listed", "is_st", "is_suspended", "is_limit_up", "is_limit_down"):
        grid[column] = grid[column].astype(bool)
    out_path = Path(args.output).expanduser()
    grid = grid[[
        "trade_date", "code", "index_code", "is_member", "is_listed", "is_st",
        "is_suspended", "is_limit_up", "is_limit_down", "build_phase",
    ]].sort_values(["trade_date", "code"]).reset_index(drop=True)
    grid.to_parquet(out_path, index=False)
    report["output"] = str(out_path)
    report["sha256"] = hashlib.sha256(out_path.read_bytes()).hexdigest()
    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    report = build(args)
    slim = {k: v for k, v in report.items() if not str(k).endswith("sample")}
    print(json.dumps(slim, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
