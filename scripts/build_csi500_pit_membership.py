#!/usr/bin/env python3
"""Normalize official CSI 500 sources into PIT membership, never fake weights.

Historical membership is reconstructed by reversing every captured official
rebalance from the latest official 500-stock snapshot.  The free official
endpoints expose only a current/month-end weight snapshot, so this script emits
that snapshot under an explicitly non-historical name and does not create
``csi500_weights_pit.parquet``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


EXTRACTOR_VERSION = "csi500_membership_normalize_v1"
INDEX_CODE = "000905"
# Matched against whitespace-stripped text: the CSI CMS inserts spaces inside
# numbers ("中证 1 000", "2 025年") in newer announcements.
EFFECTIVE_RE = re.compile(r"于(20\d{2})年(\d{1,2})月(\d{1,2})日收市后生效")
SECTION_RE = re.compile(r"中证\s*(\d+)\s*指数样本(?:临时)?调整名单")
CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


class PITSourceError(RuntimeError):
    """Official source is incomplete or internally inconsistent."""


# Exchange code changes are administrative substitutions inside an index: no
# rebalance announcement is published, so the reverse replay chain breaks
# unless old codes are canonicalized to the current code space.  Each entry
# needs official evidence.  Membership output uses the CURRENT code for the
# whole span (price vendors serve the merged history under the new code).
CODE_CHANGES = (
    {
        "old": "300114",
        "new": "302132",
        "effective": "2025-02-17",
        "evidence": (
            "中航电测重组更名中航成飞并变更证券代码 300114→302132；"
            "https://www.cnindex.com.cn/zh_information/notices_news/2025/202502/"
            "P020250212310734081331.pdf"
        ),
    },
)


def _canonical_code_map() -> dict[str, str]:
    mapping = {item["old"]: item["new"] for item in CODE_CHANGES}
    # Follow chains (A->B, B->C) and refuse cycles.
    for old in list(mapping):
        seen = {old}
        target = mapping[old]
        while target in mapping:
            if target in seen:
                raise PITSourceError(f"code-change cycle at {old}")
            seen.add(target)
            target = mapping[target]
        mapping[old] = target
    return mapping


@dataclass(frozen=True)
class RebalanceDelta:
    notice_id: int
    published_at: str
    available_at: str
    effective_after_close: pd.Timestamp
    removed: tuple[str, ...]
    added: tuple[str, ...]
    source_url: str
    source_hash: str


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description="Build official CSI500 PIT membership.")
    parser.add_argument("--raw-dir", default=str(root / "csi500_official_raw"))
    parser.add_argument("--calendar", required=True,
                        help="CSV/parquet with trade_date or date; no inferred weekdays.")
    parser.add_argument("--scope",
                        help="Optional CSV/parquet trade_date+code grid for explicit true/false output.")
    parser.add_argument("--output-dir", default=str(root))
    parser.add_argument("--expected-members", type=int, default=500)
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_source(raw_root: Path, entry: dict) -> bytes:
    path = raw_root / entry["local_path"]
    if not path.exists():
        raise PITSourceError(f"captured source is missing: {path}")
    data = path.read_bytes()
    actual = _sha256(data)
    if actual != entry.get("source_hash"):
        raise PITSourceError(f"captured source hash mismatch: {path}")
    return data


def _find_column(frame: pd.DataFrame, *terms: str) -> str:
    for column in frame.columns:
        compact = re.sub(r"\s+", "", str(column)).lower()
        if all(term.lower() in compact for term in terms):
            return column
    raise PITSourceError(f"missing column containing {terms}: {list(frame.columns)}")


def _normalize_code(value: object) -> str:
    text = str(value).strip()
    match = re.search(r"(\d{1,6})(?:\.0)?$", text)
    return match.group(1).zfill(6) if match else ""


def _read_constituent_snapshot(data: bytes, expected_members: int) -> tuple[pd.Timestamp, set[str]]:
    frame = pd.read_excel(BytesIO(data))
    date_col = _find_column(frame, "日期")
    index_col = _find_column(frame, "指数代码")
    code_col = _find_column(frame, "券代码")
    index_codes = frame[index_col].map(_normalize_code)
    frame = frame[index_codes == INDEX_CODE].copy()
    codes = {_normalize_code(value) for value in frame[code_col]}
    codes.discard("")
    if len(codes) != expected_members:
        raise PITSourceError(
            f"current CSI500 snapshot has {len(codes)} unique members, expected {expected_members}"
        )
    dates = pd.to_datetime(frame[date_col].astype(str), format="%Y%m%d", errors="coerce")
    if dates.isna().any() or dates.nunique() != 1:
        raise PITSourceError("current CSI500 snapshot has invalid or mixed dates")
    return dates.iloc[0].normalize(), codes


def _read_current_weight_snapshot(data: bytes, entry: dict) -> pd.DataFrame:
    frame = pd.read_excel(BytesIO(data))
    date_col = _find_column(frame, "日期")
    index_col = _find_column(frame, "指数代码")
    code_col = _find_column(frame, "券代码")
    weight_col = _find_column(frame, "权重")
    out = pd.DataFrame({
        "trade_date": pd.to_datetime(
            frame[date_col].astype(str), format="%Y%m%d", errors="coerce"
        ),
        "code": frame[code_col].map(_normalize_code),
        "index_code": frame[index_col].map(_normalize_code),
        "benchmark_weight": pd.to_numeric(frame[weight_col], errors="coerce") / 100.0,
    })
    out = out[out["index_code"] == INDEX_CODE].copy()
    if out.empty or out.isna().any().any() or (out["benchmark_weight"] <= 0).any():
        raise PITSourceError("current CSI500 weight snapshot is invalid")
    if not 0.995 <= float(out["benchmark_weight"].sum()) <= 1.005:
        raise PITSourceError("current CSI500 weight snapshot does not sum to one")
    out["available_at"] = entry["available_at"]
    out["source_url"] = entry["source_url"]
    out["source_hash"] = entry["source_hash"]
    out["extractor_version"] = EXTRACTOR_VERSION
    out["snapshot_only"] = True
    out["usable_for_historical_backtest"] = False
    return out.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _effective_after_close(content: str) -> pd.Timestamp:
    text = BeautifulSoup(content or "", "html.parser").get_text(" ", strip=True)
    match = EFFECTIVE_RE.search(re.sub(r"\s+", "", text))
    if not match:
        raise PITSourceError("announcement has no explicit after-close effective date")
    year, month, day = map(int, match.groups())
    return pd.Timestamp(year=year, month=month, day=day)


def _parse_html_delta(content: str) -> tuple[list[str], list[str]]:
    soup = BeautifulSoup(content or "", "html.parser")
    marker = soup.find(string=lambda value: bool(
        value and re.search(r"中证\s*500\s*指数样本(?:临时)?调整名单", value)
    ))
    if marker is None:
        return [], []
    table = marker.find_next("table")
    if table is None:
        return [], []
    removed: list[str] = []
    added: list[str] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if len(cells) < 4:
            continue
        left = CODE_RE.search(cells[0])
        right = CODE_RE.search(cells[2])
        if left and right:
            removed.append(left.group(1))
            added.append(right.group(1))
    return removed, added


def _parse_pdf_delta_text(text: str) -> tuple[list[str], list[str]]:
    headers = list(SECTION_RE.finditer(text))
    marker_index = next(
        (index for index, match in enumerate(headers) if match.group(1) == "500"), None
    )
    if marker_index is None:
        return [], []
    start = headers[marker_index].end()
    end = headers[marker_index + 1].start() if marker_index + 1 < len(headers) else len(text)
    section = text[start:end]
    removed: list[str] = []
    added: list[str] = []
    for line in section.splitlines():
        codes = CODE_RE.findall(line)
        if len(codes) == 2:
            removed.append(codes[0])
            added.append(codes[1])
    return removed, added


def _pdf_text(path: Path) -> str:
    executable = shutil.which("pdftotext")
    if not executable:
        raise PITSourceError("pdftotext is required to normalize official rebalance PDFs")
    result = subprocess.run(
        [executable, "-layout", str(path), "-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace")


def _parse_xlsx_delta(path: Path) -> tuple[list[str], list[str]]:
    book = pd.ExcelFile(path)
    result: dict[str, list[str]] = {}
    for direction in ("调出", "调入"):
        sheet = next((name for name in book.sheet_names if direction in name), None)
        if sheet is None:
            continue
        frame = pd.read_excel(path, sheet_name=sheet)
        index_col = _find_column(frame, "指数代码")
        code_col = _find_column(frame, "证券代码")
        index_codes = frame[index_col].map(_normalize_code)
        codes = frame.loc[index_codes == INDEX_CODE, code_col].map(_normalize_code)
        result[direction] = [code for code in codes if code]
    return result.get("调出", []), result.get("调入", [])


def _validate_delta(removed: list[str], added: list[str], notice_id: int) -> None:
    if not removed or len(removed) != len(added):
        raise PITSourceError(
            f"notice {notice_id} has incomplete CSI500 delta: out={len(removed)} in={len(added)}"
        )
    if len(set(removed)) != len(removed) or len(set(added)) != len(added):
        raise PITSourceError(f"notice {notice_id} has duplicate delta codes")
    if set(removed) & set(added):
        raise PITSourceError(f"notice {notice_id} has the same code on both sides")


def _load_deltas(raw_root: Path, manifest: dict) -> list[RebalanceDelta]:
    entries = manifest["sources"]
    details = {int(item["notice_id"]): item for item in entries
               if item["kind"] == "rebalance_announcement"}
    attachments: dict[int, list[dict]] = {}
    for item in entries:
        if item["kind"] == "rebalance_attachment":
            attachments.setdefault(int(item["notice_id"]), []).append(item)

    deltas: list[RebalanceDelta] = []
    for notice_id, detail_entry in sorted(details.items()):
        detail_data = _verify_source(raw_root, detail_entry)
        detail = json.loads(detail_data)
        content = str(detail.get("content") or "")
        effective = _effective_after_close(content)
        removed, added = _parse_html_delta(content)
        chosen_entry = detail_entry
        candidates: list[tuple[list[str], list[str], dict]] = []
        if removed:
            candidates.append((removed, added, detail_entry))
        for attachment_entry in attachments.get(notice_id, []):
            path = raw_root / attachment_entry["local_path"]
            _verify_source(raw_root, attachment_entry)
            suffix = path.suffix.lower()
            if suffix in {".xlsx", ".xls"}:
                parsed = _parse_xlsx_delta(path)
            elif suffix == ".pdf":
                parsed = _parse_pdf_delta_text(_pdf_text(path))
            else:
                continue
            if parsed[0]:
                candidates.append((parsed[0], parsed[1], attachment_entry))
        if not candidates:
            raise PITSourceError(f"notice {notice_id} has no parseable CSI500 delta")
        canonical = (candidates[0][0], candidates[0][1])
        for candidate_removed, candidate_added, _ in candidates[1:]:
            if (candidate_removed, candidate_added) != canonical:
                raise PITSourceError(f"notice {notice_id} HTML/attachment deltas disagree")
        removed, added, chosen_entry = candidates[-1]
        code_map = _canonical_code_map()
        removed = [code_map.get(code, code) for code in removed]
        added = [code_map.get(code, code) for code in added]
        _validate_delta(removed, added, notice_id)
        deltas.append(RebalanceDelta(
            notice_id=notice_id,
            published_at=detail_entry["published_at"],
            available_at=detail_entry["available_at"],
            effective_after_close=effective,
            removed=tuple(removed),
            added=tuple(added),
            source_url=chosen_entry["source_url"],
            source_hash=chosen_entry["source_hash"],
        ))
    effective_dates = [item.effective_after_close for item in deltas]
    if len(set(effective_dates)) != len(effective_dates):
        raise PITSourceError("multiple captured CSI500 deltas share an effective date")
    return sorted(deltas, key=lambda item: item.effective_after_close)


def _reconstruct_versions(
    anchor_codes: set[str],
    deltas: list[RebalanceDelta],
    expected_members: int,
) -> list[tuple[RebalanceDelta, frozenset[str]]]:
    if not deltas:
        raise PITSourceError("no CSI500 rebalance deltas were captured")
    state = set(anchor_codes)
    versions_desc: list[tuple[RebalanceDelta, frozenset[str]]] = []
    for delta in reversed(deltas):
        missing_added = set(delta.added) - state
        still_present_removed = set(delta.removed) & state
        if missing_added or still_present_removed:
            raise PITSourceError(
                f"notice {delta.notice_id} cannot reverse from anchor; "
                f"missing_added={sorted(missing_added)} present_removed={sorted(still_present_removed)}"
            )
        versions_desc.append((delta, frozenset(state)))
        state.difference_update(delta.added)
        state.update(delta.removed)
        if len(state) != expected_members:
            raise PITSourceError(
                f"notice {delta.notice_id} reverse produced {len(state)} members"
            )
    return list(reversed(versions_desc))


def _read_calendar(path: Path) -> pd.DatetimeIndex:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    column = "trade_date" if "trade_date" in frame.columns else "date" if "date" in frame.columns else None
    if column is None:
        raise PITSourceError("calendar must contain trade_date or date")
    dates = pd.to_datetime(frame[column], errors="coerce").dropna().dt.normalize().drop_duplicates()
    if dates.empty:
        raise PITSourceError("calendar has no valid trade dates")
    return pd.DatetimeIndex(dates.sort_values())


def _read_scope(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=["trade_date", "code"])
    else:
        frame = pd.read_csv(path, usecols=["trade_date", "code"])
    out = frame[["trade_date", "code"]].copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    out["code"] = out["code"].map(_normalize_code)
    if out.isna().any().any() or (out["code"] == "").any():
        raise PITSourceError("membership scope contains invalid trade_date/code rows")
    out = out.drop_duplicates(["trade_date", "code"])
    return out.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _explicit_membership_grid(scope: pd.DataFrame, active: pd.DataFrame) -> pd.DataFrame:
    active_dates = set(active["trade_date"].unique())
    missing_dates = sorted(set(scope["trade_date"].unique()) - active_dates)
    if missing_dates:
        raise PITSourceError(
            f"membership scope extends beyond reconstructed coverage: {missing_dates[:3]}"
        )
    metadata_columns = [
        "trade_date", "index_code", "published_at", "available_at", "effective_from",
        "source_url", "source_hash", "source_chain_hash", "extractor_version",
    ]
    metadata = active[metadata_columns].drop_duplicates("trade_date")
    true_keys = active[["trade_date", "code"]].assign(is_member=True)
    out = scope.merge(true_keys, on=["trade_date", "code"], how="left", validate="one_to_one")
    out["is_member"] = out["is_member"].fillna(False).astype(bool)
    out = out.merge(metadata, on="trade_date", how="left", validate="many_to_one")
    if out[metadata_columns[1:]].isna().any().any():
        raise PITSourceError("membership grid lost PIT version metadata")
    return out.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _daily_membership(
    calendar: pd.DatetimeIndex,
    versions: list[tuple[RebalanceDelta, frozenset[str]]],
    anchor_date: pd.Timestamp,
    anchor_hash: str,
    expected_members: int,
) -> pd.DataFrame:
    usable_dates = calendar[(calendar > versions[0][0].effective_after_close) & (calendar <= anchor_date)]
    if usable_dates.empty:
        raise PITSourceError("calendar does not overlap reconstructed CSI500 coverage")
    chain_hash = _sha256(
        (anchor_hash + "".join(delta.source_hash for delta, _ in versions)).encode("ascii")
    )
    rows: list[dict] = []
    for trade_date in usable_dates:
        eligible = [item for item in versions if trade_date > item[0].effective_after_close]
        if not eligible:
            continue
        delta, codes = eligible[-1]
        if len(codes) != expected_members:
            raise PITSourceError(f"{trade_date.date()} does not have {expected_members} members")
        effective_from = usable_dates[usable_dates > delta.effective_after_close][0]
        for code in sorted(codes):
            rows.append({
                "trade_date": trade_date,
                "code": code,
                "index_code": INDEX_CODE,
                "is_member": True,
                "published_at": delta.published_at,
                "available_at": delta.available_at,
                "effective_from": effective_from,
                "source_url": delta.source_url,
                "source_hash": delta.source_hash,
                "source_chain_hash": chain_hash,
                "extractor_version": EXTRACTOR_VERSION,
            })
    out = pd.DataFrame(rows)
    counts = out.groupby("trade_date")["code"].nunique()
    if out.duplicated(["trade_date", "code"]).any() or not counts.eq(expected_members).all():
        raise PITSourceError("daily CSI500 membership failed uniqueness/member-count checks")
    return out.sort_values(["trade_date", "code"]).reset_index(drop=True)


def build(
    raw_root: Path,
    calendar_path: Path,
    output_dir: Path,
    expected_members: int,
    scope_path: Path | None = None,
) -> dict:
    manifest_path = raw_root / "manifest.json"
    if not manifest_path.exists():
        raise PITSourceError(f"missing raw manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("index_code") != INDEX_CODE:
        raise PITSourceError("raw manifest is not CSI500")
    source_entries = manifest.get("sources") or []
    constituent_entry = next(
        (item for item in source_entries if item["kind"] == "current_constituents"), None
    )
    weight_entry = next(
        (item for item in source_entries if item["kind"] == "current_weight_snapshot"), None
    )
    if constituent_entry is None or weight_entry is None:
        raise PITSourceError("raw capture lacks current constituent/weight snapshots")
    constituent_data = _verify_source(raw_root, constituent_entry)
    weight_data = _verify_source(raw_root, weight_entry)
    anchor_date, anchor_codes = _read_constituent_snapshot(constituent_data, expected_members)
    deltas = _load_deltas(raw_root, manifest)
    if deltas[-1].effective_after_close >= anchor_date:
        raise PITSourceError("latest rebalance is not earlier than the full anchor snapshot")
    versions = _reconstruct_versions(anchor_codes, deltas, expected_members)
    calendar = _read_calendar(calendar_path)
    membership = _daily_membership(
        calendar, versions, anchor_date, constituent_entry["source_hash"], expected_members
    )
    current_weights = _read_current_weight_snapshot(weight_data, weight_entry)
    events = pd.DataFrame([
        {
            "notice_id": delta.notice_id,
            "published_at": delta.published_at,
            "available_at": delta.available_at,
            "effective_after_close": delta.effective_after_close,
            "removed_count": len(delta.removed),
            "added_count": len(delta.added),
            "removed_codes": json.dumps(delta.removed, ensure_ascii=False),
            "added_codes": json.dumps(delta.added, ensure_ascii=False),
            "source_url": delta.source_url,
            "source_hash": delta.source_hash,
            "extractor_version": EXTRACTOR_VERSION,
        }
        for delta in deltas
    ])

    output_dir.mkdir(parents=True, exist_ok=True)
    membership_path = output_dir / "csi500_membership_pit.parquet"
    grid_path = output_dir / "csi500_membership_grid_pit.parquet"
    event_path = output_dir / "csi500_rebalance_events.parquet"
    current_weight_path = output_dir / "csi500_current_weight_snapshot.parquet"
    report_path = output_dir / "csi500_pit_report.json"
    membership.to_parquet(membership_path, index=False)
    grid = None
    if scope_path is not None:
        grid = _explicit_membership_grid(_read_scope(scope_path), membership)
        grid.to_parquet(grid_path, index=False)
    events.to_parquet(event_path, index=False)
    current_weights.to_parquet(current_weight_path, index=False)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "index_code": INDEX_CODE,
        "extractor_version": EXTRACTOR_VERSION,
        "anchor_date": anchor_date.date().isoformat(),
        "anchor_members": len(anchor_codes),
        "rebalance_events": len(deltas),
        "coverage_start": membership["trade_date"].min().date().isoformat(),
        "coverage_end": membership["trade_date"].max().date().isoformat(),
        "trade_dates": int(membership["trade_date"].nunique()),
        "membership_rows": len(membership),
        "membership_active_only": True,
        "explicit_scope_grid_status": "BUILT" if grid is not None else "NOT_REQUESTED",
        "explicit_scope_grid_rows": len(grid) if grid is not None else 0,
        "historical_membership_status": "reconstructed_from_official_anchor_and_complete_delta_chain",
        "code_changes_applied": list(CODE_CHANGES),
        "historical_weight_status": "UNAVAILABLE_FAIL_CLOSED",
        "historical_weight_reason": (
            "Public CSI/AKShare endpoints expose only the current/month-end weight snapshot; "
            "no current weight is backfilled into historical dates."
        ),
        "outputs": {
            "membership": str(membership_path),
            "membership_true_false_scope_grid": str(grid_path) if grid is not None else None,
            "rebalance_events": str(event_path),
            "current_weight_snapshot_not_for_backtest": str(current_weight_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = _parse_args()
    report = build(
        Path(args.raw_dir).expanduser(),
        Path(args.calendar).expanduser(),
        Path(args.output_dir).expanduser(),
        args.expected_members,
        Path(args.scope).expanduser() if args.scope else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
