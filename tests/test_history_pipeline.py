import json
import hashlib

import pandas as pd
import pytest

import scripts.build_neutralization_exposures as exposures
from scripts.bootstrap_history import (
    _download_sina,
    _history_file_is_current,
    _index_constituents,
    _normalize_history,
    _pit_universe_scope,
    _save_benchmarks,
)
from scripts.build_training_set import (
    _apply_pit_eligibility,
    _factor_input,
    _require_legacy_training_opt_in,
    _training_stock_paths,
    _validate_training_foundation,
)


def test_training_stock_paths_follow_exact_manifest_codes(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    for code in ("000001", "000002", "000003"):
        (stock_dir / f"{code}.parquet").touch()
    (tmp_path / "history_manifest.json").write_text(json.dumps({
        "codes": ["000001", "000003"],
        "constituent_membership_kind": "current_snapshot_not_point_in_time",
    }))

    paths, meta = _training_stock_paths(tmp_path, stock_dir)

    assert [path.stem for path in paths] == ["000001", "000003"]
    assert meta["source"] == "history_manifest.codes"
    assert meta["membership_kind"] == "current_snapshot_not_point_in_time"


def test_training_stock_paths_reject_legacy_directory_without_manifest(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    for code in ("000001", "000002"):
        (stock_dir / f"{code}.parquet").touch()

    with pytest.raises(RuntimeError, match="history_manifest.json is required"):
        _training_stock_paths(tmp_path, stock_dir)


def test_training_stock_paths_reject_incomplete_snapshot(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    (stock_dir / "000001.parquet").touch()
    (tmp_path / "history_manifest.json").write_text(json.dumps({
        "codes": ["000001"],
        "failed": 1,
    }))

    with pytest.raises(RuntimeError, match="1 stock downloads failed"):
        _training_stock_paths(tmp_path, stock_dir)


def test_training_stock_paths_reject_explicitly_excluded_failure(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    for code in ("000001", "000002"):
        (stock_dir / f"{code}.parquet").touch()
    (tmp_path / "history_manifest.json").write_text(json.dumps({
        "requested_codes": ["000001", "000002"],
        "codes": ["000001"],
        "excluded_failed_codes": ["000002"],
        "failed": 1,
    }))

    with pytest.raises(RuntimeError, match="1 stock downloads failed"):
        _training_stock_paths(tmp_path, stock_dir)


def test_index_constituents_prefers_csindex():
    class Ak:
        @staticmethod
        def index_stock_cons_csindex(symbol):
            return pd.DataFrame({
                "指数代码": [symbol, symbol],
                "成分券代码": ["000001", "600000"],
            })

        @staticmethod
        def index_stock_cons(symbol):
            raise AssertionError("fallback should not be used")

    codes, source = _index_constituents(Ak, "000300")

    assert codes == ["000001", "600000"]
    assert source == "csindex"


def test_benchmark_download_uses_requested_full_date_range(tmp_path):
    class Ak:
        calls = []

        @classmethod
        def stock_zh_index_daily_em(cls, **kwargs):
            cls.calls.append(kwargs)
            return pd.DataFrame({
                "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "open": [100.0, 101.0], "close": [101.0, 102.0],
                "high": [102.0, 103.0], "low": [99.0, 100.0],
                "volume": [1.0, 1.0], "amount": [100.0, 100.0],
            })

    counts = _save_benchmarks(Ak, pd, tmp_path, "20250102", "20250103")

    assert counts == {"sh000300": 2, "sh000905": 2}
    assert [call["symbol"] for call in Ak.calls] == ["sh000300", "csi000905"]
    assert all(call["start_date"] == "20250102" for call in Ak.calls)
    saved = pd.read_parquet(tmp_path / "market_sh000905.parquet")
    assert saved["raw_open"].tolist() == [100.0, 101.0]


def test_download_sina_maps_adjusted_history():
    rows = 30

    class Ak:
        @staticmethod
        def stock_zh_a_daily(**kwargs):
            frame = pd.DataFrame({
                "date": pd.date_range("2025-01-01", periods=rows),
                "open": pd.Series(range(1, rows + 1), dtype=float),
                "high": pd.Series(range(2, rows + 2), dtype=float),
                "low": pd.Series(range(1, rows + 1), dtype=float),
                "close": pd.Series(range(1, rows + 1), dtype=float),
                "volume": 1000.0,
                "amount": 10000.0,
                "outstanding_share": 100000.0,
                "turnover": 0.01,
            })
            if kwargs["adjust"] == "hfq":
                frame[["open", "high", "low", "close"]] *= 2
            return frame

    out = _download_sina(Ak, pd, "600000", "20250101", "20251231")

    assert {"raw_close", "adj_close", "adj_factor", "volume_shares", "vwap", "adj_vwap", "turnover"}.issubset(out.columns)
    assert len(out) == rows
    assert out.loc[0, "volume_shares"] == 1000.0
    assert out.loc[0, "vwap"] == 10.0
    assert out.loc[0, "adj_vwap"] == 20.0
    assert out.loc[0, "adj_factor"] == 2.0


def test_normalize_eastmoney_history_has_explicit_units_and_price_bases():
    raw = pd.DataFrame({
        "日期": ["2025-01-02", "2025-01-03"],
        "开盘": [10.0, 11.0], "最高": [11.0, 12.0], "最低": [9.0, 10.0],
        "收盘": [10.0, 11.0], "成交量": [2.0, 4.0],
        "成交额": [2000.0, 4400.0], "换手率": [1.0, 2.0],
    })
    hfq = raw[["日期", "开盘", "最高", "最低", "收盘"]].copy()
    hfq[["开盘", "最高", "最低", "收盘"]] *= 3

    out = _normalize_history(pd, raw, hfq, "eastmoney")

    assert out["volume_lots"].tolist() == [2.0, 4.0]
    assert out["volume_shares"].tolist() == [200.0, 400.0]
    assert out["turnover"].tolist() == [0.01, 0.02]
    assert out["vwap"].tolist() == [10.0, 11.0]
    assert out["raw_close"].tolist() == [10.0, 11.0]
    assert out["adj_close"].tolist() == [30.0, 33.0]
    assert out["adj_factor"].tolist() == [3.0, 3.0]
    assert out.loc[1, "amihud_1d"] == pytest.approx(0.1 / 4400.0)


def test_normalize_history_rejects_invalid_hfq_prices():
    raw = pd.DataFrame({
        "日期": ["2025-01-02"], "开盘": [10.0], "最高": [11.0], "最低": [9.0],
        "收盘": [10.0], "成交量": [2.0], "成交额": [2000.0], "换手率": [1.0],
    })
    hfq = raw[["日期", "开盘", "最高", "最低", "收盘"]].copy()
    hfq["收盘"] = -1.0

    with pytest.raises(ValueError, match="zero or negative"):
        _normalize_history(pd, raw, hfq, "eastmoney")


def test_old_qfq_only_history_is_not_reused(tmp_path):
    path = tmp_path / "000001.parquet"
    pd.DataFrame({"trade_date": ["2025-01-02"], "close": [10.0]}).to_parquet(path)

    assert not _history_file_is_current(pd, path, "20250102", "20250102")


def test_history_reuse_checks_requested_start_and_schema_values(tmp_path):
    raw = pd.DataFrame({
        "日期": ["2025-01-02", "2025-01-03"],
        "开盘": [10.0, 11.0], "最高": [11.0, 12.0], "最低": [9.0, 10.0],
        "收盘": [10.0, 11.0], "成交量": [2.0, 4.0],
        "成交额": [2000.0, 4400.0], "换手率": [1.0, 2.0],
    })
    hfq = raw[["日期", "开盘", "最高", "最低", "收盘"]].copy()
    path = tmp_path / "000001.parquet"
    frame = _normalize_history(pd, raw, hfq, "eastmoney")
    frame.to_parquet(path, index=False)

    assert _history_file_is_current(pd, path, "20250102", "20250103")
    assert not _history_file_is_current(pd, path, "20240102", "20250103")
    frame["market_data_schema_version"] = 1
    frame.to_parquet(path, index=False)
    assert not _history_file_is_current(pd, path, "20250102", "20250103")


def _pit_frame():
    return pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-03"],
        "code": ["000001", "000001"],
        "index_code": ["000905", "000905"],
        "is_member": [True, True],
        "is_listed": [True, True],
        "is_st": [False, False],
        "is_suspended": [False, False],
        "is_limit_up": [False, True],
        "is_limit_down": [False, False],
    })


def test_pit_universe_scope_and_daily_eligibility_gate(tmp_path):
    pit = _pit_frame()
    pit.to_parquet(tmp_path / "pit_universe_daily.parquet", index=False)

    codes, meta = _pit_universe_scope(
        pd, tmp_path / "pit_universe_daily.parquet", ["000905"], {"000905": 1}
    )
    filtered, report = _apply_pit_eligibility(
        pd.DataFrame({
            "trade_date": ["2025-01-02", "2025-01-03"],
            "code": ["000001", "000001"],
            "x": [1.0, 2.0],
        }),
        tmp_path,
    )

    assert codes == ["000001"]
    assert meta["date_start"] == "2025-01-02"
    assert filtered["trade_date"].tolist() == ["2025-01-02"]
    assert report["rows_excluded"] == 1
    assert report["excluded_limit_up"] == 1


def test_pit_universe_scope_rejects_incomplete_member_count(tmp_path):
    _pit_frame().to_parquet(tmp_path / "pit_universe_daily.parquet", index=False)

    with pytest.raises(RuntimeError, match="member-count gate failed"):
        _pit_universe_scope(pd, tmp_path / "pit_universe_daily.parquet", ["000905"])


def test_pit_universe_scope_rejects_string_status(tmp_path):
    frame = _pit_frame()
    frame["is_member"] = "True"
    frame.to_parquet(tmp_path / "pit_universe_daily.parquet", index=False)

    with pytest.raises(RuntimeError, match="non-null boolean"):
        _pit_universe_scope(
            pd, tmp_path / "pit_universe_daily.parquet", ["000905"], {"000905": 1}
        )


def test_legacy_training_requires_explicit_reproduction_opt_in(monkeypatch):
    monkeypatch.delenv("STOCKWATCH_ALLOW_LEGACY_UNEXECUTABLE_LABELS", raising=False)
    with pytest.raises(RuntimeError, match="not executable"):
        _require_legacy_training_opt_in()
    monkeypatch.setenv("STOCKWATCH_ALLOW_LEGACY_UNEXECUTABLE_LABELS", "true")
    _require_legacy_training_opt_in()


def test_pit_eligibility_missing_row_fails_closed(tmp_path):
    _pit_frame().iloc[:1].to_parquet(tmp_path / "pit_universe_daily.parquet", index=False)
    data = pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-03"],
        "code": ["000001", "000001"],
    })

    with pytest.raises(RuntimeError, match="coverage gate failed"):
        _apply_pit_eligibility(data, tmp_path)


def test_foundation_rejects_current_snapshot_and_old_schema(tmp_path):
    with pytest.raises(RuntimeError, match="schema v2"):
        _validate_training_foundation(tmp_path, {
            "market_data_schema": {"version": 1},
            "membership_kind": "current_snapshot_not_point_in_time",
        })


def test_foundation_pins_pit_universe_hash(tmp_path):
    pit_path = tmp_path / "pit_universe_daily.parquet"
    _pit_frame().to_parquet(pit_path, index=False)
    digest = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    meta = {
        "market_data_schema": {"version": 2},
        "membership_kind": "point_in_time_daily",
        "pit_universe_manifest": {"sha256": digest},
    }
    result = _validate_training_foundation(tmp_path, meta)
    assert result["pit_universe_sha256"] == digest

    _pit_frame().iloc[:1].to_parquet(pit_path, index=False)
    with pytest.raises(RuntimeError, match="hash differs"):
        _validate_training_foundation(tmp_path, meta)

    with pytest.raises(RuntimeError, match="current constituent snapshots"):
        _validate_training_foundation(tmp_path, {
            "market_data_schema": {"version": 2},
            "membership_kind": "current_snapshot_not_point_in_time",
        })


def test_factor_input_uses_hfq_ohlc_but_keeps_raw_schema():
    frame = pd.DataFrame({column: [1.0] for column in {
        "open", "high", "low", "close", "raw_open", "raw_high", "raw_low", "raw_close",
        "adj_open", "adj_high", "adj_low", "adj_close", "adj_factor", "volume",
        "volume_shares", "amount", "turnover", "vwap", "adj_vwap", "amihud_1d",
    }})
    frame["trade_date"] = ["2025-01-02"]
    frame["data_source"] = ["test"]
    frame["market_data_schema_version"] = [2]
    frame["return_adjustment"] = ["hfq"]
    frame[["adj_open", "adj_high", "adj_low", "adj_close"]] = 2.0

    out = _factor_input(frame, "000001")

    assert out[["open", "high", "low", "close"]].iloc[0].tolist() == [2.0] * 4
    assert out.loc[0, "raw_close"] == 1.0
    assert out.loc[0, "amount"] == out.loc[0, "adj_vwap"] * out.loc[0, "volume_shares"]


def test_market_cap_uses_raw_close_and_total_shares(tmp_path, monkeypatch):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    pd.DataFrame({
        "trade_date": ["2025-01-02"],
        "raw_close": [10.0],
        "float_a_share": [80.0],
    }).to_parquet(stock_dir / "000001.parquet", index=False)
    shares = pd.DataFrame({
        "share_date": pd.to_datetime(["2025-01-01"]),
        "total_share": [100.0],
        "float_a_share": [80.0],
    })
    monkeypatch.setattr(exposures, "_load_or_fetch_shares", lambda *args: (shares, "cache"))

    frame, status = exposures._market_cap_for_code(
        tmp_path, tmp_path / "cache", "000001",
        pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-03"), False,
    )

    assert status == "ok"
    assert frame.loc[0, "market_cap"] == 1000.0
    assert frame.loc[0, "float_market_cap"] == 800.0
    assert frame.loc[0, "market_cap_source"] == "cache:raw_close_x_total_share"


def test_market_cap_report_fails_below_pit_coverage(tmp_path, monkeypatch):
    training = tmp_path / "training_set.parquet"
    pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-02"],
        "code": ["000001", "000002"],
    }).to_parquet(training, index=False)

    def fake_cap(root, cache_dir, code, start, end, refresh):
        if code == "000002":
            return None, "missing total-share history"
        return pd.DataFrame({
            "trade_date": pd.to_datetime(["2025-01-02"]),
            "code": [code], "market_cap": [1000.0], "float_market_cap": [800.0],
            "total_share": [100.0], "float_a_share": [80.0],
            "share_date": pd.to_datetime(["2025-01-01"]),
            "market_cap_source": ["test:raw_close_x_total_share"],
        }), "ok"

    monkeypatch.setattr(exposures, "_market_cap_for_code", fake_cap)
    report = exposures._build_market_cap(
        tmp_path, training, tmp_path / "market_cap.parquet", 0, 0, False, 0.95,
    )

    assert report["coverage"] == 0.5
    assert report["gate"] == "FAIL"


def test_sector_auto_does_not_fall_back_to_current_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(exposures, "_fetch_sw_historical", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(exposures, "_fetch_sw_current", lambda: pd.DataFrame({
        "code": ["000001"], "sector": ["bank"], "industry_code": ["1"],
        "sector_kind": ["sw_current_component"],
    }))

    with pytest.raises(RuntimeError, match="down"):
        exposures._build_sector(tmp_path / "sector.parquet", "auto")

    report = exposures._build_sector(tmp_path / "sector.parquet", "current")
    assert report["kind"] == "static_current"
    assert report["gate"] == "FAIL"


def test_historical_sector_report_has_coverage_gate(tmp_path, monkeypatch):
    training = tmp_path / "training_set.parquet"
    pd.DataFrame({
        "trade_date": ["2025-01-02", "2025-01-02"],
        "code": ["000001", "000002"],
    }).to_parquet(training, index=False)
    monkeypatch.setattr(exposures, "_fetch_sw_historical", lambda: pd.DataFrame({
        "symbol": ["000001"],
        "start_date": ["2020-01-01"],
        "industry_name": ["bank"],
        "industry_code": ["1"],
    }))

    report = exposures._build_sector(
        tmp_path / "sector.parquet", "historical", training, 0.95,
    )

    assert report["kind"] == "point_in_time"
    assert report["coverage"] == 0.5
    assert report["gate"] == "FAIL"
