"""Regression tests for the review-confirmed bug fixes and the rebuilt quant signal."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("STOCKWATCH_SKIP_REQUIRED_CONFIG", "1")


class TestReportNoLookahead:
    """analysis/report.py must price an entry from the NEXT trading day's close,
    not the (possibly intraday, not-yet-final) decision-day close."""

    def _kline(self):
        return [
            {"trade_date": "2024-01-02", "close": 10.0},
            {"trade_date": "2024-01-03", "close": 11.0},
            {"trade_date": "2024-01-04", "close": 12.0},
            {"trade_date": "2024-01-05", "close": 13.0},
        ]

    def test_entry_is_next_day_close(self):
        import pytest
        from analysis.report import _future_return
        # decision on 2024-01-02 -> entry should be 2024-01-03 close (11.0), not 10.0
        ret = _future_return(self._kline(), "2024-01-02", horizon=1)
        assert ret == pytest.approx((12.0 / 11.0 - 1) * 100)

    def test_returns_none_when_no_future_bar(self):
        from analysis.report import _future_return
        assert _future_return(self._kline(), "2024-01-05", horizon=1) is None


class TestSafeNextRedirect:
    """dashboard login must not honour off-site ?next= targets (open-redirect)."""

    def test_blocks_external_and_protocol_relative(self):
        from dashboard import _safe_next
        assert _safe_next("https://evil.com") == "/"
        assert _safe_next("//evil.com") == "/"
        assert _safe_next("javascript:alert(1)") == "/"
        assert _safe_next("\\\\evil.com") == "/"

    def test_allows_local_paths(self):
        from dashboard import _safe_next
        assert _safe_next("/settings") == "/settings"
        assert _safe_next("/") == "/"


class TestRobustFeatureSet:
    """The rebuilt model must use the sign-stable robust factor set, NOT the
    unstable risk factors (STD/BETA/RSQR/ILLIQ) that produced a negative OOS IC."""

    def test_robust_excludes_unstable_risk_factors(self):
        from analysis.factors import ROBUST_FEATURES
        for f in ROBUST_FEATURES:
            assert not f.startswith(("STD", "BETA", "RSQR", "ILLIQ")), f
        # and it keeps the validated reversal / position / turnover families
        assert "RET20" in ROBUST_FEATURES and "QTLD30" in ROBUST_FEATURES and "TURN120" in ROBUST_FEATURES

    def test_cross_sectional_rank_normalize(self):
        import pandas as pd
        from analysis.factors import cross_sectional_rank_normalize
        frame = pd.DataFrame({"RET20": [1.0, 2.0, 3.0, 4.0], "QTLD30": [4.0, 3.0, 2.0, 1.0]})
        out = cross_sectional_rank_normalize(frame, ["RET20", "QTLD30"])
        assert out["RET20"].min() == -0.25 and out["RET20"].max() == 0.5
        # a single-row batch is un-rankable -> neutral 0.0
        single = cross_sectional_rank_normalize(frame.head(1), ["RET20", "QTLD30"])
        assert (single[["RET20", "QTLD30"]] == 0.0).all().all()


class TestRegimeFeature:
    """Asymmetric regime model: bear uses the specialized model (validated OOS lift),
    every other regime uses the universal model."""

    def test_bear_set_adds_illiquidity_bull_drops_reversal(self):
        from analysis.factors import ROBUST_FEATURES, BEAR_FEATURES, BULL_FEATURES
        # bear = robust + illiquidity/oversold/drawdown
        assert set(ROBUST_FEATURES).issubset(set(BEAR_FEATURES))
        assert any(f.startswith("ILLIQ") for f in BEAR_FEATURES)
        # bull is light-touch: no short-term reversal factors
        assert not any(f.startswith(("RET", "RELV", "ROC")) for f in BULL_FEATURES)

    def test_is_bull_trend(self):
        import pandas as pd
        from analysis.regime import is_bull_trend
        assert bool(is_bull_trend(pd.Series([float(i) for i in range(250)])).iloc[-1]) is True
        assert bool(is_bull_trend(pd.Series([float(250 - i) for i in range(250)])).iloc[-1]) is False

    def test_resolve_model_path(self, tmp_path, monkeypatch):
        (tmp_path / "lgbm.txt").write_text("x")
        (tmp_path / "lgbm_bear.txt").write_text("x")
        monkeypatch.setenv("LGBM_MODEL_PATH", str(tmp_path / "lgbm.txt"))
        from config import Config
        cfg = Config()
        assert cfg.resolve_lgbm_model_path("bear").name == "lgbm_bear.txt"
        assert cfg.resolve_lgbm_model_path("bull").name == "lgbm.txt"
        assert cfg.resolve_lgbm_model_path("auto").name == "lgbm.txt"
        # falls back to universal when the bear model is absent
        (tmp_path / "lgbm_bear.txt").unlink()
        assert cfg.resolve_lgbm_model_path("bear").name == "lgbm.txt"


class TestFundamentalPipeline:
    """ocf_to_eps was validated NOT to improve the bear model OOS, so it must stay
    out of the production feature set; the PIT pipeline remains as opt-in infra."""

    def test_ocf_not_in_bear_features(self):
        from analysis.factors import BEAR_FEATURES
        assert "ocf_to_eps" not in BEAR_FEATURES

    def test_recent_quarter_ends_descending(self):
        from analysis.fundamental import _recent_quarter_ends
        ends = _recent_quarter_ends(3)
        assert len(ends) == 3
        assert ends == sorted(ends, reverse=True)  # most recent first
        assert all(len(e) == 8 and e[4:] in {"0331", "0630", "0930", "1231"} for e in ends)

    def test_merge_fundamental_graceful_without_file(self, tmp_path):
        import pandas as pd
        from scripts.build_training_set import _merge_fundamental
        data = pd.DataFrame({"trade_date": ["2024-06-01"], "code": ["600519"], "X": [1.0]})
        out, ok = _merge_fundamental(data, tmp_path)  # no fundamental parquet here
        assert ok is False and "ocf_to_eps" in out.columns and out["ocf_to_eps"].iloc[0] == 0.0


class TestEventLayer:
    """The event layer is risk/context only — its notes must never read as price
    predictions, and warnings (减持/首亏/big upcoming 解禁) sort before info."""

    def test_format_orders_warnings_first_and_is_empty_safe(self):
        from analysis.events import format_events_context
        assert format_events_context([]) == ""
        events = [
            {"type": "回购", "date": "", "level": "info", "note": "回购方案"},
            {"type": "减持", "date": "", "level": "warning", "note": "股东净减持"},
        ]
        out = format_events_context(events)
        assert "非买卖指令" in out
        assert out.index("减持") < out.index("回购")  # warning first

    def test_event_type_polarity_sets(self):
        from analysis.events import _POS_YJYG, _NEG_YJYG
        assert "首亏" in _NEG_YJYG and "续亏" in _NEG_YJYG
        assert "预增" in _POS_YJYG
        assert not (_POS_YJYG & _NEG_YJYG)  # no overlap
