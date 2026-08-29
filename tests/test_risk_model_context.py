from __future__ import annotations

import pytest

from analysis.lgbm import evaluate_model_health, format_risk_context


class TestFormatRiskContext:
    def test_pool_percentile_semantics(self):
        pool = {f"c{i:02d}": float(i) for i in range(20)}  # c00 riskiest .. c19 safest
        contexts = format_risk_context(pool)
        assert "风险最高的 10%" in contexts["c00"]
        assert "风险最高的 10%" in contexts["c01"]
        assert "风险最高" not in contexts["c19"]
        assert contexts["c19"].endswith("9.0/9")

    def test_none_scores_get_unavailable_text(self):
        contexts = format_risk_context({"a": None, "b": 1.0, "c": 2.0})
        assert "未就绪" in contexts["a"]

    def test_single_stock_never_fakes_a_percentile(self):
        contexts = format_risk_context({"a": 3.21})
        assert "单票无法横向排名" in contexts["a"]


class TestRiskModelHealthGate:
    def _meta(self, drawdown_ic):
        return {
            "model_kind": "risk",
            "test_metrics": {
                "drawdown_spearman_ic": drawdown_ic,
                # incidental alpha metrics must NOT decide a risk model's fate
                "return_spearman_ic": -0.9,
                "decile_returns": {"spread_9_minus_0": -0.9},
            },
        }

    def test_risk_model_gates_on_drawdown_ic_not_alpha_metrics(self, monkeypatch):
        monkeypatch.delenv("STOCKWATCH_RISK_MIN_TEST_DRAWDOWN_IC", raising=False)
        health = evaluate_model_health(self._meta(0.23))
        assert health["status"] == "VALIDATED"
        assert health["model_kind"] == "risk"

    def test_risk_model_fails_below_threshold(self, monkeypatch):
        monkeypatch.delenv("STOCKWATCH_RISK_MIN_TEST_DRAWDOWN_IC", raising=False)
        health = evaluate_model_health(self._meta(0.05))
        assert health["status"] == "UNVALIDATED"
        assert any("drawdown IC" in f for f in health["failures"])

    def test_missing_drawdown_ic_is_unknown_not_validated(self):
        health = evaluate_model_health({"model_kind": "risk", "test_metrics": {}})
        assert health["status"] == "UNKNOWN"

    def test_alpha_branch_untouched(self, monkeypatch):
        monkeypatch.delenv("STOCKWATCH_LGBM_MIN_TEST_RETURN_IC", raising=False)
        health = evaluate_model_health({
            "test_metrics": {"return_spearman_ic": -0.1,
                             "decile_returns": {"spread_9_minus_0": -0.1}},
        })
        assert health["status"] == "UNVALIDATED"


class TestModelScoresStorage:
    def test_roundtrip_and_full_pool_lookup(self, tmp_path):
        from utils.storage import Storage

        storage = Storage(str(tmp_path / "t.sqlite"))
        rows = [
            {"trade_date": "2026-06-10", "code": "000001",
             "alpha_model_version": None, "alpha_score": None,
             "risk_model_version": "lgbm_v2_risk:t", "risk_score": 5.5,
             "feature_contract_version": "v2.1",
             "reference_universe_sha256": "x", "scored_at": "2026-06-10T18:00:00"},
            {"trade_date": "2026-06-10", "code": "000002",
             "alpha_model_version": None, "alpha_score": None,
             "risk_model_version": "lgbm_v2_risk:t", "risk_score": 2.0,
             "feature_contract_version": "v2.1",
             "reference_universe_sha256": "x", "scored_at": "2026-06-10T18:00:00"},
        ]
        storage.upsert_model_scores(rows)
        full = storage.get_latest_model_scores()
        assert set(full) == {"000001", "000002"}
        some = storage.get_latest_model_scores(["000002", "999999"])
        assert set(some) == {"000002"}
        assert some["000002"]["risk_score"] == pytest.approx(2.0)
        # upsert replaces same-day rows
        rows[0]["risk_score"] = 6.0
        storage.upsert_model_scores(rows[:1])
        assert storage.get_latest_model_scores(["000001"])["000001"]["risk_score"] == pytest.approx(6.0)
