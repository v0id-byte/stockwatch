import math

import pandas as pd

from scripts.build_fundamental_features import (
    EXTRACTION_VERSION,
    _prepare_fundamental_frame,
)
from scripts.build_training_set import _merge_fundamental


def test_prepare_fundamental_frame_keeps_growth_and_audits_snapshot_vintage():
    raw = pd.DataFrame([{
        "股票代码": "1",
        "最新公告日期": "2024-04-20",
        "每股收益": 0.5,
        "每股经营现金流量": 0.75,
        "净资产收益率": 8.0,
        "销售毛利率": 30.0,
        "营业总收入-营业总收入": 1000.0,
        "营业总收入-同比增长": 12.0,
        "营业总收入-季度环比增长": 3.0,
        "净利润-净利润": 100.0,
        "净利润-同比增长": 20.0,
        "净利润-季度环比增长": 4.0,
        "每股净资产": 5.0,
        "所处行业": "测试行业",
    }])

    result = _prepare_fundamental_frame(
        raw, "20240331", "2024-04-21T00:00:00+08:00"
    ).iloc[0]

    assert result["code"] == "000001"
    assert result["available_at"] == pd.Timestamp("2024-04-20 15:00:01")
    assert math.isclose(result["ocf_to_eps"], 1.5)
    assert math.isclose(result["net_margin"], 0.1)
    assert result["revenue_yoy"] == 12.0
    assert result["net_profit_yoy"] == 20.0
    assert len(result["source_row_sha256"]) == 64
    assert result["extraction_version"] == EXTRACTION_VERSION
    assert not bool(result["vintage_verified"])


def test_legacy_training_does_not_enable_current_snapshot_fundamentals(tmp_path):
    pd.DataFrame([{
        "code": "000001",
        "available_at": "2024-04-20 15:00:01",
        "ocf_to_eps": 1.5,
        "vintage_verified": False,
    }]).to_parquet(tmp_path / "fundamental_features.parquet", index=False)
    training = pd.DataFrame([{"code": "000001", "trade_date": "2024-04-21"}])

    result, enabled = _merge_fundamental(training, tmp_path)

    assert not enabled
    assert result.iloc[0]["ocf_to_eps"] == 0.0
