import pandas as pd

from scripts.backfill_analyst_forecasts import (
    EXTRACTOR_VERSION,
    _normalize_report_frame,
)


def test_normalize_analyst_reports_keeps_publication_time_hash_and_target_years():
    raw = pd.DataFrame([{
        "股票代码": "1",
        "股票简称": "测试",
        "报告名称": "业绩点评",
        "东财评级": "买入",
        "机构": "测试证券",
        "行业": "银行",
        "日期": "2024-04-20",
        "报告PDF链接": "https://pdf.example/H3_AP123_1.pdf",
        "2024-盈利预测-收益": 1.2,
        "2024-盈利预测-市盈率": 10.0,
        "2025-盈利预测-收益": 1.5,
        "2025-盈利预测-市盈率": 8.0,
    }])

    result = _normalize_report_frame(raw, "2024-04-21T00:00:00+08:00")

    assert len(result) == 2
    assert set(result["target_period"]) == {"2024", "2025"}
    assert result.iloc[0]["code"] == "000001"
    assert result.iloc[0]["available_at"] == pd.Timestamp("2024-04-20 15:00:01")
    assert result.iloc[0]["report_id"] == "AP123"
    assert len(result.iloc[0]["source_row_sha256"]) == 64
    assert result.iloc[0]["extractor_version"] == EXTRACTOR_VERSION
    assert not result["source_document_verified"].any()
    assert not result["pit_verified"].any()
