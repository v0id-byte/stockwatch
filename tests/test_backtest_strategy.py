import pandas as pd
import pytest

from scripts.backtest_strategy import _benchmark_label, _load_benchmark


def test_csi500_is_the_default_named_benchmark():
    assert _benchmark_label("sh000905") == "CSI500"
    assert _benchmark_label("sh000300") == "CSI300"


def test_load_benchmark_requires_real_file(tmp_path):
    with pytest.raises(SystemExit, match="基准数据缺失"):
        _load_benchmark(tmp_path, "sh000905", hold=1)


def test_load_benchmark_builds_forward_returns(tmp_path):
    pd.DataFrame({
        "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "close": [100.0, 105.0, 110.0],
    }).to_parquet(tmp_path / "market_sh000905.parquet", index=False)

    values = _load_benchmark(tmp_path, "sh000905", hold=1)

    assert values[pd.Timestamp("2024-01-02")] == pytest.approx(0.05)
    assert values[pd.Timestamp("2024-01-03")] == pytest.approx(110 / 105 - 1)
