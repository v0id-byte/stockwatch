import numpy as np
import pandas as pd
import pytest


def _six_day_frame() -> pd.DataFrame:
    close = np.arange(1.0, 7.0)
    volume = np.array([10.0, 20.0, 40.0, 80.0, 40.0, 20.0])
    return pd.DataFrame({
        "trade_date": pd.date_range("2024-01-02", periods=6).strftime("%Y-%m-%d"),
        "open": close - 0.25,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": volume,
        "vwap": close + 0.10,
    })


def test_exact_feature_names_count_and_official_order():
    from analysis.alpha158 import (
        QLIB_ALPHA158_FEATURES,
        QLIB_ALPHA158_ROLLING_FEATURES,
        QLIB_ALPHA158_WINDOWS,
        compute_qlib_alpha158_frame,
    )

    assert len(QLIB_ALPHA158_FEATURES) == 158
    assert len(set(QLIB_ALPHA158_FEATURES)) == 158
    assert QLIB_ALPHA158_FEATURES[:13] == (
        "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
        "OPEN0", "HIGH0", "LOW0", "VWAP0",
    )
    assert QLIB_ALPHA158_FEATURES[13:18] == tuple(
        f"ROC{window}" for window in QLIB_ALPHA158_WINDOWS
    )
    assert QLIB_ALPHA158_FEATURES[-5:] == tuple(
        f"VSUMD{window}" for window in QLIB_ALPHA158_WINDOWS
    )
    assert len(QLIB_ALPHA158_ROLLING_FEATURES) == 29

    result = compute_qlib_alpha158_frame(_six_day_frame())
    assert tuple(result.columns[1:]) == QLIB_ALPHA158_FEATURES


def test_exact_alpha158_matches_manual_price_and_trend_formulas():
    from analysis.alpha158 import compute_qlib_alpha158_frame

    source = _six_day_frame()
    result = compute_qlib_alpha158_frame(source)
    row = result.iloc[-1]

    assert row["OPEN0"] == pytest.approx(5.75 / 6.0)
    assert row["HIGH0"] == pytest.approx(7.0 / 6.0)
    assert row["LOW0"] == pytest.approx(5.0 / 6.0)
    assert row["VWAP0"] == pytest.approx(6.1 / 6.0)
    assert row["ROC5"] == pytest.approx(1.0 / 6.0)
    assert row["MA5"] == pytest.approx(4.0 / 6.0)
    assert row["STD5"] == pytest.approx(np.std([2, 3, 4, 5, 6], ddof=1) / 6.0)
    assert row["BETA5"] == pytest.approx(1.0 / 6.0)
    assert row["RSQR5"] == pytest.approx(1.0)
    assert row["RESI5"] == pytest.approx(0.0, abs=1e-12)
    assert row["MAX5"] == pytest.approx(7.0 / 6.0)
    assert row["MIN5"] == pytest.approx(1.0 / 6.0)
    assert row["RANK5"] == pytest.approx(1.0)
    assert row["RSV5"] == pytest.approx(5.0 / 6.0)
    assert row["IMAX5"] == pytest.approx(1.0)
    assert row["IMIN5"] == pytest.approx(0.2)
    assert row["IMXD5"] == pytest.approx(0.8)


def test_exact_alpha158_matches_manual_change_and_volume_formulas():
    from analysis.alpha158 import compute_qlib_alpha158_frame

    source = _six_day_frame()
    row = compute_qlib_alpha158_frame(source).iloc[-1]

    assert row["CNTP5"] == pytest.approx(1.0)
    assert row["CNTN5"] == pytest.approx(0.0)
    assert row["CNTD5"] == pytest.approx(1.0)
    assert row["SUMP5"] == pytest.approx(1.0)
    assert row["SUMN5"] == pytest.approx(0.0)
    assert row["SUMD5"] == pytest.approx(1.0)
    assert row["VMA5"] == pytest.approx(2.0)
    assert row["VSTD5"] == pytest.approx(np.std([20, 40, 80, 40, 20], ddof=1) / 20.0)
    assert row["VSUMP5"] == pytest.approx(70.0 / 130.0)
    assert row["VSUMN5"] == pytest.approx(60.0 / 130.0)
    assert row["VSUMD5"] == pytest.approx(10.0 / 130.0)

    close = source["close"]
    volume = source["volume"]
    move = (close / close.shift(1) - 1).abs() * volume
    expected_wvma = move.iloc[-5:].std() / move.iloc[-5:].mean()
    expected_corr = close.iloc[-5:].corr(np.log(volume.iloc[-5:] + 1))
    expected_cord = (close / close.shift(1)).iloc[-5:].corr(
        np.log(volume / volume.shift(1) + 1).iloc[-5:]
    )
    assert row["WVMA5"] == pytest.approx(expected_wvma)
    assert row["CORR5"] == pytest.approx(expected_corr)
    assert row["CORD5"] == pytest.approx(expected_cord)


def test_exact_alpha158_preserves_raw_nan_and_requires_vwap():
    from analysis.alpha158 import compute_qlib_alpha158_frame

    source = _six_day_frame()
    result = compute_qlib_alpha158_frame(source)
    assert np.isnan(result.iloc[0]["STD5"])
    assert np.isnan(result.iloc[0]["BETA5"])
    assert np.isnan(result.iloc[0]["RSQR5"])
    assert np.isnan(result.iloc[0]["RESI5"])
    assert np.isnan(result.iloc[0]["CORR5"])
    assert np.isnan(result.iloc[0]["CORD5"])
    assert np.isnan(result.iloc[0]["WVMA5"])

    with pytest.raises(ValueError, match="missing: vwap"):
        compute_qlib_alpha158_frame(source.drop(columns="vwap"))


def test_exact_and_legacy_custom300_are_explicitly_separate():
    from analysis.alpha158 import QLIB_ALPHA158_FEATURES, compute_qlib_alpha158_frame
    from analysis.factors import (
        ALPHA158_FEATURES,
        CUSTOM_TECHNICAL_300_FEATURES,
        compute_alpha158_frame,
        compute_custom_technical_300_frame,
    )

    assert len(QLIB_ALPHA158_FEATURES) == 158
    assert len(CUSTOM_TECHNICAL_300_FEATURES) == 300
    assert ALPHA158_FEATURES == CUSTOM_TECHNICAL_300_FEATURES
    assert "BETA60" in QLIB_ALPHA158_FEATURES
    assert "BETA120" not in QLIB_ALPHA158_FEATURES
    assert "BETA120" in CUSTOM_TECHNICAL_300_FEATURES
    assert "RET20" not in QLIB_ALPHA158_FEATURES
    assert "RET20" in CUSTOM_TECHNICAL_300_FEATURES
    assert compute_custom_technical_300_frame is compute_alpha158_frame

    source = _six_day_frame()
    exact = compute_qlib_alpha158_frame(source)
    legacy_source = source.assign(amount=0.0).drop(columns="vwap")
    custom = compute_custom_technical_300_frame(legacy_source, legacy_source)
    assert exact.iloc[-1]["BETA5"] == pytest.approx(1.0 / 6.0)
    assert custom.iloc[-1]["BETA5"] == pytest.approx(1.0)


def test_exact_alpha158_sorts_and_preserves_trade_date():
    from analysis.alpha158 import compute_qlib_alpha158_frame

    source = _six_day_frame().iloc[::-1].reset_index(drop=True)
    result = compute_qlib_alpha158_frame(source)
    assert result.columns[0] == "trade_date"
    assert result["trade_date"].tolist() == sorted(source["trade_date"].tolist())
