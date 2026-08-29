from scripts.evaluate_pead_events import _parse_pead_title


def test_parse_pead_positive_direction():
    parsed = _parse_pead_title("某公司2024年年度业绩预增公告")

    assert parsed is not None
    assert parsed["sign"] == 1
    assert parsed["direction"] == "positive"


def test_parse_pead_negative_direction():
    parsed = _parse_pead_title("某公司2024年年度业绩预减公告")

    assert parsed is not None
    assert parsed["sign"] == -1
    assert parsed["direction"] == "negative"


def test_parse_pead_turnaround_overrides_loss_word():
    parsed = _parse_pead_title("某公司2024年度业绩扭亏为盈公告")

    assert parsed is not None
    assert parsed["sign"] == 1


def test_parse_pead_turn_to_loss_overrides_growth_word():
    parsed = _parse_pead_title("某公司2024年度业绩由盈转亏公告")

    assert parsed is not None
    assert parsed["sign"] == -1


def test_parse_pead_magnitude_percent():
    parsed = _parse_pead_title("某公司2024年年度业绩预增50%到80%公告")

    assert parsed is not None
    assert parsed["sign"] == 1
    assert parsed["magnitude_pct"] == 80.0
    assert parsed["signed_score"] == 80.0


def test_parse_pead_skips_directionless_title():
    assert _parse_pead_title("某公司2024年度业绩预告") is None
