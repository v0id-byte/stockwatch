import hashlib
import sqlite3

import pytest

from scripts.backfill_announcement_documents import (
    _available_at,
    _classify,
    _event_status,
    _extract_event,
    _candidate_rows,
)


@pytest.mark.parametrize(
    "title,expected",
    [
        ("2025年度业绩预告", "earnings"),
        ("关于股份回购进展的公告", "buyback"),
        ("股东减持股份计划公告", "holding_change"),
        ("重大合同中标公告", "major_contract"),
        ("收到监管警示函的公告", "inquiry_penalty"),
        ("2025年度权益分派实施公告", "capital_action"),
    ],
)
def test_six_document_categories(title, expected):
    assert _classify(title) == expected


def test_midnight_timestamp_is_conservatively_post_close():
    assert _available_at("2025-01-02 00:00:00") == "2025-01-02 15:00:01"
    assert _available_at("2025-01-02 10:30:00") == "2025-01-02 10:30:00"


def test_event_extracts_direction_status_and_magnitude():
    row = {
        "source": "cninfo",
        "announcement_id": "1",
        "code": "000001",
        "category": "holding_change",
        "title": "股东拟减持不超过3%股份的公告",
        "published_at": "2025-01-02 00:00:00",
    }
    event = _extract_event(row, "拟减持金额不超过2亿元，占总股本3%", "a" * 64, 1.0)

    assert event["direction"] == -1
    assert event["event_status"] == "planned"
    assert event["magnitude_percent"] == 3.0
    assert event["magnitude_value"] == 200_000_000
    assert event["signed_score"] < 0
    assert event["available_at"] == "2025-01-02 15:00:01"
    assert event["title_fingerprint"] == hashlib.sha256(
        "股东拟减持不超过3%股份".encode("utf-8")
    ).hexdigest()


def test_cancelled_status_beats_generic_plan_word():
    assert _event_status("buyback", "拟终止回购计划") == "cancelled"


def test_capital_action_ignores_incidental_negative_body_words():
    row = {
        "source": "cninfo",
        "announcement_id": "2",
        "code": "000002",
        "category": "capital_action",
        "title": "2025年度利润分配预案",
        "published_at": "2025-01-02 20:00:00",
    }
    event = _extract_event(row, "若发生终止上市风险，公司将另行公告。每10股派2元。", "b" * 64, 1.0)

    assert event["direction"] == 1
    assert event["event_status"] == "planned"


def test_cancelled_reduction_is_not_treated_as_fresh_negative_sale():
    row = {
        "source": "cninfo",
        "announcement_id": "3",
        "code": "000003",
        "category": "holding_change",
        "title": "关于提前终止减持计划的公告",
        "published_at": "2025-01-02 20:00:00",
    }
    event = _extract_event(row, "原计划减持3%，现决定终止。", "c" * 64, 1.0)

    assert event["direction"] == 1
    assert event["event_status"] == "cancelled"


def test_candidate_limit_is_time_stratified(tmp_path):
    source = tmp_path / "source.sqlite"
    with sqlite3.connect(source) as conn:
        conn.execute(
            """CREATE TABLE announcements (
                source TEXT, announcement_id TEXT, code TEXT, title TEXT,
                published_at TEXT, url TEXT
            )"""
        )
        conn.executemany(
            "INSERT INTO announcements VALUES ('cninfo',?,?,?,?,?)",
            [
                (str(index), "000001", "业绩预告", f"2025-{index:02d}-01 10:00:00", "u")
                for index in range(1, 13)
            ],
        )

    rows = _candidate_rows(source, ["earnings"], "2025-01-01", "2025-12-31", 3)

    assert [row["published_at"][:7] for row in rows] == ["2025-01", "2025-07", "2025-12"]
