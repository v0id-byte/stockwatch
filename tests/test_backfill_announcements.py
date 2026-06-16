"""Tests for resumable CNINFO announcement backfill helpers."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.backfill_announcements import _progress_summary, _select_tasks


class FakeStorage:
    def __init__(self, rows):
        self.rows = rows

    def get_announcement_progress(self):
        return self.rows


def _task(code, start, end):
    return {
        "mode": "by-code",
        "code": code,
        "org_id": "gssz0000001",
        "chunk_start": start,
        "chunk_end": end,
    }


def _progress(code, start, end, status, fetched_count=0):
    return {
        "code": code,
        "chunk_start": start,
        "chunk_end": end,
        "status": status,
        "attempts": 1,
        "fetched_count": fetched_count,
        "error": "timeout" if status == "failed" else None,
        "updated_at": f"{end} 12:00:00",
    }


def test_select_tasks_resumes_non_done_chunks():
    tasks = [
        _task("000001", "2024-01-01", "2024-01-31"),
        _task("000001", "2024-02-01", "2024-02-29"),
        _task("000001", "2024-03-01", "2024-03-31"),
        _task("000001", "2024-04-01", "2024-04-30"),
    ]
    rows = [
        _progress("000001", "2024-01-01", "2024-01-31", "done"),
        _progress("000001", "2024-02-01", "2024-02-29", "pending"),
        _progress("000001", "2024-03-01", "2024-03-31", "failed"),
    ]

    selected = _select_tasks(tasks, FakeStorage(rows), force=False)

    assert [task["chunk_start"] for task in selected] == ["2024-02-01", "2024-03-01", "2024-04-01"]


def test_select_tasks_only_failed_keeps_retry_scope_small():
    tasks = [
        _task("000001", "2024-01-01", "2024-01-31"),
        _task("000001", "2024-02-01", "2024-02-29"),
        _task("000001", "2024-03-01", "2024-03-31"),
    ]
    rows = [
        _progress("000001", "2024-01-01", "2024-01-31", "done"),
        _progress("000001", "2024-02-01", "2024-02-29", "pending"),
        _progress("000001", "2024-03-01", "2024-03-31", "failed"),
    ]

    selected = _select_tasks(tasks, FakeStorage(rows), force=False, only_failed=True)

    assert [task["chunk_start"] for task in selected] == ["2024-03-01"]


def test_progress_summary_counts_expected_grid():
    tasks = [
        _task("000001", "2024-01-01", "2024-01-31"),
        _task("000001", "2024-02-01", "2024-02-29"),
        _task("000001", "2024-03-01", "2024-03-31"),
        _task("000001", "2024-04-01", "2024-04-30"),
    ]
    rows = [
        _progress("000001", "2024-01-01", "2024-01-31", "done", fetched_count=3),
        _progress("000001", "2024-02-01", "2024-02-29", "pending"),
        _progress("000001", "2024-03-01", "2024-03-31", "failed"),
    ]

    summary = _progress_summary(tasks, rows)

    assert summary["total"] == 4
    assert summary["done"] == 1
    assert summary["pending"] == 1
    assert summary["failed"] == 1
    assert summary["missing"] == 1
    assert summary["rows_done"] == 3
    assert len(summary["failed_examples"]) == 1
