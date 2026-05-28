"""SQLite 持久化层"""
import json
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from config import get_config


def _row_to_dict(row: tuple, cols: list[str]) -> dict:
    return dict(zip(cols, row))


class Storage:
    def __init__(self, db_path: Path | None = None):
        cfg = get_config()
        self.db_path = db_path or cfg.db_path
        self.db_path.parent.mkdir(exist_ok=True, parents=True)
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS daily_kline (
                code TEXT NOT NULL,
                trade_date TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume REAL, amount REAL,
                PRIMARY KEY (code, trade_date)
            );
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL, title TEXT, content TEXT,
                source TEXT, ts TEXT, sentiment_score REAL,
                UNIQUE(code, title, ts)
            );
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL, run_ts TEXT NOT NULL,
                code TEXT NOT NULL, name TEXT,
                action TEXT, confidence REAL,
                target_price REAL, stop_loss REAL,
                reasons_json TEXT, risks_json TEXT,
                one_liner TEXT, pushed INTEGER DEFAULT 0, push_error TEXT
            );
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, run_ts TEXT NOT NULL,
                stocks_analyzed INTEGER, llm_calls INTEGER,
                tokens_used INTEGER, pushed_count INTEGER, pushed_ok INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions(run_id);
            CREATE INDEX IF NOT EXISTS idx_news_code ON news(code);
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), check_same_thread=False)

    def upsert_kline(self, code: str, trade_date: str, row: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO daily_kline
                (code, trade_date, open, high, low, close, volume, amount)
                VALUES (:code, :trade_date, :open, :high, :low, :close, :volume, :amount)
            """, {"code": code, "trade_date": trade_date, **row})

    def get_kline(self, code: str, start_date: str, end_date: str) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM daily_kline WHERE code=? AND trade_date BETWEEN ? AND ? ORDER BY trade_date ASC",
                [code, start_date, end_date]
            ).fetchall()
        return [dict(r) for r in rows]

    def kline_cached_today(self, code: str) -> bool:
        today = date.today().isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM daily_kline WHERE code=? AND trade_date=? LIMIT 1",
                [code, today]
            ).fetchone()
        return row is not None

    def upsert_news(self, code: str, items: list[dict]):
        if not items: return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO news (code, title, content, source, ts, sentiment_score)
                VALUES (:code, :title, :content, :source, :ts, NULL)
            """, [{"code": code, **item} for item in items])

    def get_news_since(self, code: str, days: int = 7) -> list[dict]:
        from datetime import datetime as dt
        cutoff = (dt.now().replace(hour=0, minute=0, second=0) - dt.timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM news WHERE code=? AND ts>=? ORDER BY ts DESC",
                [code, cutoff]
            ).fetchall()
        return [dict(r) for r in rows]

    def update_news_sentiment(self, code: str, title: str, ts: str, score: float):
        with self._conn() as conn:
            conn.execute(
                "UPDATE news SET sentiment_score=? WHERE code=? AND title=? AND ts=?",
                [score, code, title, ts]
            )

    def insert_decision(self, run_id: str, run_ts: str, dec: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO decisions
                (run_id, run_ts, code, name, action, confidence, target_price, stop_loss, reasons_json, risks_json, one_liner, pushed, push_error)
                VALUES (:run_id, :run_ts, :code, :name, :action, :confidence, :target_price, :stop_loss, :reasons_json, :risks_json, :one_liner, 0, NULL)
            """, {"run_id": run_id, "run_ts": run_ts, **dec})

    def get_decisions_by_run(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM decisions WHERE run_id=?",
                [run_id]).fetchall()
        return [dict(r) for r in rows]

    def mark_decision_pushed(self, run_id: str, code: str, ok: bool, error: str = ""):
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET pushed=?, push_error=? WHERE run_id=? AND code=?",
                [1 if ok else -1, error or None, run_id, code]
            )

    def insert_run(self, run_id: str, stats: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs (run_id, run_ts, stocks_analyzed, llm_calls, tokens_used, pushed_count, pushed_ok)
                VALUES (:run_id, :run_ts, :stocks_analyzed, :llm_calls, :tokens_used, :pushed_count, :pushed_ok)
            """, {"run_id": run_id, "run_ts": datetime.now().isoformat(), **stats})

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_ts DESC LIMIT ?", [limit]
            ).fetchall()
        return [dict(r) for r in rows]
