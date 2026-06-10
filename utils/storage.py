"""SQLite 持久化层"""
import json
import sqlite3
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Optional
from loguru import logger

from config import get_config


def _row_to_dict(row: tuple, cols: list[str]) -> dict:
    return dict(zip(cols, row))


class Storage:
    def __init__(self, db_path: Path | None = None):
        if db_path is None:
            cfg = get_config()
            self.db_path = cfg.db_path
        else:
            self.db_path = Path(db_path).expanduser()
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
            self._migrate_v2(conn)

    def _migrate_v2(self, conn: sqlite3.Connection):
        self._add_column(conn, "decisions", "raw_confidence", "raw_confidence REAL")
        self._add_column(conn, "decisions", "calibrated_confidence", "calibrated_confidence REAL")
        self._add_column(conn, "decisions", "resolved_at", "resolved_at TEXT")
        self._add_column(conn, "decisions", "success", "success INTEGER")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS calibration_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            trained_at TEXT NOT NULL,
            sample_size INTEGER,
            coef REAL,
            intercept REAL,
            auc REAL,
            notes TEXT
        );
        CREATE TABLE IF NOT EXISTS market_regime_history (
            trade_date TEXT PRIMARY KEY,
            vol_20d REAL,
            regime TEXT,
            percentile REAL
        );
        CREATE TABLE IF NOT EXISTS stock_sector_map (
            code TEXT PRIMARY KEY,
            sector TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS sector_strength (
            sector TEXT,
            trade_date TEXT,
            return_5d REAL,
            excess_return_5d REAL,
            PRIMARY KEY (sector, trade_date)
        );
        CREATE TABLE IF NOT EXISTS tracked_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            chat_id TEXT,
            code TEXT NOT NULL,
            name TEXT,
            buy_price REAL,
            quantity REAL,
            stop_loss REAL,
            target_price REAL,
            status TEXT NOT NULL DEFAULT 'active',
            opened_at TEXT NOT NULL,
            closed_at TEXT,
            last_notified_at TEXT
        );
        CREATE TABLE IF NOT EXISTS bot_events (
            event_id TEXT PRIMARY KEY,
            message_id TEXT,
            handled_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            chat_id TEXT,
            code TEXT NOT NULL,
            name TEXT,
            trigger_price REAL NOT NULL,
            direction TEXT NOT NULL DEFAULT 'below',
            quantity REAL,
            status TEXT NOT NULL DEFAULT 'active',
            note TEXT,
            created_at TEXT NOT NULL,
            closed_at TEXT,
            last_notified_at TEXT
        );
        CREATE TABLE IF NOT EXISTS alert_events (
            event_key TEXT PRIMARY KEY,
            event_type TEXT,
            code TEXT,
            title TEXT,
            sent_at TEXT NOT NULL
        );
        """)

    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    def _add_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str):
        if not self._column_exists(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

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
        cutoff = (datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=days)).strftime("%Y-%m-%d")
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
                (run_id, run_ts, code, name, action, confidence, raw_confidence, calibrated_confidence,
                 target_price, stop_loss, reasons_json, risks_json, one_liner, pushed, push_error)
                VALUES (:run_id, :run_ts, :code, :name, :action, :confidence, :raw_confidence,
                        :calibrated_confidence, :target_price, :stop_loss, :reasons_json, :risks_json,
                        :one_liner, 0, NULL)
            """, {
                "raw_confidence": dec.get("raw_confidence"),
                "calibrated_confidence": dec.get("calibrated_confidence", dec.get("confidence")),
                "run_id": run_id,
                "run_ts": run_ts,
                **dec,
            })

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

    def upsert_market_regime_history(self, rows: list[dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO market_regime_history (trade_date, vol_20d, regime, percentile)
                VALUES (:trade_date, :vol_20d, :regime, :percentile)
            """, rows)

    def get_cached_stock_sectors(self, codes: list[str], max_age_days: int = 30) -> dict[str, str]:
        if not codes:
            return {}
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        placeholders = ",".join("?" for _ in codes)
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT code, sector FROM stock_sector_map WHERE code IN ({placeholders}) AND updated_at>=?",
                [*codes, cutoff],
            ).fetchall()
        return {code: sector for code, sector in rows}

    def upsert_stock_sectors(self, mapping: dict[str, str]):
        if not mapping:
            return
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO stock_sector_map (code, sector, updated_at)
                VALUES (?, ?, ?)
            """, [(code, sector, now) for code, sector in mapping.items()])

    def upsert_sector_strength(self, rows: list[dict]):
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO sector_strength (sector, trade_date, return_5d, excess_return_5d)
                VALUES (:sector, :trade_date, :return_5d, :excess_return_5d)
            """, rows)

    def get_sector_strength(self, sector: str) -> dict | None:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM sector_strength WHERE sector=? ORDER BY trade_date DESC LIMIT 1
            """, [sector]).fetchone()
        return dict(row) if row else None

    def get_latest_calibration_model(self, action: str) -> dict | None:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM calibration_model WHERE action=?
                ORDER BY trained_at DESC, id DESC LIMIT 1
            """, [action]).fetchone()
        return dict(row) if row else None

    def insert_calibration_model(self, row: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO calibration_model (action, trained_at, sample_size, coef, intercept, auc, notes)
                VALUES (:action, :trained_at, :sample_size, :coef, :intercept, :auc, :notes)
            """, row)

    def mark_decision_resolved(self, decision_id: int, success: int | None):
        with self._conn() as conn:
            conn.execute(
                "UPDATE decisions SET resolved_at=?, success=? WHERE id=?",
                [datetime.now().isoformat(), success, decision_id],
            )

    def get_unresolved_action_decisions(self, limit: int = 500) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM decisions
                WHERE action IN ('BUY', 'SELL') AND resolved_at IS NULL
                ORDER BY run_ts ASC LIMIT ?
            """, [limit]).fetchall()
        return [dict(r) for r in rows]

    def get_calibration_samples(self, action: str) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT confidence, raw_confidence, success FROM decisions
                WHERE action=? AND success IN (0, 1)
                ORDER BY run_ts ASC
            """, [action]).fetchall()
        return [dict(r) for r in rows]

    def is_bot_event_handled(self, event_id: str) -> bool:
        if not event_id:
            return False
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM bot_events WHERE event_id=? LIMIT 1",
                [event_id],
            ).fetchone()
        return row is not None

    def mark_bot_event_handled(self, event_id: str, message_id: str = ""):
        if not event_id:
            return
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO bot_events (event_id, message_id, handled_at)
                VALUES (?, ?, ?)
            """, [event_id, message_id, datetime.now().isoformat()])

    def upsert_tracked_position(self, row: dict):
        now = datetime.now().isoformat()
        payload = {
            "user_id": row.get("user_id", ""),
            "chat_id": row.get("chat_id", ""),
            "code": row["code"],
            "name": row.get("name", row["code"]),
            "buy_price": row.get("buy_price", 0),
            "quantity": row.get("quantity"),
            "stop_loss": row.get("stop_loss", 0),
            "target_price": row.get("target_price", 0),
            "opened_at": row.get("opened_at", now),
        }
        with self._conn() as conn:
            existing = conn.execute("""
                SELECT id FROM tracked_positions
                WHERE user_id=? AND code=? AND status='active'
                ORDER BY opened_at DESC LIMIT 1
            """, [payload["user_id"], payload["code"]]).fetchone()
            if existing:
                conn.execute("""
                    UPDATE tracked_positions
                    SET chat_id=:chat_id, name=:name, buy_price=:buy_price,
                        quantity=:quantity, stop_loss=:stop_loss,
                        target_price=:target_price, opened_at=:opened_at,
                        closed_at=NULL
                    WHERE id=:id
                """, {**payload, "id": existing[0]})
            else:
                conn.execute("""
                    INSERT INTO tracked_positions
                    (user_id, chat_id, code, name, buy_price, quantity, stop_loss, target_price, status, opened_at)
                    VALUES (:user_id, :chat_id, :code, :name, :buy_price, :quantity, :stop_loss, :target_price, 'active', :opened_at)
                """, payload)

    def close_tracked_position(self, user_id: str, code: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                UPDATE tracked_positions
                SET status='closed', closed_at=?
                WHERE user_id=? AND code=? AND status='active'
            """, [datetime.now().isoformat(), user_id or "", code])
            return cur.rowcount

    def get_active_tracked_positions(self) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM tracked_positions
                WHERE status='active'
                ORDER BY opened_at ASC
            """).fetchall()
        return [dict(r) for r in rows]

    def mark_position_notified(self, position_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE tracked_positions SET last_notified_at=? WHERE id=?",
                [datetime.now().isoformat(), position_id],
            )

    def upsert_price_alert(self, row: dict):
        now = datetime.now().isoformat()
        payload = {
            "user_id": row.get("user_id", ""),
            "chat_id": row.get("chat_id", ""),
            "code": row["code"],
            "name": row.get("name", row["code"]),
            "trigger_price": row.get("trigger_price", 0),
            "direction": row.get("direction", "below"),
            "quantity": row.get("quantity"),
            "note": row.get("note", ""),
            "created_at": row.get("created_at", now),
        }
        with self._conn() as conn:
            existing = conn.execute("""
                SELECT id FROM price_alerts
                WHERE user_id=? AND code=? AND status='active'
                ORDER BY created_at DESC LIMIT 1
            """, [payload["user_id"], payload["code"]]).fetchone()
            if existing:
                conn.execute("""
                    UPDATE price_alerts
                    SET chat_id=:chat_id, name=:name, trigger_price=:trigger_price,
                        direction=:direction, quantity=:quantity, note=:note,
                        created_at=:created_at, closed_at=NULL, last_notified_at=NULL
                    WHERE id=:id
                """, {**payload, "id": existing[0]})
            else:
                conn.execute("""
                    INSERT INTO price_alerts
                    (user_id, chat_id, code, name, trigger_price, direction, quantity,
                     status, note, created_at)
                    VALUES (:user_id, :chat_id, :code, :name, :trigger_price, :direction,
                            :quantity, 'active', :note, :created_at)
                """, payload)

    def close_price_alert(self, user_id: str, code: str) -> int:
        with self._conn() as conn:
            cur = conn.execute("""
                UPDATE price_alerts
                SET status='closed', closed_at=?
                WHERE user_id=? AND code=? AND status='active'
            """, [datetime.now().isoformat(), user_id or "", code])
            return cur.rowcount

    def get_active_price_alerts(self) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM price_alerts
                WHERE status='active'
                ORDER BY created_at ASC
            """).fetchall()
        return [dict(r) for r in rows]

    def mark_price_alert_notified(self, alert_id: int):
        with self._conn() as conn:
            conn.execute(
                "UPDATE price_alerts SET last_notified_at=? WHERE id=?",
                [datetime.now().isoformat(), alert_id],
            )

    def alert_event_exists(self, event_key: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM alert_events WHERE event_key=? LIMIT 1",
                [event_key],
            ).fetchone()
        return row is not None

    def mark_alert_event(self, event_key: str, event_type: str, code: str = "", title: str = ""):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO alert_events (event_key, event_type, code, title, sent_at)
                VALUES (?, ?, ?, ?, ?)
            """, [event_key, event_type, code, title, datetime.now().isoformat()])
