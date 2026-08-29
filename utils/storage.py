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
                fetched_at TEXT, available_at TEXT, sentiment_model_version TEXT,
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
        # 运行级健康：status=ok/partial/failed（区别于 pushed_ok——安静但健康的一天
        # pushed_ok=0 却 status=ok）；"最近一次成功运行" 应按 status='ok' 判定。
        self._add_column(conn, "runs", "status", "status TEXT DEFAULT 'ok'")
        self._add_column(conn, "runs", "error", "error TEXT")
        # 模型信号复盘列：决策落库时记录当时的模型分数与版本，事后可对齐真实
        # 收益做 paper-monitor（此前这些值只进 LLM prompt，落库即丢）。
        self._add_column(conn, "decisions", "tech_score", "tech_score REAL")
        self._add_column(conn, "decisions", "lgbm_score", "lgbm_score REAL")
        self._add_column(conn, "decisions", "risk_score", "risk_score REAL")
        self._add_column(conn, "decisions", "lgbm_context", "lgbm_context TEXT")
        self._add_column(conn, "decisions", "alpha_model_version", "alpha_model_version TEXT")
        self._add_column(conn, "decisions", "risk_model_version", "risk_model_version TEXT")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS model_scores (
            trade_date TEXT NOT NULL,
            code TEXT NOT NULL,
            alpha_model_version TEXT,
            alpha_score REAL,
            risk_model_version TEXT,
            risk_score REAL,
            feature_contract_version TEXT,
            reference_universe_sha256 TEXT,
            scored_at TEXT NOT NULL,
            PRIMARY KEY (trade_date, code)
        );
        """)
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
        CREATE TABLE IF NOT EXISTS alert_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            source TEXT,
            code TEXT,
            label TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS announcements (
            source TEXT NOT NULL,
            announcement_id TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT,
            title TEXT,
            published_at TEXT,
            url TEXT,
            org_id TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (source, announcement_id)
        );
        CREATE INDEX IF NOT EXISTS idx_announcements_code_published
            ON announcements(code, published_at);
        CREATE TABLE IF NOT EXISTS announcement_fetch_progress (
            code TEXT NOT NULL,
            chunk_start TEXT NOT NULL,
            chunk_end TEXT NOT NULL,
            status TEXT NOT NULL,
            attempts INTEGER DEFAULT 0,
            fetched_count INTEGER DEFAULT 0,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (code, chunk_start, chunk_end)
        );
        CREATE INDEX IF NOT EXISTS idx_announcement_progress_status
            ON announcement_fetch_progress(status);
        CREATE TABLE IF NOT EXISTS source_health (
            source TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_ok_at TEXT,
            last_error_at TEXT,
            last_error TEXT,
            last_latency_ms REAL,
            last_count INTEGER,
            ok_count INTEGER DEFAULT 0,
            fail_count INTEGER DEFAULT 0,
            updated_at TEXT NOT NULL
        );
        -- 盘中监控（monitor）在 NOTIFY_CHANNEL=web 时没有飞书推送面，把触发的
        -- 盯价/卖压/重大消息提醒落到这张表，首页「盘中提醒」feed 渲染。event_key
        -- 唯一以去重（与 alert_events 同款 key），INSERT OR IGNORE 保证幂等。
        CREATE TABLE IF NOT EXISTS web_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_key TEXT UNIQUE,
            category TEXT,
            level TEXT,
            code TEXT,
            name TEXT,
            title TEXT,
            body TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_web_alerts_created ON web_alerts(created_at);
        """)
        self._add_column(conn, "price_alerts", "direction", "direction TEXT NOT NULL DEFAULT 'below'")
        self._add_column(conn, "news", "fetched_at", "fetched_at TEXT")
        self._add_column(conn, "news", "available_at", "available_at TEXT")
        self._add_column(conn, "news", "sentiment_model_version", "sentiment_model_version TEXT")

    @staticmethod
    def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row[1] == column for row in rows)

    def _add_column(self, conn: sqlite3.Connection, table: str, column: str, ddl: str):
        if not self._column_exists(conn, table, column):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30, check_same_thread=False)
        # 守护进程（runner/monitor）与 Web 控制台是两个独立进程共用同一个 SQLite，
        # 又各自多线程访问。默认 rollback-journal 模式下并发写极易 "database is locked"。
        # 开 WAL 让读写不互斥，busy_timeout 让偶发写锁自动等待而非立刻报错。
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

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
        now = datetime.now().isoformat(timespec="seconds")
        rows = []
        for item in items:
            fetched_at = item.get("fetched_at") or now
            rows.append({
                "code": code,
                "title": item.get("title"),
                "content": item.get("content"),
                "source": item.get("source"),
                "ts": item.get("ts"),
                "fetched_at": fetched_at,
                "available_at": item.get("available_at") or fetched_at,
                "sentiment_model_version": item.get("sentiment_model_version"),
            })
        with self._conn() as conn:
            conn.executemany("""
                INSERT OR IGNORE INTO news
                (code, title, content, source, ts, sentiment_score,
                 fetched_at, available_at, sentiment_model_version)
                VALUES (:code, :title, :content, :source, :ts, NULL,
                        :fetched_at, :available_at, :sentiment_model_version)
            """, rows)

    def get_news_since(self, code: str, days: int = 7) -> list[dict]:
        cutoff = (datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=days)).strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM news WHERE code=? AND ts>=? ORDER BY ts DESC",
                [code, cutoff]
            ).fetchall()
        return [dict(r) for r in rows]

    def update_news_sentiment(self, code: str, title: str, ts: str, score: float,
                              model_version: str | None = None):
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE news
                SET sentiment_score=?,
                    sentiment_model_version=COALESCE(?, sentiment_model_version)
                WHERE code=? AND title=? AND ts=?
                """,
                [score, model_version, code, title, ts]
            )

    def upsert_announcements(self, rows: list[dict]):
        if not rows:
            return
        now = datetime.now().isoformat(timespec="seconds")
        payload = []
        for row in rows:
            announcement_id = row.get("announcement_id")
            raw_code = str(row.get("code") or "").strip()
            if not announcement_id or not raw_code:
                continue
            code = raw_code.zfill(6)
            payload.append({
                "source": row.get("source") or "cninfo",
                "announcement_id": str(announcement_id),
                "code": code,
                "name": row.get("name"),
                "title": row.get("title"),
                "published_at": row.get("published_at"),
                "url": row.get("url"),
                "org_id": row.get("org_id"),
                "fetched_at": row.get("fetched_at") or now,
            })
        if not payload:
            return
        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO announcements
                (source, announcement_id, code, name, title, published_at, url, org_id, fetched_at)
                VALUES (:source, :announcement_id, :code, :name, :title, :published_at, :url, :org_id, :fetched_at)
                ON CONFLICT(source, announcement_id) DO UPDATE SET
                    code=excluded.code,
                    name=excluded.name,
                    title=excluded.title,
                    published_at=excluded.published_at,
                    url=excluded.url,
                    org_id=excluded.org_id,
                    fetched_at=excluded.fetched_at
            """, payload)

    def get_announcements(self, codes: list[str] | None = None,
                          start: str | None = None,
                          end: str | None = None) -> list[dict]:
        clauses = []
        params: list[Any] = []
        if codes is not None:
            codes = [str(code).strip().zfill(6) for code in codes if str(code).strip()]
            if not codes:
                return []
            placeholders = ",".join("?" for _ in codes)
            clauses.append(f"code IN ({placeholders})")
            params.extend(codes)
        if start:
            clauses.append("published_at >= ?")
            params.append(start)
        if end:
            clauses.append("published_at <= ?")
            params.append(end)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"""
                SELECT * FROM announcements
                {where}
                ORDER BY published_at ASC, code ASC
            """, params).fetchall()
        return [dict(r) for r in rows]

    def get_announcement_progress(self, codes: list[str] | None = None) -> list[dict]:
        params: list[Any] = []
        where = ""
        if codes is not None:
            codes = [str(code).strip().zfill(6) for code in codes if str(code).strip()]
            if not codes:
                return []
            placeholders = ",".join("?" for _ in codes)
            where = f"WHERE code IN ({placeholders})"
            params.extend(codes)
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"""
                SELECT * FROM announcement_fetch_progress
                {where}
            """, params).fetchall()
        return [dict(r) for r in rows]

    def mark_announcement_chunk(self, code: str, chunk_start: str, chunk_end: str,
                                status: str, attempts: int, fetched_count: int = 0,
                                error: str | None = None):
        now = datetime.now().isoformat(timespec="seconds")
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO announcement_fetch_progress
                (code, chunk_start, chunk_end, status, attempts, fetched_count, error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code, chunk_start, chunk_end) DO UPDATE SET
                    status=excluded.status,
                    attempts=excluded.attempts,
                    fetched_count=excluded.fetched_count,
                    error=excluded.error,
                    updated_at=excluded.updated_at
            """, [
                str(code).strip().zfill(6), chunk_start, chunk_end, status,
                attempts, fetched_count, error, now,
            ])

    def insert_decision(self, run_id: str, run_ts: str, dec: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO decisions
                (run_id, run_ts, code, name, action, confidence, raw_confidence, calibrated_confidence,
                 target_price, stop_loss, reasons_json, risks_json, one_liner, pushed, push_error,
                 tech_score, lgbm_score, risk_score, lgbm_context,
                 alpha_model_version, risk_model_version)
                VALUES (:run_id, :run_ts, :code, :name, :action, :confidence, :raw_confidence,
                        :calibrated_confidence, :target_price, :stop_loss, :reasons_json, :risks_json,
                        :one_liner, 0, NULL,
                        :tech_score, :lgbm_score, :risk_score, :lgbm_context,
                        :alpha_model_version, :risk_model_version)
            """, {
                "raw_confidence": dec.get("raw_confidence"),
                "calibrated_confidence": dec.get("calibrated_confidence", dec.get("confidence")),
                "tech_score": dec.get("tech_score"),
                "lgbm_score": dec.get("lgbm_score"),
                "risk_score": dec.get("risk_score"),
                "lgbm_context": dec.get("lgbm_context"),
                "alpha_model_version": dec.get("alpha_model_version"),
                "risk_model_version": dec.get("risk_model_version"),
                "run_id": run_id,
                "run_ts": run_ts,
                **dec,
            })

    def upsert_model_scores(self, rows: list[dict]):
        """Nightly batch scores over the reference universe (one row per stock)."""
        with self._conn() as conn:
            conn.executemany("""
                INSERT INTO model_scores
                (trade_date, code, alpha_model_version, alpha_score,
                 risk_model_version, risk_score, feature_contract_version,
                 reference_universe_sha256, scored_at)
                VALUES (:trade_date, :code, :alpha_model_version, :alpha_score,
                        :risk_model_version, :risk_score, :feature_contract_version,
                        :reference_universe_sha256, :scored_at)
                ON CONFLICT(trade_date, code) DO UPDATE SET
                    alpha_model_version=excluded.alpha_model_version,
                    alpha_score=excluded.alpha_score,
                    risk_model_version=excluded.risk_model_version,
                    risk_score=excluded.risk_score,
                    feature_contract_version=excluded.feature_contract_version,
                    reference_universe_sha256=excluded.reference_universe_sha256,
                    scored_at=excluded.scored_at
            """, rows)

    def get_latest_model_scores(self, codes: list[str]) -> dict[str, dict]:
        """Latest scored trade date's rows for the requested codes."""
        if not codes:
            return {}
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            latest = conn.execute("SELECT MAX(trade_date) FROM model_scores").fetchone()[0]
            if not latest:
                return {}
            marks = ",".join("?" for _ in codes)
            rows = conn.execute(
                f"SELECT * FROM model_scores WHERE trade_date=? AND code IN ({marks})",
                [latest, *codes],
            ).fetchall()
        return {row["code"]: dict(row) for row in rows}

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
        payload = {
            "run_id": run_id,
            "run_ts": datetime.now().isoformat(),
            "status": "ok",
            "error": None,
            **stats,
        }
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs
                (run_id, run_ts, stocks_analyzed, llm_calls, tokens_used, pushed_count, pushed_ok, status, error)
                VALUES (:run_id, :run_ts, :stocks_analyzed, :llm_calls, :tokens_used, :pushed_count, :pushed_ok, :status, :error)
            """, payload)

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY run_ts DESC LIMIT ?", [limit]
            ).fetchall()
        return [dict(r) for r in rows]

    def get_last_successful_run_ts(self) -> str | None:
        """最近一次 data-fetch 成功的运行时间。按 status='ok' 判定，而非 pushed_ok——
        安静但健康的一天没有可推送信号（pushed_ok=0）依然是一次成功运行。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT run_ts FROM runs WHERE status='ok' ORDER BY run_ts DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def upsert_source_health(self, source: str, ok: bool, latency_ms: float = 0.0,
                             error: str | None = None, count: int = 0):
        """记录单个数据源一次拉取的健康状况（成功/失败/延迟/条数）。供首页健康横幅展示。"""
        now = datetime.now().isoformat(timespec="seconds")
        row = {
            "source": source,
            "status": "ok" if ok else "error",
            "last_ok_at": now if ok else None,
            "last_error_at": None if ok else now,
            "last_error": None if ok else (str(error)[:300] if error else "未知错误"),
            "last_latency_ms": round(float(latency_ms), 1),
            "last_count": int(count or 0),
            "ok_inc": 1 if ok else 0,
            "fail_inc": 0 if ok else 1,
            "updated_at": now,
        }
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO source_health
                (source, status, last_ok_at, last_error_at, last_error, last_latency_ms,
                 last_count, ok_count, fail_count, updated_at)
                VALUES (:source, :status, :last_ok_at, :last_error_at, :last_error, :last_latency_ms,
                        :last_count, :ok_inc, :fail_inc, :updated_at)
                ON CONFLICT(source) DO UPDATE SET
                    status=excluded.status,
                    last_ok_at=COALESCE(excluded.last_ok_at, source_health.last_ok_at),
                    last_error_at=COALESCE(excluded.last_error_at, source_health.last_error_at),
                    last_error=COALESCE(excluded.last_error, source_health.last_error),
                    last_latency_ms=excluded.last_latency_ms,
                    last_count=excluded.last_count,
                    ok_count=source_health.ok_count + excluded.ok_count,
                    fail_count=source_health.fail_count + excluded.fail_count,
                    updated_at=excluded.updated_at
            """, row)

    def get_source_health(self) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM source_health ORDER BY source").fetchall()
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

    def get_all_stock_sectors(self, max_age_days: int = 45) -> dict[str, str]:
        """Whole cached code->sector map (used for event sector-propagation)."""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT code, sector FROM stock_sector_map WHERE updated_at>=?", [cutoff]
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
                WHERE user_id=? AND code=? AND direction=? AND status='active'
                ORDER BY created_at DESC LIMIT 1
            """, [payload["user_id"], payload["code"], payload["direction"]]).fetchone()
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

    def record_web_alert(self, event_key: str, category: str, level: str,
                         code: str, name: str, title: str, body: str):
        """盘中监控在 web 模式下把一条触发的提醒写入首页「盘中提醒」feed。
        event_key 唯一去重，重复触发静默忽略。"""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO web_alerts
                (event_key, category, level, code, name, title, body, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [event_key, category, level, code, name, title, body,
                  datetime.now().isoformat()])

    def get_recent_web_alerts(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM web_alerts ORDER BY created_at DESC, id DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [dict(r) for r in rows]

    def insert_alert_feedback(self, row: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO alert_feedback (user_id, source, code, label, note, created_at)
                VALUES (:user_id, :source, :code, :label, :note, :created_at)
            """, {
                "user_id": row.get("user_id", ""),
                "source": row.get("source", ""),
                "code": row.get("code", ""),
                "label": row["label"],
                "note": row.get("note", ""),
                "created_at": row.get("created_at", datetime.now().isoformat()),
            })

    def get_recent_alert_feedback(self, limit: int = 30) -> list[dict]:
        with self._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM alert_feedback
                ORDER BY created_at DESC, id DESC LIMIT ?
            """, [limit]).fetchall()
        return [dict(r) for r in rows]

    def get_misreported_codes(self, days: int = 30, min_count: int = 2) -> dict[str, dict]:
        """近 days 天内被标记「误报」占多数的个股，用于回灌推送门槛。
        返回 {code: {"bad": n, "good": m}}，仅含 误报>=min_count 且 误报>有用 的代码。"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT code,
                       SUM(CASE WHEN label='误报' THEN 1 ELSE 0 END) AS bad,
                       SUM(CASE WHEN label='有用' THEN 1 ELSE 0 END) AS good
                FROM alert_feedback
                WHERE code IS NOT NULL AND code != '' AND created_at >= ?
                GROUP BY code
            """, [cutoff]).fetchall()
        result = {}
        for code, bad, good in rows:
            bad, good = int(bad or 0), int(good or 0)
            if bad >= min_count and bad > good:
                result[str(code)] = {"bad": bad, "good": good}
        return result
