#!/usr/bin/env python3
"""Score announcement titles/full texts with MiniMax-M3 per docs/llm_scoring_spec_v1.md.

Credentials come ONLY from ``MINIMAX_API_KEY`` / ``MINIMAX_BASE_URL`` (optionally
loaded from ~/.stockwatch/research.env).  Never write them anywhere.

Scores are cached permanently in sqlite keyed by
``(prompt_version, model_id, content_sha256)``; a provider-side model update
never triggers re-scoring of already-scored history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT_VERSION = "ann_score_v1"
MODEL_ID = "MiniMax-M3"
PARSED_SCHEMA_VERSION = 1
MAX_TOKENS = 2000  # thinking model: budget covers reasoning + the small JSON
FULLTEXT_CHAR_LIMIT = 6000

EVENT_TYPES = {
    "减持", "增持", "回购", "业绩预告", "业绩快报", "定期报告", "问询监管", "处罚立案",
    "诉讼仲裁", "担保质押", "解禁限售", "重大合同", "资产运作", "退市风险", "停复牌",
    "更正致歉", "人事变动", "其他",
}
HORIZONS = {"short", "medium", "long"}

# Frozen prefilter vocabulary — mirror of docs/llm_scoring_spec_v1.md §2.
PREFILTER_TERMS = (
    "减持 增持 回购 业绩预告 业绩快报 预增 预减 扭亏 首亏 续亏 预盈 预亏 "
    "问询 关注函 监管函 警示函 立案 处罚 诉讼 仲裁 担保 冻结 质押 解押 "
    "限售 解禁 重大合同 中标 资产重组 收购 出售 定增 配股 可转债 "
    "退市 风险警示 ST 摘帽 停牌 复牌 商誉 减值 违约 破产 清算 占用 "
    "辞职 变更 更正 致歉"
).split()

SYSTEM_PROMPT = (
    "你是A股公告分析员。仅根据给出的公告标题（或正文摘录）判断该公告对上市公司股东价值的"
    "方向与量级。只输出一个JSON对象，不要输出任何其他文字。不知道或无法判断时"
    "direction=0、severity=0、is_substantive=false。字段与取值必须严格符合给定枚举。"
)

USER_TEMPLATE = (
    "公告标题：{title}\n{body_block}"
    "输出JSON字段：event_type(减持|增持|回购|业绩预告|业绩快报|定期报告|问询监管|处罚立案|"
    "诉讼仲裁|担保质押|解禁限售|重大合同|资产运作|退市风险|停复牌|更正致歉|人事变动|其他), "
    "direction(-2..2), severity(0..3), horizon(short|medium|long), "
    "is_substantive(true|false), confidence(0..1)"
)


def _parse_args() -> argparse.Namespace:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("title", "fulltext"), required=True)
    parser.add_argument("--announcements-db", default=str(Path("~/.stockwatch/db.sqlite").expanduser()))
    parser.add_argument("--event-library", default=str(root / "announcement_event_library.sqlite"))
    parser.add_argument("--documents-dir", default=str(root / "announcement_documents"))
    parser.add_argument("--universe", default=str(root / "pit_universe_daily.parquet"))
    parser.add_argument("--cache", default=str(root / "announcement_llm_scores.sqlite"))
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--limit", type=int, default=0, help="score at most N new items (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="only report corpus/cache counts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--audit-sample", type=int, default=0,
                        help="export a stratified sample CSV of cached scores and exit")
    parser.add_argument("--repeat-check", type=int, default=0,
                        help="re-score N cached items and report agreement, without touching the cache")
    return parser.parse_args()


def _load_research_env() -> None:
    env_file = Path("~/.stockwatch/research.env").expanduser()
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


def _client():
    import anthropic

    api_key = os.environ.get("MINIMAX_API_KEY")
    base_url = os.environ.get("MINIMAX_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("set MINIMAX_API_KEY and MINIMAX_BASE_URL (or ~/.stockwatch/research.env)")
    return anthropic.Anthropic(api_key=api_key, base_url=base_url)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _open_cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS scores (
            prompt_version TEXT NOT NULL,
            model_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            tier TEXT NOT NULL,
            code TEXT NOT NULL,
            published_at TEXT NOT NULL,
            publication_time_quality TEXT NOT NULL,
            title TEXT NOT NULL,
            event_type TEXT, direction INTEGER, severity INTEGER, horizon TEXT,
            is_substantive INTEGER, confidence REAL,
            parse_status TEXT NOT NULL,
            parsed_schema_version INTEGER NOT NULL,
            raw_response_sha256 TEXT,
            api_model_identifier TEXT,
            usage_input_tokens INTEGER, usage_output_tokens INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            scored_at TEXT NOT NULL,
            PRIMARY KEY (prompt_version, model_id, content_sha256)
        )"""
    )
    conn.commit()
    return conn


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in response")
    return json.loads(text[start:end + 1])


def _validate(payload: dict) -> dict:
    event_type = str(payload["event_type"])
    if event_type not in EVENT_TYPES:
        raise ValueError(f"bad event_type {event_type!r}")
    direction = int(payload["direction"])
    severity = int(payload["severity"])
    horizon = str(payload["horizon"])
    if direction not in (-2, -1, 0, 1, 2) or severity not in (0, 1, 2, 3) or horizon not in HORIZONS:
        raise ValueError("enum out of range")
    return {
        "event_type": event_type,
        "direction": direction,
        "severity": severity,
        "horizon": horizon,
        "is_substantive": 1 if bool(payload["is_substantive"]) else 0,
        "confidence": max(0.0, min(1.0, float(payload["confidence"]))),
    }


def _score_one(client, item: dict) -> dict:
    body_block = ""
    if item.get("body"):
        body_block = f"正文摘录：{item['body'][:FULLTEXT_CHAR_LIMIT]}\n"
    user = USER_TEMPLATE.format(title=item["title"], body_block=body_block)
    retries = 0
    last_error = None
    for attempt in range(6):
        try:
            message = client.messages.create(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                # anthropic>=1.2 dropped the top-level temperature kwarg; the
                # MiniMax endpoint still honors it through the request body.
                extra_body={"temperature": 0},
            )
            text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
            parsed = _validate(_extract_json(text))
            usage = getattr(message, "usage", None)
            return {
                **item, **parsed,
                "parse_status": "ok",
                "raw_response_sha256": _sha256(text),
                "api_model_identifier": getattr(message, "model", MODEL_ID),
                "usage_input_tokens": getattr(usage, "input_tokens", None),
                "usage_output_tokens": getattr(usage, "output_tokens", None),
                "retry_count": retries,
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retries += 1
            if "RateLimit" in type(exc).__name__ or "429" in str(exc):
                time.sleep(min(10 * (attempt + 1), 60))
            else:
                time.sleep(min(2 ** attempt, 8))
    return {
        **item,
        "event_type": None, "direction": None, "severity": None, "horizon": None,
        "is_substantive": None, "confidence": None,
        "parse_status": f"failed:{type(last_error).__name__}",
        "raw_response_sha256": None, "api_model_identifier": None,
        "usage_input_tokens": None, "usage_output_tokens": None,
        "retry_count": retries,
    }


def _title_corpus(args: argparse.Namespace) -> list[dict]:
    import pandas as pd

    universe = pd.read_parquet(args.universe, columns=["code", "is_member"])
    codes = set(universe.loc[universe["is_member"], "code"].unique())
    conn = sqlite3.connect(args.announcements_db)
    try:
        frame = pd.read_sql_query(
            "SELECT code, title, published_at FROM announcements WHERE published_at >= ?",
            conn, params=(args.start,),
        )
    finally:
        conn.close()
    frame["code"] = frame["code"].astype(str).str.extract(r"(\d{6})")[0]
    frame = frame.dropna(subset=["code", "title"])
    frame = frame[frame["code"].isin(codes)]
    pattern = "|".join(map(re.escape, PREFILTER_TERMS))
    frame = frame[frame["title"].str.contains(pattern, regex=True, na=False)]
    # De-duplicate by normalized title content (announcement titles are highly
    # templated): one score per distinct normalized title per code.
    frame["norm_title"] = frame["title"].str.replace(r"\s+", "", regex=True)
    frame = frame.drop_duplicates(["code", "norm_title"])
    items = []
    for _, row in frame.iterrows():
        items.append({
            "tier": "title",
            "code": row["code"],
            "title": str(row["title"]).strip(),
            "published_at": str(row["published_at"]),
            "publication_time_quality": "DATE_ONLY"
            if str(row["published_at"]).endswith("00:00:00") or len(str(row["published_at"])) <= 10
            else "EXACT_TIMESTAMP",
            "body": None,
            "content_sha256": _sha256(f"title|{row['code']}|{row['norm_title']}"),
        })
    return items


def _fulltext_corpus(args: argparse.Namespace) -> list[dict]:
    import gzip

    conn = sqlite3.connect(args.event_library)
    try:
        rows = conn.execute(
            "SELECT code, title, published_at, text_path FROM documents WHERE status='done'"
        ).fetchall()
    finally:
        conn.close()
    items = []
    for code, title, published_at, text_path in rows:
        path = Path(args.documents_dir).parent / text_path if not Path(text_path).is_absolute() else Path(text_path)
        if not path.exists():
            continue
        if path.suffix == ".gz":
            body = gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")
        else:
            body = path.read_text(encoding="utf-8", errors="replace")
        body = body[:FULLTEXT_CHAR_LIMIT]
        code = re.search(r"(\d{6})", str(code)).group(1)
        items.append({
            "tier": "fulltext",
            "code": code,
            "title": str(title).strip(),
            "published_at": str(published_at),
            "publication_time_quality": "DATE_ONLY",
            "body": body,
            "content_sha256": _sha256(f"fulltext|{code}|{_sha256(body)}"),
        })
    return items


def main() -> None:
    args = _parse_args()
    _load_research_env()
    cache = _open_cache(Path(args.cache).expanduser())

    if args.audit_sample:
        import pandas as pd

        frame = pd.read_sql_query(
            "SELECT * FROM scores WHERE parse_status='ok'", cache)
        sample = (
            frame.groupby(["event_type", "direction"], group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, args.audit_sample // 20)), random_state=7))
            .head(args.audit_sample)
        )
        out = Path(args.cache).with_suffix(".audit_sample.csv")
        sample[["code", "published_at", "title", "event_type", "direction",
                "severity", "horizon", "is_substantive", "confidence"]].to_csv(out, index=False)
        print(json.dumps({"audit_sample": str(out), "rows": len(sample)}, ensure_ascii=False))
        return

    items = _title_corpus(args) if args.tier == "title" else _fulltext_corpus(args)
    cached = {row[0] for row in cache.execute(
        "SELECT content_sha256 FROM scores WHERE prompt_version=? AND model_id=? AND parse_status='ok'",
        (PROMPT_VERSION, MODEL_ID))}
    todo = [item for item in items if item["content_sha256"] not in cached]
    summary = {
        "tier": args.tier, "corpus": len(items), "already_cached": len(items) - len(todo),
        "to_score": len(todo),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False))
        return
    if args.limit:
        todo = todo[: args.limit]

    client = _client()
    lock = threading.Lock()
    done = 0
    usage_in = usage_out = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_score_one, client, item): item for item in todo}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                cache.execute(
                    """INSERT OR REPLACE INTO scores VALUES
                       (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        PROMPT_VERSION, MODEL_ID, result["content_sha256"], result["tier"],
                        result["code"], result["published_at"], result["publication_time_quality"],
                        result["title"], result["event_type"], result["direction"],
                        result["severity"], result["horizon"], result["is_substantive"],
                        result["confidence"], result["parse_status"], PARSED_SCHEMA_VERSION,
                        result["raw_response_sha256"], result["api_model_identifier"],
                        result["usage_input_tokens"], result["usage_output_tokens"],
                        result["retry_count"],
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                    ),
                )
                cache.commit()
                done += 1
                usage_in += result["usage_input_tokens"] or 0
                usage_out += result["usage_output_tokens"] or 0
                if done % 50 == 0:
                    rate = done / max(time.time() - started, 1)
                    print(f"scored {done}/{len(todo)} rate={rate:.2f}/s "
                          f"in_tok={usage_in} out_tok={usage_out}", flush=True)
    elapsed = time.time() - started
    ok = cache.execute(
        "SELECT COUNT(*) FROM scores WHERE parse_status='ok' AND prompt_version=?",
        (PROMPT_VERSION,)).fetchone()[0]
    print(json.dumps({
        **summary, "scored_now": done, "elapsed_s": round(elapsed, 1),
        "usage_input_tokens": usage_in, "usage_output_tokens": usage_out,
        "cache_ok_total": ok,
        "per_item_tokens": {
            "input": round(usage_in / max(done, 1)), "output": round(usage_out / max(done, 1)),
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
