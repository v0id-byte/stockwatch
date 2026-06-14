"""数据源健康记录（best-effort）。

健康记录是观测用的副作用，绝不能影响主流程：任何异常都吞掉。
用模块级缓存的 Storage，调用方无需自己持有 Storage 也能记录。
"""
import time
from contextlib import contextmanager
from functools import lru_cache

from utils.storage import Storage


@lru_cache(maxsize=1)
def _storage() -> Storage:
    return Storage()


def record_source_health(source: str, ok: bool, latency_ms: float = 0.0,
                         error: str | None = None, count: int = 0) -> None:
    try:
        _storage().upsert_source_health(source, ok, latency_ms, error, count)
    except Exception:
        pass


@contextmanager
def track(source: str, *, count_from=None):
    """计时并记录一次数据源拉取。

    用法：
        with track("行情") as h:
            result = fetch()
            h["count"] = len(result)
            h["ok"] = bool(result)   # 可选：默认未抛异常即视为成功
    抛异常 → 记为失败并把异常信息写入 last_error，然后继续抛出。
    """
    started = time.perf_counter()
    state: dict = {"ok": None, "count": 0, "error": None}
    try:
        yield state
    except Exception as exc:  # 记失败后照常抛出，不吞业务异常
        latency = (time.perf_counter() - started) * 1000
        record_source_health(source, False, latency, str(exc), 0)
        raise
    else:
        latency = (time.perf_counter() - started) * 1000
        ok = state["ok"] if state["ok"] is not None else True
        record_source_health(source, bool(ok), latency,
                             state.get("error"), int(state.get("count") or 0))
