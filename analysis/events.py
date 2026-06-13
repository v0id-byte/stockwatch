"""Structured corporate-event layer — RISK / CONTEXT, not return-prediction alpha.

A PIT event study on this project's own data (see scripts/event_study and the README)
found that public A-share events have NO clean tradeable post-event abnormal return —
they are priced in by the announcement date (earnings preannouncements and lockup
expiries are null; insider buying is only a faint, fast-fading +0.8% over 5 days). So
this module deliberately does NOT predict price moves. It surfaces events as risk
heads-ups and honest context for alerts and Q&A:

  - 解禁 (lockup expiry): forward-looking calendar — historically raises volatility.
  - 业绩预告 (earnings preannounce): 首亏/预减/续亏 = risk flag; 预增/扭亏 = info
    (with the honest note that the public reaction is mostly already priced in).
  - 增减持 (insider trades): 减持 = caution; 增持 = mild positive context.
  - 回购 (buyback plan): mild positive context.

All fetches are cached per day and fail soft (a network error just drops that event
source for the run). Output is plain-language context, never a buy/sell instruction.
"""
from __future__ import annotations

from datetime import date, timedelta

from loguru import logger

_CACHE: dict[str, dict] = {}

LOCKUP_WARN_PCT = 3.0          # 解禁占流通市值比例阈值（%）
LOOKBACK_DAYS = 30             # 近期事件回看
LOOKAHEAD_DAYS = 45            # 解禁日历前瞻

_POS_YJYG = {"预增", "略增", "扭亏", "减亏"}
_NEG_YJYG = {"预减", "首亏", "增亏", "略减", "续亏", "续盈"}


def _today() -> date:
    # date.today() avoided in some sandboxes; loguru-safe wrapper
    from datetime import datetime
    return datetime.now().date()


def _cache_for(key: str, builder):
    day = _today().isoformat()
    bucket = _CACHE.get(day)
    if bucket is None:
        _CACHE.clear()
        bucket = _CACHE[day] = {}
    if key not in bucket:
        try:
            bucket[key] = builder()
        except Exception as e:
            logger.debug(f"事件源 {key} 获取失败，跳过: {e}")
            bucket[key] = None
    return bucket[key]


def _fmt_pct(value) -> str:
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "?"


def _lockups():
    import akshare as ak
    import pandas as pd
    today = _today()
    start = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end = (today + timedelta(days=LOOKAHEAD_DAYS)).strftime("%Y%m%d")
    df = ak.stock_restricted_release_detail_em(start_date=start, end_date=end)
    df = df.rename(columns={"股票代码": "code", "解禁时间": "rdate",
                            "占解禁前流通市值比例": "pct", "限售股类型": "rtype"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["rdate"] = pd.to_datetime(df["rdate"], errors="coerce")
    df["pct"] = pd.to_numeric(df["pct"], errors="coerce") * 100  # 比例->百分数
    return df.dropna(subset=["code", "rdate"])


def _preannounce():
    import akshare as ak
    import pandas as pd
    today = _today()
    # most recent 1-2 report periods
    periods = []
    for year in (today.year, today.year - 1):
        for md in ("1231", "0930", "0630", "0331"):
            d = date(year, int(md[:2]), int(md[2:]))
            if d <= today:
                periods.append(f"{year}{md}")
    frames = []
    for p in periods[:2]:
        try:
            x = ak.stock_yjyg_em(date=p)
            x = x[x["预测指标"].astype(str).str.contains("归属于上市公司股东的净利润", na=False)]
            frames.append(x.rename(columns={"股票代码": "code", "预告类型": "etype", "公告日期": "adate"}))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["adate"] = pd.to_datetime(df["adate"], errors="coerce")
    return df.dropna(subset=["code", "adate", "etype"]).sort_values("adate").drop_duplicates("code", keep="last")


def _insider():
    import akshare as ak
    import pandas as pd
    df = ak.stock_hold_management_detail_em()
    df = df.rename(columns={"日期": "adate", "代码": "code", "变动股数": "shares", "变动比例": "pct"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["adate"] = pd.to_datetime(df["adate"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    cutoff = pd.Timestamp(_today() - timedelta(days=LOOKBACK_DAYS))
    return df.dropna(subset=["code", "adate", "shares"])[lambda d: d["adate"] >= cutoff]


def _buybacks():
    import akshare as ak
    import pandas as pd
    df = ak.stock_repurchase_em()
    df = df.rename(columns={"股票代码": "code", "最新公告日期": "adate", "实施进度": "stage"})
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["adate"] = pd.to_datetime(df["adate"], errors="coerce")
    cutoff = pd.Timestamp(_today() - timedelta(days=LOOKBACK_DAYS))
    return df.dropna(subset=["code", "adate"])[lambda d: d["adate"] >= cutoff]


def collect_events(codes: list[str]) -> dict[str, list[dict]]:
    """Return {code: [event, ...]}. Each event: {type, date, level, note}."""
    import pandas as pd
    wanted = {str(c).zfill(6) for c in codes}
    out: dict[str, list[dict]] = {c: [] for c in wanted}
    today = pd.Timestamp(_today())

    lk = _cache_for("lockup", _lockups)
    if lk is not None:
        for r in lk[lk["code"].isin(wanted)].itertuples():
            upcoming = r.rdate >= today
            big = pd.notna(r.pct) and r.pct >= LOCKUP_WARN_PCT
            when = "未来" if upcoming else "近期"
            level = "warning" if (upcoming and big) else "info"
            out[r.code].append({
                "type": "解禁", "date": r.rdate.date().isoformat(), "level": level,
                "note": f"{when}{r.rdate.date()}解禁，占流通约{_fmt_pct(r.pct)}（历史上解禁前后波动加大，非涨跌预测）",
            })

    pa = _cache_for("preannounce", _preannounce)
    if pa is not None:
        for r in pa[pa["code"].isin(wanted)].itertuples():
            neg = r.etype in _NEG_YJYG
            level = "warning" if neg else "info"
            tail = "属风险信号" if neg else "公开反应多已被price-in"
            out[r.code].append({
                "type": "业绩预告", "date": r.adate.date().isoformat(), "level": level,
                "note": f"业绩预告：{r.etype}（{r.adate.date()}，{tail}）",
            })

    ins = _cache_for("insider", _insider)
    if ins is not None and len(ins):
        agg = ins[ins["code"].isin(wanted)].groupby("code")["shares"].sum()
        for code, net in agg.items():
            if net > 0:
                out[code].append({"type": "增持", "date": "", "level": "info",
                                  "note": "近期董监高/股东净增持（短期略偏积极，影响有限）"})
            elif net < 0:
                out[code].append({"type": "减持", "date": "", "level": "warning",
                                  "note": "近期董监高/股东净减持（资金面偏谨慎）"})

    bb = _cache_for("buyback", _buybacks)
    if bb is not None:
        for code in bb[bb["code"].isin(wanted)]["code"].unique():
            out[code].append({"type": "回购", "date": "", "level": "info",
                              "note": "近期发布回购方案（管理层信心信号）"})
    return out


def format_events_context(events: list[dict]) -> str:
    """Plain-language one-block summary for alert / Q&A context. Empty if no events."""
    if not events:
        return ""
    order = {"warning": 0, "info": 1}
    events = sorted(events, key=lambda e: order.get(e.get("level"), 2))
    lines = [f"  - {e['note']}" for e in events[:6]]
    return "事件提醒（风险/情境，非买卖指令）:\n" + "\n".join(lines)
