"""财报解析：按用户选择的维度抓取 A 股财报数据并交给 LLM 解读。

合规与免责声明：
- 本模块仅做公开财报数据的二次整理与摘要，不构成投资建议。
- 数据源为 AKShare 公开接口（东方财富 / 巨潮资讯），存在延迟与口径差异，
  最终数据请以公司在巨潮资讯网披露的原文为准。
- LLM 解读受模型与 prompt 限制，不能替代审计报告或独立分析；涉及重大决策请
  自行核对原文并咨询持牌专业人士。
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

import akshare as ak

from bot.research import (
    StockRef,
    _ak_call,
    _display_text,
    _market_secid,
    _safe_float,
)
from utils.llm import get_llm_client


# ============ 选项菜单（4 个维度，编号 1-N 或字母 a-z） ============

REPORT_TRIGGER_KEYWORDS = ("财报解析", "财报", "财务报告", "年报解析", "看财报")


CONTENT_OPTIONS: list[tuple[str, str]] = [
    ("业绩快报/业绩预告（最新一期）", "快速看最新一期的主营收入、归母净利、扣非净利、每股收益等关键数"),
    ("资产负债表（最新一期）", "货币资金、应收账款、存货、总资产、负债结构、股东权益"),
    ("利润表（最新一期）", "营业总收入、营业总成本、销售/管理/财务费用、营业利润、净利润"),
    ("现金流量表（最新一期）", "经营/投资/筹资活动现金流净额与结构"),
    ("关键财务指标（最新一期）", "ROE、毛利率、净利率、资产负债率、每股收益、每股净资产"),
    ("全部（三大表 + 关键指标）", "上述 1-5 项汇总"),
    ("最近 4 期趋势（横截面对比）", "用于观察同比/环比变化，默认 1-5 的全部指标"),
]

FOCUS_OPTIONS: list[tuple[str, str]] = [
    ("同比/环比变化", "和去年同期 / 上期对比，标注同比、环比变化方向与幅度"),
    ("与公司历史均值对比", "和最近 3-5 年同口径历史均值对比，看当前处于什么分位"),
    ("同行业横向对比（参考）", "和申万二级行业均值做参考对比，数据不可得时跳过"),
    ("异常项/重大变动", "标记同比变动超过 ±30% 或绝对值异常的科目，并解释可能原因"),
    ("趋势分析（最近 4 期）", "按时间序列列出最近 4 期关键指标，看拐点和加速度"),
    ("经营讨论要点（年报 MD&A）", "从巨潮年报原文摘要经营情况、行业地位、风险因素（仅做事实摘录）"),
    ("综合解读（全部维度）", "a-f 全要，输出会较长"),
]

FORMAT_OPTIONS: list[tuple[str, str]] = [
    ("表格为主", "Markdown 表格列出关键数字与变化，便于一眼对比"),
    ("文字摘要（通俗易懂）", "一段段白话描述，避免大段术语和数字堆砌"),
    ("重点标注（关键数字 + 简短点评）", "只列核心数字和 1-2 句点评，其它略过"),
    ("问答式", "按问题逐条回答（每个维度一个小标题 + 2-4 句回答）"),
    ("结构化清单（分点列出）", "分点短句，不成段，方便复制和快速阅读"),
]

SUGGEST_OPTIONS: list[tuple[str, str]] = [
    ("仅做事实陈述，不给建议", "完全中立，不输出任何主观判断或风险描述之外的提示"),
    ("主要风险点", "结合财报数据，列出 2-4 条值得后续跟踪的风险点（例如商誉占比、应收账款增速等）"),
    ("观察指标清单", "给出 3-5 个后续需要继续跟踪的指标或事件（例如下季现金流、存货周转等）"),
    ("综合（风险点 + 观察指标）", "b 和 c 的组合，但**不给出任何买卖、价格目标、仓位建议**"),
]


@dataclass(frozen=True)
class ChoiceDimension:
    key: str
    title: str
    options: tuple[tuple[str, str], ...]
    default: int = 1


CONTENT_DIM = ChoiceDimension("content", "解析财报里的什么内容", tuple(CONTENT_OPTIONS))
FOCUS_DIM = ChoiceDimension("focus", "具体想看什么", tuple(FOCUS_OPTIONS))
FORMAT_DIM = ChoiceDimension("format", "呈现方式偏向什么", tuple(FORMAT_OPTIONS))
SUGGEST_DIM = ChoiceDimension("suggest", "想要有什么建议内容", tuple(SUGGEST_OPTIONS))

DIMENSIONS: tuple[ChoiceDimension, ...] = (CONTENT_DIM, FOCUS_DIM, FORMAT_DIM, SUGGEST_DIM)


def _format_dim_line(dim: ChoiceDimension, idx: int, show_default_marker: bool = True) -> str:
    marker = "（默认）" if show_default_marker and dim.default == idx else ""
    label, hint = dim.options[idx - 1]
    return f"  {idx}. {label}{marker}\n     ↳ {hint}"


def render_option_menu() -> str:
    """返回给用户看的选项菜单（飞书卡片正文 / Web 控制台面板共用）。"""
    lines = ["请按顺序回复 4 个选择（数字 1-N 或字母 a-z，空格或逗号分隔，例如 `1 2 3 4` / `a b c d`）：", ""]
    for dim in DIMENSIONS:
        lines.append(f"【{dim.title}】")
        for i in range(1, len(dim.options) + 1):
            lines.append(_format_dim_line(dim, i))
        lines.append("")
    lines.append(
        "回复说明："
        "① 只回 4 个选择时按顺序对应 4 个维度；"
        "② 也可以只回 1-2 个数字，未给出的维度使用默认（带『默认』字样的选项）；"
        "③ 任意一行写 `默认` 全部采用默认。"
    )
    return "\n".join(lines)


# ============ 用户回复解析 ============

_LETTER_RE = re.compile(r"[a-zA-Z]")


def _normalize_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token or "").strip().lower()


def parse_user_reply(text: str) -> dict[str, int] | None:
    """解析用户的回复文本，映射成 {dim_key: 1-based option index}。

    支持的格式：
    - "1 2 3 4" / "1,2,3,4" / "1|2|3|4"：按顺序对应 4 个维度
    - "a b c d" / "A B C D"：字母对应每个维度内部的选项
    - "内容1 角度2 形式3 建议4"：维度前缀 + 数字，忽略顺序
    - "默认"：返回空 dict `{}`，调用方走全部默认
    - 只给 1-2 个选择时，缺失维度使用维度默认
    - 命名 + 位置可混用：命名的填到指定维度，剩下的按位置补
    - 无法解析成上述任何格式时返回 `None`，调用方应视为非回复（例如新问题）
    """
    if not text:
        return None
    cleaned = unicodedata.normalize("NFKC", text).strip()
    if not cleaned:
        return None
    if cleaned in {"默认", "default", "默认全部", "use default"}:
        return {}

    tokens = re.split(r"[\s,，;；|、]+", cleaned)
    tokens = [t for t in tokens if t]
    if not tokens:
        return None

    # 命名前缀：单字母 c/f/p/s 故意不放在这里，否则会和字母选项（如 "c"=3）冲突。
    # 维度前缀仅支持中文（"内容/财报/角度/看/形式/呈现/建议"）。
    dim_key_by_alias = {
        "内容": CONTENT_DIM.key, "财报": CONTENT_DIM.key,
        "角度": FOCUS_DIM.key, "看": FOCUS_DIM.key,
        "形式": FORMAT_DIM.key, "呈现": FORMAT_DIM.key,
        "建议": SUGGEST_DIM.key,
    }

    result: dict[str, int] = {}

    for token in tokens:
        token = _normalize_token(token)
        if not token:
            continue

        # 维度前缀写法：内容1 / 角度2 ...
        match = re.match(r"^(内容|财报|角度|看|形式|呈现|建议)(.*)$", token)
        if match:
            prefix, rest = match.group(1), match.group(2)
            dim_key = dim_key_by_alias[prefix]
            value = _coerce_choice_token(rest)
            if value is None:
                return None  # 命名格式但数字/字母无法解析，整条拒绝
            if not _set_dim(dim_key, value):
                return None
            result[dim_key] = value
            continue

        value = _coerce_choice_token(token)
        if value is None:
            continue
        # 位置写法：填到下一个尚未设置的维度（跳过已被命名的）。
        for dim in DIMENSIONS:
            if dim.key not in result:
                if not _set_dim(dim.key, value):
                    return None
                result[dim.key] = value
                break
        else:
            # 命名 + 位置已经填满了所有 4 个维度，剩余 token 忽略。
            continue

    if not result:
        return None
    return result


def _coerce_choice_token(token: str) -> int | None:
    if not token:
        return None
    if token.isdigit():
        return int(token)
    letter = _LETTER_RE.fullmatch(token)
    if letter:
        return ord(letter.group(0).lower()) - ord("a") + 1
    return None


def _set_dim(dim_key: str, value: int) -> bool:
    """Validate ``value`` against the dimension's option count; return False when
    the value is out of range. Caller decides whether to reject the whole reply
    or simply skip this token."""
    dim = next(d for d in DIMENSIONS if d.key == dim_key)
    if not 1 <= value <= len(dim.options):
        logger.debug(f"财报选项越界，忽略: {dim_key}={value}")
        return False
    return True


def resolve_choices(selections: dict[str, int] | None) -> dict[str, tuple[ChoiceDimension, int]]:
    """把用户选择（含默认）落地成 {dim_key: (dim, 1-based option index)}。"""
    out: dict[str, tuple[ChoiceDimension, int]] = {}
    for dim in DIMENSIONS:
        idx = (selections or {}).get(dim.key, dim.default)
        if not 1 <= idx <= len(dim.options):
            idx = dim.default
        out[dim.key] = (dim, idx)
    return out


# ============ 财报数据抓取 ============

@dataclass
class ReportData:
    """聚合的财报快照；不存在的部分保持空列表。"""

    stock: StockRef
    announcements: list[dict] = field(default_factory=list)
    performance_express: list[dict] = field(default_factory=list)
    performance_forecast: list[dict] = field(default_factory=list)
    balance_sheet: list[dict] = field(default_factory=list)
    income_statement: list[dict] = field(default_factory=list)
    cash_flow: list[dict] = field(default_factory=list)
    key_indicators: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (
            self.announcements
            or self.performance_express
            or self.performance_forecast
            or self.balance_sheet
            or self.income_statement
            or self.cash_flow
            or self.key_indicators
        )


def _filter_df_by_code(df, code: str) -> "object | None":
    """Filter a DataFrame to rows matching the given 6-digit stock code. Returns
    None when the input is empty, not a DataFrame, lacks a code column, or has
    no matching rows — callers use the None return to skip the period candidate
    and try the next one."""
    if df is None:
        return None
    if not hasattr(df, "columns") or not hasattr(df, "empty"):
        return None
    if df.empty:
        return None
    code_col = None
    for col in ("股票代码", "代码"):
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        return None
    matches = df[df[code_col].astype(str).str.zfill(6) == code]
    return matches if not matches.empty else None


def _summarize_row(row: dict, label: str, value_fields: list[str]) -> dict:
    out = {"label": label}
    for field_name in value_fields:
        if field_name not in row:
            continue
        raw = row.get(field_name)
        num = _safe_float(raw)
        out[field_name] = num if num is not None else _display_text(raw, 40)
    if "公告日期" in row:
        out["公告日期"] = _display_text(row.get("公告日期"), 40)
    return out


def fetch_report_data(stock: StockRef) -> ReportData:
    """从公开数据源抓取与目标股票相关的财报数据；不抛异常，失败则记 warning。"""
    data = ReportData(stock=stock)

    # 1) 巨潮原文：年报 / 半年报 / 一季报 / 三季报 / 业绩快报 / 业绩预告
    try:
        df = _ak_call(
            ak.stock_zh_a_disclosure_report_cninfo,
            symbol=stock.code,
            keyword="年度报告",
            start_date="20220101",
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        year_count = 0
        for _, row in df.head(8).iterrows():
            title = _display_text(row.get("公告标题"), 200)
            if "年度报告" not in title:
                continue
            data.announcements.append({
                "type": "年报",
                "title": title,
                "date": _display_text(row.get("公告时间"), 40),
                "url": str(row.get("公告链接", "")),
            })
            year_count += 1
            if year_count >= 3:
                break
    except Exception as e:
        logger.debug(f"财报公告-年报获取失败 {stock.code}: {e}")
        data.warnings.append(f"年报公告拉取失败：{e}")

    for keyword, label in (("半年度报告", "半年报"), ("第一季度报告", "一季报"), ("第三季度报告", "三季报")):
        try:
            df = _ak_call(
                ak.stock_zh_a_disclosure_report_cninfo,
                symbol=stock.code,
                keyword=keyword,
                start_date="20220101",
                end_date=datetime.now().strftime("%Y%m%d"),
            )
            # 巨潮的 keyword 是模糊匹配，结果里偶尔会有"近义"标题混进来。
            # 严格按标题包含关键词过滤一下，保证三季报不会被一季报顶掉。
            matched = None
            for _, row in df.head(5).iterrows():
                title = _display_text(row.get("公告标题"), 200)
                if keyword in title:
                    matched = row
                    break
            if matched is not None and not any(a.get("type") == label for a in data.announcements):
                data.announcements.append({
                    "type": label,
                    "title": _display_text(matched.get("公告标题"), 160),
                    "date": _display_text(matched.get("公告时间"), 40),
                    "url": str(matched.get("公告链接", "")),
                })
        except Exception as e:
            logger.debug(f"财报公告-{label}获取失败 {stock.code}: {e}")

    # 2) 业绩快报
    for period in _candidate_report_dates("yjkb"):
        try:
            df = _ak_call(ak.stock_yjkb_em, date=period)
            row_df = _filter_df_by_code(df, stock.code)
            if row_df is None:
                continue
            row = row_df.iloc[0].to_dict()
            data.performance_express.append({
                "period": period,
                "revenue": _summarize_row(row, "营收", ["营业收入-营业收入", "营业收入-同比增长", "营业收入-季度环比增长"]),
                "net_profit": _summarize_row(row, "净利", ["净利润-净利润", "净利润-同比增长", "净利润-季度环比增长"]),
                "eps": _safe_float(row.get("每股收益")),
                "bps": _safe_float(row.get("每股净资产")),
                "roe": _safe_float(row.get("净资产收益率")),
                "industry": _display_text(row.get("所处行业"), 40),
                "ann_date": _display_text(row.get("公告日期"), 40),
            })
            break
        except Exception as e:
            logger.debug(f"业绩快报 {period} 失败 {stock.code}: {e}")

    # 3) 业绩预告
    for period in _candidate_report_dates("yjyg"):
        try:
            df = _ak_call(ak.stock_yjyg_em, date=period)
            row_df = _filter_df_by_code(df, stock.code)
            if row_df is None:
                continue
            row = row_df.iloc[0].to_dict()
            data.performance_forecast.append({
                "period": period,
                "indicator": _display_text(row.get("预测指标"), 60),
                "change": _display_text(row.get("业绩变动"), 80),
                "forecast_value": _display_text(row.get("预测数值"), 80),
                "change_range": _display_text(row.get("业绩变动幅度"), 80),
                "reason": _display_text(row.get("业绩变动原因"), 200),
                "forecast_type": _display_text(row.get("预告类型"), 30),
                "ann_date": _display_text(row.get("公告日期"), 40),
            })
            break
        except Exception as e:
            logger.debug(f"业绩预告 {period} 失败 {stock.code}: {e}")

    # 4) 资产负债表
    for period in _candidate_report_dates("zcfz"):
        try:
            df = _ak_call(ak.stock_zcfz_em, date=period)
            row_df = _filter_df_by_code(df, stock.code)
            if row_df is None:
                continue
            row = row_df.iloc[0].to_dict()
            data.balance_sheet.append({
                "period": period,
                "items": _summarize_row(row, "资产负债", [
                    "资产-货币资金",
                    "资产-应收账款",
                    "资产-存货",
                    "资产-总资产",
                    "资产-总资产同比",
                    "负债-应付账款",
                    "负债-总负债",
                    "负债-总负债同比",
                    "资产负债率",
                    "股东权益合计",
                ]),
                "ann_date": _display_text(row.get("公告日期"), 40),
            })
            break
        except Exception as e:
            logger.debug(f"资产负债表 {period} 失败 {stock.code}: {e}")

    # 5) 利润表
    for period in _candidate_report_dates("lrb"):
        try:
            df = _ak_call(ak.stock_lrb_em, date=period)
            row_df = _filter_df_by_code(df, stock.code)
            if row_df is None:
                continue
            row = row_df.iloc[0].to_dict()
            data.income_statement.append({
                "period": period,
                "items": _summarize_row(row, "利润表", [
                    "营业总收入",
                    "营业总收入同比",
                    "营业总支出-营业支出",
                    "营业总支出-销售费用",
                    "营业总支出-管理费用",
                    "营业总支出-财务费用",
                    "营业利润",
                    "利润总额",
                    "净利润",
                    "净利润同比",
                ]),
                "ann_date": _display_text(row.get("公告日期"), 40),
            })
            break
        except Exception as e:
            logger.debug(f"利润表 {period} 失败 {stock.code}: {e}")

    # 6) 现金流量表
    for period in _candidate_report_dates("xjll"):
        try:
            df = _ak_call(ak.stock_xjll_em, date=period)
            row_df = _filter_df_by_code(df, stock.code)
            if row_df is None:
                continue
            row = row_df.iloc[0].to_dict()
            data.cash_flow.append({
                "period": period,
                "items": _summarize_row(row, "现金流", [
                    "净现金流-净现金流",
                    "净现金流-同比增长",
                    "经营性现金流-现金流量净额",
                    "经营性现金流-净现金流占比",
                    "投资性现金流-现金流量净额",
                    "融资性现金流-现金流量净额",
                ]),
                "ann_date": _display_text(row.get("公告日期"), 40),
            })
            break
        except Exception as e:
            logger.debug(f"现金流量表 {period} 失败 {stock.code}: {e}")

    # 7) 关键财务指标：取最新一期 + 最近 4 期（用于趋势/同比）
    try:
        df = _ak_call(
            ak.stock_financial_analysis_indicator_em,
            symbol=_market_secid(stock.code),
            indicator="按报告期",
        )
        if df is not None and not df.empty:
            cols_map = {
                "REPORT_DATE_NAME": "报告期",
                "EPSJB": "每股收益",
                "BPS": "每股净资产",
                "TOTALOPERATEREVETZ": "营收同比",
                "PARENTNETPROFITTZ": "归母净利同比",
                "ROEJQ": "净资产收益率",
                "XSMLL": "销售毛利率",
                "ZCFZL": "资产负债率",
            }
            recent = df.head(8)
            for _, row in recent.iterrows():
                entry = {"报告期": _display_text(row.get("REPORT_DATE_NAME"), 30)}
                for src, label in cols_map.items():
                    if src == "REPORT_DATE_NAME":
                        continue
                    if src not in row.index:
                        continue
                    num = _safe_float(row.get(src))
                    entry[label] = num if num is not None else _display_text(row.get(src), 40)
                data.key_indicators.append(entry)
    except Exception as e:
        logger.debug(f"关键财务指标失败 {stock.code}: {e}")
        data.warnings.append(f"关键财务指标拉取失败：{e}")

    return data


def _candidate_report_dates(kind: str) -> list[str]:
    """生成从最近一期往前推的若干财报期候选。

    季度规则：03-31 / 06-30 / 09-30；年度规则：12-31。
    """
    today = datetime.now()
    year = today.year
    if kind == "yjyg":
        return [f"{y}0331" for y in (year, year - 1)]
    if kind == "yjkb":
        return [f"{y}1231" for y in (year, year - 1, year - 2)]
    return [
        f"{y}{suf}"
        for y in (year, year - 1)
        for suf in ("1231", "0930", "0630", "0331")
    ]


# ============ LLM prompt + 输出 ============

_RISK_DISCLAIMER = (
    "⚠️ 免责声明：本回答基于公开财报数据与模型整理，仅供学习研究使用，"
    "不构成任何投资建议、不承诺收益。A 股投资有风险，财报数据可能存在"
    "口径调整、追溯修订或延迟披露，重大决策请以公司在巨潮资讯网披露"
    "的原文为准，并自行核对或咨询持牌专业人士。"
)


def _summarize_choices(choices: dict[str, tuple[ChoiceDimension, int]]) -> str:
    lines = []
    for dim, idx in choices.values():
        label, hint = dim.options[idx - 1]
        lines.append(f"- {dim.title}：{label}（{hint}）")
    return "\n".join(lines)


def _build_data_pack(stock: StockRef, report: ReportData, choices: dict) -> dict:
    return {
        "stock": {"code": stock.code, "name": stock.name},
        "user_choices": {
            dim.key: {"title": dim.title, "label": dim.options[idx - 1][0], "hint": dim.options[idx - 1][1]}
            for dim, idx in choices.values()
        },
        "announcements": report.announcements,
        "performance_express": report.performance_express,
        "performance_forecast": report.performance_forecast,
        "balance_sheet": report.balance_sheet,
        "income_statement": report.income_statement,
        "cash_flow": report.cash_flow,
        "key_indicators": report.key_indicators,
        "fetch_warnings": report.warnings,
    }


def _format_report_fallback(report: ReportData) -> str:
    """LLM 不可用时的纯模板兜底。"""
    lines = [
        f"结论：{report.stock.name}({report.stock.code}) 财报数据已整理如下，"
        "未走 LLM 解读，建议核对原文确认。",
        "",
    ]
    if report.announcements:
        lines.append("【财报原文公告】")
        for item in report.announcements:
            lines.append(f"- {item.get('type', '')} {item.get('date', '')} {item.get('title', '')}")
            if item.get("url"):
                lines.append(f"  详情：{item['url']}")
        lines.append("")
    if report.performance_express:
        lines.append("【业绩快报】")
        for item in report.performance_express:
            rev = item.get("revenue", {})
            np_ = item.get("net_profit", {})
            lines.append(
                f"- {item.get('period', '')}：营收 {rev.get('营业收入-营业收入', '未取到')} "
                f"（同比 {rev.get('营业收入-同比增长', '—')}），"
                f"净利 {np_.get('净利润-净利润', '未取到')} "
                f"（同比 {np_.get('净利润-同比增长', '—')}），"
                f"EPS {item.get('eps', '—')}，ROE {item.get('roe', '—')}"
            )
        lines.append("")
    if report.balance_sheet:
        lines.append("【资产负债表】")
        for item in report.balance_sheet:
            items = item.get("items", {})
            lines.append(
                f"- {item.get('period', '')}："
                f"总资产 {items.get('资产-总资产', '—')}（同比 {items.get('资产-总资产同比', '—')}），"
                f"总负债 {items.get('负债-总负债', '—')}（同比 {items.get('负债-总负债同比', '—')}），"
                f"资产负债率 {items.get('资产负债率', '—')}，"
                f"货币资金 {items.get('资产-货币资金', '—')}，"
                f"存货 {items.get('资产-存货', '—')}"
            )
        lines.append("")
    if report.income_statement:
        lines.append("【利润表】")
        for item in report.income_statement:
            items = item.get("items", {})
            lines.append(
                f"- {item.get('period', '')}："
                f"营收 {items.get('营业总收入', '—')}（同比 {items.get('营业总收入同比', '—')}），"
                f"净利润 {items.get('净利润', '—')}（同比 {items.get('净利润同比', '—')}），"
                f"营业利润 {items.get('营业利润', '—')}，"
                f"销售费用 {items.get('营业总支出-销售费用', '—')}，"
                f"管理费用 {items.get('营业总支出-管理费用', '—')}，"
                f"财务费用 {items.get('营业总支出-财务费用', '—')}"
            )
        lines.append("")
    if report.cash_flow:
        lines.append("【现金流量表】")
        for item in report.cash_flow:
            items = item.get("items", {})
            lines.append(
                f"- {item.get('period', '')}："
                f"经营现金流 {items.get('经营性现金流-现金流量净额', '—')}，"
                f"投资现金流 {items.get('投资性现金流-现金流量净额', '—')}，"
                f"筹资现金流 {items.get('融资性现金流-现金流量净额', '—')}，"
                f"净现金流 {items.get('净现金流-净现金流', '—')}（同比 {items.get('净现金流-同比增长', '—')}）"
            )
        lines.append("")
    if report.key_indicators:
        lines.append("【关键财务指标（最近多期）】")
        for item in report.key_indicators:
            lines.append(
                f"- {item.get('报告期', '')}：EPS {item.get('每股收益', '—')}，"
                f"ROE {item.get('净资产收益率', '—')}，"
                f"毛利率 {item.get('销售毛利率', '—')}，"
                f"资产负债率 {item.get('资产负债率', '—')}，"
                f"营收同比 {item.get('营收同比', '—')}，"
                f"归母净利同比 {item.get('归母净利同比', '—')}"
            )
        lines.append("")
    if report.performance_forecast:
        lines.append("【业绩预告】")
        for item in report.performance_forecast:
            lines.append(
                f"- {item.get('period', '')} {item.get('forecast_type', '')}："
                f"{item.get('change', '')} {item.get('change_range', '')}"
                f"；原因：{item.get('reason', '')}"
            )
        lines.append("")
    if report.warnings:
        lines.append("【拉取告警】")
        for warn in report.warnings:
            lines.append(f"- {warn}")
        lines.append("")
    lines.append(_RISK_DISCLAIMER)
    return "\n".join(lines)


def answer_financial_report(
    stock: StockRef,
    selections: dict[str, int] | None = None,
) -> tuple[str, ReportData, dict[str, tuple[ChoiceDimension, int]]]:
    """主入口：抓数据 → 拼 prompt → 调 LLM → 兜底文本。

    Returns: (final_text, report_data, resolved_choices)
    """
    report = fetch_report_data(stock)
    choices = resolve_choices(selections)

    if report.is_empty():
        text = (
            f"{stock.name}({stock.code}) 暂未取到任何公开财报数据。"
            "可能是新股、已退市或数据源暂时不可用，请稍后重试或检查代码。\n\n"
            + _RISK_DISCLAIMER
        )
        return text, report, choices

    system_prompt = (
        "你是给非专业家人使用的 A 股财报助手。只根据提供的数据回答，"
        "不要编造未提供的科目或数字；数据为空时明确写『未取到』而不是猜测。"
        "绝对不能给出确定性买卖指令、价格目标、仓位建议或收益承诺。"
        "可以使用风险点、观察指标、异常科目等中性描述，但措辞要谨慎。"
        "回答结构清晰，先结论再分点，符合用户选择的呈现方式。"
    )
    choices_text = _summarize_choices(choices)
    user_prompt = f"""请按以下用户选择和提供的数据回答：

用户股票：{stock.name}({stock.code})

用户选择的解读维度：
{choices_text}

合规与免责：
- 数据源仅为公开接口（AKShare / 巨潮资讯），存在延迟与口径差异。
- 不能输出买卖、价格目标、仓位、收益承诺。
- 必须列出主要数据的数据源/口径，避免读者误以为是公司原文。

输出要求：
1. 先给一句结论（事实陈述 + 风险提示）。
2. 按用户选的『解析内容』逐项给出关键数字与变化。
3. 按用户选的『具体看什么』做对比（同比/环比/历史均值/行业参考等）。
4. 按用户选的『呈现方式』组织排版（表格 / 摘要 / 重点标注 / 问答 / 分点）。
5. 按用户选的『建议内容』结尾（如选 a 则不放任何建议）。
6. 不要使用 Markdown 表格水平分隔线，飞书卡片里避免大段 markdown。
7. 公告/原文必须附 URL，方便用户去巨潮核对。
8. 末尾固定加一句免责声明。

数据（已 JSON 化）：
{json.dumps(_build_data_pack(stock, report, choices), ensure_ascii=False, indent=2)[:9000]}
"""

    try:
        client = get_llm_client()
        raw = client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ], temperature=0.2, max_tokens=2200)
        answer = client._strip_think(raw).strip()
        if answer:
            final = answer + "\n\n" + _format_sources_section(report) + "\n\n" + _RISK_DISCLAIMER
            return final, report, choices
    except Exception as e:
        logger.warning(f"财报 LLM 解读失败 {stock.code}: {e}")

    return _format_report_fallback(report), report, choices


def _format_sources_section(report: ReportData) -> str:
    if not report.announcements:
        return "资料来源\n- 暂未取到对应公告原文"
    lines = ["资料来源"]
    for i, item in enumerate(report.announcements[:6], 1):
        lines.append(
            f"- [公告{i}] {item.get('date', '')} {item.get('type', '')}: {item.get('title', '')}"
        )
        url = str(item.get("url") or "").strip()
        if url.startswith(("http://", "https://")):
            lines.append(f"  详情：{url}")
    return "\n".join(lines)


# ============ 帮助/触发检测 ============

def is_financial_report_intent(text: str) -> bool:
    if not text:
        return False
    cleaned = unicodedata.normalize("NFKC", text).strip()
    return any(kw in cleaned for kw in REPORT_TRIGGER_KEYWORDS)


def help_financial_report() -> list[str]:
    return [
        "**财报解析**：",
        "用法：`财报解析 600519`，机器人会先列出 4 个维度选项，"
        "你回 `1 2 3 4`（数字或字母均可）就开始拉数据并解读。",
        "也可以只回部分选择，剩余维度使用默认（带『默认』字样的选项）。",
        "本功能仅做公开财报数据的二次整理，不构成投资建议，最终数据以巨潮原文为准。",
    ]