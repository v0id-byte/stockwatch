"""飞书自建应用客户端 + 卡片渲染"""
import json
import time
from datetime import datetime
from pathlib import Path
from loguru import logger

from config import get_config


class FeishuClient:
    """飞书自建应用消息推送（tenant_access_token 缓存2h）"""

    BASE_URL = "https://open.feishu.cn/open-apis"

    def __init__(self):
        self.cfg = get_config()
        self._token = None
        self._token_expires_at = 0

    def _get_token(self) -> str:
        """获取 tenant_access_token，缓存2h"""
        if self._token and time.time() < self._token_expires_at - 60:
            return self._token

        url = f"{self.BASE_URL}/auth/v3/tenant_access_token/internal"
        resp = self._post(url, {
            "app_id": self.cfg.feishu_app_id,
            "app_secret": self.cfg.feishu_app_secret,
        })
        self._token = resp.get("tenant_access_token", "")
        self._token_expires_at = time.time() + 7200
        if not self._token:
            raise RuntimeError("飞书 token 获取失败")
        logger.info("飞书 tenant_access_token 已刷新")
        return self._token

    def _post(self, url: str, data: dict) -> dict:
        """POST JSON，失败重试一次（token失效场景）"""
        import urllib.request
        body = json.dumps(data).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except Exception as e:
            # 如果是 token 失效错误，重试一次
            if "99991663" in str(e) or "token" in str(e).lower():
                self._token = None
                body2 = json.dumps(data).encode()
                req2 = urllib.request.Request(url, data=body2, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req2, timeout=15) as r2:
                    return json.loads(r2.read())
            raise

    def send_message(self, content: dict) -> bool:
        """发送卡片消息给所有接收人，content 是卡片 JSON"""
        # 收集所有接收人
        receivers = [self.cfg.feishu_receive_id]
        if self.cfg.feishu_receive_id_2:
            receivers.append(self.cfg.feishu_receive_id_2)

        success = True
        for rid in receivers:
            if self.send_card_to(rid, content, self.cfg.feishu_receive_id_type):
                continue
            success = False
        return success

    def send_card_to(self, receive_id: str, content: dict, receive_id_type: str = "chat_id") -> bool:
        """按指定 receive_id_type 发送交互式卡片。"""
        url = f"{self.BASE_URL}/im/v1/messages?receive_id_type={receive_id_type}"
        import urllib.request

        def _send() -> dict:
            payload = {
                "receive_id": receive_id,
                "msg_type": "interactive",
                "content": json.dumps(content, ensure_ascii=False),
            }
            headers = {
                "Authorization": f"Bearer {self._get_token()}",
                "Content-Type": "application/json",
            }
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())

        try:
            result = _send()
            if result.get("code") == 99991663:
                logger.info("飞书 token 失效，刷新后重试")
                self._token = None
                result = _send()
            if result.get("code") == 0:
                logger.info(f"飞书消息发送成功 -> {receive_id}")
                return True
            logger.warning(f"飞书发送失败 -> {receive_id} code={result.get('code')}: {result.get('msg')}")
        except Exception as e:
            logger.error(f"飞书发送异常 -> {receive_id}: {e}")
        return False


def _format_price(value) -> str:
    try:
        price = float(value or 0)
    except (TypeError, ValueError):
        price = 0
    return f"{price:.2f}元" if price > 0 else "—"


def _format_score(value) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return f"{score:+.2f}"


def _compact_line(value: str) -> str:
    lines = [line.strip(" -") for line in str(value or "").splitlines() if line.strip()]
    return "；".join(lines)


def _json_items(value: str, limit: int) -> list[str]:
    try:
        items = json.loads(value or "[]")
    except json.JSONDecodeError:
        items = []
    return [str(item) for item in items[:limit]]


def _action_label(action: str) -> str:
    labels = {
        "BUY": "机会观察",
        "SELL": "风险复核",
        "HOLD": "继续观察",
    }
    return labels.get(str(action or "").upper(), str(action or "-"))


def _advice_line(d: dict) -> str:
    action = d.get("action", "HOLD")
    if action == "BUY":
        return (
            f"观察：参考价 {_format_price(d.get('trade_price'))} 附近再看，"
            f"跌破 {_format_price(d.get('stop_loss'))} 复核风险"
        )
    if action == "SELL":
        return f"风险：参考价 {_format_price(d.get('trade_price'))} 附近复核仓位"
    return "观察：暂不需要额外盯盘，继续跟踪"


def _analysis_lines(d: dict) -> list[str]:
    lines = []
    if "tech_score" in d:
        lines.append(
            f"技术面 {_format_score(d.get('tech_score'))}/1："
            f"{d.get('tech_summary') or '暂无细节'}"
        )
    if "sentiment_score" in d:
        lines.append(
            f"消息面指数 {_format_score(d.get('sentiment_score'))}/1："
            f"{d.get('sentiment_summary') or '中性'}"
        )
    sentiment_context = _compact_line(d.get("sentiment_context", ""))
    if sentiment_context:
        lines.append(sentiment_context)
    alpha = _compact_line(d.get("alpha_summary", ""))
    if alpha:
        lines.append(alpha)
    lgbm = _compact_line(d.get("lgbm_context", ""))
    if lgbm:
        lines.append(lgbm)
    return lines


def _reason_line(d: dict) -> str:
    reasons = _json_items(d.get("reasons_json"), 2)
    return "为什么：" + "；".join(reasons) if reasons else ""


def _family_brief_line(d: dict) -> str:
    try:
        if not get_config().enable_family_brief:
            return ""
    except Exception:
        return ""
    if d.get("family_brief"):
        return f"给家人看的结论：{d['family_brief']}"
    name = d.get("name") or d.get("code") or "这只股票"
    action = d.get("action", "HOLD")
    one_liner = str(d.get("one_liner") or "")
    if "跌破止损" in one_liner:
        return f"给家人看的结论：{name} 跌到风险位了，今天需要看一下。"
    if action == "SELL":
        return f"给家人看的结论：{name} 风险变高，建议今天复核仓位。"
    if action == "BUY":
        return f"给家人看的结论：{name} 出现可关注机会，先看风险位，不用追涨。"
    return f"给家人看的结论：{name} 暂时没有必须操作的信号，不用一直盯盘。"


def _position_lines(d: dict) -> list[str]:
    action = d.get("action", "HOLD")
    if action == "BUY":
        trade_line = (
            f"观察价 {_format_price(d.get('trade_price'))} | "
            f"压力位 {_format_price(d.get('target_price'))} | "
            f"风险价 {_format_price(d.get('stop_loss'))}"
        )
    elif action == "SELL":
        trade_line = (
            f"复核价 {_format_price(d.get('trade_price'))} | "
            f"下方支撑 {_format_price(d.get('target_price'))} | "
            f"风控线 {_format_price(d.get('stop_loss'))}"
        )
    else:
        trade_line = ""
    levels_line = (
        f"支撑 {_format_price(d.get('support_price'))} | "
        f"压力 {_format_price(d.get('resistance_price'))}"
    )
    basis = d.get("position_basis") or d.get("one_liner", "")
    basis_line = f"依据：{basis}" if basis else ""
    return [line for line in (trade_line, levels_line, basis_line) if line]


def render_text_card(title: str, lines: list[str], template: str = "blue") -> dict:
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": template,
        },
        "elements": [{
            "tag": "div",
            "text": {"tag": "lark_md", "content": "\n".join(lines)},
        }],
    }


def render_single_decision_card(decision: dict, title: str = "股票即时分析",
                                extra_lines: list[str] | None = None) -> dict:
    action = decision.get("action", "HOLD")
    template = "green" if action == "BUY" else "orange" if action == "SELL" else "blue"
    conf = float(decision.get("confidence") or 0)
    raw_conf = decision.get("raw_confidence")
    raw_text = f"（原始 {raw_conf:.0%}）" if raw_conf is not None and abs(float(raw_conf) - conf) > 0.005 else ""
    reasons = json.loads(decision.get("reasons_json") or "[]")
    risks = json.loads(decision.get("risks_json") or "[]")
    lines = [
        f"**{decision.get('name', decision.get('code'))}**({decision.get('code')}) {_action_label(action)}",
        f"置信度 {conf:.0%}{raw_text}",
        _advice_line(decision),
        *_position_lines(decision),
        *_analysis_lines(decision),
    ]
    if extra_lines:
        lines.extend(extra_lines)
    family_brief = _family_brief_line(decision)
    if family_brief:
        lines.append(family_brief)
    if decision.get("one_liner"):
        lines.append(f"一句话：{decision['one_liner']}")
    if reasons:
        lines.append("原因：" + "；".join(str(item) for item in reasons[:3]))
    if risks:
        lines.append("风险：" + "；".join(str(item) for item in risks[:2]))
    lines.append("*仅供研究参考，不构成投资建议*")
    return render_text_card(title, lines, template=template)


def render_card(run_id: str, decisions: list[dict], regime_info: dict | None = None) -> dict:
    """
    渲染飞书交互式卡片，返回卡片 JSON payload
    decisions: 已过滤并按 confidence + action 排序的决策列表
    """
    if not decisions:
        return None

    # 分组
    strong_signals = [d for d in decisions if d.get("action") in ("BUY", "SELL") and d.get("confidence", 0) >= 0.75]
    watch_signals = [d for d in decisions if d.get("action") in ("BUY", "SELL") and 0.6 <= d.get("confidence", 0) < 0.75]
    hold_notes = [d for d in decisions if d.get("action") == "HOLD" and d.get("confidence", 0) >= 0.65]
    has_visible_signal = bool(strong_signals or watch_signals or hold_notes)
    if not has_visible_signal and not (regime_info and regime_info.get("regime") == "crisis"):
        logger.info("飞书卡片无可展示信号，跳过发送")
        return None

    # 卡片头颜色：优先高波动风险，再展示风险复核和机会观察。
    if regime_info and regime_info.get("regime") == "crisis":
        header_color = "red"
        header_title = "⚠️ 高波动风险提示"
    elif any(d.get("action") == "SELL" for d in strong_signals):
        header_color = "orange"
        header_title = "⚠️ 今日风险复核"
    elif any(d.get("action") == "BUY" for d in strong_signals):
        header_color = "green"
        header_title = "📌 今日机会观察"
    else:
        header_color = "blue"
        header_title = "📌 今日持仓观察"

    elements = []
    if regime_info and regime_info.get("regime") == "crisis":
        elements.append({
            "tag": "div",
            "text": {
                "tag": "lark_md",
                "content": f"⚠️ {regime_info.get('context', '大盘波动处于高位')}"
            }
        })

    # 函数：添加信号分组块
    def add_signal_group(title_emoji: str, title: str, items: list):
        if not items:
            return
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**{title_emoji} {title}**"}
        })
        for d in items:
            action = d.get("action", "HOLD")
            conf = d.get("confidence", 0)
            raw_conf = d.get("raw_confidence")
            stars = "⭐" * max(1, min(5, int(conf * 5)))
            raw_str = f" 原始 {raw_conf:.0%}" if raw_conf is not None and abs(raw_conf - conf) > 0.005 else ""
            body_lines = [
                f"**{d['name']}**({d['code']}) {_action_label(action)}",
                f"{stars} 校准置信度 {conf:.0%}{raw_str}",
                _advice_line(d),
            ]
            body_lines.extend(_position_lines(d))
            body_lines.extend(_analysis_lines(d))
            reason_text = _reason_line(d)
            if reason_text:
                body_lines.append(reason_text)
            family_brief = _family_brief_line(d)
            if family_brief:
                body_lines.append(family_brief)
            if d.get("one_liner"):
                body_lines.append(f"📝 {d.get('one_liner', '—')}")
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": "\n".join(body_lines)
                }
            })

    add_signal_group("⚠️", "重点提醒", strong_signals)
    add_signal_group("📌", "一般提醒", watch_signals)
    add_signal_group("📌", "继续观察", hold_notes)

    # 卡片尾
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append({"tag": "hr"})
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": (
                f"📅 更新时间：{run_time}\n"
                f"{'📊 ' + regime_info.get('context', '') + chr(10) if regime_info else ''}"
                f"📡 数据来源：AKShare + MiniMax\n"
                f"⚠️ *本推送仅供研究参考，不构成投资建议，A 股投资有风险，入市需谨慎*"
            )
        }
    })

    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {"tag": "plain_text", "content": header_title},
            "template": header_color,
        },
        "elements": elements,
    }
