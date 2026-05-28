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
        """发送卡片消息，content 是卡片 JSON"""
        url = f"{self.BASE_URL}/im/v1/messages?receive_id_type={self.cfg.feishu_receive_id_type}"
        payload = {
            "receive_id": self.cfg.feishu_receive_id,
            "msg_type": "interactive",
            "content": json.dumps(content, ensure_ascii=False),
        }
        headers = {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }
        import urllib.request
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                result = json.loads(r.read())
                if result.get("code") == 0:
                    logger.info("飞书消息发送成功")
                    return True
                logger.warning(f"飞书发送失败 code={result.get('code')}: {result.get('msg')}")
                return False
        except Exception as e:
            logger.error(f"飞书发送异常: {e}")
            return False


def render_card(run_id: str, decisions: list[dict]) -> dict:
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

    # 卡片头颜色：优先红>绿>蓝
    if any(d.get("action") == "BUY" for d in strong_signals):
        header_color = "red"
        header_title = "🔥 今日强烈推荐信号"
    elif any(d.get("action") == "SELL" for d in strong_signals):
        header_color = "orange"
        header_title = "⚠️ 今日卖出信号"
    else:
        header_color = "blue"
        header_title = "📌 今日持仓观察"

    elements = []

    # 函数：添加信号分组块
    def add_signal_group(title_emoji: str, title: str, items: list):
        if not items:
            return
        # 分隔线：普通文本行
        elements.append({
            "tag": "div",
            "text": {"tag": "plain_text", "content": "─────────────────────"}
        })
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"**{title_emoji} {title}**"}
        })
        for d in items:
            action = d.get("action", "HOLD")
            conf = d.get("confidence", 0)
            stars = "⭐" * max(1, min(5, int(conf * 5)))
            price_str = f'目标价 {d.get("target_price", "—")}元' if d.get("target_price") else ""
            stop_str = f'止损 {d.get("stop_loss", "—")}元' if d.get("stop_loss") else ""
            elements.append({
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": (
                        f"**{d['name']}**({d['code']}) {action}\n"
                        f"{stars} 置信度 {conf:.0%} {price_str} {stop_str}\n"
                        f"📝 {d.get('one_liner', '—')}"
                    )
                }
            })

    add_signal_group("🔥", "强烈推荐", strong_signals)
    add_signal_group("⚠️", "关注信号", watch_signals)
    add_signal_group("📌", "持有不动", hold_notes)

    # 卡片尾
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    elements.append({
        "tag": "div",
        "text": {"tag": "plain_text", "content": "─────────────────────"}
    })
    elements.append({
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": (
                f"📅 更新时间：{run_time}\n"
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
