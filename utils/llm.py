"""MiniMax LLM 客户端（OpenAI 兼容，重试 + JSON 解析）"""
import json
import re
import tenacity
from openai import OpenAI
from loguru import logger

from config import get_config


class MiniMaxClient:
    def __init__(self):
        cfg = get_config()
        self.client = OpenAI(
            api_key=cfg.minimax_api_key,
            base_url=cfg.minimax_base_url,
            timeout=60,
            max_retries=0,
        )
        self.model = cfg.minimax_model

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=30),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def chat_json(self, messages: list[dict]) -> dict:
        """调用并强制解析为 JSON，超时兜底返回空 dict"""
        raw = self.chat(messages)
        raw = self._strip_think(raw)
        raw = raw.strip()
        # 去掉 markdown 围栏
        if raw.startswith("```"):
            parts = raw.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                elif p.startswith("}"):
                    pass
                if p.startswith("{") and p.endswith("}"):
                    try:
                        return json.loads(p)
                    except Exception:
                        pass
            raw = parts[-1] if parts else raw
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 尝试提取第一个 {...} 块
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except Exception:
                    pass
            logger.warning(f"JSON 解析失败: {raw[:100]}")
            return {}

    @staticmethod
    def _strip_think(text: str) -> str:
        """去掉 <think>...</think> 标签内容"""
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text


def get_llm_client() -> MiniMaxClient:
    return MiniMaxClient()
