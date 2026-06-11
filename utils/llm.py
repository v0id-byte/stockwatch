"""LLM 客户端：OpenAI-compatible / Anthropic，重试 + JSON 解析。"""
import json
import re
import threading
import urllib.request

import tenacity
from openai import OpenAI
from loguru import logger

from config import get_config


_TOKEN_LOCK = threading.Lock()
_TOKEN_USAGE_TOTAL = 0


def _record_token_usage(response):
    global _TOKEN_USAGE_TOTAL
    usage = getattr(response, "usage", None)
    total = getattr(usage, "total_tokens", 0) if usage else 0
    if not total and isinstance(response, dict):
        usage = response.get("usage") or {}
        total = int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    if not total:
        return
    with _TOKEN_LOCK:
        _TOKEN_USAGE_TOTAL += int(total)


def reset_token_usage():
    global _TOKEN_USAGE_TOTAL
    with _TOKEN_LOCK:
        _TOKEN_USAGE_TOTAL = 0


def get_token_usage() -> int:
    with _TOKEN_LOCK:
        return _TOKEN_USAGE_TOTAL


class BaseLLMClient:
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


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self):
        cfg = get_config()
        self.client = OpenAI(
            api_key=cfg.llm_api_key_or_placeholder,
            base_url=cfg.llm_base_url,
            timeout=60,
            max_retries=0,
        )
        self.model = cfg.llm_model
        self.provider = cfg.llm_provider

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
        _record_token_usage(response)
        return response.choices[0].message.content or ""


class AnthropicClient(BaseLLMClient):
    def __init__(self):
        cfg = get_config()
        self.api_key = cfg.llm_api_key
        self.base_url = cfg.llm_base_url
        self.model = cfg.llm_model

    def _messages_endpoint(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts = []
        converted = []
        for message in messages:
            role = message.get("role", "user")
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                converted.append({"role": role, "content": content})
            else:
                converted.append({"role": "user", "content": content})
        return "\n\n".join(system_parts), converted

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=2, min=4, max=30),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    def chat(self, messages: list[dict], temperature: float = 0.3, max_tokens: int = 2048) -> str:
        system_prompt, payload_messages = self._split_system(messages)
        payload = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt
        body = json.dumps(payload, ensure_ascii=False).encode()
        req = urllib.request.Request(
            self._messages_endpoint(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read())
        _record_token_usage(result)
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)


MiniMaxClient = OpenAICompatibleClient


def get_llm_client() -> BaseLLMClient:
    cfg = get_config()
    provider = cfg.llm_provider
    if provider == "anthropic":
        return AnthropicClient()
    if provider in {"openai", "openai-compatible", "minimax", "custom", "local"}:
        return OpenAICompatibleClient()
    raise ValueError(f"不支持的 LLM_PROVIDER: {provider}")
