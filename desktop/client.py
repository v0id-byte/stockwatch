"""Small stdlib client for the local StockWatch Agent API."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class AgentUnavailable(RuntimeError):
    pass


class AgentClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765", token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.token = token

    def request(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["X-StockWatch-Token"] = self.token
        req = Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=4) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise AgentUnavailable(str(exc)) from exc

    def status(self) -> dict:
        return self.request("GET", "/api/v1/status")

    def settings(self) -> dict:
        return self.request("GET", "/api/v1/settings")

    def save_settings(self, updates: dict) -> dict:
        return self.request("POST", "/api/v1/settings", updates)

    def run(self, action: str) -> dict:
        return self.request("POST", "/api/v1/run", {"action": action})
