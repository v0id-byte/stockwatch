"""File-backed settings used by both the dashboard and desktop app.

The implementation deliberately keeps the existing ``KEY=value`` format so
source installations remain compatible.  It separates persistence from the
web UI, which allows a future Keychain/Credential Manager backend without
teaching the dashboard how secrets are stored.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping


_ENV_ASSIGN_RE = re.compile(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*?)(\r?\n)?$")


def stockwatch_home() -> Path:
    raw = os.getenv("STOCKWATCH_HOME", "").strip()
    path = Path(raw).expanduser() if raw else Path.home() / ".stockwatch"
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_path(project_dir: Path | None = None) -> Path:
    explicit = os.getenv("STOCKWATCH_ENV_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if project_dir is not None:
        legacy = project_dir / ".env"
        if legacy.exists():
            return legacy
    return stockwatch_home() / "settings.env"


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if not value or value.startswith("#"):
        return ""
    if value[0] in {"'", '"'} and value[-1:] == value[0]:
        try:
            return json.loads(value) if value[0] == '"' else value[1:-1]
        except json.JSONDecodeError:
            return value[1:-1]
    if " #" in value:
        value = value.split(" #", 1)[0].strip()
    return value


def _serialize_value(value: str) -> str:
    value = str(value)
    if value == "" or re.fullmatch(r"[A-Za-z0-9_./:@,+~=-]+", value):
        return value
    return json.dumps(value, ensure_ascii=False)


class FileSettingsStore:
    """Atomic, allow-listed settings persistence."""

    def __init__(self, defaults: Mapping[str, str], project_dir: Path | None = None):
        self.defaults = dict(defaults)
        self.project_dir = project_dir

    @property
    def path(self) -> Path:
        return settings_path(self.project_dir)

    def _read_file(self) -> dict[str, str]:
        values: dict[str, str] = {}
        if not self.path.exists():
            return values
        for line in self.path.read_text(encoding="utf-8").splitlines():
            match = _ENV_ASSIGN_RE.match(line)
            if match:
                values[match.group(2)] = _parse_value(match.group(4))
        return values

    def load(self) -> dict[str, str]:
        values = dict(self.defaults)
        for key in values:
            if os.getenv(key) is not None:
                values[key] = os.getenv(key, "")
        file_values = self._read_file()
        for key in values:
            if key in file_values:
                values[key] = file_values[key]
        if not values.get("LLM_API_KEY"):
            alias = "ANTHROPIC_API_KEY" if values.get("LLM_PROVIDER") == "anthropic" else "MINIMAX_API_KEY"
            values["LLM_API_KEY"] = file_values.get(alias, os.getenv(alias, ""))
        return values

    def save(self, updates: Mapping[str, str]) -> None:
        cleaned = {
            key: str(value).strip()
            for key, value in updates.items()
            if key in self.defaults
        }
        if not cleaned:
            return
        path = self.path
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.exists() else []
        written: set[str] = set()
        output: list[str] = []
        for line in lines:
            match = _ENV_ASSIGN_RE.match(line)
            if match and match.group(2) in cleaned:
                key = match.group(2)
                output.append(f"{key}={_serialize_value(cleaned[key])}{match.group(5) or chr(10)}")
                written.add(key)
            else:
                output.append(line)
        missing = [key for key in cleaned if key not in written]
        if missing:
            if output and not output[-1].endswith(("\n", "\r\n")):
                output[-1] += "\n"
            if output:
                output.append("\n")
            output.append("# ===== StockWatch settings =====\n")
            output.extend(f"{key}={_serialize_value(cleaned[key])}\n" for key in missing)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text("".join(output), encoding="utf-8")
        temp.chmod(0o600)
        temp.replace(path)
        for key, value in cleaned.items():
            os.environ[key] = value

