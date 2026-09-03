"""Per-user background-agent registration without administrator rights."""
from __future__ import annotations

import os
import sys
from pathlib import Path


MACOS_PLIST = "com.melspectrum.stockwatch.agent.plist"


def set_enabled(enabled: bool) -> tuple[bool, str]:
    if sys.platform == "darwin":
        return _set_macos(enabled)
    if os.name == "nt":
        return _set_windows(enabled)
    return False, "Automatic login start is supported on packaged macOS and Windows builds."


def _set_macos(enabled: bool) -> tuple[bool, str]:
    try:
        import ServiceManagement

        service = ServiceManagement.SMAppService.agentServiceWithPlistName_(MACOS_PLIST)
        if enabled:
            ok, error = service.registerAndReturnError_(None)
        else:
            ok, error = service.unregisterAndReturnError_(None)
        return bool(ok), "" if ok else str(error)
    except Exception as exc:
        return False, f"SMAppService unavailable: {exc}"


def _set_windows(enabled: bool) -> tuple[bool, str]:
    appdata = os.getenv("APPDATA", "").strip()
    if not appdata:
        return False, "Windows Startup folder is unavailable."
    startup = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup.mkdir(parents=True, exist_ok=True)
    launcher = startup / "StockWatch Agent.cmd"
    if not enabled:
        launcher.unlink(missing_ok=True)
        return True, ""
    executable = Path(sys.executable).resolve()
    launcher.write_text(f'@start "" "{executable}" --background-agent\r\n', encoding="utf-8")
    return True, ""
