"""Runtime assets shared by source and frozen desktop builds."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from core.settings import stockwatch_home


def bundled_resource_dir() -> Path | None:
    root = getattr(sys, "_MEIPASS", None)
    if not root:
        return None
    candidate = Path(root) / "models"
    return candidate if candidate.is_dir() else None


def install_bundled_models() -> list[Path]:
    """Copy signed bundle models once; never overwrite user-managed models."""
    source = bundled_resource_dir()
    if source is None:
        return []
    target = stockwatch_home() / "models"
    target.mkdir(parents=True, exist_ok=True)
    installed: list[Path] = []
    for model in source.iterdir():
        if not model.is_file():
            continue
        destination = target / model.name
        if destination.exists():
            continue
        shutil.copy2(model, destination)
        installed.append(destination)
    return installed
