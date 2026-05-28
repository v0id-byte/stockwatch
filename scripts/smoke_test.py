#!/usr/bin/env python3
"""Run a one-stock v2 smoke test with all feature flags enabled."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "600519"
    os.environ["WATCHLIST"] = code
    os.environ["MAX_STOCKS_PER_RUN"] = "1"
    for name in [
        "ENABLE_CALIBRATION",
        "ENABLE_ALPHA158",
        "ENABLE_LGBM",
        "ENABLE_REGIME",
        "ENABLE_SECTOR",
    ]:
        os.environ.setdefault(name, "true")

    from main import once
    once()


if __name__ == "__main__":
    main()
