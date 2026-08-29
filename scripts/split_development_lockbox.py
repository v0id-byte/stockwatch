#!/usr/bin/env python3
"""Physically split a panel into development and lockbox files.

Enforcement today is distributed date guards: every development-stage script
hard-fails on any row with signal_date >= LOCKBOX_START, so pointing one at
the lockbox file (or a file containing lockbox rows) raises immediately.  The
one-shot unlock script (with frozen go/no-go criteria) is added only at the
lockbox-evaluation stage.  Only four facts about the lockbox are printed:
existence, date range, row count, sha256.  No statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

LOCKBOX_START = "2026-06-12"


def main() -> None:
    root = Path(os.getenv("STOCKWATCH_HISTORY_DIR", "~/.stockwatch/history")).expanduser()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default=str(root / "oos_baseline_panel.parquet"))
    parser.add_argument("--dev-out", default=str(root / "development_panel.parquet"))
    parser.add_argument("--lockbox-out", default=str(root / "lockbox_panel.parquet"))
    parser.add_argument("--lockbox-start", default=LOCKBOX_START)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    signal = pd.to_datetime(frame["signal_date"])
    cutoff = pd.Timestamp(args.lockbox_start)
    dev = frame[signal < cutoff]
    lockbox = frame[signal >= cutoff]

    dev_path, lockbox_path = Path(args.dev_out), Path(args.lockbox_out)
    dev.to_parquet(dev_path, index=False)
    lockbox.to_parquet(lockbox_path, index=False)

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    print(json.dumps({
        "development": {
            "path": str(dev_path), "rows": int(len(dev)),
            "date_range": [str(signal[signal < cutoff].min().date()) if len(dev) else None,
                           str(signal[signal < cutoff].max().date()) if len(dev) else None],
            "sha256": _sha(dev_path),
        },
        "lockbox": {
            "path": str(lockbox_path), "rows": int(len(lockbox)),
            "date_range": [str(signal[signal >= cutoff].min().date()) if len(lockbox) else None,
                           str(signal[signal >= cutoff].max().date()) if len(lockbox) else None],
            "sha256": _sha(lockbox_path),
            "note": "sealed: no statistics beyond these four facts until the one-shot unlock",
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
