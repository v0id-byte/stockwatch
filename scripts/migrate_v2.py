#!/usr/bin/env python3
"""Run StockWatch v2 SQLite migrations."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.storage import Storage


def main():
    storage = Storage()
    print(f"v2 migration complete: {storage.db_path}")


if __name__ == "__main__":
    main()
