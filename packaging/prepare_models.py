"""Validate and stage inference-only models for a desktop build."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_FILES = (
    "lgbm.txt",
    "lgbm_bear.txt",
    "lgbm_meta.json",
    "lgbm_v2_risk.txt",
    "lgbm_v2_risk.meta.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("packaging/runtime-models"))
    parser.add_argument("--require-risk", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for name in MODEL_FILES:
        source = args.source / name
        if source.is_file():
            target = args.output / name
            shutil.copy2(source, target)
            copied[name] = sha256(target)
    if args.require_risk and "lgbm_v2_risk.txt" not in copied:
        raise SystemExit("release build requires lgbm_v2_risk.txt")
    (args.output / "bundle-manifest.json").write_text(
        json.dumps({"sha256": copied}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"staged {len(copied)} inference assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
