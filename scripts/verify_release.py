#!/usr/bin/env python3
"""Verify that every released checkpoint matches the immutable manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "weights/manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = ROOT / "weights" / relative
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"checkpoint hash differs: {relative}")
        print(f"ok  {relative}  {observed}")


if __name__ == "__main__":
    main()
