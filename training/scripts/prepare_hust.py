#!/usr/bin/env python3
"""Create the deterministic raw-HUST uniform-subband cache used by HOKO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hoko.data.hust import discover_hust_records
from hoko.common import sha256
from hoko.dynamics.train import materialize_uniform_subband_envelopes


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/hust/dynamics.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/hust/subband_cache"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {args.output}")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    bearings = tuple(str(value) for value in config["bearings"])
    rows = discover_hust_records(args.data_root, bearings)
    materialize_uniform_subband_envelopes(rows, args.data_root.resolve(), config)
    args.output.mkdir(parents=True, exist_ok=False)
    shapes = {}
    for row in rows:
        record_id = str(row["record_id"])
        values = np.asarray(row.pop("uniform_subband_envelopes"), dtype=np.float32)
        np.save(args.output / f"{record_id}.npy", values, allow_pickle=False)
        shapes[record_id] = list(values.shape)
    manifest = {
        "schema": "hoko-hust-uniform-subband-cache-v1",
        "config_sha256": sha256(args.config.resolve()),
        "data_root_record_count": len(rows),
        "resources": {str(row["record_id"]): row["resource_sha256"] for row in rows},
        "shapes": shapes,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
