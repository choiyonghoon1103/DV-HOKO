"""Raw HUST adapter for the public four-bearing N/I/O/B benchmark.

The adapter only enumerates records and validates the measured shaft-frequency
field.  Signal-derived subband envelopes are materialized by the training
module so the exact model observation remains in one place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from hoko.common import sha256


CLASS_NAMES = ("N", "I", "O", "B")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
LOADS = ("00", "02", "04")


def discover_hust_records(
    data_root: Path,
    bearings: tuple[str, ...] = ("6205", "6206", "6207", "6208"),
) -> list[dict]:
    """Return the complete record index without computing target statistics."""

    root = Path(data_root).resolve()
    rows: list[dict] = []
    missing: list[str] = []
    for bearing in bearings:
        suffix = bearing[-1]
        for class_name in CLASS_NAMES:
            for load in LOADS:
                record_id = f"{class_name}{suffix}{load}"
                path = root / f"{record_id}.mat"
                if not path.is_file():
                    missing.append(path.name)
                    continue
                archive = loadmat(path, variable_names=("data", "fs"))
                waveform = np.asarray(archive["data"]).reshape(-1)
                shaft_hz = float(np.asarray(archive["fs"]).squeeze())
                if waveform.size < 51_200 or not np.isfinite(shaft_hz) or shaft_hz <= 0:
                    raise ValueError(f"invalid HUST record: {path}")
                rows.append(
                    {
                        "record_id": record_id,
                        "bearing": str(bearing),
                        "class_name": class_name,
                        "class_index": CLASS_INDEX[class_name],
                        "load": load,
                        "shaft_frequency_hz": shaft_hz,
                        "wall_seconds": int(waveform.size // 51_200),
                        "resource_path": str(path),
                        "resource_sha256": sha256(path),
                    }
                )
    if missing:
        raise FileNotFoundError(f"HUST data root lacks {len(missing)} records: {missing[:6]}")
    expected = len(bearings) * len(CLASS_NAMES) * len(LOADS)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} HUST records, found {len(rows)}")
    return rows


def load_subband_cache(
    rows: list[dict], cache_root: Path, *, allow_manifest_subset: bool = False
) -> None:
    """Attach a manifest-validated envelope cache to an existing record index.

    ``allow_manifest_subset`` is used by strict held-bearing experiments: the
    fit process may load only source records even though the immutable cache
    manifest also contains held-bearing entries.  Every requested record must
    still match its raw-resource hash; extra manifest entries are never loaded.
    """

    import json

    root = Path(cache_root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(row["record_id"]): str(row["resource_sha256"]) for row in rows}
    if manifest.get("schema") != "hoko-hust-uniform-subband-cache-v1":
        raise ValueError("unexpected HUST subband-cache schema")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise ValueError("HUST subband cache lacks a resource manifest")
    if allow_manifest_subset:
        if any(resources.get(record_id) != digest for record_id, digest in expected.items()):
            raise ValueError("HUST subband cache and requested raw resources differ")
    elif resources != expected:
        raise ValueError("HUST subband cache and raw-resource manifest differ")
    for row in rows:
        values = np.load(root / f"{row['record_id']}.npy", allow_pickle=False)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError(f"invalid subband cache for {row['record_id']}")
        row["uniform_subband_envelopes"] = values
