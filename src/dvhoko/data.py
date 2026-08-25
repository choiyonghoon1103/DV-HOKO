"""Small dataset readers used by the public evaluation scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


HUST_CLASSES = ("N", "I", "O", "B")
FINAL_CLASSES = ("BACKGROUND", "LEAK")


@dataclass(frozen=True)
class FinalStream:
    record_id: str
    role: str
    observations: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    packet_counts: np.ndarray


def load_hust_target(data_root: str | Path, bearing: str):
    root = Path(data_root)
    digit = int(bearing) - 6200
    rows = []
    for label_index, label in enumerate(HUST_CLASSES):
        for load in (0, 2, 4):
            path = root / f"{label}{digit}0{load}.mat"
            archive = loadmat(path, variable_names=("data", "fs"))
            rows.append({
                "record_id": path.stem, "class_index": label_index, "load": load,
                "waveform": np.asarray(archive["data"], dtype=np.float32).reshape(-1),
                "shaft_frequency_hz": float(np.asarray(archive["fs"]).squeeze()),
            })
    return rows


def load_final_streams(data_root: str | Path) -> list[FinalStream]:
    root = Path(data_root)
    with (root / "index.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    streams = []
    for row in rows:
        record_id = row["id"]
        spectra = np.load(root / "spectra" / f"{record_id}.npy", mmap_mode="r")
        labels = np.load(root / "labels" / f"{record_id}.npy", mmap_mode="r")
        timestamps = np.load(root / "timestamps" / f"{record_id}.npy", mmap_mode="r")
        starts = np.r_[0, np.flatnonzero(np.diff(timestamps) != 0) + 1].astype(np.int64)
        ends = np.r_[starts[1:], len(timestamps)].astype(np.int64)
        observations = np.empty((len(starts), 320), dtype=np.float32)
        second_labels = np.empty(len(starts), dtype=np.int64)
        for index, (start, end) in enumerate(zip(starts, ends)):
            observations[index] = np.asarray(spectra[start:end]).mean(0)
            unique = np.unique(labels[start:end])
            if len(unique) != 1 or int(unique[0]) not in (0, 1):
                raise ValueError(f"ambiguous external label in record {record_id}")
            second_labels[index] = int(unique[0])
        streams.append(FinalStream(
            record_id, row["role"], observations, second_labels,
            np.asarray(timestamps[starts], dtype=np.int64), (ends - starts).astype(np.int64),
        ))
    return streams
