"""Native Data_final loading with an explicit one-observation-per-second contract."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


BIN_COUNT = 320
CLASS_NAMES = ("BACKGROUND", "LEAK")


@dataclass(frozen=True)
class SpectrumEpisode:
    record_id: str
    session: str
    equipment: str
    label: int
    spectra: np.ndarray = field(repr=False, compare=False)
    starts: np.ndarray = field(repr=False, compare=False)
    ends: np.ndarray = field(repr=False, compare=False)
    timestamps: np.ndarray = field(repr=False, compare=False)

    @property
    def event_count(self) -> int:
        return len(self.starts)

    def observation(self, index: int) -> np.ndarray:
        start, end = int(self.starts[index]), int(self.ends[index])
        return np.asarray(self.spectra[start:end], dtype=np.float32).mean(axis=0)

    def causal_window(self, endpoint: int, maximum_events: int) -> np.ndarray:
        if endpoint < 0 or endpoint >= self.event_count or maximum_events < 1:
            raise ValueError("invalid causal endpoint")
        start = endpoint
        while (
            start > 0
            and endpoint - start + 1 < maximum_events
            and int(self.timestamps[start]) == int(self.timestamps[start - 1]) + 1
        ):
            start -= 1
        return np.stack([self.observation(i) for i in range(start, endpoint + 1)])


@dataclass(frozen=True)
class ExternalStream:
    record_id: str
    role: str
    equipment: str
    observations: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    packet_counts: np.ndarray


def _index_rows(root: Path) -> list[dict[str, str]]:
    with (root / "index.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _groups(timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if timestamps.ndim != 1 or len(timestamps) < 1 or np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be a nonempty chronological vector")
    starts = np.r_[0, np.flatnonzero(np.diff(timestamps) != 0) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(timestamps)].astype(np.int64)
    return starts, ends


def _sessions(root: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    ordered = []
    for row in rows:
        timestamps = np.load(root / "timestamps" / f"{row['id']}.npy", mmap_mode="r")
        ordered.append((int(timestamps[0]), row["id"], row["equipment"]))
    ordered.sort()
    groups: list[list[tuple[int, str, str]]] = []
    for item in ordered:
        if not groups or item[0] - groups[-1][0][0] > 30:
            groups.append([item])
        else:
            groups[-1].append(item)
    if len(groups) != 24:
        raise ValueError(f"expected 24 source sessions, found {len(groups)}")
    result = {}
    for index, group in enumerate(groups, start=1):
        equipments = [item[2] for item in group]
        if len(equipments) != len(set(equipments)):
            raise ValueError("a source session repeats an equipment")
        for _, record_id, _ in group:
            result[record_id] = f"S{index:02d}"
    return result


def load_source_episodes(root: str | Path) -> list[SpectrumEpisode]:
    """Load only 2024 valid constant-label episodes; `-1` is never an observation."""

    root = Path(root)
    rows = _index_rows(root)
    sessions = _sessions(root, rows)
    episodes: list[SpectrumEpisode] = []
    for row in rows:
        record_id = row["id"]
        spectra = np.load(root / "spectra" / f"{record_id}.npy", mmap_mode="r")
        labels = np.load(root / "labels" / f"{record_id}.npy", mmap_mode="r")
        timestamps = np.load(root / "timestamps" / f"{record_id}.npy", mmap_mode="r")
        if spectra.shape != (len(labels), BIN_COUNT) or len(labels) != len(timestamps):
            raise ValueError(f"unaligned source record {record_id}")
        starts, ends = _groups(timestamps)
        group_labels = np.full(len(starts), -1, dtype=np.int8)
        for index, (start, end) in enumerate(zip(starts, ends)):
            unique = np.unique(labels[int(start) : int(end)])
            if len(unique) == 1 and int(unique[0]) in (0, 1):
                group_labels[index] = int(unique[0])
        block_start = 0
        while block_start < len(starts):
            if int(group_labels[block_start]) == -1:
                block_start += 1
                continue
            label = int(group_labels[block_start])
            block_end = block_start + 1
            while block_end < len(starts) and int(group_labels[block_end]) != -1:
                if int(group_labels[block_end]) != label:
                    raise ValueError(f"valid source block changes class in {record_id}")
                block_end += 1
            episodes.append(
                SpectrumEpisode(
                    record_id=record_id,
                    session=sessions[record_id],
                    equipment=row["equipment"],
                    label=label,
                    spectra=spectra,
                    starts=starts[block_start:block_end],
                    ends=ends[block_start:block_end],
                    timestamps=np.asarray(timestamps[starts[block_start:block_end]], dtype=np.int64),
                )
            )
            block_start = block_end
    if len(episodes) != 236 or {episode.label for episode in episodes} != {0, 1}:
        raise ValueError("source universe must contain 236 binary episodes")
    return episodes


def load_external_streams(root: str | Path) -> list[ExternalStream]:
    root = Path(root)
    streams = []
    for row in _index_rows(root):
        record_id = row["id"]
        spectra = np.load(root / "spectra" / f"{record_id}.npy", mmap_mode="r")
        labels = np.load(root / "labels" / f"{record_id}.npy", mmap_mode="r")
        timestamps = np.load(root / "timestamps" / f"{record_id}.npy", mmap_mode="r")
        starts, ends = _groups(timestamps)
        observations = np.empty((len(starts), BIN_COUNT), dtype=np.float32)
        second_labels = np.empty(len(starts), dtype=np.int64)
        for index, (start, end) in enumerate(zip(starts, ends)):
            observations[index] = np.asarray(spectra[int(start) : int(end)]).mean(axis=0)
            unique = np.unique(labels[int(start) : int(end)])
            if len(unique) != 1 or int(unique[0]) not in (0, 1):
                raise ValueError(f"external second has ambiguous label in {record_id}")
            second_labels[index] = int(unique[0])
        streams.append(
            ExternalStream(
                record_id=record_id,
                role=row["role"],
                equipment=row["equipment"],
                observations=observations,
                labels=second_labels,
                timestamps=np.asarray(timestamps[starts], dtype=np.int64),
                packet_counts=(ends - starts).astype(np.int64),
            )
        )
    return streams
