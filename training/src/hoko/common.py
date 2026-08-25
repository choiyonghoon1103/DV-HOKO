"""Small deterministic primitives shared by HOKO training stages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_prefix_sum(local: Tensor) -> Tensor:
    """Accumulate ordered evidence without CUDA's nondeterministic cumsum path."""

    if local.ndim != 2 or len(local) < 1:
        raise ValueError("operation evidence must be a nonempty [time,class] tensor")
    running = torch.zeros_like(local[0])
    prefixes = []
    for value in local.unbind(dim=0):
        running = running + value
        prefixes.append(running)
    return torch.stack(prefixes)


def order_grid(config: dict) -> np.ndarray:
    observation = config["observation"]
    return np.arange(
        float(observation["minimum_order"]),
        float(observation["maximum_order"])
        + 0.5 * float(observation["order_step"]),
        float(observation["order_step"]),
        dtype=np.float32,
    )

