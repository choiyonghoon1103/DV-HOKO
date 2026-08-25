"""Dataset-native observation adapters; the DV-HOKO core is unchanged."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from dvhoko.model import BalancedSubbandAttentionMixer


def mode_grid(config: dict, device: torch.device) -> Tensor:
    section = config["observation"]
    values = np.arange(
        float(section["minimum_mode"]),
        float(section["maximum_mode"]) + 0.5 * float(section["mode_step"]),
        float(section["mode_step"]), dtype=np.float32,
    )
    return torch.from_numpy(values).to(device)


def _selected_dft(windows: Tensor, indices: Tensor, taper: Tensor) -> Tensor:
    length = windows.shape[-1]
    position = torch.arange(length, device=windows.device, dtype=windows.dtype)
    phase = 2.0 * torch.pi * indices.to(windows.dtype)[:, None] * position[None] / length
    weighted = windows * taper
    scale = taper.square().sum().sqrt()
    real = torch.einsum("...n,kn->...k", weighted, torch.cos(phase)) / scale
    imaginary = -torch.einsum("...n,kn->...k", weighted, torch.sin(phase)) / scale
    return torch.stack((real, imaginary), dim=-1)


def coordinate_paths_and_state(
    mixer: BalancedSubbandAttentionMixer, coordinates: Tensor, config: dict
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert `[observation,atom,coordinate]` values to the shared model interface."""

    section = config["observation"]
    local = torch.log1p(mixer(coordinates.clamp_min(0.0)))
    level = local.mean(-1)
    centered = local - level[..., None]
    spread = centered.square().mean(-1).add(1e-8).sqrt()
    state = torch.cat((level, spread), dim=-1)
    samples_per_unit = int(section["samples_per_unit"])
    window = int(round(float(section["window_units"]) * samples_per_unit))
    hop = int(round(float(section["hop_units"]) * samples_per_unit))
    windows = centered.unfold(-1, window, hop)
    taper = torch.hann_window(window, periodic=False, device=coordinates.device, dtype=coordinates.dtype)
    modes = mode_grid(config, coordinates.device)
    indices = torch.round(modes * window / samples_per_unit).long()
    if int(indices.max()) > window // 2:
        raise ValueError("requested mode exceeds local Fourier support")
    selected = _selected_dft(windows, indices, taper)
    paths = selected.permute(0, 2, 1, 3, 4).contiguous()
    return paths, state, modes


def hust_uniform_envelopes(
    waveform: np.ndarray, *, shaft_frequency_hz: float, config: dict
) -> np.ndarray:
    """Convert complete 51.2-kHz seconds to generic angular subband envelopes."""

    section, observation = config["learned_filterbank"], config["observation"]
    sample_count = int(section["sample_count"])
    atom_count = int(observation["spectral_atoms"])
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    count = len(values) // sample_count
    if count < 1 or shaft_frequency_hz <= 0:
        raise ValueError("HUST record lacks a complete second or valid shaft clock")
    blocks = values[: count * sample_count].reshape(count, sample_count)
    blocks = blocks - blocks.mean(1, keepdims=True)
    spectrum = np.fft.rfft(blocks, axis=-1)
    bins = spectrum.shape[-1]
    edges = np.linspace(0, bins, atom_count + 1, dtype=np.int64)
    analytic = np.zeros((count, atom_count, sample_count), dtype=np.complex64)
    for atom, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        local = np.zeros((count, sample_count), dtype=np.complex64)
        local[:, left:right] = spectrum[:, left:right]
        local[:, max(left, 1) : min(right, bins - 1)] *= 2.0
        analytic[:, atom] = np.fft.ifft(local, axis=-1)
    envelopes = np.abs(analytic).astype(np.float32)
    samples_per_unit = int(observation["samples_per_unit"])
    units = int(np.floor((sample_count - 1) * shaft_frequency_hz / sample_count))
    coordinate_count = units * samples_per_unit
    positions = np.arange(coordinate_count) * sample_count / (
        samples_per_unit * shaft_frequency_hz
    )
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, sample_count - 1)
    fraction = (positions - left).astype(np.float32)
    return envelopes[..., left] * (1.0 - fraction) + envelopes[..., right] * fraction


def final_coordinates(spectra: Tensor, source_scale: float, config: dict) -> Tensor:
    section = config["observation"]
    atoms = int(section["spectral_atoms"])
    width = int(section["coordinates_per_atom"])
    if spectra.ndim != 2 or spectra.shape[1] != atoms * width or source_scale <= 0:
        raise ValueError("invalid Data_final spectrum geometry or source scale")
    return spectra.reshape(len(spectra), atoms, width) / source_scale
