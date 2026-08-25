"""Thin native-spectrum adapter for the shared DV-HOKO core."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from dvhoko.model import BalancedSubbandAttentionMixer, DualViewKoopmanMoriField


def mode_grid(config: dict) -> np.ndarray:
    observation = config["observation"]
    return np.arange(
        float(observation["minimum_mode"]),
        float(observation["maximum_mode"]) + 0.5 * float(observation["mode_step"]),
        float(observation["mode_step"]),
        dtype=np.float32,
    )


def build_model(config: dict, device: torch.device) -> DualViewKoopmanMoriField:
    observation, architecture = config["observation"], config["architecture"]
    return DualViewKoopmanMoriField(
        carrier_bands=int(observation["carrier_bands"]),
        field_width=int(architecture["field_width"]),
        embedding_width=int(architecture["embedding_width"]),
        operation_count=int(architecture["operation_count"]),
        attention_heads=int(architecture["attention_heads"]),
        temporal_layers=int(architecture["temporal_layers"]),
        feedforward_width=int(architecture["feedforward_width"]),
        mixture_count=int(architecture["mixture_count"]),
        minimum_query_width=float(architecture["minimum_query_width_modes"]),
        maximum_query_width=float(architecture["maximum_query_width_modes"]),
        hop_revolutions=float(observation["hop_units"]),
        forecast_horizons=int(architecture["forecast_horizons"]),
        state_width=int(architecture["state_width"]),
    ).to(device)


def build_mixer(config: dict, device: torch.device) -> BalancedSubbandAttentionMixer:
    observation, mixer = config["observation"], config["learned_filterbank"]
    return BalancedSubbandAttentionMixer(
        atom_count=int(observation["spectral_atoms"]),
        band_count=int(observation["carrier_bands"]),
        harmonics=int(mixer["coordinate_harmonics"]),
        iterations=int(mixer["sinkhorn_iterations"]),
    ).to(device)


def spectrum_paths_and_state(
    mixer: BalancedSubbandAttentionMixer,
    spectra: Tensor,
    source_scale: float,
    config: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map a native spectrum to the exact path/state interface used by DV-HOKO.

    The coordinate is normalized spectral position rather than shaft angle.  This is
    intentionally confined to the adapter: the Koopman--Mori model is unchanged.
    """

    observation = config["observation"]
    atoms = int(observation["spectral_atoms"])
    coordinates = int(observation["coordinates_per_atom"])
    if spectra.ndim != 2 or spectra.shape[1] != atoms * coordinates:
        raise ValueError("Final spectrum does not match its registered native geometry")
    if source_scale <= 0.0:
        raise ValueError("source amplitude scale must be positive")
    native = spectra.clamp_min(0.0).reshape(len(spectra), atoms, coordinates)
    local = torch.log1p(mixer(native / source_scale))
    level = local.mean(dim=-1)
    centered = local - level[..., None]
    spread = centered.square().mean(dim=-1).add(1e-8).sqrt()
    state_observation = torch.cat((level, spread), dim=-1)

    samples_per_unit = int(observation["samples_per_unit"])
    window = int(round(float(observation["window_units"]) * samples_per_unit))
    hop = int(round(float(observation["hop_units"]) * samples_per_unit))
    if window > coordinates or min(window, hop) < 1:
        raise ValueError("invalid Final coordinate window")
    windows = centered.unfold(-1, window, hop)
    taper = torch.hann_window(
        window, periodic=False, device=spectra.device, dtype=spectra.dtype
    )
    orders = torch.from_numpy(mode_grid(config)).to(spectra.device)
    indices = torch.round(orders * window / samples_per_unit).long()
    if int(indices.max()) > window // 2:
        raise ValueError("registered Final mode exceeds the local rFFT support")
    # Compute only the registered real-FFT modes.  This is algebraically identical to
    # indexing torch.fft.rfft but avoids platform-specific cuFFT failures and keeps a
    # fully differentiable path from the forecast loss to the learned subband mixer.
    position = torch.arange(window, device=spectra.device, dtype=spectra.dtype)
    phase = 2.0 * torch.pi * indices.to(spectra.dtype)[:, None] * position[None] / window
    weighted = windows * taper
    normalization = taper.square().sum().sqrt()
    real = torch.einsum("...n,kn->...k", weighted, torch.cos(phase)) / normalization
    imaginary = -torch.einsum("...n,kn->...k", weighted, torch.sin(phase)) / normalization
    selected = torch.stack((real, imaginary), dim=-1)
    paths = selected.permute(0, 2, 1, 3, 4).contiguous()
    return paths, state_observation, orders
