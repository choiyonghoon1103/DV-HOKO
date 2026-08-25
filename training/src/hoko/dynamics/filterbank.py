"""Class-independent balanced frequency-attention carrier filter bank."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BalancedFrequencyAttentionFilterBank(nn.Module):
    """Learn a smooth partition of a one-sided spectrum without fixed band edges.

    Column and row Sinkhorn normalization makes every frequency fully assigned
    while preventing unused carrier channels.  No class or fault coordinate is
    an input to this module.
    """

    def __init__(
        self,
        *,
        sample_count: int,
        band_count: int,
        coordinate_harmonics: int,
        sinkhorn_iterations: int,
    ) -> None:
        super().__init__()
        if (
            sample_count < 16
            or sample_count % 2
            or band_count < 2
            or coordinate_harmonics < 1
            or sinkhorn_iterations < 1
        ):
            raise ValueError("invalid learned carrier filter-bank geometry")
        self.sample_count = int(sample_count)
        self.band_count = int(band_count)
        self.coordinate_harmonics = int(coordinate_harmonics)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        frequency = torch.linspace(0.0, 1.0, sample_count // 2 + 1)
        features = [torch.ones_like(frequency), frequency, frequency.square()]
        for harmonic in range(1, coordinate_harmonics + 1):
            features.extend(
                (
                    torch.sin(math.pi * harmonic * frequency),
                    torch.cos(math.pi * harmonic * frequency),
                )
            )
        coordinate = torch.stack(features, dim=-1)
        coordinate = coordinate / coordinate.square().mean(dim=0, keepdim=True).sqrt().clamp_min(
            1e-6
        )
        self.register_buffer("frequency_coordinate", coordinate)
        self.band_queries = nn.Parameter(torch.empty(band_count, coordinate.shape[-1]))
        nn.init.normal_(self.band_queries, std=coordinate.shape[-1] ** -0.5)

    def masks(self) -> Tensor:
        logits = self.band_queries @ self.frequency_coordinate.T
        logits = logits / math.sqrt(self.frequency_coordinate.shape[-1])
        values = torch.exp(logits - logits.max()).clamp_min(torch.finfo(logits.dtype).tiny)
        target_row_mass = values.new_tensor(values.shape[1] / values.shape[0])
        for _ in range(self.sinkhorn_iterations):
            values = values / values.sum(dim=0, keepdim=True).clamp_min(1e-12)
            values = values * (
                target_row_mass / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
            )
        # The final column projection enforces exact spectral partition.  Row
        # masses converge to equal use under the preceding alternating steps.
        return values / values.sum(dim=0, keepdim=True).clamp_min(1e-12)

    def analytic_envelopes(self, one_sided_spectrum: Tensor) -> Tensor:
        """Return analytic-signal magnitudes for every learned carrier."""

        expected = self.sample_count // 2 + 1
        if (
            one_sided_spectrum.ndim != 2
            or one_sided_spectrum.shape[1] != expected
            or not torch.is_complex(one_sided_spectrum)
        ):
            raise ValueError("spectrum must be complex [second,rfft_bin]")
        masked = one_sided_spectrum[:, None] * self.masks()[None]
        analytic_spectrum = masked.new_zeros(
            (len(masked), self.band_count, self.sample_count)
        )
        analytic_spectrum[..., 0] = masked[..., 0]
        analytic_spectrum[..., 1 : expected - 1] = 2.0 * masked[..., 1:-1]
        analytic_spectrum[..., expected - 1] = masked[..., -1]
        return torch.fft.ifft(analytic_spectrum, dim=-1).abs()


__all__ = ["BalancedFrequencyAttentionFilterBank"]


class BalancedSubbandAttentionMixer(nn.Module):
    """Learn a balanced grouping of a generic uniform subband envelope bank."""

    def __init__(
        self,
        *,
        atom_count: int,
        band_count: int,
        coordinate_harmonics: int,
        sinkhorn_iterations: int,
    ) -> None:
        super().__init__()
        if min(atom_count, band_count, coordinate_harmonics, sinkhorn_iterations) < 1:
            raise ValueError("invalid subband mixer geometry")
        self.atom_count = int(atom_count)
        self.band_count = int(band_count)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        coordinate = torch.linspace(0.0, 1.0, atom_count)
        features = [torch.ones_like(coordinate), coordinate, coordinate.square()]
        for harmonic in range(1, coordinate_harmonics + 1):
            features.extend(
                (torch.sin(math.pi * harmonic * coordinate), torch.cos(math.pi * harmonic * coordinate))
            )
        features = torch.stack(features, dim=-1)
        features = features / features.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
        self.register_buffer("subband_coordinate", features)
        self.band_queries = nn.Parameter(torch.empty(band_count, features.shape[-1]))
        nn.init.normal_(self.band_queries, std=features.shape[-1] ** -0.5)

    def masks(self) -> Tensor:
        logits = self.band_queries @ self.subband_coordinate.T
        values = torch.exp(logits / math.sqrt(self.subband_coordinate.shape[-1]) - logits.max())
        target = values.new_tensor(values.shape[1] / values.shape[0])
        for _ in range(self.sinkhorn_iterations):
            values = values / values.sum(dim=0, keepdim=True).clamp_min(1e-12)
            values = values * target / values.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return values / values.sum(dim=0, keepdim=True).clamp_min(1e-12)

    def forward(self, envelopes: Tensor) -> Tensor:
        if envelopes.ndim != 3 or envelopes.shape[1] != self.atom_count:
            raise ValueError("subband envelopes must be [second,atom,angular_sample]")
        return torch.einsum("ba,san->sbn", self.masks(), envelopes)


__all__.append("BalancedSubbandAttentionMixer")
