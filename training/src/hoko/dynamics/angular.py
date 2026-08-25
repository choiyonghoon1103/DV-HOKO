"""Class-free causal continuous-order observation for angular vibration streams."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LearnedCausalAngularModeBank(nn.Module):
    """Learn continuous angular frequencies and causal memory scales.

    One coefficient is emitted at each completed revolution.  Every coefficient
    uses only samples at or before that endpoint.  Positive frequency increments
    keep the mode coordinate ordered and span the complete observable angular
    Nyquist interval without a hand-selected fault-order crop.
    """

    def __init__(
        self,
        *,
        mode_count: int,
        samples_per_revolution: int,
        initial_memory_revolutions: float = 1.0,
    ) -> None:
        super().__init__()
        if mode_count < 4 or samples_per_revolution < 8:
            raise ValueError("angular mode-bank geometry is too small")
        if initial_memory_revolutions <= 0.0:
            raise ValueError("initial causal memory must be positive")
        self.mode_count = int(mode_count)
        self.samples_per_revolution = int(samples_per_revolution)
        # Equal positive increments initialize a full-range, class-independent
        # coordinate.  Learning changes the spacing but cannot collapse or swap
        # the modes.
        initial_increment = math.log(math.expm1(1.0))
        self.raw_frequency_increments = nn.Parameter(
            torch.full((mode_count,), initial_increment)
        )
        initial_scale = math.log(math.expm1(float(initial_memory_revolutions)))
        self.raw_memory_scales = nn.Parameter(torch.full((mode_count,), initial_scale))

    @property
    def nyquist_order(self) -> float:
        return 0.5 * self.samples_per_revolution

    def orders(self) -> Tensor:
        increments = F.softplus(self.raw_frequency_increments) + 1e-6
        # CUDA cumsum currently has no deterministic backward.  This short
        # elementwise recurrence is algebraically identical and deterministic.
        running = increments[0]
        cumulative_values = [running]
        for index in range(1, self.mode_count):
            running = running + increments[index]
            cumulative_values.append(running)
        cumulative = torch.stack(cumulative_values)
        return self.nyquist_order * cumulative / cumulative[-1]

    def memory_scales(self) -> Tensor:
        return F.softplus(self.raw_memory_scales) + 1e-3

    def forward(self, angular: Tensor) -> Tensor:
        """Return `[second,revolution,band,mode,complex_pair]` paths."""

        if angular.ndim != 3:
            raise ValueError("angular observations must be [second,band,sample]")
        seconds, bands, samples = angular.shape
        revolutions = samples // self.samples_per_revolution
        if revolutions < 2:
            raise ValueError("learned angular modes require two completed revolutions")
        usable = revolutions * self.samples_per_revolution
        values = angular[..., :usable]
        sample_index = torch.arange(usable, device=angular.device, dtype=angular.dtype)
        endpoints = (
            torch.arange(1, revolutions + 1, device=angular.device, dtype=angular.dtype)
            * self.samples_per_revolution
            - 1
        )
        lag = (endpoints[:, None] - sample_index[None]) / self.samples_per_revolution
        causal = lag >= 0
        nonnegative_lag = lag.clamp_min(0)
        orders = self.orders().to(dtype=angular.dtype)
        scales = self.memory_scales().to(dtype=angular.dtype)
        taper = torch.exp(
            -0.5
            * (
                nonnegative_lag[None]
                / scales[:, None, None]
            ).square()
        ) * causal[None]
        phase = 2.0 * math.pi * orders[:, None, None] * nonnegative_lag[None]
        scale = taper.square().sum(dim=-1).clamp_min(1e-8).sqrt()
        real_kernel = taper * torch.cos(phase) / scale[..., None]
        imaginary_kernel = taper * torch.sin(phase) / scale[..., None]
        real = torch.einsum("sbl,mrl->srbm", values, real_kernel)
        imaginary = torch.einsum("sbl,mrl->srbm", values, imaginary_kernel)
        return torch.stack((real, imaginary), dim=-1)
