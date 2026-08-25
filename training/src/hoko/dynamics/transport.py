"""Resolved Koopman transport operators for complex angular paths."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _rotate(values: Tensor, angle: Tensor, radius: Tensor | float = 1.0) -> Tensor:
    """Apply a broadcast complex rotation to a final real/imaginary pair."""

    real, imaginary = values[..., 0], values[..., 1]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack(
        (
            radius * (cosine * real - sine * imaginary),
            radius * (sine * real + cosine * imaginary),
        ),
        dim=-1,
    )


class FixedRotationTransport(nn.Module):
    """The original analytic order-coordinate rotation comparator."""

    def __init__(self, hop_revolutions: float) -> None:
        super().__init__()
        self.hop_revolutions = float(hop_revolutions)

    def generator(self, orders: Tensor, bands: int) -> tuple[Tensor, Tensor]:
        phase = 2.0 * math.pi * self.hop_revolutions * orders
        return phase[None].expand(bands, -1), torch.zeros(
            bands, len(orders), device=orders.device, dtype=orders.dtype
        )

    def forward(self, values: Tensor, orders: Tensor, offset: int = 1) -> Tensor:
        phase, _ = self.generator(orders, values.shape[-3])
        angle = float(offset) * phase
        return _rotate(values, angle[None, None])

    def phase_lock(self, values: Tensor, orders: Tensor) -> Tensor:
        windows = values.shape[1]
        time = torch.arange(windows, device=values.device, dtype=values.dtype)
        phase, _ = self.generator(orders, values.shape[-3])
        angle = -time[:, None, None] * phase[None]
        return _rotate(values, angle[None])


class LearnedStableTransport(nn.Module):
    """Class-free stable continuous-time transport learned from source paths.

    A small coordinate network emits one complex generator per carrier band and
    observed order coordinate.  Its real part is non-positive, so every horizon
    belongs to the same contractive semigroup.  The imaginary part and damping
    are learned rather than copied from the manually specified rotation law.
    """

    def __init__(
        self,
        *,
        carrier_bands: int,
        hidden_width: int,
        coordinate_harmonics: int = 4,
        initial_decay: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(carrier_bands, hidden_width, coordinate_harmonics) < 1:
            raise ValueError("invalid learned Koopman transport geometry")
        if initial_decay <= 0.0:
            raise ValueError("initial Koopman damping must be positive")
        self.carrier_bands = int(carrier_bands)
        self.coordinate_harmonics = int(coordinate_harmonics)
        coordinate_width = 2 + 2 * self.coordinate_harmonics
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(coordinate_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, 2 * carrier_bands),
        )
        final = self.coordinate_encoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        inverse_softplus = math.log(math.expm1(float(initial_decay)))
        with torch.no_grad():
            final.bias[carrier_bands:].fill_(inverse_softplus)

    def _coordinates(self, orders: Tensor) -> Tensor:
        scale = orders.detach().abs().max().clamp_min(1.0)
        value = orders / scale
        features = [torch.ones_like(value), value]
        for harmonic in range(1, self.coordinate_harmonics + 1):
            features.extend(
                (torch.sin(math.pi * harmonic * value), torch.cos(math.pi * harmonic * value))
            )
        return torch.stack(features, dim=-1)

    def generator(self, orders: Tensor, bands: int) -> tuple[Tensor, Tensor]:
        if bands != self.carrier_bands:
            raise ValueError("carrier-band count differs from learned transport")
        raw = self.coordinate_encoder(self._coordinates(orders))
        raw_phase, raw_decay = raw.split(self.carrier_bands, dim=-1)
        # One-step phase is unrestricted on the circle.  Damping is non-negative,
        # hence exp(-h*damping) has spectral radius at most one at every horizon.
        phase = math.pi * torch.tanh(raw_phase).T
        decay = F.softplus(raw_decay).T
        return phase, decay

    def forward(self, values: Tensor, orders: Tensor, offset: int = 1) -> Tensor:
        if offset < 1:
            raise ValueError("Koopman transport offset must be positive")
        phase, decay = self.generator(orders, values.shape[-3])
        horizon = float(offset)
        angle = horizon * phase
        radius = torch.exp(-horizon * decay)
        return _rotate(values, angle[None, None], radius[None, None])

    def phase_lock(self, values: Tensor, orders: Tensor) -> Tensor:
        windows = values.shape[1]
        time = torch.arange(windows, device=values.device, dtype=values.dtype)
        phase, _ = self.generator(orders, values.shape[-3])
        angle = -time[:, None, None] * phase[None]
        return _rotate(values, angle[None])


class LearnedUnitaryTransport(nn.Module):
    """Source-learned energy-preserving complex Koopman transport.

    The model learns phase evolution from the order coordinate but cannot lower
    forecast loss by shrinking every state toward zero.  Attenuation and other
    unresolved effects remain the responsibility of the Mori closure.
    """

    def __init__(
        self,
        *,
        carrier_bands: int,
        hidden_width: int,
        coordinate_harmonics: int = 4,
    ) -> None:
        super().__init__()
        if min(carrier_bands, hidden_width, coordinate_harmonics) < 1:
            raise ValueError("invalid learned unitary transport geometry")
        self.carrier_bands = int(carrier_bands)
        self.coordinate_harmonics = int(coordinate_harmonics)
        coordinate_width = 2 + 2 * self.coordinate_harmonics
        self.coordinate_encoder = nn.Sequential(
            nn.Linear(coordinate_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, carrier_bands),
        )
        nn.init.normal_(self.coordinate_encoder[-1].weight, std=1e-2)
        nn.init.zeros_(self.coordinate_encoder[-1].bias)

    def _coordinates(self, orders: Tensor) -> Tensor:
        scale = orders.detach().abs().max().clamp_min(1.0)
        value = orders / scale
        features = [torch.ones_like(value), value]
        for harmonic in range(1, self.coordinate_harmonics + 1):
            features.extend(
                (torch.sin(math.pi * harmonic * value), torch.cos(math.pi * harmonic * value))
            )
        return torch.stack(features, dim=-1)

    def generator(self, orders: Tensor, bands: int) -> tuple[Tensor, Tensor]:
        if bands != self.carrier_bands:
            raise ValueError("carrier-band count differs from learned transport")
        phase = math.pi * torch.tanh(self.coordinate_encoder(self._coordinates(orders))).T
        decay = torch.zeros_like(phase)
        return phase, decay

    def forward(self, values: Tensor, orders: Tensor, offset: int = 1) -> Tensor:
        if offset < 1:
            raise ValueError("Koopman transport offset must be positive")
        phase, _ = self.generator(orders, values.shape[-3])
        return _rotate(values, (float(offset) * phase)[None, None])

    def phase_lock(self, values: Tensor, orders: Tensor) -> Tensor:
        windows = values.shape[1]
        time = torch.arange(windows, device=values.device, dtype=values.dtype)
        phase, _ = self.generator(orders, values.shape[-3])
        return _rotate(values, (-time[:, None, None] * phase[None])[None])


class EquivariantResidualTransport(nn.Module):
    """Exact angular group action plus a source-learned unitary detuning field.

    The analytic term is a coordinate identity of the complex order transform,
    not a class-specific fault rule.  The neural residual learns deviations from
    ideal angular equivariance while retaining the original model as the exact
    zero-residual initialization.
    """

    def __init__(
        self,
        *,
        carrier_bands: int,
        hop_revolutions: float,
        hidden_width: int,
        coordinate_harmonics: int = 4,
    ) -> None:
        super().__init__()
        if min(carrier_bands, hidden_width, coordinate_harmonics) < 1:
            raise ValueError("invalid equivariant residual transport geometry")
        self.carrier_bands = int(carrier_bands)
        self.hop_revolutions = float(hop_revolutions)
        self.coordinate_harmonics = int(coordinate_harmonics)
        coordinate_width = 2 + 2 * self.coordinate_harmonics
        self.detuning = nn.Sequential(
            nn.Linear(coordinate_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, carrier_bands),
        )
        nn.init.zeros_(self.detuning[-1].weight)
        nn.init.zeros_(self.detuning[-1].bias)

    def _coordinates(self, orders: Tensor) -> Tensor:
        scale = orders.detach().abs().max().clamp_min(1.0)
        value = orders / scale
        features = [torch.ones_like(value), value]
        for harmonic in range(1, self.coordinate_harmonics + 1):
            features.extend(
                (torch.sin(math.pi * harmonic * value), torch.cos(math.pi * harmonic * value))
            )
        return torch.stack(features, dim=-1)

    def generator(self, orders: Tensor, bands: int) -> tuple[Tensor, Tensor]:
        if bands != self.carrier_bands:
            raise ValueError("carrier-band count differs from residual transport")
        base = 2.0 * math.pi * self.hop_revolutions * orders[None]
        correction = math.pi * torch.tanh(self.detuning(self._coordinates(orders))).T
        phase = base + correction
        return phase, torch.zeros_like(phase)

    def forward(self, values: Tensor, orders: Tensor, offset: int = 1) -> Tensor:
        if offset < 1:
            raise ValueError("Koopman transport offset must be positive")
        phase, _ = self.generator(orders, values.shape[-3])
        return _rotate(values, (float(offset) * phase)[None, None])

    def phase_lock(self, values: Tensor, orders: Tensor) -> Tensor:
        windows = values.shape[1]
        time = torch.arange(windows, device=values.device, dtype=values.dtype)
        phase, _ = self.generator(orders, values.shape[-3])
        return _rotate(values, (-time[:, None, None] * phase[None])[None])


def build_transport(
    kind: str,
    *,
    carrier_bands: int,
    hop_revolutions: float,
    hidden_width: int,
    coordinate_harmonics: int,
    initial_decay: float,
) -> nn.Module:
    if kind == "fixed_rotation":
        return FixedRotationTransport(hop_revolutions)
    if kind == "learned_stable":
        return LearnedStableTransport(
            carrier_bands=carrier_bands,
            hidden_width=hidden_width,
            coordinate_harmonics=coordinate_harmonics,
            initial_decay=initial_decay,
        )
    if kind == "learned_unitary":
        return LearnedUnitaryTransport(
            carrier_bands=carrier_bands,
            hidden_width=hidden_width,
            coordinate_harmonics=coordinate_harmonics,
        )
    if kind == "equivariant_residual":
        return EquivariantResidualTransport(
            carrier_bands=carrier_bands,
            hop_revolutions=hop_revolutions,
            hidden_width=hidden_width,
            coordinate_harmonics=coordinate_harmonics,
        )
    raise ValueError(f"unknown Koopman transport: {kind}")


__all__ = [
    "FixedRotationTransport",
    "LearnedStableTransport",
    "LearnedUnitaryTransport",
    "EquivariantResidualTransport",
    "build_transport",
]
