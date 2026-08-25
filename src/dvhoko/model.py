"""Deployable DV-HOKO core shared by HUST and Data_final."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class BalancedSubbandAttentionMixer(nn.Module):
    """Learn a balanced grouping of generic contiguous spectral atoms."""

    def __init__(self, atom_count: int, band_count: int, harmonics: int, iterations: int):
        super().__init__()
        if min(atom_count, band_count, harmonics, iterations) < 1:
            raise ValueError("invalid mixer geometry")
        self.atom_count, self.band_count = int(atom_count), int(band_count)
        self.sinkhorn_iterations = int(iterations)
        coordinate = torch.linspace(0.0, 1.0, atom_count)
        features = [torch.ones_like(coordinate), coordinate, coordinate.square()]
        for harmonic in range(1, harmonics + 1):
            features.extend((
                torch.sin(math.pi * harmonic * coordinate),
                torch.cos(math.pi * harmonic * coordinate),
            ))
        features = torch.stack(features, dim=-1)
        features = features / features.square().mean(0, keepdim=True).sqrt().clamp_min(1e-6)
        self.register_buffer("subband_coordinate", features)
        self.band_queries = nn.Parameter(torch.empty(band_count, features.shape[-1]))
        nn.init.normal_(self.band_queries, std=features.shape[-1] ** -0.5)

    def masks(self) -> Tensor:
        logits = self.band_queries @ self.subband_coordinate.T
        values = torch.exp(logits / math.sqrt(self.subband_coordinate.shape[-1]) - logits.max())
        target = values.new_tensor(values.shape[1] / values.shape[0])
        for _ in range(self.sinkhorn_iterations):
            values = values / values.sum(0, keepdim=True).clamp_min(1e-12)
            values = values * target / values.sum(1, keepdim=True).clamp_min(1e-12)
        return values / values.sum(0, keepdim=True).clamp_min(1e-12)

    def forward(self, envelopes: Tensor) -> Tensor:
        if envelopes.ndim != 3 or envelopes.shape[1] != self.atom_count:
            raise ValueError("expected [observation,atom,coordinate]")
        return torch.einsum("ba,san->sbn", self.masks(), envelopes)


class DualViewKoopmanMoriField(nn.Module):
    """One shared neural trunk with centered dynamics and preserved state views."""

    def __init__(
        self, *, carrier_bands: int, field_width: int, state_width: int,
        embedding_width: int, operation_count: int, attention_heads: int,
        temporal_layers: int, feedforward_width: int, mixture_count: int,
        minimum_query_width: float, maximum_query_width: float,
        hop_revolutions: float, forecast_horizons: int,
    ) -> None:
        super().__init__()
        self.carrier_bands = int(carrier_bands)
        self.field_width = int(field_width)
        self.state_width = int(state_width)
        self.embedding_width = int(embedding_width)
        self.operation_count = int(operation_count)
        self.mixture_count = int(mixture_count)
        self.minimum_query_width = float(minimum_query_width)
        self.maximum_query_width = float(maximum_query_width)
        self.hop_revolutions = float(hop_revolutions)
        self.forecast_horizons = int(forecast_horizons)

        self.path_embedding = nn.Sequential(
            nn.Linear(6 * carrier_bands, embedding_width), nn.GELU(),
            nn.LayerNorm(embedding_width),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(5, embedding_width), nn.GELU(), nn.Linear(embedding_width, embedding_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_width, nhead=attention_heads,
            dim_feedforward=feedforward_width, dropout=0.0,
            batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.order_embedding = nn.Sequential(
            nn.Linear(5, embedding_width), nn.GELU(), nn.Linear(embedding_width, embedding_width)
        )
        self.field_decoder = nn.Sequential(
            nn.LayerNorm(embedding_width), nn.Linear(embedding_width, feedforward_width),
            nn.GELU(), nn.Linear(feedforward_width, field_width),
        )

        # These operation-reader parameters are retained for checkpoint compatibility.
        self.mode_score = nn.Sequential(
            nn.Linear(field_width, embedding_width), nn.GELU(),
            nn.LayerNorm(embedding_width), nn.Linear(embedding_width, 1),
        )
        self.class_embedding = nn.Parameter(torch.empty(operation_count, embedding_width))
        nn.init.normal_(self.class_embedding, std=embedding_width ** -0.5)
        self.query_generator = nn.Sequential(
            nn.Linear(embedding_width, embedding_width), nn.GELU(),
            nn.Linear(embedding_width, 3 * mixture_count),
        )
        self.context_mode_embedding = nn.Sequential(
            nn.LayerNorm(field_width), nn.Linear(field_width, embedding_width),
            nn.GELU(), nn.LayerNorm(embedding_width),
        )
        self.context_order_embedding = nn.Sequential(
            nn.Linear(5, embedding_width), nn.GELU(), nn.Linear(embedding_width, embedding_width)
        )
        self.context_query = nn.Parameter(torch.empty(embedding_width))
        nn.init.normal_(self.context_query, std=embedding_width ** -0.5)
        self.memory_forecast_head = nn.Sequential(
            nn.LayerNorm(field_width),
            nn.Linear(field_width, forecast_horizons * 2 * carrier_bands),
        )
        state_input = embedding_width + 2 * carrier_bands
        self.state_decoder = None
        if self.state_width > 0:
            self.state_decoder = nn.Sequential(
                nn.LayerNorm(state_input), nn.Linear(state_input, embedding_width),
                nn.GELU(), nn.Linear(embedding_width, state_width),
            )

    @staticmethod
    def _coordinate(value: Tensor, divisor: float) -> Tensor:
        normalized = value / divisor
        return torch.stack((
            normalized, torch.sin(math.pi * normalized), torch.cos(math.pi * normalized),
            torch.sin(2.0 * math.pi * normalized), torch.cos(2.0 * math.pi * normalized),
        ), dim=-1)

    def _causal_states(self, paths: Tensor, valid: Tensor, orders: Tensor):
        batch, windows, bands, order_count, _ = paths.shape
        power = paths.square().sum(dim=-1)
        weighted = power * valid[:, :, None, None]
        power_steps, count_steps = [], []
        running_power = weighted.new_zeros((batch, bands))
        running_count = weighted.new_zeros((batch,))
        for step in range(windows):
            running_power = running_power + weighted[:, step].sum(dim=2)
            running_count = running_count + valid[:, step].to(weighted.dtype)
            power_steps.append(running_power); count_steps.append(running_count)
        cumulative_power = torch.stack(power_steps, dim=1)
        cumulative_count = torch.stack(count_steps, dim=1).clamp_min(1) * order_count
        scale = (cumulative_power / cumulative_count[:, :, None]).add(1e-6).sqrt()
        normalized = paths / scale[:, :, :, None, None]

        time = torch.arange(windows, device=paths.device, dtype=paths.dtype)
        lock_angle = -2.0 * math.pi * time[:, None] * orders[None] * self.hop_revolutions
        lock_cos, lock_sin = torch.cos(lock_angle), torch.sin(lock_angle)
        locked = torch.stack((
            lock_cos[None, :, None] * normalized[..., 0]
            - lock_sin[None, :, None] * normalized[..., 1],
            lock_sin[None, :, None] * normalized[..., 0]
            + lock_cos[None, :, None] * normalized[..., 1],
        ), dim=-1)
        angle = 2.0 * math.pi * orders * self.hop_revolutions
        predicted = torch.stack((
            torch.cos(angle)[None, None, None] * normalized[:, :-1, ..., 0]
            - torch.sin(angle)[None, None, None] * normalized[:, :-1, ..., 1],
            torch.sin(angle)[None, None, None] * normalized[:, :-1, ..., 0]
            + torch.cos(angle)[None, None, None] * normalized[:, :-1, ..., 1],
        ), dim=-1)
        innovation = normalized.new_zeros(normalized.shape)
        innovation[:, 1:] = normalized[:, 1:] - predicted
        features = torch.cat((normalized, locked, innovation), dim=-1)
        tokens = features.permute(0, 3, 1, 2, 4).reshape(
            batch * order_count, windows, 6 * bands
        )
        tokens = self.path_embedding(tokens)
        tokens = tokens + self.time_embedding(self._coordinate(time, 32.0))[None]
        expanded_valid = valid[:, None].expand(-1, order_count, -1).reshape(
            batch * order_count, windows
        )
        mask = torch.triu(torch.ones(windows, windows, dtype=torch.bool, device=paths.device), 1)
        encoded = self.temporal_encoder(
            tokens, mask=mask, src_key_padding_mask=~expanded_valid
        ).reshape(batch, order_count, windows, self.embedding_width)
        return normalized, encoded

    def _joint_order_states(self, encoded: Tensor, orders: Tensor) -> Tensor:
        context = self.order_embedding(
            self._coordinate(orders, max(float(orders.max()), 1.0))
        )[None, :, None]
        return encoded + context

    def _field_and_forecast(self, paths: Tensor, valid: Tensor, orders: Tensor):
        normalized, encoded = self._causal_states(paths, valid, orders)
        batch, windows, bands, order_count, _ = normalized.shape
        state_field = self.field_decoder(self._joint_order_states(encoded, orders))
        raw = self.memory_forecast_head(state_field).reshape(
            batch, order_count, windows, self.forecast_horizons, bands, 2
        )
        losses = []
        for offset in range(1, self.forecast_horizons + 1):
            if offset >= windows:
                continue
            angle = 2.0 * math.pi * orders * self.hop_revolutions * offset
            source = normalized[:, :-offset]
            resolved = torch.stack((
                torch.cos(angle)[None, None, None] * source[..., 0]
                - torch.sin(angle)[None, None, None] * source[..., 1],
                torch.sin(angle)[None, None, None] * source[..., 0]
                + torch.cos(angle)[None, None, None] * source[..., 1],
            ), dim=-1)
            memory = raw[:, :, :-offset, offset - 1].permute(0, 2, 3, 1, 4)
            pair = valid[:, :-offset] & valid[:, offset:]
            error = resolved + memory - normalized[:, offset:]
            mask = pair[:, :, None, None, None]
            denominator = mask.sum().clamp_min(1) * bands * order_count * 2
            losses.append((error.square() * mask).sum() / denominator)
        if not losses:
            raise ValueError("trajectory is shorter than the forecast horizon")
        last = valid.sum(1).clamp_min(1) - 1
        gather = last[:, None, None, None].expand(-1, order_count, 1, self.field_width)
        field = state_field.gather(2, gather).squeeze(2)
        field = field - field.mean(1, keepdim=True)
        field = field / field.square().mean(1, keepdim=True).add(1e-6).sqrt()
        return field, torch.stack(losses).mean()

    def _field(self, paths: Tensor, valid: Tensor, orders: Tensor) -> Tensor:
        return self._field_and_forecast(paths, valid, orders)[0]

    def dynamics_loss(self, paths: Tensor, valid: Tensor, orders: Tensor) -> Tensor:
        return self._field_and_forecast(paths, valid, orders)[1]

    def state_base(
        self, paths: Tensor, valid: Tensor, orders: Tensor, state_observation: Tensor
    ) -> Tensor:
        if state_observation.shape != (len(paths), 2 * self.carrier_bands):
            raise ValueError("state observation does not match the carrier count")
        _, encoded = self._causal_states(paths, valid, orders)
        joint = self._joint_order_states(encoded, orders)
        last = valid.sum(1).clamp_min(1) - 1
        gather = last[:, None, None, None].expand(-1, joint.shape[1], 1, joint.shape[-1])
        temporal = joint.gather(2, gather).squeeze(2).mean(1)
        return torch.cat((temporal, state_observation), dim=-1)

    def state_mode_base(
        self, paths: Tensor, valid: Tensor, orders: Tensor, state_observation: Tensor
    ) -> Tensor:
        """Return every resolved mode for the learned health-state attention."""

        if state_observation.shape != (len(paths), 2 * self.carrier_bands):
            raise ValueError("state observation does not match the carrier count")
        _, encoded = self._causal_states(paths, valid, orders)
        joint = self._joint_order_states(encoded, orders)
        last = valid.sum(1).clamp_min(1) - 1
        gather = last[:, None, None, None].expand(
            -1, joint.shape[1], 1, joint.shape[-1]
        )
        temporal = joint.gather(2, gather).squeeze(2)
        state = state_observation[:, None].expand(-1, temporal.shape[1], -1)
        return torch.cat((temporal, state), dim=-1)

    def state_view(self, base: Tensor) -> Tensor:
        if self.state_decoder is None:
            raise RuntimeError("this checkpoint uses the attentive health readout")
        values = self.state_decoder(base)
        return torch.cat((values, torch.ones_like(values[..., :1])), dim=-1)


def build_model(config: dict, device: torch.device) -> DualViewKoopmanMoriField:
    observation, architecture = config["observation"], config["architecture"]
    return DualViewKoopmanMoriField(
        carrier_bands=int(observation["carrier_bands"]),
        field_width=int(architecture["field_width"]),
        state_width=int(architecture["state_width"]),
        embedding_width=int(architecture["embedding_width"]),
        operation_count=int(architecture["operation_count"]),
        attention_heads=int(architecture["attention_heads"]),
        temporal_layers=int(architecture["temporal_layers"]),
        feedforward_width=int(architecture["feedforward_width"]),
        mixture_count=int(architecture["mixture_count"]),
        minimum_query_width=float(architecture["minimum_query_width"]),
        maximum_query_width=float(architecture["maximum_query_width"]),
        hop_revolutions=float(observation["hop_units"]),
        forecast_horizons=int(architecture["forecast_horizons"]),
    ).to(device)


def build_mixer(config: dict, device: torch.device) -> BalancedSubbandAttentionMixer:
    observation, mixer = config["observation"], config["learned_filterbank"]
    return BalancedSubbandAttentionMixer(
        int(observation["spectral_atoms"]), int(observation["carrier_bands"]),
        int(mixer["coordinate_harmonics"]), int(mixer["sinkhorn_iterations"]),
    ).to(device)
