"""Teacher-free spectral field learned by Koopman transport and Mori memory."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from hoko.dynamics.closure import (
    CenteredConditionalOperationClosure,
    CenteredFactorClosure,
)
from hoko.dynamics.backbone import SpectralFieldDistilledKoopman
from hoko.dynamics.transport import build_transport


@dataclass(frozen=True)
class KoopmanMoriSpectralOutput:
    logits: Tensor
    field: Tensor
    class_attention: Tensor
    forecast_loss: Tensor


class KoopmanMoriSpectralField(SpectralFieldDistilledKoopman):
    """Learn a class-free field from causal multi-step complex trajectory prediction.

    The resolved Koopman transport may be the original analytic rotation or a
    source-learned stable semigroup.  A causal Transformer represents the
    finite-memory Mori--Zwanzig closure.  No dense HOKO field or class label is
    required by :meth:`dynamics_loss`.
    """

    def __init__(
        self,
        *,
        forecast_horizons: int,
        forecast_distribution: str = "deterministic",
        order_interaction_layers: int = 0,
        factorized_operation_closure: bool = False,
        operation_closure_type: str = "linear",
        transport_type: str = "fixed_rotation",
        transport_hidden_width: int = 24,
        transport_coordinate_harmonics: int = 4,
        transport_initial_decay: float = 1e-3,
        operator_loss_weight: float = 0.0,
        **kwargs,
    ) -> None:
        attention_heads = int(kwargs["attention_heads"])
        feedforward_width = int(kwargs["feedforward_width"])
        super().__init__(**kwargs)
        if forecast_horizons < 2:
            raise ValueError("Koopman-Mori training requires at least two horizons")
        self.forecast_horizons = int(forecast_horizons)
        self.forecast_distribution = str(forecast_distribution)
        if self.forecast_distribution not in {"deterministic", "isotropic_gaussian"}:
            raise ValueError("unknown Koopman-Mori forecast distribution")
        if order_interaction_layers < 0:
            raise ValueError("order interaction layer count cannot be negative")
        self.order_interaction_layers = int(order_interaction_layers)
        self.order_interaction_encoder: nn.Module | None = None
        if self.order_interaction_layers:
            interaction_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_width,
                nhead=attention_heads,
                dim_feedforward=feedforward_width,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.order_interaction_encoder = nn.TransformerEncoder(
                interaction_layer,
                num_layers=self.order_interaction_layers,
            )
        forecast_components = 2 if self.forecast_distribution == "deterministic" else 3
        self.memory_forecast_head = nn.Sequential(
            nn.LayerNorm(self.field_width),
            nn.Linear(
                self.field_width,
                self.forecast_horizons * forecast_components * self.carrier_bands,
            ),
        )
        self.factorized_operation_closure = bool(factorized_operation_closure)
        self.operation_closure_type = str(operation_closure_type)
        if self.operation_closure_type not in {"linear", "conditional_mlp"}:
            raise ValueError("unknown operation closure type")
        if operator_loss_weight < 0.0:
            raise ValueError("operator-only loss weight cannot be negative")
        self.transport_type = str(transport_type)
        self.operator_loss_weight = float(operator_loss_weight)
        self.transport = build_transport(
            self.transport_type,
            carrier_bands=self.carrier_bands,
            hop_revolutions=self.hop_revolutions,
            hidden_width=int(transport_hidden_width),
            coordinate_harmonics=int(transport_coordinate_harmonics),
            initial_decay=float(transport_initial_decay),
        )
        self.operation_forecast_head: nn.Module | None = None
        if self.factorized_operation_closure:
            output_width = self.forecast_horizons * forecast_components * self.carrier_bands
            if self.operation_closure_type == "linear":
                self.operation_forecast_head = CenteredFactorClosure(
                    self.operation_count, self.field_width, output_width
                )
            else:
                self.operation_forecast_head = CenteredConditionalOperationClosure(
                    self.operation_count,
                    self.field_width,
                    self.embedding_width,
                    output_width,
                )

    def field_parameters(self):
        yield from super().field_parameters()
        yield from self.transport.parameters()
        if self.order_interaction_encoder is not None:
            yield from self.order_interaction_encoder.parameters()
        yield from self.memory_forecast_head.parameters()

    def operation_dynamics_parameters(self):
        if self.operation_forecast_head is not None:
            yield from self.operation_forecast_head.parameters()

    def _causal_states(
        self, paths: Tensor, valid: Tensor, orders: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, windows, bands, order_count, _ = paths.shape
        power = paths.square().sum(dim=-1)
        weighted = power * valid[:, :, None, None]
        cumulative_power_steps = []
        cumulative_count_steps = []
        running_power = weighted.new_zeros((batch, bands))
        running_count = weighted.new_zeros((batch,))
        for step in range(windows):
            running_power = running_power + weighted[:, step].sum(dim=2)
            running_count = running_count + valid[:, step].to(weighted.dtype)
            cumulative_power_steps.append(running_power)
            cumulative_count_steps.append(running_count)
        cumulative_power = torch.stack(cumulative_power_steps, dim=1)
        cumulative_count = torch.stack(cumulative_count_steps, dim=1).clamp_min(1)
        cumulative_count = cumulative_count * order_count
        scale = (cumulative_power / cumulative_count[:, :, None]).add(1e-6).sqrt()
        normalized = paths / scale[:, :, :, None, None]

        time = torch.arange(windows, device=paths.device, dtype=paths.dtype)
        locked = self.transport.phase_lock(normalized, orders)
        predicted = self.transport(normalized[:, :-1], orders, 1)
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
        causal_mask = torch.triu(
            torch.ones(windows, windows, dtype=torch.bool, device=paths.device), diagonal=1
        )
        encoded = self.temporal_encoder(
            tokens, mask=causal_mask, src_key_padding_mask=~expanded_valid
        )
        encoded = encoded.reshape(batch, order_count, windows, self.embedding_width)
        return normalized, encoded

    def _joint_order_states(self, encoded: Tensor, orders: Tensor) -> Tensor:
        """Couple resolved order modes at each causal time prefix.

        Temporal history has already been encoded independently for every order by
        :meth:`_causal_states`.  Attention here runs only across the order axis at
        the same wall-time index, so it cannot access a future observation.
        """
        batch, order_count, windows, width = encoded.shape
        order_context = self.order_embedding(
            self._coordinate(orders, max(float(orders.max()), 1.0))
        )[None, :, None]
        states = encoded + order_context
        if self.order_interaction_encoder is None:
            return states
        order_tokens = states.permute(0, 2, 1, 3).reshape(
            batch * windows, order_count, width
        )
        coupled = self.order_interaction_encoder(order_tokens)
        return coupled.reshape(batch, windows, order_count, width).permute(0, 2, 1, 3)

    def state_base(
        self,
        paths: Tensor,
        valid: Tensor,
        orders: Tensor,
        state_observation: Tensor,
    ) -> Tensor:
        """Expose an uncentered equilibrium view from the shared causal trunk.

        The returned tensor is deliberately a feature base, not a health
        prediction.  A small source-meta-trained readout decides which parts of
        the preserved level/spread and shared temporal state transfer.
        """

        expected = (len(paths), 2 * self.carrier_bands)
        if tuple(state_observation.shape) != expected:
            raise ValueError("state observation does not align with the path batch")
        _, encoded = self._causal_states(paths, valid, orders)
        joint = self._joint_order_states(encoded, orders)
        last_index = valid.sum(dim=1).clamp_min(1) - 1
        gather = last_index[:, None, None, None].expand(
            -1, joint.shape[1], 1, joint.shape[-1]
        )
        temporal = joint.gather(dim=2, index=gather).squeeze(2).mean(dim=1)
        return torch.cat((temporal, state_observation), dim=-1)

    def state_mode_base(
        self,
        paths: Tensor,
        valid: Tensor,
        orders: Tensor,
        state_observation: Tensor,
    ) -> Tensor:
        """Preserve every resolved mode for a learned downstream state readout."""

        expected = (len(paths), 2 * self.carrier_bands)
        if tuple(state_observation.shape) != expected:
            raise ValueError("state observation does not align with the path batch")
        _, encoded = self._causal_states(paths, valid, orders)
        joint = self._joint_order_states(encoded, orders)
        last_index = valid.sum(dim=1).clamp_min(1) - 1
        gather = last_index[:, None, None, None].expand(
            -1, joint.shape[1], 1, joint.shape[-1]
        )
        temporal = joint.gather(dim=2, index=gather).squeeze(2)
        state = state_observation[:, None].expand(-1, temporal.shape[1], -1)
        return torch.cat((temporal, state), dim=-1)

    def _field_and_forecast(
        self,
        paths: Tensor,
        valid: Tensor,
        orders: Tensor,
        environment_residual_head: nn.Module | None = None,
        environment_index: int | None = None,
        operation_index: int | Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        normalized, encoded = self._causal_states(paths, valid, orders)
        batch, windows, bands, order_count, _ = normalized.shape
        state_field = self.field_decoder(self._joint_order_states(encoded, orders))
        forecast_components = 2 if self.forecast_distribution == "deterministic" else 3
        normalized_state = self.memory_forecast_head[0](state_field)
        raw_memory = self.memory_forecast_head[1](normalized_state)
        if environment_residual_head is not None:
            if environment_index is None:
                raise ValueError("environment residual closure requires a source index")
            raw_memory = raw_memory + environment_residual_head(
                normalized_state, environment_index
            )
        elif environment_index is not None:
            raise ValueError("environment index supplied without a residual closure")
        if self.operation_forecast_head is not None and operation_index is not None:
            raw_memory = raw_memory + self.operation_forecast_head(
                normalized_state, operation_index
            )
        elif self.operation_forecast_head is None and operation_index is not None:
            raise ValueError("operation index supplied without a factorized closure")
        raw_memory = raw_memory.reshape(
            batch,
            order_count,
            windows,
            self.forecast_horizons,
            bands,
            forecast_components,
        )
        losses = []
        for offset in range(1, self.forecast_horizons + 1):
            if offset >= windows:
                continue
            source = normalized[:, :-offset]
            resolved = self.transport(source, orders, offset)
            raw_step = raw_memory[:, :, :-offset, offset - 1]
            memory = raw_step[..., :2].permute(0, 2, 3, 1, 4)
            target = normalized[:, offset:]
            valid_pair = valid[:, :-offset] & valid[:, offset:]
            error = resolved + memory - target
            if self.forecast_distribution == "deterministic":
                mask = valid_pair[:, :, None, None, None]
                squared = error.square() * mask
                denominator = mask.sum().clamp_min(1) * bands * order_count * 2
                full_loss = squared.sum() / denominator
                operator_error = resolved - target
                operator_squared = operator_error.square() * mask
                operator_loss = operator_squared.sum() / denominator
                losses.append(full_loss + self.operator_loss_weight * operator_loss)
            else:
                log_variance = raw_step[..., 2].permute(0, 2, 3, 1)
                log_variance = 8.0 * torch.tanh(log_variance / 8.0)
                energy = 0.5 * error.square().sum(dim=-1) * torch.exp(-log_variance)
                gaussian_nll = energy + log_variance
                mask = valid_pair[:, :, None, None]
                denominator = mask.sum().clamp_min(1) * bands * order_count
                full_loss = (gaussian_nll * mask).sum() / denominator
                operator_error = resolved - target
                operator_energy = operator_error.square().sum(dim=-1)
                operator_loss = (operator_energy * mask).sum() / denominator
                losses.append(full_loss + self.operator_loss_weight * operator_loss)
        if not losses:
            raise ValueError("trajectory is too short for the forecast horizons")

        last_index = valid.sum(dim=1).clamp_min(1) - 1
        field_gather = last_index[:, None, None, None].expand(
            -1, order_count, 1, self.field_width
        )
        raw_field = state_field.gather(dim=2, index=field_gather).squeeze(2)
        centered = raw_field - raw_field.mean(dim=1, keepdim=True)
        field = centered / centered.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()
        return field, torch.stack(losses).mean()

    def _field(self, paths: Tensor, valid: Tensor, orders: Tensor) -> Tensor:
        field, _ = self._field_and_forecast(paths, valid, orders)
        return field

    def innovation_field(self, paths: Tensor, valid: Tensor, orders: Tensor) -> Tensor:
        """Return order-resolved energy/correlation of the learned Mori innovation."""

        normalized, encoded = self._causal_states(paths, valid, orders)
        batch, windows, bands, order_count, _ = normalized.shape
        state_field = self.field_decoder(self._joint_order_states(encoded, orders))
        normalized_state = self.memory_forecast_head[0](state_field)
        forecast_components = 2 if self.forecast_distribution == "deterministic" else 3
        raw_memory = self.memory_forecast_head[1](normalized_state).reshape(
            batch,
            order_count,
            windows,
            self.forecast_horizons,
            bands,
            forecast_components,
        )
        signatures = []
        for offset in range(1, self.forecast_horizons + 1):
            if offset >= windows:
                continue
            source = normalized[:, :-offset]
            resolved = self.transport(source, orders, offset)
            memory = raw_memory[:, :, :-offset, offset - 1, :, :2].permute(
                0, 2, 3, 1, 4
            )
            error = resolved + memory - normalized[:, offset:]
            valid_pair = valid[:, :-offset] & valid[:, offset:]
            mask = valid_pair[:, :, None, None].to(error.dtype)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            energy = (error.square().sum(dim=-1) * mask).sum(dim=1) / denominator
            log_energy = torch.log1p(energy).permute(0, 2, 1)

            if error.shape[1] > 1:
                adjacent = valid_pair[:, :-1] & valid_pair[:, 1:]
                adjacent_mask = adjacent[:, :, None, None].to(error.dtype)
                left, right = error[:, :-1], error[:, 1:]
                dot = (left * right).sum(dim=-1)
                scale = (
                    left.square().sum(dim=-1) * right.square().sum(dim=-1)
                ).add(1e-8).sqrt()
                correlation = (
                    (dot / scale) * adjacent_mask
                ).sum(dim=1) / adjacent_mask.sum(dim=1).clamp_min(1.0)
                correlation = correlation.permute(0, 2, 1)
            else:
                correlation = log_energy.new_zeros(log_energy.shape)
            signatures.extend((log_energy, correlation))

        if not signatures:
            raise ValueError("trajectory is too short for an innovation signature")
        signature = torch.cat(signatures, dim=-1)
        reference = signature.new_ones((batch, order_count, 1))
        return torch.cat((signature, reference), dim=-1)

    def dynamics_loss(
        self,
        paths: Tensor,
        valid: Tensor,
        orders: Tensor,
        environment_residual_head: nn.Module | None = None,
        environment_index: int | None = None,
        operation_index: int | Tensor | None = None,
    ) -> Tensor:
        return self._field_and_forecast(
            paths,
            valid,
            orders,
            environment_residual_head=environment_residual_head,
            environment_index=environment_index,
            operation_index=operation_index,
        )[1]

    def operation_forecast_nll(
        self,
        paths: Tensor,
        valid: Tensor,
        orders: Tensor,
        environment_residual_head: nn.Module | None = None,
        environment_count: int | None = None,
        environment_indices: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return one causal forecast energy per sample and operation hypothesis.

        Without an environment head, each column uses the centred shared-system
        closure.  With a fitted source-environment head, source systems are
        equal-prior nuisance hypotheses and are marginalized by log-mean-exp.
        """

        if self.operation_forecast_head is None:
            raise ValueError("operation forecast scoring requires a factorized closure")
        normalized, encoded = self._causal_states(paths, valid, orders)
        batch, windows, bands, order_count, _ = normalized.shape
        state_field = self.field_decoder(self._joint_order_states(encoded, orders))
        normalized_state = self.memory_forecast_head[0](state_field)
        shared = self.memory_forecast_head[1](normalized_state)
        forecast_components = 2 if self.forecast_distribution == "deterministic" else 3

        def forecast_nll(raw: Tensor) -> Tensor:
            raw = raw.reshape(
                batch,
                order_count,
                windows,
                self.forecast_horizons,
                bands,
                forecast_components,
            )
            horizon_losses = []
            for offset in range(1, self.forecast_horizons + 1):
                if offset >= windows:
                    continue
                source = normalized[:, :-offset]
                resolved = self.transport(source, orders, offset)
                raw_step = raw[:, :, :-offset, offset - 1]
                memory = raw_step[..., :2].permute(0, 2, 3, 1, 4)
                error = resolved + memory - normalized[:, offset:]
                pair = valid[:, :-offset] & valid[:, offset:]
                if self.forecast_distribution == "deterministic":
                    local = error.square().sum(dim=(-1, -2, -3))
                    numerator = (local * pair).sum(dim=1)
                    denominator = pair.sum(dim=1).clamp_min(1) * bands * order_count * 2
                else:
                    log_variance = raw_step[..., 2].permute(0, 2, 3, 1)
                    log_variance = 8.0 * torch.tanh(log_variance / 8.0)
                    local = (
                        0.5 * error.square().sum(dim=-1) * torch.exp(-log_variance)
                        + log_variance
                    ).sum(dim=(-1, -2))
                    numerator = (local * pair).sum(dim=1)
                    denominator = pair.sum(dim=1).clamp_min(1) * bands * order_count
                horizon_losses.append(numerator / denominator)
            if not horizon_losses:
                raise ValueError("trajectory is too short for operation forecast scoring")
            return torch.stack(horizon_losses).mean(dim=0)

        if environment_residual_head is None:
            if environment_count is not None or environment_indices is not None:
                raise ValueError("environment marginalization requires a residual head")
            selected_environments = None
        else:
            if environment_count is not None and environment_indices is not None:
                raise ValueError("specify an environment count or explicit indices, not both")
            if environment_indices is not None:
                selected_environments = tuple(int(value) for value in environment_indices)
            elif environment_count is not None:
                selected_environments = tuple(range(int(environment_count)))
            else:
                raise ValueError("environment marginalization requires count or indices")
            if not selected_environments or any(
                value < 0 or value >= environment_residual_head.environment_count
                for value in selected_environments
            ):
                raise ValueError("environment hypothesis indices are invalid")
        candidates = []
        for operation_index in range(self.operation_count):
            operation_raw = self.operation_forecast_head(normalized_state, operation_index)
            if selected_environments is None:
                candidates.append(forecast_nll(shared + operation_raw))
                continue
            environment_nll = torch.stack(
                [
                    forecast_nll(
                        shared
                        + operation_raw
                        + environment_residual_head(normalized_state, environment_index)
                    )
                    for environment_index in selected_environments
                ],
                dim=-1,
            )
            candidates.append(
                -torch.logsumexp(-environment_nll, dim=-1)
                + math.log(len(selected_environments))
            )
        return torch.stack(candidates, dim=-1)

    def forward(
        self, paths: Tensor, valid: Tensor, orders: Tensor
    ) -> KoopmanMoriSpectralOutput:
        if (
            paths.ndim != 5
            or paths.shape[2] != self.carrier_bands
            or paths.shape[-1] != 2
            or valid.shape != paths.shape[:2]
            or valid.dtype != torch.bool
            or orders.ndim != 1
            or len(orders) != paths.shape[3]
        ):
            raise ValueError("invalid Koopman-Mori spectral path input")
        field, forecast_loss = self._field_and_forecast(paths, valid, orders)
        logits, attention = self._operation(field, orders)
        return KoopmanMoriSpectralOutput(logits, field, attention, forecast_loss)
