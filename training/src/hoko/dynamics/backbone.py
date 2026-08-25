"""Spectral-path neural estimator of a class-independent Koopman field."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class SpectralFieldDistilledOutput:
    logits: Tensor
    field: Tensor
    class_attention: Tensor


class SpectralFieldDistilledKoopman(nn.Module):
    """Learn field statistics from class-free complex order trajectories.

    The input stem performs only a shared order scan and the corresponding
    Koopman phase transport.  A neural temporal set operator learns the field
    statistics from source-only teacher values.  Class labels reach only the
    downstream readout after the field estimator is frozen.
    """

    def __init__(
        self,
        *,
        carrier_bands: int,
        field_width: int,
        embedding_width: int,
        operation_count: int,
        attention_heads: int,
        temporal_layers: int,
        feedforward_width: int,
        mixture_count: int,
        minimum_query_width: float,
        maximum_query_width: float,
        hop_revolutions: float,
        operation_reader: str = "shared_scalar_mixture",
        operation_score_rank: int = 1,
    ) -> None:
        super().__init__()
        if min(
            carrier_bands,
            field_width,
            embedding_width,
            operation_count,
            attention_heads,
            temporal_layers,
            feedforward_width,
            mixture_count,
        ) < 1 or embedding_width % attention_heads or hop_revolutions <= 0.0:
            raise ValueError("invalid spectral field-distillation architecture")
        if not 0.0 < minimum_query_width < maximum_query_width:
            raise ValueError("invalid operation query width")
        self.carrier_bands = int(carrier_bands)
        self.field_width = int(field_width)
        self.embedding_width = int(embedding_width)
        self.operation_count = int(operation_count)
        self.mixture_count = int(mixture_count)
        self.minimum_query_width = float(minimum_query_width)
        self.maximum_query_width = float(maximum_query_width)
        self.hop_revolutions = float(hop_revolutions)
        self.operation_reader = str(operation_reader)
        self.operation_score_rank = int(operation_score_rank)
        if self.operation_reader not in {
            "shared_scalar_mixture",
            "self_contextual_mixture",
            "low_rank_mixture",
            "class_query_energy",
        }:
            raise ValueError("unknown spectral Koopman operation reader")
        if self.operation_score_rank < 1:
            raise ValueError("operation score rank must be positive")

        self.path_embedding = nn.Sequential(
            nn.Linear(6 * carrier_bands, embedding_width),
            nn.GELU(),
            nn.LayerNorm(embedding_width),
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(5, embedding_width), nn.GELU(), nn.Linear(embedding_width, embedding_width)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_width,
            nhead=attention_heads,
            dim_feedforward=feedforward_width,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.order_embedding = nn.Sequential(
            nn.Linear(5, embedding_width), nn.GELU(), nn.Linear(embedding_width, embedding_width)
        )
        self.field_decoder = nn.Sequential(
            nn.LayerNorm(embedding_width),
            nn.Linear(embedding_width, feedforward_width),
            nn.GELU(),
            nn.Linear(feedforward_width, field_width),
        )
        if self.operation_reader in {
            "shared_scalar_mixture",
            "self_contextual_mixture",
            "low_rank_mixture",
        }:
            score_rank = (
                1 if self.operation_reader == "shared_scalar_mixture" else self.operation_score_rank
            )
            self.mode_score = nn.Sequential(
                nn.Linear(field_width, embedding_width),
                nn.GELU(),
                nn.LayerNorm(embedding_width),
                nn.Linear(embedding_width, score_rank),
            )
            self.class_embedding = nn.Parameter(torch.empty(operation_count, embedding_width))
            nn.init.normal_(self.class_embedding, std=embedding_width**-0.5)
            query_width = (
                3 + self.operation_score_rank
                if self.operation_reader == "low_rank_mixture"
                else 3
            )
            self.query_generator = nn.Sequential(
                nn.Linear(embedding_width, embedding_width),
                nn.GELU(),
                nn.Linear(embedding_width, query_width * mixture_count),
            )
            if self.operation_reader == "self_contextual_mixture":
                self.context_mode_embedding = nn.Sequential(
                    nn.LayerNorm(field_width),
                    nn.Linear(field_width, embedding_width),
                    nn.GELU(),
                    nn.LayerNorm(embedding_width),
                )
                self.context_order_embedding = nn.Sequential(
                    nn.Linear(5, embedding_width),
                    nn.GELU(),
                    nn.Linear(embedding_width, embedding_width),
                )
                self.context_query = nn.Parameter(torch.empty(embedding_width))
                nn.init.normal_(self.context_query, std=embedding_width**-0.5)
        else:
            self.operation_mode_embedding = nn.Sequential(
                nn.LayerNorm(field_width),
                nn.Linear(field_width, embedding_width),
                nn.GELU(),
                nn.LayerNorm(embedding_width),
            )
            self.operation_order_embedding = nn.Sequential(
                nn.Linear(5, embedding_width),
                nn.GELU(),
                nn.Linear(embedding_width, embedding_width),
            )
            self.operation_queries = nn.Parameter(
                torch.empty(operation_count, mixture_count, embedding_width)
            )
            nn.init.normal_(self.operation_queries, std=embedding_width**-0.5)
            self.operation_key = nn.Linear(embedding_width, embedding_width, bias=False)

    @staticmethod
    def _coordinate(value: Tensor, divisor: float) -> Tensor:
        normalized = value / divisor
        return torch.stack(
            (
                normalized,
                torch.sin(math.pi * normalized),
                torch.cos(math.pi * normalized),
                torch.sin(2.0 * math.pi * normalized),
                torch.cos(2.0 * math.pi * normalized),
            ),
            dim=-1,
        )

    def field_parameters(self):
        for module in (
            self.path_embedding,
            self.time_embedding,
            self.temporal_encoder,
            self.order_embedding,
            self.field_decoder,
        ):
            yield from module.parameters()

    def operation_parameters(self):
        if self.operation_reader in {
            "shared_scalar_mixture",
            "self_contextual_mixture",
            "low_rank_mixture",
        }:
            yield self.class_embedding
            yield from self.mode_score.parameters()
            yield from self.query_generator.parameters()
            if self.operation_reader == "self_contextual_mixture":
                yield self.context_query
                yield from self.context_mode_embedding.parameters()
                yield from self.context_order_embedding.parameters()
        else:
            yield self.operation_queries
            for module in (
                self.operation_mode_embedding,
                self.operation_order_embedding,
                self.operation_key,
            ):
                yield from module.parameters()

    def _field(self, paths: Tensor, valid: Tensor, orders: Tensor) -> Tensor:
        batch, windows, bands, order_count, _ = paths.shape
        power = paths.square().sum(dim=-1)
        weight = valid[:, :, None, None]
        scale = (
            (power * weight).sum(dim=(1, 3))
            / (weight.sum(dim=1).squeeze(1) * order_count).clamp_min(1)
        ).add(1e-6).sqrt()
        normalized = paths / scale[:, None, :, None, None]

        time = torch.arange(windows, device=paths.device, dtype=paths.dtype)
        lock_angle = -2.0 * math.pi * time[:, None] * orders[None] * self.hop_revolutions
        lock_cos, lock_sin = torch.cos(lock_angle), torch.sin(lock_angle)
        locked = torch.stack(
            (
                lock_cos[None, :, None] * normalized[..., 0]
                - lock_sin[None, :, None] * normalized[..., 1],
                lock_sin[None, :, None] * normalized[..., 0]
                + lock_cos[None, :, None] * normalized[..., 1],
            ),
            dim=-1,
        )

        angle = 2.0 * math.pi * orders * self.hop_revolutions
        predicted = torch.stack(
            (
                torch.cos(angle)[None, None, None] * normalized[:, :-1, ..., 0]
                - torch.sin(angle)[None, None, None] * normalized[:, :-1, ..., 1],
                torch.sin(angle)[None, None, None] * normalized[:, :-1, ..., 0]
                + torch.cos(angle)[None, None, None] * normalized[:, :-1, ..., 1],
            ),
            dim=-1,
        )
        residual = normalized.new_zeros(normalized.shape)
        residual[:, 1:] = normalized[:, 1:] - predicted
        features = torch.cat((normalized, locked, residual), dim=-1)
        tokens = features.permute(0, 3, 1, 2, 4).reshape(
            batch * order_count, windows, 6 * bands
        )
        tokens = self.path_embedding(tokens)
        tokens = tokens + self.time_embedding(self._coordinate(time, 32.0))[None]
        expanded_valid = valid[:, None].expand(-1, order_count, -1).reshape(
            batch * order_count, windows
        )
        encoded = self.temporal_encoder(tokens, src_key_padding_mask=~expanded_valid)
        pooled = (encoded * expanded_valid[..., None]).sum(dim=1) / expanded_valid.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        modes = pooled.reshape(batch, order_count, self.embedding_width)
        modes = modes + self.order_embedding(
            self._coordinate(orders, max(float(orders.max()), 1.0))
        )[None]
        raw_field = self.field_decoder(modes)
        centered = raw_field - raw_field.mean(dim=1, keepdim=True)
        return centered / centered.square().mean(dim=1, keepdim=True).add(1e-6).sqrt()

    def _operation(self, field: Tensor, orders: Tensor) -> tuple[Tensor, Tensor]:
        if self.operation_reader == "class_query_energy":
            tokens = self.operation_mode_embedding(field)
            tokens = tokens + self.operation_order_embedding(
                self._coordinate(orders, max(float(orders.max()), 1.0))
            )[None]
            keys = self.operation_key(tokens)
            scale = self.embedding_width**-0.5
            energy = scale * torch.einsum(
                "crd,bod->bcro", self.operation_queries, keys
            )
            logits = torch.logsumexp(energy.flatten(start_dim=2), dim=-1)
            logits = logits - math.log(self.mixture_count * len(orders))
            order_energy = torch.logsumexp(energy, dim=2)
            attention = torch.softmax(order_energy, dim=-1)
            return logits, attention
        if self.operation_reader == "low_rank_mixture":
            raw = self.query_generator(self.class_embedding).reshape(
                self.operation_count,
                self.mixture_count,
                3 + self.operation_score_rank,
            )
            centers = orders.min() + (orders.max() - orders.min()) * torch.sigmoid(raw[..., 0])
            widths = self.minimum_query_width + (
                self.maximum_query_width - self.minimum_query_width
            ) * torch.sigmoid(raw[..., 1])
            mixture = torch.softmax(raw[..., 2], dim=-1)
            rank_weights = raw[..., 3:]
            rank_weights = rank_weights / rank_weights.square().sum(dim=-1, keepdim=True).add(
                1e-6
            ).sqrt()
            distance = (orders[None, None] - centers[..., None]) / widths[..., None]
            component_attention = torch.softmax(-0.5 * distance.square(), dim=-1)
            mode_scores = self.mode_score(field)
            localized = torch.einsum("cqo,bor->bcqr", component_attention, mode_scores)
            component_scores = (localized * rank_weights[None]).sum(dim=-1)
            logits = (mixture[None] * component_scores).sum(dim=-1)
            attention = (mixture[..., None] * component_attention).sum(dim=1)
            return logits, attention[None].expand(len(field), -1, -1)
        if self.operation_reader == "self_contextual_mixture":
            context_tokens = self.context_mode_embedding(field)
            context_tokens = context_tokens + self.context_order_embedding(
                self._coordinate(orders, max(float(orders.max()), 1.0))
            )[None]
            context_attention = torch.softmax(
                self.embedding_width**-0.5
                * torch.einsum("d,bod->bo", self.context_query, context_tokens),
                dim=-1,
            )
            context = torch.einsum("bo,bod->bd", context_attention, context_tokens)
            queries = self.class_embedding[None] + context[:, None]
            raw = self.query_generator(queries).reshape(
                len(field), self.operation_count, self.mixture_count, 3
            )
            centers = orders.min() + (orders.max() - orders.min()) * torch.sigmoid(raw[..., 0])
            widths = self.minimum_query_width + (
                self.maximum_query_width - self.minimum_query_width
            ) * torch.sigmoid(raw[..., 1])
            mixture = torch.softmax(raw[..., 2], dim=-1)
            distance = (orders[None, None, None] - centers[..., None]) / widths[..., None]
            component_attention = torch.softmax(-0.5 * distance.square(), dim=-1)
            attention = (mixture[..., None] * component_attention).sum(dim=2)
            mode_score = self.mode_score(field).squeeze(-1)
            logits = (attention * mode_score[:, None]).sum(dim=-1)
            return logits, attention
        raw = self.query_generator(self.class_embedding).reshape(
            self.operation_count, self.mixture_count, 3
        )
        centers = orders.min() + (orders.max() - orders.min()) * torch.sigmoid(raw[..., 0])
        widths = self.minimum_query_width + (
            self.maximum_query_width - self.minimum_query_width
        ) * torch.sigmoid(raw[..., 1])
        mixture = torch.softmax(raw[..., 2], dim=-1)
        distance = (orders[None, None] - centers[..., None]) / widths[..., None]
        attention = (mixture[..., None] * torch.softmax(-0.5 * distance.square(), dim=-1)).sum(
            dim=1
        )
        mode_score = self.mode_score(field).squeeze(-1)
        logits = (attention[None] * mode_score[:, None]).sum(dim=-1)
        return logits, attention[None].expand(len(field), -1, -1)

    def forward(
        self, paths: Tensor, valid: Tensor, orders: Tensor
    ) -> SpectralFieldDistilledOutput:
        if (
            paths.ndim != 5
            or paths.shape[2] != self.carrier_bands
            or paths.shape[-1] != 2
            or valid.shape != paths.shape[:2]
            or valid.dtype != torch.bool
            or orders.ndim != 1
            or len(orders) != paths.shape[3]
        ):
            raise ValueError("invalid spectral Koopman path input")
        field = self._field(paths, valid, orders)
        logits, attention = self._operation(field, orders)
        return SpectralFieldDistilledOutput(logits, field, attention)
