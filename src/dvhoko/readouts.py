"""Source-trained health and operation readouts used by released HUST folds."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def prefix_sum(values: Tensor) -> Tensor:
    """Deterministic ordered evidence accumulation."""

    running = torch.zeros_like(values[0])
    output = []
    for value in values.unbind(0):
        running = running + value
        output.append(running)
    return torch.stack(output)


class AttentiveStateViewDecoder(nn.Module):
    """Pool full-Nyquist state modes into binary health evidence."""

    def __init__(
        self, input_width: int, hidden_width: int, output_width: int, attention_heads: int
    ) -> None:
        super().__init__()
        if hidden_width % attention_heads:
            raise ValueError("health attention width must divide its head count")
        self.token = nn.Sequential(
            nn.LayerNorm(input_width), nn.Linear(input_width, hidden_width), nn.GELU()
        )
        self.query = nn.Parameter(torch.empty(1, 1, hidden_width))
        nn.init.normal_(self.query, std=hidden_width**-0.5)
        self.attention = nn.MultiheadAttention(
            hidden_width, attention_heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_width), nn.Linear(hidden_width, output_width)
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("health modes must be [second,mode,feature]")
        tokens = self.token(values)
        query = self.query.expand(len(tokens), -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        decoded = self.output(pooled[:, 0])
        return F.normalize(
            torch.cat((decoded, torch.ones_like(decoded[..., :1])), dim=-1), dim=-1
        )


class _SetAttentionPool(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("attention width must divide its head count")
        self.query = nn.Parameter(torch.zeros(1, 1, width))
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.norm_feedforward = nn.LayerNorm(width)

    def forward(self, values: Tensor) -> Tensor:
        query = self.query.expand(len(values), -1, -1)
        attended, _ = self.attention(query, values, values, need_weights=False)
        state = self.norm_attention(query + attended)
        return self.norm_feedforward(state + self.feedforward(state))[:, 0]


class QueryAdaptiveOperationMetric(nn.Module):
    """Source-support-conditioned full-mode metric with per-query reliability."""

    def __init__(
        self, statistic_width: int, hidden_width: int, attention_heads: int
    ) -> None:
        super().__init__()
        self.statistic_width = int(statistic_width)
        self.second_encoder = nn.Sequential(
            nn.Linear(statistic_width, hidden_width), nn.GELU(),
            nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width),
        )
        self.second_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.cell_encoder = nn.Sequential(
            nn.GELU(), nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width)
        )
        self.environment_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.class_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.raw_reliability = nn.Sequential(
            nn.Linear(hidden_width + 5, hidden_width), nn.GELU(),
            nn.LayerNorm(hidden_width), nn.Linear(hidden_width, 1),
        )
        self.feature_metric = nn.Linear(statistic_width, statistic_width, bias=False)
        self.query_reliability = nn.Sequential(
            nn.Linear(statistic_width + 5, hidden_width), nn.GELU(),
            nn.LayerNorm(hidden_width), nn.Linear(hidden_width, 1),
        )

    @staticmethod
    def _coordinate(orders: Tensor) -> Tensor:
        value = orders / orders.max().clamp_min(1.0)
        return torch.stack((
            value, torch.sin(math.pi * value), torch.cos(math.pi * value),
            torch.sin(2 * math.pi * value), torch.cos(2 * math.pi * value),
        ), dim=-1)

    @staticmethod
    def _unit(values: Tensor) -> Tensor:
        return values / values.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()

    def order_weights(
        self, support: tuple[tuple[Tensor, ...], ...], orders: Tensor
    ) -> Tensor:
        environments, classes, modes = len(support), len(support[0]), len(orders)
        rows = []
        for environment in support:
            local = []
            for values in environment:
                encoded = self.second_encoder(values).permute(1, 0, 2)
                local.append(self.cell_encoder(self.second_pool(encoded)))
            rows.append(torch.stack(local))
        cells = torch.stack(rows)
        environment_sets = cells.permute(1, 2, 0, 3).reshape(
            classes * modes, environments, -1
        )
        class_tokens = self.environment_pool(environment_sets).reshape(classes, modes, -1)
        support_token = self.class_pool(class_tokens.permute(1, 0, 2))
        logits = self.raw_reliability(
            torch.cat((support_token, self._coordinate(orders)), dim=-1)
        ).squeeze(-1)
        return torch.softmax(logits, dim=0) * len(logits)

    def forward(
        self, support: tuple[tuple[Tensor, ...], ...], query: Tensor, orders: Tensor
    ) -> Tensor:
        prior = self.order_weights(support, orders)
        coordinate = self._coordinate(orders)[None].expand(len(query), -1, -1)
        residual = self.query_reliability(torch.cat((query, coordinate), dim=-1)).squeeze(-1)
        weights = torch.softmax(prior.clamp_min(torch.finfo(prior.dtype).tiny).log()[None] + residual, dim=-1)
        weights = weights * len(orders)
        transformed_query = self.feature_metric(query)
        transformed_support = [
            self.feature_metric(torch.cat([environment[index] for environment in support]))
            for index in range(len(support[0]))
        ]
        logits = []
        for second, local_weights in enumerate(weights):
            scale = local_weights.clamp_min(torch.finfo(local_weights.dtype).tiny).sqrt()[None, :, None]
            local_query = self._unit((transformed_query[second:second + 1] * scale).reshape(1, -1))
            centroids = []
            for values in transformed_support:
                embedded = self._unit((values * scale).reshape(len(values), -1))
                centroids.append(self._unit(embedded.mean(0, keepdim=True))[0])
            logits.append(local_query @ torch.stack(centroids).T)
        return torch.cat(logits)
