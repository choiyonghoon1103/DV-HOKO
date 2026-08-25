"""Class-permutation-equivariant metric attention over frozen dynamical fields."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FieldMetricOutput:
    logits: Tensor
    order_weights: Tensor
    class_centroids: Tensor


class SupportConditionedFieldMetric(nn.Module):
    """Learn a support-statistic-conditioned nonnegative metric over orders.

    The input field is never compressed before the metric is formed.  Support
    class labels only group examples; the same reliability network is used for
    every class and every order.  A zero-initialized output layer makes the
    initial metric exactly the identity, reproducing ordinary cosine memory.
    """

    def __init__(
        self,
        *,
        statistic_width: int,
        hidden_width: int,
        order_weight_normalization: str = "softmax",
    ) -> None:
        super().__init__()
        if statistic_width < 1 or hidden_width < 1:
            raise ValueError("invalid field-metric architecture")
        self.statistic_width = int(statistic_width)
        self.order_weight_normalization = str(order_weight_normalization)
        if self.order_weight_normalization not in {"softmax", "sparsemax"}:
            raise ValueError("unknown order-weight normalization")
        # Four support reliability summaries plus five continuous coordinates.
        self.reliability = nn.Sequential(
            nn.Linear(9, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.reliability[-1].weight)
        nn.init.zeros_(self.reliability[-1].bias)

    @staticmethod
    def _unit_rows(values: Tensor) -> Tensor:
        return values / values.square().sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()

    @staticmethod
    def _coordinate(orders: Tensor) -> Tensor:
        normalized = orders / orders.max().clamp_min(1.0)
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

    @staticmethod
    def _standardize_across_orders(values: Tensor) -> Tensor:
        mean = values.mean(dim=0, keepdim=True)
        scale = values.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        return (values - mean) / scale

    @staticmethod
    def _sqrt_metric_weights(weights: Tensor) -> Tensor:
        """Take a zero-safe square root for a nonnegative diagonal metric.

        Sparsemax intentionally emits exact zeros.  Clamping only inside this
        square root avoids the undefined derivative at zero while preserving
        the sparse weights themselves and a zero gradient for inactive modes.
        """

        return weights.clamp_min(torch.finfo(weights.dtype).tiny).sqrt()

    def _normalize_order_logits(self, logits: Tensor) -> Tensor:
        """Return nonnegative mode weights with exact mean one."""

        if self.order_weight_normalization == "softmax":
            return torch.softmax(logits, dim=0) * len(logits)
        shifted = logits - logits.max()
        sorted_logits, _ = torch.sort(shifted, descending=True)
        rank = torch.arange(
            1, len(logits) + 1, device=logits.device, dtype=logits.dtype
        )
        # CUDA cumsum has no deterministic implementation in the supported
        # PyTorch build.  This small lower-triangular product is algebraically
        # identical and keeps exact deterministic training for the mode axis.
        cumulative = torch.tril(
            torch.ones(
                len(logits), len(logits), device=logits.device, dtype=logits.dtype
            )
        ) @ sorted_logits
        support = 1.0 + rank * sorted_logits > cumulative
        support_size = support.sum().clamp_min(1)
        threshold = (cumulative[support_size - 1] - 1.0) / support_size
        probability = torch.clamp(shifted - threshold, min=0.0)
        return probability / probability.sum().clamp_min(1e-12) * len(logits)

    def order_weights(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        orders: Tensor,
    ) -> Tensor:
        """Return mean-one nonnegative weights from support-cell statistics.

        ``support_cells[environment][class]`` is ``[second,order,feature]``.
        Environment and class axes are both aggregated symmetrically.
        """

        if len(support_cells) < 1 or len(support_cells[0]) < 2:
            raise ValueError("metric attention needs support environments and classes")
        class_count = len(support_cells[0])
        if any(len(environment) != class_count for environment in support_cells):
            raise ValueError("support environments must contain the same class set")
        order_count = len(orders)
        for environment in support_cells:
            for values in environment:
                if (
                    values.ndim != 3
                    or values.shape[1] != order_count
                    or values.shape[2] != self.statistic_width
                    or len(values) < 1
                ):
                    raise ValueError("invalid support field cell")

        cell_means = torch.stack(
            [
                torch.stack([values.mean(dim=0) for values in environment], dim=0)
                for environment in support_cells
            ],
            dim=0,
        )
        class_means = cell_means.mean(dim=0)
        between_class = class_means.var(dim=0, unbiased=False).mean(dim=-1)
        environment_shift = (
            cell_means - class_means[None]
        ).square().mean(dim=(0, 1, 3))
        within_cells = []
        for environment_index, environment in enumerate(support_cells):
            for class_index, values in enumerate(environment):
                within_cells.append(
                    (values - cell_means[environment_index, class_index][None])
                    .square()
                    .mean(dim=(0, 2))
                )
        within_class = torch.stack(within_cells).mean(dim=0)
        support_power = cell_means.square().mean(dim=(0, 1, 3))
        summaries = torch.stack(
            (
                torch.log1p(between_class),
                torch.log1p(within_class),
                torch.log1p(environment_shift),
                torch.log1p(support_power),
            ),
            dim=-1,
        )
        features = torch.cat(
            (self._standardize_across_orders(summaries), self._coordinate(orders)),
            dim=-1,
        )
        logits = self.reliability(features).squeeze(-1)
        return self._normalize_order_logits(logits)

    def induce(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        orders: Tensor,
    ) -> tuple[Tensor, Tensor]:
        weights = self.order_weights(support_cells, orders)
        return self.induce_with_weights(support_cells, weights), weights

    def induce_with_weights(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        weights: Tensor,
    ) -> Tensor:
        """Induce class centroids under an explicit nonnegative order metric."""

        if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("explicit order weights must be a nonnegative nonzero vector")
        if any(
            values.shape[1] != len(weights)
            for environment in support_cells
            for values in environment
        ):
            raise ValueError("explicit metric differs from the support order axis")
        metric_scale = self._sqrt_metric_weights(weights)[None, :, None]
        class_count = len(support_cells[0])
        centroids = []
        for class_index in range(class_count):
            seconds = torch.cat(
                [environment[class_index] for environment in support_cells], dim=0
            )
            embedded = (seconds * metric_scale).reshape(len(seconds), -1)
            centroid = self._unit_rows(embedded).mean(dim=0, keepdim=True)
            centroids.append(self._unit_rows(centroid)[0])
        return torch.stack(centroids)

    def score_queries(
        self, class_centroids: Tensor, order_weights: Tensor, query_fields: Tensor
    ) -> FieldMetricOutput:
        if query_fields.ndim != 3 or query_fields.shape[-1] != self.statistic_width:
            raise ValueError("query fields must be [second,order,feature]")
        if len(order_weights) != query_fields.shape[1]:
            raise ValueError("query order axis differs from the support metric")
        embedded = (
            query_fields * self._sqrt_metric_weights(order_weights)[None, :, None]
        ).reshape(len(query_fields), -1)
        logits = self._unit_rows(embedded) @ class_centroids.T
        return FieldMetricOutput(logits, order_weights, class_centroids)

    def forward(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        query_fields: Tensor,
        orders: Tensor,
    ) -> FieldMetricOutput:
        centroids, weights = self.induce(support_cells, orders)
        return self.score_queries(centroids, weights, query_fields)


class NeuralSupportConditionedFieldMetric(SupportConditionedFieldMetric):
    """Infer order reliability directly from nested raw support-field sets.

    The three mean aggregations implement a nested DeepSets hierarchy:
    seconds -> environment/class cell -> class across environments -> all
    classes. Nonlinear element maps occur before every mean, so the network can
    learn dispersion and interaction summaries without receiving handcrafted
    between/within/power statistics.
    """

    def __init__(
        self,
        *,
        statistic_width: int,
        hidden_width: int,
        order_weight_normalization: str = "softmax",
    ) -> None:
        super().__init__(
            statistic_width=statistic_width,
            hidden_width=hidden_width,
            order_weight_normalization=order_weight_normalization,
        )
        del self.reliability
        self.second_encoder = nn.Sequential(
            nn.Linear(statistic_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
        )
        self.cell_encoder = nn.Sequential(
            nn.GELU(), nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width)
        )
        self.environment_encoder = nn.Sequential(
            nn.GELU(), nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width)
        )
        self.class_encoder = nn.Sequential(
            nn.GELU(), nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width)
        )
        self.raw_reliability = nn.Sequential(
            nn.Linear(hidden_width + 5, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.raw_reliability[-1].weight)
        nn.init.zeros_(self.raw_reliability[-1].bias)

    def order_weights(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        orders: Tensor,
    ) -> Tensor:
        if len(support_cells) < 1 or len(support_cells[0]) < 2:
            raise ValueError("neural metric needs support environments and classes")
        class_count = len(support_cells[0])
        order_count = len(orders)
        if any(len(environment) != class_count for environment in support_cells):
            raise ValueError("support environments must contain the same class set")
        cell_rows = []
        for environment in support_cells:
            local = []
            for values in environment:
                if (
                    values.ndim != 3
                    or values.shape[1] != order_count
                    or values.shape[2] != self.statistic_width
                    or len(values) < 1
                ):
                    raise ValueError("invalid raw support field cell")
                # [second,order,feature] -> [order,hidden]
                token = self.second_encoder(values).mean(dim=0)
                local.append(self.cell_encoder(token))
            cell_rows.append(torch.stack(local, dim=0))
        # [environment,class,order,hidden]
        cells = torch.stack(cell_rows, dim=0)
        # Apply a nonlinear map before each symmetric aggregation. This keeps
        # both environment and class enumeration irrelevant to the result.
        class_tokens = self.environment_encoder(cells).mean(dim=0)
        support_token = self.class_encoder(class_tokens).mean(dim=0)
        features = torch.cat((support_token, self._coordinate(orders)), dim=-1)
        logits = self.raw_reliability(features).squeeze(-1)
        return self._normalize_order_logits(logits)


class _SetAttentionPool(nn.Module):
    """Permutation-invariant learned-query pooling for a batch of finite sets."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("attention width must be divisible by head count")
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
        if values.ndim != 3 or values.shape[1] < 1:
            raise ValueError("set attention expects batch x element x feature")
        query = self.query.expand(len(values), -1, -1)
        attended, _ = self.attention(query, values, values, need_weights=False)
        state = self.norm_attention(query + attended)
        state = self.norm_feedforward(state + self.feedforward(state))
        return state[:, 0]


class AttentiveSupportConditionedFieldMetric(SupportConditionedFieldMetric):
    """Infer the field metric with nested attention over raw support sets."""

    def __init__(
        self,
        *,
        statistic_width: int,
        hidden_width: int,
        attention_heads: int,
        order_weight_normalization: str = "softmax",
    ) -> None:
        super().__init__(
            statistic_width=statistic_width,
            hidden_width=hidden_width,
            order_weight_normalization=order_weight_normalization,
        )
        del self.reliability
        self.second_encoder = nn.Sequential(
            nn.Linear(statistic_width, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
        )
        self.second_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.cell_encoder = nn.Sequential(
            nn.GELU(), nn.LayerNorm(hidden_width), nn.Linear(hidden_width, hidden_width)
        )
        self.environment_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.class_pool = _SetAttentionPool(hidden_width, attention_heads)
        self.raw_reliability = nn.Sequential(
            nn.Linear(hidden_width + 5, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.raw_reliability[-1].weight)
        nn.init.zeros_(self.raw_reliability[-1].bias)

    def order_weights(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        orders: Tensor,
    ) -> Tensor:
        if len(support_cells) < 1 or len(support_cells[0]) < 2:
            raise ValueError("attentive metric needs support environments and classes")
        environment_count = len(support_cells)
        class_count = len(support_cells[0])
        order_count = len(orders)
        if any(len(environment) != class_count for environment in support_cells):
            raise ValueError("support environments must contain the same class set")
        cell_rows = []
        for environment in support_cells:
            local = []
            for values in environment:
                if (
                    values.ndim != 3
                    or values.shape[1] != order_count
                    or values.shape[2] != self.statistic_width
                    or len(values) < 1
                ):
                    raise ValueError("invalid raw support field cell")
                encoded = self.second_encoder(values).permute(1, 0, 2)
                local.append(self.cell_encoder(self.second_pool(encoded)))
            cell_rows.append(torch.stack(local, dim=0))
        cells = torch.stack(cell_rows, dim=0)
        # Pool source environments separately inside each class/order cell.
        environment_sets = cells.permute(1, 2, 0, 3).reshape(
            class_count * order_count, environment_count, -1
        )
        class_tokens = self.environment_pool(environment_sets).reshape(
            class_count, order_count, -1
        )
        # Then pool the unordered known-class set for every order.
        support_token = self.class_pool(class_tokens.permute(1, 0, 2))
        features = torch.cat((support_token, self._coordinate(orders)), dim=-1)
        logits = self.raw_reliability(features).squeeze(-1)
        return self._normalize_order_logits(logits)


class AttentiveSupportConditionedMahalanobisMetric(
    AttentiveSupportConditionedFieldMetric
):
    """Learn a shared feature geometry in addition to attentive order weights.

    The bias-free square transform is initialized to the identity.  Its cosine
    product therefore defines a learned positive-semidefinite metric while the
    initial model is exactly the v17 attentive metric.  The transform is shared
    across every order, class, environment, support sample, and query sample.
    """

    def __init__(
        self,
        *,
        statistic_width: int,
        hidden_width: int,
        attention_heads: int,
        order_weight_normalization: str = "softmax",
    ) -> None:
        super().__init__(
            statistic_width=statistic_width,
            hidden_width=hidden_width,
            attention_heads=attention_heads,
            order_weight_normalization=order_weight_normalization,
        )
        self.feature_metric = nn.Linear(
            statistic_width, statistic_width, bias=False
        )
        nn.init.eye_(self.feature_metric.weight)

    def _metric_embed(self, fields: Tensor, order_weights: Tensor) -> Tensor:
        transformed = self.feature_metric(fields)
        return (
            transformed * self._sqrt_metric_weights(order_weights)[None, :, None]
        ).reshape(len(transformed), -1)

    def induce_with_weights(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        weights: Tensor,
    ) -> Tensor:
        if weights.ndim != 1 or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("explicit order weights must be a nonnegative nonzero vector")
        if any(
            values.shape[1] != len(weights)
            for environment in support_cells
            for values in environment
        ):
            raise ValueError("explicit metric differs from the support order axis")
        centroids = []
        for class_index in range(len(support_cells[0])):
            seconds = torch.cat(
                [environment[class_index] for environment in support_cells], dim=0
            )
            embedded = self._metric_embed(seconds, weights)
            centroid = self._unit_rows(embedded).mean(dim=0, keepdim=True)
            centroids.append(self._unit_rows(centroid)[0])
        return torch.stack(centroids)

    def score_queries(
        self, class_centroids: Tensor, order_weights: Tensor, query_fields: Tensor
    ) -> FieldMetricOutput:
        if query_fields.ndim != 3 or query_fields.shape[-1] != self.statistic_width:
            raise ValueError("query fields must be [second,order,feature]")
        if len(order_weights) != query_fields.shape[1]:
            raise ValueError("query order axis differs from the support metric")
        embedded = self._metric_embed(query_fields, order_weights)
        logits = self._unit_rows(embedded) @ class_centroids.T
        return FieldMetricOutput(logits, order_weights, class_centroids)


class QueryAdaptiveSupportConditionedMahalanobisMetric(
    AttentiveSupportConditionedMahalanobisMetric
):
    """Choose the order metric causally for each observed query field.

    The support set still induces the class memories and a source reliability
    prior.  A shared query network then updates that prior from the current
    field only.  No query statistic is stored and no parameter is adapted at
    inference time.  Zero initialization makes the initial function exactly
    the support-conditioned global metric.
    """

    def __init__(
        self,
        *,
        statistic_width: int,
        hidden_width: int,
        attention_heads: int,
        order_weight_normalization: str = "softmax",
    ) -> None:
        if order_weight_normalization != "softmax":
            raise ValueError("query-adaptive metric currently requires softmax weights")
        super().__init__(
            statistic_width=statistic_width,
            hidden_width=hidden_width,
            attention_heads=attention_heads,
            order_weight_normalization=order_weight_normalization,
        )
        self.query_reliability = nn.Sequential(
            nn.Linear(statistic_width + 5, hidden_width),
            nn.GELU(),
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.query_reliability[-1].weight)
        nn.init.zeros_(self.query_reliability[-1].bias)

    def query_order_weights(
        self, query_fields: Tensor, support_weights: Tensor, orders: Tensor
    ) -> Tensor:
        if query_fields.ndim != 3 or query_fields.shape[-1] != self.statistic_width:
            raise ValueError("query fields must be [second,order,feature]")
        if query_fields.shape[1] != len(orders) or len(support_weights) != len(orders):
            raise ValueError("query-adaptive order axes differ")
        coordinates = self._coordinate(orders)[None].expand(len(query_fields), -1, -1)
        features = torch.cat((query_fields, coordinates), dim=-1)
        residual = self.query_reliability(features).squeeze(-1)
        prior = support_weights.clamp_min(torch.finfo(support_weights.dtype).tiny).log()
        return torch.softmax(prior[None] + residual, dim=-1) * len(orders)

    def forward(
        self,
        support_cells: tuple[tuple[Tensor, ...], ...],
        query_fields: Tensor,
        orders: Tensor,
    ) -> FieldMetricOutput:
        support_weights = self.order_weights(support_cells, orders)
        query_weights = self.query_order_weights(
            query_fields, support_weights, orders
        )
        transformed_query = self.feature_metric(query_fields)
        transformed_support = [
            self.feature_metric(
                torch.cat(
                    [environment[class_index] for environment in support_cells],
                    dim=0,
                )
            )
            for class_index in range(len(support_cells[0]))
        ]
        logits = []
        for second, weights in enumerate(query_weights):
            scale = self._sqrt_metric_weights(weights)[None, :, None]
            query = self._unit_rows(
                (transformed_query[second : second + 1] * scale).reshape(1, -1)
            )
            centroids = []
            for values in transformed_support:
                embedded = self._unit_rows((values * scale).reshape(len(values), -1))
                centroid = self._unit_rows(embedded.mean(dim=0, keepdim=True))
                centroids.append(centroid[0])
            logits.append(query @ torch.stack(centroids).T)
        diagnostic_centroids = self.induce_with_weights(
            support_cells, support_weights
        )
        return FieldMetricOutput(
            torch.cat(logits, dim=0), query_weights, diagnostic_centroids
        )


__all__ = [
    "AttentiveSupportConditionedMahalanobisMetric",
    "AttentiveSupportConditionedFieldMetric",
    "FieldMetricOutput",
    "NeuralSupportConditionedFieldMetric",
    "QueryAdaptiveSupportConditionedMahalanobisMetric",
    "SupportConditionedFieldMetric",
]
