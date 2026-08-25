"""Binary health evidence induced from normal and fault field hypotheses."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from hoko.memory.model import SupportConditionedFieldMetric


def binary_health_logits(class_logits: Tensor) -> Tensor:
    """Map N/I/O/B field scores to N/F without collapsing fault modes.

    The fault score is the equal-mixture log evidence of every non-normal
    hypothesis.  This keeps I/O/B exchangeable and introduces no fault-location
    preference into the binary health task.
    """

    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError("health scoring needs normal plus fault hypotheses")
    normal = class_logits[:, :1]
    fault = torch.logsumexp(class_logits[:, 1:], dim=-1, keepdim=True)
    fault = fault - math.log(class_logits.shape[1] - 1)
    return torch.cat((normal, fault), dim=-1)


def score_health(
    model: SupportConditionedFieldMetric,
    support_cells: tuple[tuple[Tensor, ...], ...],
    query_fields: Tensor,
    orders: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return local binary logits and the induced positive order weights."""

    output = model(support_cells, query_fields, orders)
    return binary_health_logits(output.logits), output.order_weights


__all__ = ["binary_health_logits", "score_health"]
