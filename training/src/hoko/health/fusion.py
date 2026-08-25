"""Source-domain-balanced calibration for two frozen binary health observers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Tuple

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from torch.nn import functional as F


Record = Tuple[torch.Tensor, torch.Tensor, int]


def binary_group_nlls(
    domains: Mapping[str, Sequence[Record]], alpha: float
) -> dict[tuple[str, int], float]:
    """Return every bearing-by-health balanced record/time log loss."""

    output = {}
    for domain, records in domains.items():
        for label in (0, 1):
            local = []
            for left, right, current in records:
                if int(current) != label:
                    continue
                left_cpu = left.detach().cpu().to(torch.float64)
                right_cpu = right.detach().cpu().to(torch.float64)
                fused = float(alpha) * left_cpu + (1.0 - float(alpha)) * right_cpu
                local.append(
                    F.binary_cross_entropy_with_logits(
                        fused, torch.full_like(fused, float(label))
                    )
                )
            if not local:
                raise ValueError("every calibration domain must contain both health states")
            output[(str(domain), label)] = float(torch.stack(local).mean())
    if not output:
        raise ValueError("at least one calibration domain is required")
    return output


def hierarchical_binary_nll(domains: Mapping[str, Sequence[Record]], alpha: float) -> float:
    """Bearing -> health -> record -> time balanced log loss."""

    domain_losses = []
    for records in domains.values():
        health_losses = []
        for label in (0, 1):
            local = []
            for left, right, current in records:
                if int(current) != label:
                    continue
                left_cpu = left.detach().cpu().to(torch.float64)
                right_cpu = right.detach().cpu().to(torch.float64)
                fused = float(alpha) * left_cpu + (1.0 - float(alpha)) * right_cpu
                local.append(
                    F.binary_cross_entropy_with_logits(
                        fused, torch.full_like(fused, float(label))
                    )
                )
            if not local:
                raise ValueError("every calibration domain must contain both health states")
            health_losses.append(torch.stack(local).mean())
        domain_losses.append(torch.stack(health_losses).mean())
    if not domain_losses:
        raise ValueError("at least one calibration domain is required")
    return float(torch.stack(domain_losses).mean())


def fit_convex_logit_pool(domains: Mapping[str, Sequence[Record]]) -> dict:
    """Fit the single global left-observer weight on source-only cross-predictions."""

    result = minimize_scalar(
        lambda value: hierarchical_binary_nll(domains, float(value)),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 256},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"health fusion calibration failed: {result.message}")
    candidates = [
        (float(result.x), float(result.fun)),
        (0.0, hierarchical_binary_nll(domains, 0.0)),
        (1.0, hierarchical_binary_nll(domains, 1.0)),
    ]
    alpha, loss = min(candidates, key=lambda item: (item[1], item[0]))
    return {
        "left_weight": alpha,
        "right_weight": 1.0 - alpha,
        "calibration_nll": loss,
        "equal_weight_nll": hierarchical_binary_nll(domains, 0.5),
        "optimizer": "bounded_scalar_convex_logit_pool",
    }


def fit_source_safe_convex_logit_pool(
    domains: Mapping[str, Sequence[Record]],
) -> dict:
    """Minimize worst group regret relative to the exact equal-logit rule."""

    reference = binary_group_nlls(domains, 0.5)

    def objective(value: float) -> float:
        risks = binary_group_nlls(domains, float(value))
        return max(risks[key] - reference[key] for key in reference)

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1e-10, "maxiter": 256},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"source-safe health fusion failed: {result.message}")
    candidates = [(float(result.x), objective(float(result.x)))]
    candidates.extend((value, objective(value)) for value in (0.0, 0.5, 1.0))
    alpha, maximum_regret = min(
        candidates, key=lambda item: (item[1], abs(item[0] - 0.5), item[0])
    )
    return {
        "left_weight": alpha,
        "right_weight": 1.0 - alpha,
        "calibration_nll": hierarchical_binary_nll(domains, alpha),
        "equal_weight_nll": hierarchical_binary_nll(domains, 0.5),
        "maximum_equal_fusion_regret": maximum_regret,
        "optimizer": "bounded_scalar_maximum_equal_fusion_group_regret",
    }


def fuse_logits(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], alpha: float
) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        raise ValueError("health observers cover different records")
    return {
        key: float(alpha) * left[key].detach().cpu()
        + (1.0 - float(alpha)) * right[key].detach().cpu()
        for key in left
    }
