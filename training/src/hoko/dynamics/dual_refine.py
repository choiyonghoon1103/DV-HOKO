"""Joint source-meta refinement of the shared state/dynamics representation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from hoko.common import deterministic_prefix_sum
from hoko.dynamics.factory import build_model
from hoko.dynamics.train import (
    _observation_orders,
    angular_paths_and_state,
    build_mixer,
)
from hoko.memory.train import _class_prefix_risks, _query_batch, _support_cells


LABELS = (0, 1, 2, 3)
FAULTS = (1, 2, 3)


def differentiable_dual_views(
    rows: list[dict],
    bearings: tuple[str, ...],
    model,
    mixer,
    config: dict,
    device: torch.device,
) -> tuple[dict, dict, Tensor]:
    """Return differentiable state/dynamics views and fault forecast risk."""

    mode_bank = getattr(model, "angular_mode_bank", None)
    orders = _observation_orders(config, device, mode_bank)
    fields: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    bases: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    forecast: dict[tuple[str, int], list[Tensor]] = defaultdict(list)

    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing not in bearings or label not in LABELS:
            continue
        envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)

        def record_forward(values: Tensor) -> tuple[Tensor, Tensor, Tensor]:
            paths, state = angular_paths_and_state(mixer, values, config, mode_bank)
            valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
            field, risk = model._field_and_forecast(paths, valid, orders)
            base = model.state_base(paths, valid, orders, state)
            return field, base, risk

        field, base, risk = activation_checkpoint(
            record_forward, envelopes, use_reentrant=False
        )
        fields[(bearing, label)].append(field)
        bases[(bearing, label)].append(base)
        if label in FAULTS:
            forecast[(bearing, label)].append(risk)

    expected = {(bearing, label) for bearing in bearings for label in LABELS}
    if set(fields) != expected or set(bases) != expected:
        raise ValueError("joint refinement requires every source bearing/class cell")
    fault_expected = {(bearing, label) for bearing in bearings for label in FAULTS}
    if set(forecast) != fault_expected:
        raise ValueError("joint refinement fault forecast cells are incomplete")
    bearing_risks = []
    for bearing in bearings:
        bearing_risks.append(
            torch.stack(
                [torch.stack(forecast[(bearing, label)]).mean() for label in FAULTS]
            ).mean()
        )
    return dict(fields), dict(bases), torch.stack(bearing_risks).mean()


def _health_centroids(decoder, bases: dict, bearings: tuple[str, ...]) -> Tensor:
    by_health: dict[int, list[Tensor]] = defaultdict(list)
    for bearing in bearings:
        for label in LABELS:
            for record in bases[(bearing, label)]:
                by_health[int(label != 0)].append(decoder(record))
    return F.normalize(
        torch.stack([torch.cat(by_health[index]).mean(dim=0) for index in (0, 1)]),
        dim=-1,
    )


def health_meta_query_risk(
    bases: dict,
    bearings: tuple[str, ...],
    decoder,
) -> tuple[Tensor, dict[str, float]]:
    risks, per_bearing = [], {}
    for pseudoheld in bearings:
        support = tuple(value for value in bearings if value != pseudoheld)
        centroids = _health_centroids(decoder, bases, support)
        class_risks = []
        for label in LABELS:
            record_risks = []
            for record in bases[(pseudoheld, label)]:
                logits = deterministic_prefix_sum(decoder(record) @ centroids.T)
                target = torch.full(
                    (len(logits),), int(label != 0), dtype=torch.long, device=logits.device
                )
                record_risks.append(F.cross_entropy(logits, target))
            class_risks.append(torch.stack(record_risks).mean())
        local = torch.stack(class_risks).mean()
        risks.append(local)
        per_bearing[pseudoheld] = float(local.detach())
    return torch.stack(risks).mean(), per_bearing


def operation_meta_query_group_risks(
    fields: dict,
    bearings: tuple[str, ...],
    metric,
    orders: Tensor,
) -> tuple[Tensor, dict[str, list[float]]]:
    """Return pseudo-held-bearing by operation-class prefix risks.

    Keeping the groups separate lets a source-only minimax refinement prevent
    easy operation cells from hiding a transported I/O/B decision boundary.
    """

    groups, per_bearing = [], {}
    for pseudoheld in bearings:
        support = _support_cells(
            fields,
            tuple(value for value in bearings if value != pseudoheld),
            FAULTS,
        )
        query, records = _query_batch(fields, pseudoheld, FAULTS)
        local = _class_prefix_risks(metric(support, query, orders).logits, records)
        groups.append(local)
        per_bearing[pseudoheld] = [float(value) for value in local.detach()]
    return torch.stack(groups), per_bearing


def refine_dual_fold(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    model,
    mixer,
    operation_metric,
    state_decoder,
    dynamics_config: dict,
    refinement_config: dict,
    device: torch.device,
    checkpoint: Path,
):
    """Refine the trunk under normalized health, operation and forecast risks."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite joint-refined dynamics: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    torch.manual_seed(
        int(refinement_config["optimization"]["seed"]) + bearings.index(held)
    )
    operation_metric.eval().requires_grad_(False)
    state_decoder.eval().requires_grad_(False)
    model.train(); mixer.train()
    parameters = list(model.field_parameters()) + list(mixer.parameters())
    mode_bank = getattr(model, "angular_mode_bank", None)
    if mode_bank is not None:
        parameters += list(mode_bank.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(refinement_config["optimization"]["learning_rate"]),
        weight_decay=float(refinement_config["optimization"]["weight_decay"]),
    )
    orders = _observation_orders(dynamics_config, device, mode_bank)
    initial = None
    trace = []
    updates = int(refinement_config["optimization"]["updates"])
    log_every = int(refinement_config["optimization"]["log_every"])
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        fields, bases, forecast = differentiable_dual_views(
            rows, sources, model, mixer, dynamics_config, device
        )
        operation_fields = {
            key: value for key, value in fields.items() if key[1] in FAULTS
        }
        operation_groups, operation_groups_by_bearing = operation_meta_query_group_risks(
            operation_fields, sources, operation_metric, orders
        )
        operation_reduction = str(
            refinement_config.get("objective", {}).get(
                "operation_meta_reduction", "mean"
            )
        )
        if operation_reduction == "mean":
            operation = operation_groups.mean()
        elif operation_reduction == "maximum_group":
            operation = operation_groups.max()
        else:
            raise ValueError("unknown operation meta-risk reduction")
        operation_mean = operation_groups.mean()
        health, health_by_bearing = health_meta_query_risk(
            bases, sources, state_decoder
        )
        if initial is None:
            initial = tuple(
                value.detach().clamp_min(1e-12)
                for value in (operation, health, forecast)
            )
        objective = (
            operation / initial[0] + health / initial[1] + forecast / initial[2]
        ) / 3.0
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(refinement_config["optimization"]["gradient_clip_norm"]),
        )
        optimizer.step()
        if update == 1 or update % log_every == 0 or update == updates:
            item = {
                "update": update,
                "objective": float(objective.detach()),
                "source_operation_nll": float(operation.detach()),
                "source_mean_operation_nll": float(operation_mean.detach()),
                "source_maximum_group_operation_nll": float(
                    operation_groups.max().detach()
                ),
                "source_health_nll": float(health.detach()),
                "class_free_forecast_risk": float(forecast.detach()),
                "normalized_operation_risk": float((operation / initial[0]).detach()),
                "normalized_health_risk": float((health / initial[1]).detach()),
                "normalized_forecast_risk": float((forecast / initial[2]).detach()),
                "per_pseudoheld_operation_group_nll": operation_groups_by_bearing,
                "per_pseudoheld_health_nll": health_by_bearing,
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"DUAL-REFINE held={held} update={update}/{updates} "
                f"op={item['source_operation_nll']:.6f} "
                f"health={item['source_health_nll']:.6f} "
                f"forecast={item['class_free_forecast_risk']:.6f}",
                flush=True,
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": refinement_config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "mixer_state_dict": {key: value.detach().cpu() for key, value in mixer.state_dict().items()},
            "trace": trace,
            "initial_source_operation_nll": float(initial[0]),
            "initial_source_health_nll": float(initial[1]),
            "initial_class_free_forecast_risk": float(initial[2]),
            "operation_metric_parameters_updated": False,
            "state_decoder_parameters_updated": False,
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return model.eval(), mixer.eval(), trace


def load_dual_refined_fold(
    held: str,
    bearings: tuple[str, ...],
    dynamics_config: dict,
    refinement_config: dict,
    device: torch.device,
    checkpoint: Path,
):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    sources = tuple(value for value in bearings if value != held)
    if (
        payload.get("schema") != refinement_config["schema"]
        or str(payload.get("held_bearing")) != held
        or tuple(payload.get("source_bearings", ())) != sources
    ):
        raise ValueError("joint-refined dynamics checkpoint contract differs")
    model, mixer = build_model(dynamics_config, device), build_mixer(dynamics_config, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    return model.eval(), mixer.eval(), payload


__all__ = [
    "differentiable_dual_views",
    "health_meta_query_risk",
    "load_dual_refined_fold",
    "operation_meta_query_group_risks",
    "refine_dual_fold",
]
