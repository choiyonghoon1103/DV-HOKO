"""Source-only meta-query refinement of a pretrained dynamical field.

The class memory is frozen.  Whole-source pseudo-held classification losses
are differentiated through that memory into the Koopman--Mori field encoder
and learned subband mixer.  The objective can retain a normalized forecast
risk, or set its declared weight to zero after source dynamics pretraining.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from hoko.dynamics.train import angular_paths, _observation_orders
from hoko.dynamics.factory import build_model
from hoko.dynamics.train import build_mixer
from hoko.memory.train import _class_prefix_risks, _query_batch, _support_cells


def normalized_refinement_objective(
    meta: Tensor,
    forecast: Tensor,
    initial_meta: Tensor,
    initial_forecast: Tensor,
    objective_config: dict,
) -> Tensor:
    """Combine normalized task and dynamics risks with declared fixed weights."""

    meta_weight = float(objective_config.get("meta_query_weight", 0.5))
    forecast_weight = float(objective_config.get("forecast_retention_weight", 0.5))
    if meta_weight < 0.0 or forecast_weight < 0.0:
        raise ValueError("refinement objective weights cannot be negative")
    total = meta_weight + forecast_weight
    if total <= 0.0:
        raise ValueError("at least one refinement objective weight must be positive")
    return (
        meta_weight * (meta / initial_meta)
        + forecast_weight * (forecast / initial_forecast)
    ) / total


def differentiable_fields(
    rows: list[dict],
    bearings: tuple[str, ...],
    model,
    mixer,
    config: dict,
    device: torch.device,
    *,
    labels: tuple[int, ...] = (1, 2, 3),
    rematerialize: bool = True,
) -> tuple[dict[tuple[str, int], list[Tensor]], Tensor]:
    """Return source record fields and a bearing/class/record-balanced risk."""

    mode_bank = getattr(model, "angular_mode_bank", None)
    orders = _observation_orders(config, device, mode_bank)
    fields: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    forecast: dict[tuple[str, int], list[Tensor]] = defaultdict(list)

    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing not in bearings or label not in labels:
            continue
        envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)

        def record_forward(values: Tensor) -> tuple[Tensor, Tensor]:
            paths = angular_paths(mixer, values, config, mode_bank)
            valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
            return model._field_and_forecast(paths, valid, orders)

        if rematerialize:
            field, risk = activation_checkpoint(
                record_forward, envelopes, use_reentrant=False
            )
        else:
            field, risk = record_forward(envelopes)
        fields[(bearing, label)].append(field)
        forecast[(bearing, label)].append(risk)

    expected = {(bearing, label) for bearing in bearings for label in labels}
    if set(fields) != expected or set(forecast) != expected:
        raise ValueError("meta refinement requires every source bearing/fault cell")
    bearing_risks = []
    for bearing in bearings:
        class_risks = [
            torch.stack(forecast[(bearing, label)]).mean()
            for label in labels
        ]
        bearing_risks.append(torch.stack(class_risks).mean())
    return dict(fields), torch.stack(bearing_risks).mean()


def meta_query_risk(
    fields: dict[tuple[str, int], list[Tensor]],
    bearings: tuple[str, ...],
    metric,
    orders: Tensor,
    labels: tuple[int, ...] = (1, 2, 3),
) -> tuple[Tensor, dict[str, float]]:
    """Compute the exact whole-bearing LOEO risk of a frozen class memory."""

    risks = []
    per_bearing = {}
    for pseudoheld in bearings:
        support = _support_cells(
            fields,
            tuple(value for value in bearings if value != pseudoheld),
            labels,
        )
        query, records = _query_batch(fields, pseudoheld, labels)
        local = _class_prefix_risks(metric(support, query, orders).logits, records).mean()
        risks.append(local)
        per_bearing[pseudoheld] = float(local.detach())
    return torch.stack(risks).mean(), per_bearing


def refine_fold(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    model,
    mixer,
    metric,
    dynamics_config: dict,
    refinement_config: dict,
    device: torch.device,
    checkpoint: Path,
):
    """Refine only the dynamical representation using source LOEO gradients."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite meta-refined dynamics: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    labels = tuple(
        int(value)
        for value in refinement_config.get("objective", {}).get(
            "candidate_labels", (1, 2, 3)
        )
    )
    seed = int(refinement_config["optimization"]["seed"]) + bearings.index(held)
    torch.manual_seed(seed)

    # The memory remains an immutable source-trained decision rule.  Autograd
    # still traverses its operations to reach the support and query fields.
    metric.eval().requires_grad_(False)
    model.train()
    mixer.train()
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
    updates = int(refinement_config["optimization"]["updates"])
    log_every = int(refinement_config["optimization"]["log_every"])
    trace = []
    initial_meta = None
    initial_forecast = None
    objective_config = refinement_config.get("objective", {})

    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        fields, forecast = differentiable_fields(
            rows,
            sources,
            model,
            mixer,
            dynamics_config,
            device,
            labels=labels,
            rematerialize=True,
        )
        meta, per_bearing = meta_query_risk(
            fields, sources, metric, orders, labels
        )
        if initial_meta is None:
            initial_meta = meta.detach().clamp_min(1e-12)
            initial_forecast = forecast.detach().clamp_min(1e-12)
        objective = normalized_refinement_objective(
            meta,
            forecast,
            initial_meta,
            initial_forecast,
            objective_config,
        )
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(refinement_config["optimization"]["gradient_clip_norm"]),
        )
        optimizer.step()
        if update == 1 or update % log_every == 0 or update == updates:
            item = {
                "stage": "source_loeo_dynamics_meta_refinement",
                "update": update,
                "objective": float(objective.detach()),
                "source_meta_query_nll": float(meta.detach()),
                "class_free_forecast_risk": float(forecast.detach()),
                "normalized_meta_query_risk": float((meta / initial_meta).detach()),
                "normalized_forecast_risk": float(
                    (forecast / initial_forecast).detach()
                ),
                "per_pseudoheld_nll": per_bearing,
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"META-DYNAMICS held={held} update={update}/{updates} "
                f"nll={item['source_meta_query_nll']:.6f} "
                f"forecast={item['class_free_forecast_risk']:.6f}",
                flush=True,
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": refinement_config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "seed": seed,
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "mixer_state_dict": {
                key: value.detach().cpu() for key, value in mixer.state_dict().items()
            },
            "trace": trace,
            "initial_source_meta_query_nll": float(initial_meta),
            "initial_class_free_forecast_risk": float(initial_forecast),
            "metric_parameters_updated": False,
            "candidate_labels": labels,
            "meta_query_weight": float(objective_config.get("meta_query_weight", 0.5)),
            "forecast_retention_weight": float(
                objective_config.get("forecast_retention_weight", 0.5)
            ),
            "class_loss_gradient_reached_dynamics": True,
            "class_loss_gradient_reached_filterbank": True,
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return model.eval(), mixer.eval(), trace


def load_refined_fold(
    held: str,
    bearings: tuple[str, ...],
    dynamics_config: dict,
    refinement_config: dict,
    device: torch.device,
    checkpoint: Path,
):
    """Load an immutable source-meta-refined dynamics checkpoint."""

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    sources = tuple(value for value in bearings if value != held)
    if (
        payload.get("schema") != refinement_config["schema"]
        or str(payload.get("held_bearing")) != held
        or tuple(payload.get("source_bearings", ())) != sources
    ):
        raise ValueError("meta-refined dynamics checkpoint contract differs")
    model = build_model(dynamics_config, device)
    mixer = build_mixer(dynamics_config, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    return model.eval(), mixer.eval(), payload


__all__ = [
    "differentiable_fields",
    "load_refined_fold",
    "meta_query_risk",
    "normalized_refinement_objective",
    "refine_fold",
]
