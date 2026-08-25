"""Frozen-trunk fitting of operation-conditional Koopman--Mori closures.

The validated source-only representation is treated as an immutable observation
model.  Only a centred operation closure and a training-only centred source
environment closure are fitted.  This prevents the operation labels from
silently redesigning the filterbank or the common dynamical state.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from hoko.common import sha256
from hoko.dynamics.closure import (
    CenteredConditionalOperationClosure,
    CenteredFactorClosure,
)
from hoko.dynamics.environment import CenteredEnvironmentClosure
from hoko.dynamics.train import angular_paths, _observation_orders


def _causal_cumsum(values: torch.Tensor) -> torch.Tensor:
    """Ordered differentiable prefix sum compatible with deterministic CUDA."""

    running = torch.zeros_like(values[0])
    prefixes = []
    for step in range(len(values)):
        running = running + values[step]
        prefixes.append(running)
    return torch.stack(prefixes, dim=0)


def install_operation_hypotheses(
    model,
    *,
    environment_count: int,
    closure_type: str,
    device: torch.device,
) -> CenteredEnvironmentClosure:
    """Freeze a fitted trunk and attach the only two trainable closure modules."""

    if model.operation_forecast_head is not None:
        raise ValueError("operation hypotheses are already installed")
    forecast_components = 2 if model.forecast_distribution == "deterministic" else 3
    output_width = model.forecast_horizons * forecast_components * model.carrier_bands
    if closure_type == "linear":
        operation = CenteredFactorClosure(
            model.operation_count, model.field_width, output_width
        )
    elif closure_type == "conditional_mlp":
        operation = CenteredConditionalOperationClosure(
            model.operation_count,
            model.field_width,
            model.embedding_width,
            output_width,
        )
    else:
        raise ValueError("unknown operation hypothesis closure type")

    model.requires_grad_(False)
    model.operation_forecast_head = operation.to(device)
    model.factorized_operation_closure = True
    model.operation_closure_type = closure_type
    model.operation_forecast_head.requires_grad_(True)
    environment = CenteredEnvironmentClosure(
        environment_count=environment_count,
        input_width=model.field_width,
        output_width=output_width,
    ).to(device)
    return environment


def fit_frozen_operation_hypotheses(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    model,
    mixer,
    dynamics_config: dict,
    hypothesis_config: dict,
    device: torch.device,
    base_checkpoint: Path,
    checkpoint: Path,
):
    """Fit only operation and source-environment closures on source faults."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite operation hypotheses: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    seed = int(hypothesis_config["optimization"]["seed"]) + bearings.index(held)
    torch.manual_seed(seed)
    closure_type = str(hypothesis_config["closure"]["operation_type"])
    environment = install_operation_hypotheses(
        model,
        environment_count=len(sources),
        closure_type=closure_type,
        device=device,
    )
    mixer.eval().requires_grad_(False)
    model.eval()

    parameters = list(model.operation_dynamics_parameters()) + list(
        environment.parameters()
    )
    if not parameters or any(not parameter.requires_grad for parameter in parameters):
        raise RuntimeError("closure-only parameter contract is invalid")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(hypothesis_config["optimization"]["learning_rate"]),
        weight_decay=float(hypothesis_config["optimization"]["weight_decay"]),
    )

    mode_bank = getattr(model, "angular_mode_bank", None)
    orders = _observation_orders(dynamics_config, device, mode_bank)
    # Preserve record boundaries because deployment decisions accumulate local
    # forecast evidence causally within each record.  The frozen observation map
    # is materialized once per record, avoiding repeated selected DFT work.
    cached: dict[tuple[str, int], list[tuple[str, torch.Tensor, torch.Tensor]]] = {}
    with torch.inference_mode():
        for bearing in sources:
            for label in (1, 2, 3):
                records = []
                for row in rows:
                    if str(row["bearing"]) != bearing or int(row["class_index"]) != label:
                        continue
                    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
                    paths = angular_paths(mixer, envelopes, dynamics_config, mode_bank)
                    valid = torch.ones(
                        paths.shape[:2], dtype=torch.bool, device=paths.device
                    )
                    records.append((str(row["record_id"]), paths, valid))
                if not records:
                    raise ValueError("operation hypothesis source cell is empty")
                cached[(bearing, label)] = records

    updates = int(hypothesis_config["optimization"]["updates"])
    log_every = int(hypothesis_config["optimization"]["log_every"])
    trace = []
    initial_forecast = None
    initial_meta = None
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        forecast_losses = []
        meta_losses = []
        per_cell = {}
        for environment_index, bearing in enumerate(sources):
            support_environment_indices = tuple(
                index for index in range(len(sources)) if index != environment_index
            )
            for label in (1, 2, 3):
                record_forecasts = []
                record_meta = []
                record_items = {}
                for record_id, paths, valid in cached[(bearing, label)]:
                    local_nll = model.operation_forecast_nll(
                        paths,
                        valid,
                        orders,
                        environment_residual_head=environment,
                        environment_indices=support_environment_indices,
                    )
                    cumulative_logits = -_causal_cumsum(local_nll)
                    target = torch.full(
                        (len(local_nll),), label - 1, dtype=torch.long, device=device
                    )
                    forecast = local_nll[:, label - 1].mean()
                    meta = F.cross_entropy(cumulative_logits, target)
                    record_forecasts.append(forecast)
                    record_meta.append(meta)
                    record_items[record_id] = {
                        "forecast_nll": float(forecast.detach()),
                        "causal_prefix_operation_ce": float(meta.detach()),
                    }
                forecast = torch.stack(record_forecasts).mean()
                meta = torch.stack(record_meta).mean()
                forecast_losses.append(forecast)
                meta_losses.append(meta)
                per_cell[f"{bearing}:{label}"] = {
                    "held_environment_forecast_nll": float(forecast.detach()),
                    "held_environment_causal_prefix_operation_ce": float(meta.detach()),
                    "records": record_items,
                }
        forecast_risk = torch.stack(forecast_losses).mean()
        meta_risk = torch.stack(meta_losses).mean()
        if initial_forecast is None:
            initial_forecast = forecast_risk.detach().clamp_min(1e-12)
            initial_meta = meta_risk.detach().clamp_min(1e-12)
        objective = 0.5 * (
            forecast_risk / initial_forecast + meta_risk / initial_meta
        )
        objective.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters,
            float(hypothesis_config["optimization"]["gradient_clip_norm"]),
        )
        optimizer.step()
        if update == 1 or update % log_every == 0 or update == updates:
            item = {
                "stage": "frozen_trunk_operation_hypothesis_fit",
                "update": update,
                "source_loeo_operation_forecast_nll": float(forecast_risk.detach()),
                "source_loeo_operation_ce": float(meta_risk.detach()),
                "normalized_forecast_risk": float(
                    (forecast_risk / initial_forecast).detach()
                ),
                "normalized_meta_risk": float((meta_risk / initial_meta).detach()),
                "equal_normalized_objective": float(objective.detach()),
                "per_pseudoheld_bearing_operation_risk": per_cell,
                "gradient_norm_before_clip": float(gradient),
                "trunk_parameters_updated": False,
                "mixer_parameters_updated": False,
            }
            trace.append(item)
            print(
                f"OP-CLOSURE held={held} update={update}/{updates} "
                f"nll={item['source_loeo_operation_forecast_nll']:.6f} "
                f"ce={item['source_loeo_operation_ce']:.6f}",
                flush=True,
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": hypothesis_config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "seed": seed,
            "base_dynamics_checkpoint_sha256": sha256(base_checkpoint),
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "mixer_state_dict": {
                key: value.detach().cpu() for key, value in mixer.state_dict().items()
            },
            "environment_closure_state_dict": {
                key: value.detach().cpu() for key, value in environment.state_dict().items()
            },
            "trace": trace,
            "operation_closure_type": closure_type,
            "trained_parameter_count": sum(p.numel() for p in parameters),
            "trunk_parameters_updated": False,
            "mixer_parameters_updated": False,
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    # Attach the nuisance bank only for immediate evaluation.  It is stored as
    # a separate training artifact above and is not duplicated in model_state.
    model.source_environment_closure = environment.eval()
    return model.eval(), mixer.eval(), trace


__all__ = ["fit_frozen_operation_hypotheses", "install_operation_hypotheses"]
