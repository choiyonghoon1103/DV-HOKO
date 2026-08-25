"""Neural operation readout trained on an immutable Koopman--Mori field."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from hoko.common import deterministic_prefix_sum, sha256


def fit_frozen_operation_reader(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    model,
    orders: torch.Tensor,
    config: dict,
    device: torch.device,
    base_checkpoint: Path,
    checkpoint: Path,
):
    """Fit only the attention-based operation reader on frozen field streams."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite operation reader: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    seed = int(config["optimization"]["seed"]) + bearings.index(held)
    torch.manual_seed(seed)
    model.requires_grad_(False)
    parameters = list(model.operation_parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    if not parameters:
        raise RuntimeError("model has no neural operation-reader parameters")
    model.eval()
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    cells: dict[tuple[str, int], list[tuple[str, torch.Tensor]]] = {}
    for bearing in sources:
        for label in (1, 2, 3):
            records = [
                (
                    str(row["record_id"]),
                    torch.from_numpy(row["frozen_field"]).to(device),
                )
                for row in rows
                if str(row["bearing"]) == bearing
                and int(row["class_index"]) == label
            ]
            if not records:
                raise ValueError("operation reader source cell is empty")
            cells[(bearing, label)] = records

    updates = int(config["optimization"]["updates"])
    log_every = int(config["optimization"]["log_every"])
    trace = []
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        bearing_losses = []
        per_bearing = {}
        for bearing in sources:
            class_losses = []
            for label in (1, 2, 3):
                record_losses = []
                for _, field in cells[(bearing, label)]:
                    local_logits, _ = model._operation(field, orders)
                    prefix_logits = deterministic_prefix_sum(local_logits)
                    target = torch.full(
                        (len(prefix_logits),),
                        label - 1,
                        dtype=torch.long,
                        device=device,
                    )
                    record_losses.append(F.cross_entropy(prefix_logits, target))
                class_losses.append(torch.stack(record_losses).mean())
            local = torch.stack(class_losses).mean()
            bearing_losses.append(local)
            per_bearing[bearing] = float(local.detach())
        loss = torch.stack(bearing_losses).mean()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            parameters, float(config["optimization"]["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % log_every == 0 or update == updates:
            item = {
                "stage": "frozen_koopman_mori_field_neural_operation_reader",
                "update": update,
                "source_balanced_causal_prefix_nll": float(loss.detach()),
                "per_source_bearing_nll": per_bearing,
                "gradient_norm_before_clip": float(gradient),
                "trunk_parameters_updated": False,
            }
            trace.append(item)
            print(
                f"OP-READER held={held} update={update}/{updates} "
                f"nll={item['source_balanced_causal_prefix_nll']:.6f}",
                flush=True,
            )

    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "seed": seed,
            "base_dynamics_checkpoint_sha256": sha256(base_checkpoint),
            "operation_reader_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
                if any(key == name or key.startswith(name + ".") for name in (
                    "mode_score",
                    "class_embedding",
                    "query_generator",
                    "context_query",
                    "context_mode_embedding",
                    "context_order_embedding",
                ))
            },
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "trace": trace,
            "trained_parameter_count": sum(p.numel() for p in parameters),
            "trunk_parameters_updated": False,
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return model.eval(), trace


def operation_reader_streams(model, rows, held: str, orders: torch.Tensor, device):
    streams = {}
    with torch.inference_mode():
        for row in rows:
            if str(row["bearing"]) != held or int(row["class_index"]) not in (1, 2, 3):
                continue
            field = torch.from_numpy(row["frozen_field"]).to(device)
            local, _ = model._operation(field, orders)
            streams[str(row["record_id"])] = deterministic_prefix_sum(local).cpu()
    return streams


__all__ = ["fit_frozen_operation_reader", "operation_reader_streams"]
