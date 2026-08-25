"""Whole-bearing HUST-only meta-training for binary health evidence."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from hoko.common import deterministic_prefix_sum
from hoko.health.model import binary_health_logits
from hoko.memory.train import build_model


LABELS = (0, 1, 2, 3)


def _cells(rows: list[dict], bearings: tuple[str, ...], device: torch.device) -> dict:
    cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing in bearings and label in LABELS:
            cells[(bearing, label)].append(
                torch.from_numpy(np.asarray(row["frozen_field"], dtype=np.float32)).to(device)
            )
    expected = {(bearing, label) for bearing in bearings for label in LABELS}
    if set(cells) != expected:
        raise ValueError("health meta-training requires every bearing/class cell")
    return dict(cells)


def _support_cells(cells: dict, bearings: tuple[str, ...]):
    return tuple(
        tuple(torch.cat(cells[(bearing, label)], dim=0) for label in LABELS)
        for bearing in bearings
    )


def _query_batch(cells: dict, bearing: str) -> tuple[Tensor, list[dict]]:
    fields, records, cursor = [], [], 0
    for label in LABELS:
        for values in cells[(bearing, label)]:
            start = cursor
            fields.append(values)
            cursor += len(values)
            records.append(
                {
                    "class_index": label,
                    "health_label": int(label != 0),
                    "slice": slice(start, cursor),
                }
            )
    return torch.cat(fields), records


def _health_prefix_risks(local_logits: Tensor, records: list[dict]) -> Tensor:
    by_health: dict[int, list[Tensor]] = defaultdict(list)
    for record in records:
        prefix = deterministic_prefix_sum(local_logits[record["slice"]])
        label = int(record["health_label"])
        target = torch.full((len(prefix),), label, dtype=torch.long, device=prefix.device)
        by_health[label].append(F.cross_entropy(prefix, target))
    if set(by_health) != {0, 1}:
        raise ValueError("health query must contain normal and fault records")
    return torch.stack(
        [torch.stack(by_health[label]).mean() for label in (0, 1)]
    )


def fit_fold(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    orders: Tensor,
    config: dict,
    device: torch.device,
    checkpoint: Path,
):
    """Fit one binary health metric without any external dataset."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite health metric: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    section = config["health_optimization"]
    seed = int(section["seed"]) + bearings.index(held)
    torch.manual_seed(seed)
    model = build_model({"meta_metric": config["health_metric"]}, device)
    cells = _cells(rows, sources, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    updates = int(section["updates"])
    objective = str(section.get("objective", "maximum_identity_regret"))
    if objective not in {"mean_nll", "maximum_identity_regret"}:
        raise ValueError("unknown health metric objective")
    trace = []
    model.train()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        learned_risks, identity_risks = [], []
        for pseudoheld in sources:
            support = _support_cells(
                cells, tuple(value for value in sources if value != pseudoheld)
            )
            query, records = _query_batch(cells, pseudoheld)
            output = model(support, query, orders)
            learned = _health_prefix_risks(binary_health_logits(output.logits), records)
            identity_weights = torch.ones_like(output.order_weights)
            identity_centroids = model.induce_with_weights(support, identity_weights)
            identity_scores = model.score_queries(
                identity_centroids, identity_weights, query
            ).logits
            identity = _health_prefix_risks(
                binary_health_logits(identity_scores), records
            ).detach()
            learned_risks.append(learned)
            identity_risks.append(identity)
        learned_matrix = torch.stack(learned_risks)
        identity_matrix = torch.stack(identity_risks)
        regret = learned_matrix - identity_matrix
        loss = learned_matrix.mean() if objective == "mean_nll" else regret.max()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(section["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % int(section["log_every"]) == 0 or update == updates:
            item = {
                "stage": "source_only_health_field_metric",
                "update": update,
                "objective": objective,
                "objective_value": float(loss.detach()),
                "meta_query_binary_nll": float(learned_matrix.mean().detach()),
                "maximum_identity_regret": float(regret.max().detach()),
                "per_pseudoheld_nll": {
                    bearing: float(risk.mean().detach())
                    for bearing, risk in zip(sources, learned_risks)
                },
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"HEALTH held={held} update={update}/{updates} "
                f"nll={item['meta_query_binary_nll']:.6f} "
                f"max_regret={item['maximum_identity_regret']:.6f}",
                flush=True,
            )
    # A dedicated health dynamics checkpoint may already occupy the same fold
    # directory.  The checkpoint itself is still protected by the fail-closed
    # existence check at the start of this function.
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    support = _support_cells(cells, sources)
    with torch.no_grad():
        _, weights = model.induce(support, orders)
    torch.save(
        {
            "schema": config["schema"],
            "component": "source_only_health_field_metric",
            "held_bearing": held,
            "source_bearings": sources,
            "external_health_sources": [],
            "seed": seed,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "trace": trace,
            "all_source_order_weights": weights.detach().cpu(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return model.eval(), trace, weights.detach()


def health_streams(model, rows, held, sources, orders, device):
    """Return cumulative fault-vs-normal log odds for every held record."""

    cells = _cells(rows, sources, device)
    support = _support_cells(cells, sources)
    centroids, weights = model.induce(support, orders)
    streams = {}
    with torch.inference_mode():
        for row in rows:
            if str(row["bearing"]) != held:
                continue
            values = torch.from_numpy(
                np.asarray(row["frozen_field"], dtype=np.float32)
            ).to(device)
            scores = model.score_queries(centroids, weights, values).logits
            binary = binary_health_logits(scores)
            prefix = deterministic_prefix_sum(binary)
            streams[str(row["record_id"])] = (prefix[:, 1] - prefix[:, 0]).cpu()
    return streams, weights.detach().cpu()


__all__ = ["fit_fold", "health_streams"]
