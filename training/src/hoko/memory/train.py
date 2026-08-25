"""Whole-bearing episodic training for the support-conditioned field metric."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from hoko.memory.model import (
    AttentiveSupportConditionedFieldMetric,
    AttentiveSupportConditionedMahalanobisMetric,
    NeuralSupportConditionedFieldMetric,
    QueryAdaptiveSupportConditionedMahalanobisMetric,
    SupportConditionedFieldMetric,
)
from hoko.common import deterministic_prefix_sum


def build_model(config: dict, device: torch.device) -> SupportConditionedFieldMetric:
    architecture = config["meta_metric"]
    metric_type = str(architecture.get("type", "manual_reliability_summary"))
    classes = {
        "manual_reliability_summary": SupportConditionedFieldMetric,
        "nested_deepset_raw_field": NeuralSupportConditionedFieldMetric,
        "hierarchical_attention_raw_field": AttentiveSupportConditionedFieldMetric,
        "hierarchical_attention_raw_field_learned_feature_metric": (
            AttentiveSupportConditionedMahalanobisMetric
        ),
        "query_adaptive_hierarchical_attention_raw_field_learned_feature_metric": (
            QueryAdaptiveSupportConditionedMahalanobisMetric
        ),
    }
    if metric_type not in classes:
        raise ValueError("unknown support-conditioned field metric type")
    kwargs = dict(
        statistic_width=int(architecture["statistic_width"]),
        hidden_width=int(architecture["hidden_width"]),
        order_weight_normalization=str(
            architecture.get("order_weight_normalization", "softmax")
        ),
    )
    if metric_type in {
        "hierarchical_attention_raw_field",
        "hierarchical_attention_raw_field_learned_feature_metric",
        "query_adaptive_hierarchical_attention_raw_field_learned_feature_metric",
    }:
        kwargs["attention_heads"] = int(architecture["attention_heads"])
    return classes[metric_type](**kwargs).to(device)


def _cells(
    rows: list[dict],
    bearings: tuple[str, ...],
    device: torch.device,
    labels: tuple[int, ...] = (1, 2, 3),
) -> dict:
    cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing in bearings and label in labels:
            cells[(bearing, label)].append(
                torch.from_numpy(np.asarray(row["frozen_field"], dtype=np.float32)).to(device)
            )
    expected = {(bearing, label) for bearing in bearings for label in labels}
    if set(cells) != expected:
        raise ValueError("meta field metric requires every bearing/fault cell")
    return dict(cells)


def _support_cells(
    cells: dict,
    bearings: tuple[str, ...],
    labels: tuple[int, ...] = (1, 2, 3),
):
    return tuple(
        tuple(torch.cat(cells[(bearing, label)], dim=0) for label in labels)
        for bearing in bearings
    )


def _query_batch(
    cells: dict,
    bearing: str,
    labels: tuple[int, ...] = (1, 2, 3),
) -> tuple[Tensor, list[dict]]:
    fields, records, cursor = [], [], 0
    for target_index, label in enumerate(labels):
        for values in cells[(bearing, label)]:
            start = cursor
            fields.append(values)
            cursor += len(values)
            records.append(
                {
                    "class_index": label,
                    "target_index": target_index,
                    "slice": slice(start, cursor),
                }
            )
    return torch.cat(fields), records


def _class_prefix_risks(logits: Tensor, records: list[dict]) -> Tensor:
    by_class: dict[int, list[Tensor]] = defaultdict(list)
    for record in records:
        local = deterministic_prefix_sum(logits[record["slice"]])
        label = int(record.get("target_index", int(record["class_index"]) - 1))
        target = torch.full((len(local),), label, dtype=torch.long, device=local.device)
        by_class[label].append(F.cross_entropy(local, target))
    return torch.stack(
        [torch.stack(by_class[label]).mean() for label in range(logits.shape[1])]
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
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite meta metric: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    labels = tuple(int(value) for value in config["meta_metric"].get(
        "candidate_labels", (1, 2, 3)
    ))
    seed = int(config["optimization"]["metric_seed"]) + bearings.index(held)
    torch.manual_seed(seed)
    model = build_model(config, device)
    cells = _cells(rows, sources, device, labels)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optimization"]["metric_learning_rate"]),
        weight_decay=float(config["optimization"]["metric_weight_decay"]),
    )
    updates = int(config["optimization"]["metric_updates"])
    objective = str(config["optimization"].get("metric_objective", "mean_nll"))
    if objective not in {"mean_nll", "maximum_identity_regret"}:
        raise ValueError("unknown meta metric objective")
    trace = []
    model.train()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        learned_risks, identity_risks, regrets = [], [], []
        for pseudoheld in sources:
            support = _support_cells(
                cells,
                tuple(value for value in sources if value != pseudoheld),
                labels,
            )
            query, records = _query_batch(cells, pseudoheld, labels)
            output = model(support, query, orders)
            learned = _class_prefix_risks(output.logits, records)
            # The identity control is defined on the common order axis.  A
            # query-adaptive model returns one diagnostic metric per second,
            # so copying the output shape would incorrectly create a matrix.
            identity_weights = torch.ones_like(orders)
            identity_centroids = model.induce_with_weights(support, identity_weights)
            identity_logits = model.score_queries(
                identity_centroids, identity_weights, query
            ).logits
            identity = _class_prefix_risks(identity_logits, records).detach()
            learned_risks.append(learned)
            identity_risks.append(identity)
            regrets.append(learned - identity)
        learned_matrix = torch.stack(learned_risks)
        identity_matrix = torch.stack(identity_risks)
        regret_matrix = torch.stack(regrets)
        loss = (
            learned_matrix.mean()
            if objective == "mean_nll"
            else regret_matrix.reshape(-1).max()
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["optimization"]["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % int(config["optimization"]["log_every"]) == 0 or update == updates:
            item = {
                "stage": "whole_bearing_support_conditioned_field_metric",
                "update": update,
                "objective": objective,
                "objective_value": float(loss.detach()),
                "meta_query_prefix_nll": float(learned_matrix.mean().detach()),
                "maximum_identity_regret": float(regret_matrix.max().detach()),
                "per_pseudoheld_nll": {
                    bearing: float(risk.mean().detach())
                    for bearing, risk in zip(sources, learned_risks)
                },
                "per_pseudoheld_class_regret": {
                    bearing: {
                        str(label): float(value.detach())
                        for label, value in zip(labels, local)
                    }
                    for bearing, local in zip(sources, regrets)
                },
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"META-METRIC held={held} update={update}/{updates} "
                f"nll={item['meta_query_prefix_nll']:.6f} "
                f"max_regret={item['maximum_identity_regret']:.6f}",
                flush=True,
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    support = _support_cells(cells, sources, labels)
    with torch.no_grad():
        _, weights = model.induce(support, orders)
    torch.save(
        {
            "schema": config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "trace": trace,
            "all_source_order_weights": weights.detach().cpu(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "metric_objective": objective,
            "class_embeddings": False,
            "candidate_labels": labels,
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return model.eval(), trace, weights.detach()


def candidate_streams(
    model,
    rows,
    held,
    sources,
    orders,
    device,
    labels=(1, 2, 3),
):
    labels = tuple(int(value) for value in labels)
    cells = _cells(rows, sources, device, labels)
    support = _support_cells(cells, sources, labels)
    centroids, weights = model.induce(support, orders)
    streams = {}
    query_weights = []
    with torch.inference_mode():
        for row in rows:
            # A conditional fault-location model must also score a normal
            # record: the hierarchical health branch decides whether those
            # conditional logits are used.  Filtering N here makes the joint
            # N/I/O/B probability algebra incomplete.
            if str(row["bearing"]) != held:
                continue
            values = torch.from_numpy(
                np.asarray(row["frozen_field"], dtype=np.float32)
            ).to(device)
            if isinstance(model, QueryAdaptiveSupportConditionedMahalanobisMetric):
                output = model(support, values, orders)
                local = output.logits
                query_weights.append(output.order_weights)
            else:
                local = model.score_queries(centroids, weights, values).logits
            streams[str(row["record_id"])] = deterministic_prefix_sum(local).cpu()
    diagnostic_weights = (
        torch.cat(query_weights, dim=0).mean(dim=0)
        if query_weights
        else weights
    )
    return streams, diagnostic_weights.detach().cpu()


def operation_streams(model, rows, held, sources, orders, device):
    return candidate_streams(
        model, rows, held, sources, orders, device, labels=(1, 2, 3)
    )


def load_fold(config: dict, device: torch.device, checkpoint: Path):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != config["schema"]:
        raise ValueError("meta metric checkpoint schema differs")
    model = build_model(config, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), payload


__all__ = ["candidate_streams", "fit_fold", "load_fold", "operation_streams"]
