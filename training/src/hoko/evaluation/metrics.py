"""Record- and class-balanced metrics for conditional I/O/B prediction."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def summarize_multiclass(
    rows: list[dict],
    logits: dict[str, torch.Tensor],
    labels: tuple[int, ...],
) -> dict:
    """Record- and candidate-balanced causal multiclass metrics."""

    labels = tuple(int(value) for value in labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    by_class: dict[int, list[dict]] = defaultdict(list)
    records = []
    for row in rows:
        label = int(row["class_index"])
        if label not in labels:
            continue
        values = logits[str(row["record_id"])].detach().cpu().numpy().astype(np.float64)
        probabilities = np.exp(values - values.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        prediction = np.asarray(labels)[probabilities.argmax(axis=1)]
        target_index = label_to_index[label]
        item = {
            "record_id": str(row["record_id"]),
            "class_index": label,
            "final_prediction": int(prediction[-1]),
            "final_correct": float(prediction[-1] == label),
            "prefix_accuracy": float(np.mean(prediction == label)),
            "prequential_nll": float(
                -np.log(np.maximum(probabilities[:, target_index], 1e-12)).mean()
            ),
        }
        records.append(item)
        by_class[label].append(item)

    if set(by_class) != set(labels):
        raise ValueError("multiclass summary requires every candidate label")

    def balanced(key: str) -> float:
        return float(
            np.mean(
                [np.mean([item[key] for item in by_class[label]]) for label in labels]
            )
        )

    return {
        "final_balanced_accuracy": balanced("final_correct"),
        "balanced_prefix_accuracy": balanced("prefix_accuracy"),
        "hierarchical_prequential_nll": balanced("prequential_nll"),
        "records": records,
    }


def summarize_operation(rows: list[dict], logits: dict[str, torch.Tensor]) -> dict:
    return summarize_multiclass(rows, logits, (1, 2, 3))


def aggregate_folds(folds: dict[str, dict]) -> dict:
    values = [fold["meta_metric"] for fold in folds.values()]
    return {
        "mean_final_balanced_accuracy": float(
            np.mean([item["final_balanced_accuracy"] for item in values])
        ),
        "worst_final_balanced_accuracy": float(
            np.min([item["final_balanced_accuracy"] for item in values])
        ),
        "mean_balanced_prefix_accuracy": float(
            np.mean([item["balanced_prefix_accuracy"] for item in values])
        ),
        "mean_hierarchical_prequential_nll": float(
            np.mean([item["hierarchical_prequential_nll"] for item in values])
        ),
    }


def summarize_health(rows: list[dict], logits: dict[str, torch.Tensor]) -> dict:
    """Record- and health-state-balanced binary causal metrics."""

    by_health: dict[int, list[dict]] = defaultdict(list)
    records = []
    for row in rows:
        label = int(int(row["class_index"]) != 0)
        values = logits[str(row["record_id"])].detach().cpu().to(torch.float64)
        prediction = (values >= 0).to(torch.int64)
        item = {
            "record_id": str(row["record_id"]),
            "health_label": label,
            "final_prediction": int(prediction[-1]),
            "final_correct": float(int(prediction[-1]) == label),
            "prefix_accuracy": float((prediction == label).to(torch.float64).mean()),
            "prequential_nll": float(
                F.binary_cross_entropy_with_logits(
                    values, torch.full_like(values, float(label))
                )
            ),
            "final_fault_probability": float(torch.sigmoid(values[-1])),
        }
        records.append(item)
        by_health[label].append(item)
    if set(by_health) != {0, 1}:
        raise ValueError("health summary requires normal and fault records")

    def balanced(key: str) -> float:
        return float(
            np.mean(
                [np.mean([item[key] for item in by_health[label]]) for label in (0, 1)]
            )
        )

    return {
        "final_balanced_accuracy": balanced("final_correct"),
        "balanced_prefix_accuracy": balanced("prefix_accuracy"),
        "hierarchical_prequential_nll": balanced("prequential_nll"),
        "records": records,
    }


def summarize_joint(
    rows: list[dict],
    health_logits: dict[str, torch.Tensor],
    operation_logits: dict[str, torch.Tensor],
) -> dict:
    """Compose P(N) and P(F)P(I/O/B|F) and report four-class metrics."""

    by_class: dict[int, list[dict]] = defaultdict(list)
    records = []
    for row in rows:
        record_id = str(row["record_id"])
        label = int(row["class_index"])
        health = health_logits[record_id].detach().cpu().to(torch.float64)
        operation = operation_logits[record_id].detach().cpu().to(torch.float64)
        if health.ndim != 1 or operation.ndim != 2 or operation.shape[1] != 3:
            raise ValueError("invalid hierarchical evidence shape")
        if len(health) != len(operation):
            raise ValueError(f"health/operation time mismatch for {record_id}")
        log_normal = F.logsigmoid(-health)[:, None]
        log_fault = F.logsigmoid(health)[:, None]
        joint = torch.cat((log_normal, log_fault + F.log_softmax(operation, dim=-1)), dim=-1)
        prediction = joint.argmax(dim=-1)
        item = {
            "record_id": record_id,
            "class_index": label,
            "final_prediction": int(prediction[-1]),
            "final_correct": float(int(prediction[-1]) == label),
            "prefix_accuracy": float((prediction == label).to(torch.float64).mean()),
            "prequential_nll": float((-joint[:, label]).mean()),
            "final_true_probability": float(joint[-1, label].exp()),
            "final_fault_probability": float(torch.sigmoid(health[-1])),
        }
        records.append(item)
        by_class[label].append(item)
    if set(by_class) != {0, 1, 2, 3}:
        raise ValueError("joint summary requires N/I/O/B records")

    def balanced(key: str) -> float:
        return float(
            np.mean(
                [
                    np.mean([item[key] for item in by_class[label]])
                    for label in (0, 1, 2, 3)
                ]
            )
        )

    return {
        "final_balanced_accuracy": balanced("final_correct"),
        "balanced_prefix_accuracy": balanced("prefix_accuracy"),
        "hierarchical_prequential_nll": balanced("prequential_nll"),
        "records": records,
    }
