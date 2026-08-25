"""Gauge-safe diagnostics for Koopman-order latent feature fields.

The tensors called ``field`` in HOKO are vector-valued descriptors over an
order coordinate.  They are not flow-matching vector fields.  This module
compares their record geometry without fitting on held-bearing observations.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def record_embedding(field: np.ndarray) -> np.ndarray:
    """Average a seconds x orders x features field and flatten its order map."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all() or len(values) < 1:
        raise ValueError("field must be a finite seconds x orders x features tensor")
    return values.mean(axis=0).reshape(-1)


def unit_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("expected a matrix")
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("zero-norm representation is not identifiable")
    return matrix / norm


def centroid_transfer(
    source: np.ndarray,
    source_labels: np.ndarray,
    query: np.ndarray,
    query_labels: np.ndarray,
) -> dict:
    """Cosine nearest-source-centroid transfer with no query fitting."""

    source_u, query_u = unit_rows(source), unit_rows(query)
    source_labels = np.asarray(source_labels, dtype=np.int64)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    classes = np.unique(source_labels)
    if set(classes.tolist()) != set(np.unique(query_labels).tolist()):
        raise ValueError("source and query class universes differ")
    centroids = unit_rows(
        np.stack([source_u[source_labels == label].mean(axis=0) for label in classes])
    )
    similarities = query_u @ centroids.T
    predictions = classes[similarities.argmax(axis=1)]
    true_index = np.asarray([np.flatnonzero(classes == label)[0] for label in query_labels])
    true_similarity = similarities[np.arange(len(query_labels)), true_index]
    wrong = similarities.copy()
    wrong[np.arange(len(query_labels)), true_index] = -np.inf
    margins = true_similarity - wrong.max(axis=1)
    by_class = {
        str(int(label)): float(np.mean(predictions[query_labels == label] == label))
        for label in classes
    }
    return {
        "accuracy": float(np.mean(predictions == query_labels)),
        "balanced_accuracy": float(np.mean(list(by_class.values()))),
        "mean_true_class_cosine_margin": float(margins.mean()),
        "minimum_true_class_cosine_margin": float(margins.min()),
        "per_class_recall": by_class,
        "predictions": predictions.tolist(),
        "margins": margins.tolist(),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def representational_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation between cosine-distance geometries.

    This remains meaningful when two latent fields use different coordinate
    gauges, unlike coordinate-wise MSE.
    """

    left_u, right_u = unit_rows(left), unit_rows(right)
    if len(left_u) != len(right_u) or len(left_u) < 3:
        raise ValueError("paired representations require at least three records")
    triangle = np.triu_indices(len(left_u), k=1)
    left_distance = (1.0 - left_u @ left_u.T)[triangle]
    right_distance = (1.0 - right_u @ right_u.T)[triangle]
    left_rank, right_rank = _average_ranks(left_distance), _average_ranks(right_distance)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.linalg.norm(left_rank) * np.linalg.norm(right_rank)
    if denominator <= 1e-12:
        raise ValueError("constant representational geometry is not comparable")
    return float(left_rank @ right_rank / denominator)


def coordinate_alignment(predicted: list[np.ndarray], teacher: list[np.ndarray]) -> dict:
    """Coordinate alignment for a student explicitly distilled into teacher gauge."""

    if len(predicted) != len(teacher) or not predicted:
        raise ValueError("predicted and teacher field lists must be paired")
    cosine, nrmse = [], []
    for student, target in zip(predicted, teacher):
        student = np.asarray(student, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if student.shape != target.shape:
            raise ValueError("distilled and teacher fields must share coordinates")
        student_flat, target_flat = student.reshape(len(student), -1), target.reshape(len(target), -1)
        numerator = (student_flat * target_flat).sum(axis=1)
        denominator = np.linalg.norm(student_flat, axis=1) * np.linalg.norm(target_flat, axis=1)
        cosine.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
        reference = np.sqrt(np.mean(np.square(target_flat), axis=1))
        error = np.sqrt(np.mean(np.square(student_flat - target_flat), axis=1))
        nrmse.extend((error / np.maximum(reference, 1e-12)).tolist())
    return {
        "mean_per_second_cosine": float(np.mean(cosine)),
        "minimum_per_second_cosine": float(np.min(cosine)),
        "mean_normalized_rmse": float(np.mean(nrmse)),
    }


def causal_centroid_transfer(
    source_fields: list[np.ndarray],
    source_labels: np.ndarray,
    query_fields: list[np.ndarray],
    query_labels: np.ndarray,
) -> dict:
    """Diagnose per-second source-centroid evidence with causal prefix accumulation.

    This is deliberately a diagnostic rather than a learned replacement head.
    It tests whether a latent field already contains transferable class evidence.
    """

    source_labels = np.asarray(source_labels, dtype=np.int64)
    query_labels = np.asarray(query_labels, dtype=np.int64)
    if len(source_fields) != len(source_labels) or len(query_fields) != len(query_labels):
        raise ValueError("field lists and label vectors must align")
    classes = np.unique(source_labels)
    if set(classes.tolist()) != set(np.unique(query_labels).tolist()):
        raise ValueError("source and query class universes differ")
    second_values, second_labels = [], []
    for field, label in zip(source_fields, source_labels):
        values = np.asarray(field, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("every field must be seconds x orders x features")
        second_values.extend(values.reshape(len(values), -1))
        second_labels.extend([int(label)] * len(values))
    source_u = unit_rows(np.stack(second_values))
    second_labels = np.asarray(second_labels, dtype=np.int64)
    centroids = unit_rows(
        np.stack([source_u[second_labels == label].mean(axis=0) for label in classes])
    )
    by_class: dict[int, list[dict]] = defaultdict(list)
    records = []
    for field, label in zip(query_fields, query_labels):
        values = np.asarray(field, dtype=np.float64).reshape(len(field), -1)
        logits = unit_rows(values) @ centroids.T
        prefix = np.cumsum(logits, axis=0)
        prefix -= prefix.max(axis=1, keepdims=True)
        probabilities = np.exp(prefix)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        true_index = int(np.flatnonzero(classes == label)[0])
        predictions = classes[probabilities.argmax(axis=1)]
        item = {
            "final_correct": float(predictions[-1] == label),
            "prefix_accuracy": float(np.mean(predictions == label)),
            "prequential_nll": float(-np.log(np.maximum(probabilities[:, true_index], 1e-12)).mean()),
            "final_prediction": int(predictions[-1]),
        }
        by_class[int(label)].append(item)
        records.append(item)

    def balanced(name: str) -> float:
        return float(
            np.mean(
                [np.mean([record[name] for record in by_class[int(label)]]) for label in classes]
            )
        )

    return {
        "final_balanced_accuracy": balanced("final_correct"),
        "balanced_prefix_accuracy": balanced("prefix_accuracy"),
        "hierarchical_prequential_nll": balanced("prequential_nll"),
        "final_predictions": [record["final_prediction"] for record in records],
    }


def causal_centroid_logits(
    source_fields: list[np.ndarray],
    source_labels: np.ndarray,
    query_fields: list[np.ndarray],
) -> list[np.ndarray]:
    """Return cumulative cosine evidence for a fixed source-induced class memory."""

    source_labels = np.asarray(source_labels, dtype=np.int64)
    if len(source_fields) != len(source_labels):
        raise ValueError("source fields and labels must align")
    classes = np.unique(source_labels)
    seconds, labels = [], []
    for field, label in zip(source_fields, source_labels):
        values = np.asarray(field, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError("every field must be seconds x orders x features")
        seconds.extend(values.reshape(len(values), -1))
        labels.extend([int(label)] * len(values))
    seconds_u = unit_rows(np.stack(seconds))
    labels = np.asarray(labels, dtype=np.int64)
    centroids = unit_rows(
        np.stack([seconds_u[labels == label].mean(axis=0) for label in classes])
    )
    output = []
    for field in query_fields:
        values = np.asarray(field, dtype=np.float64).reshape(len(field), -1)
        output.append(np.cumsum(unit_rows(values) @ centroids.T, axis=0))
    return output


def group_record_embeddings(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for record in records:
        grouped[int(record["class_index"])].append(record_embedding(record[key]))
    labels, values = [], []
    for label in sorted(grouped):
        for value in grouped[label]:
            labels.append(label)
            values.append(value)
    return np.stack(values), np.asarray(labels, dtype=np.int64)
