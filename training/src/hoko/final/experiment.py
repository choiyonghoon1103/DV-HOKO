"""Staged source-only DV-HOKO training and pragmatic Final evaluation."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from hoko.final.adapter import build_mixer, build_model, spectrum_paths_and_state
from hoko.final.data import CLASS_NAMES, ExternalStream, SpectrumEpisode


def source_amplitude_scale(
    episodes: list[SpectrumEpisode], *, samples_per_episode: int
) -> float:
    values = []
    for episode in episodes:
        count = min(samples_per_episode, episode.event_count)
        indices = np.linspace(0, episode.event_count - 1, count, dtype=np.int64)
        values.append(np.stack([episode.observation(int(index)) for index in indices]).reshape(-1))
    pooled = np.concatenate(values)
    positive = pooled[pooled > 0]
    if not len(positive):
        raise ValueError("source spectra contain no positive magnitude")
    return float(np.median(positive))


def _cell_lookup(episodes: list[SpectrumEpisode]):
    lookup: dict[tuple[str, int], list[SpectrumEpisode]] = defaultdict(list)
    for episode in episodes:
        lookup[(episode.session, episode.label)].append(episode)
    sessions = sorted({episode.session for episode in episodes})
    if len(sessions) != 24 or set(lookup) != {(session, label) for session in sessions for label in (0, 1)}:
        raise ValueError("Final source sessions do not form complete binary cells")
    return lookup, sessions


def _sample_cells(lookup, sessions, rng: np.random.Generator) -> Tensor:
    rows = []
    for session in sessions:
        for label in (0, 1):
            choices = lookup[(session, label)]
            episode = choices[int(rng.integers(len(choices)))]
            rows.append(episode.observation(int(rng.integers(episode.event_count))))
    return torch.from_numpy(np.stack(rows))


def _views(model, mixer, spectra: Tensor, scale: float, config: dict):
    paths, state_observation, orders = spectrum_paths_and_state(
        mixer, spectra, scale, config
    )
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=paths.device)
    dynamics = model._field(paths, valid, orders)
    base = model.state_base(paths, valid, orders, state_observation)
    return paths, valid, orders, dynamics, base


def _pseudo_loso_state_risk(model, base: Tensor, session_count: int) -> Tensor:
    values = F.normalize(model.state_view(base), dim=-1).reshape(session_count, 2, -1)
    risks = []
    for query in range(session_count):
        support = torch.cat((values[:query], values[query + 1 :]), dim=0)
        centroids = F.normalize(support.mean(dim=0), dim=-1)
        logits = values[query] @ centroids.T
        risks.append(F.cross_entropy(logits, torch.tensor((0, 1), device=base.device)))
    return torch.stack(risks).mean()


def fit_source(
    episodes: list[SpectrumEpisode], config: dict[str, Any], *, device: torch.device,
):
    lookup, sessions = _cell_lookup(episodes)
    scale = source_amplitude_scale(
        episodes, samples_per_episode=int(config["observation"]["scale_samples_per_episode"])
    )
    optimization = config["optimization"]
    torch.manual_seed(int(optimization["field_seed"]))
    rng = np.random.default_rng(int(optimization["field_seed"]))
    model, mixer = build_model(config, device), build_mixer(config, device)
    field_parameters = list(model.field_parameters()) + list(mixer.parameters())
    optimizer = torch.optim.AdamW(
        field_parameters, lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    field_trace = []
    started = time.perf_counter()
    model.train(); mixer.train()
    for update in range(1, int(optimization["field_updates"]) + 1):
        spectra = _sample_cells(lookup, sessions, rng).to(device)
        paths, _, orders = spectrum_paths_and_state(mixer, spectra, scale, config)
        valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.dynamics_loss(paths, valid, orders)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            field_parameters, float(optimization["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % 50 == 0 or update == int(optimization["field_updates"]):
            field_trace.append({
                "update": update, "forecast_loss": float(loss.detach()),
                "gradient_norm": float(gradient),
            })
            print(
                f"FIELD {update:04d}/{optimization['field_updates']} "
                f"forecast={field_trace[-1]['forecast_loss']:.6f}", flush=True,
            )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in mixer.parameters():
        parameter.requires_grad_(False)
    state = config["state_training"]
    torch.manual_seed(int(state["seed"]))
    fresh = build_model(config, device)
    model.state_decoder.load_state_dict(fresh.state_decoder.state_dict(), strict=True)
    for parameter in model.state_parameters():
        parameter.requires_grad_(True)
    state_parameters = list(model.state_parameters())
    state_optimizer = torch.optim.AdamW(
        state_parameters, lr=float(state["learning_rate"]),
        weight_decay=float(state["weight_decay"]),
    )
    state_rng = np.random.default_rng(int(state["seed"]))
    state_trace = []
    model.train(); mixer.eval()
    for update in range(1, int(state["updates"]) + 1):
        spectra = _sample_cells(lookup, sessions, state_rng).to(device)
        with torch.no_grad():
            _, _, _, _, base = _views(model, mixer, spectra, scale, config)
        state_optimizer.zero_grad(set_to_none=True)
        loss = _pseudo_loso_state_risk(model, base, len(sessions))
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            state_parameters, float(state["gradient_clip_norm"])
        )
        state_optimizer.step()
        if update == 1 or update % 50 == 0 or update == int(state["updates"]):
            state_trace.append({
                "update": update, "pseudo_loso_health_nll": float(loss.detach()),
                "gradient_norm": float(gradient),
            })
            print(
                f"STATE {update:04d}/{state['updates']} "
                f"nll={state_trace[-1]['pseudo_loso_health_nll']:.6f}", flush=True,
            )
    model.eval()

    centroid_rng = np.random.default_rng(int(config["centroids"]["seed"]))
    per_cell = int(config["centroids"]["samples_per_session_class"])
    cell_means = torch.empty((len(sessions), 2, model.state_width + 1), device=device)
    with torch.inference_mode():
        for session_index, session in enumerate(sessions):
            for label in (0, 1):
                rows = []
                choices = lookup[(session, label)]
                for _ in range(per_cell):
                    episode = choices[int(centroid_rng.integers(len(choices)))]
                    rows.append(episode.observation(int(centroid_rng.integers(episode.event_count))))
                spectra = torch.from_numpy(np.stack(rows)).to(device)
                _, _, _, _, base = _views(model, mixer, spectra, scale, config)
                cell_means[session_index, label] = F.normalize(
                    model.state_view(base), dim=-1
                ).mean(dim=0)
        centroids = F.normalize(cell_means.mean(dim=0), dim=-1)
    history = {
        "source_amplitude_scale": scale,
        "source_episode_count": len(episodes), "source_session_count": len(sessions),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in mixer.parameters()),
        "field_trace": field_trace, "state_trace": state_trace,
        "wall_time_seconds": time.perf_counter() - started,
    }
    return model, mixer, centroids, history


def _stream_logits(model, mixer, centroids, stream, scale, config, device):
    local = []
    with torch.inference_mode():
        for start in range(0, len(stream.observations), 2048):
            spectra = torch.from_numpy(stream.observations[start : start + 2048]).to(device)
            _, _, _, _, base = _views(model, mixer, spectra, scale, config)
            view = F.normalize(model.state_view(base), dim=-1)
            local.append((view @ centroids.T).cpu().numpy())
    local = np.concatenate(local)
    horizon = int(config["inference"]["evidence_window_seconds"])
    result = np.empty_like(local)
    running = np.zeros(2, dtype=np.float64)
    left = 0
    for index, timestamp in enumerate(stream.timestamps):
        running += local[index]
        while int(timestamp) - int(stream.timestamps[left]) >= horizon:
            running -= local[left]
            left += 1
        result[index] = running
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _metrics(probability, labels, weights):
    prediction = probability.argmax(axis=1)
    confusion = np.zeros((2, 2), dtype=np.int64)
    recall, nll = {}, {}
    for label, name in enumerate(CLASS_NAMES):
        selected = labels == label
        if not np.any(selected):
            recall[name] = None
            nll[name] = None
            continue
        confusion[label] = np.bincount(
            prediction[selected], weights=weights[selected], minlength=2
        ).astype(np.int64)
        recall[name] = float(np.average(prediction[selected] == label, weights=weights[selected]))
        nll[name] = float(np.average(
            -np.log(probability[selected, label].clip(1e-12)), weights=weights[selected]
        ))
    available_recall = [value for value in recall.values() if value is not None]
    available_nll = [value for value in nll.values() if value is not None]
    return {
        "balanced_accuracy": (
            float(np.mean(available_recall)) if len(available_recall) == 2 else None
        ),
        "class_balanced_nll": (
            float(np.mean(available_nll)) if len(available_nll) == 2 else None
        ),
        "available_class_mean_accuracy": float(np.mean(available_recall)),
        "available_class_mean_nll": float(np.mean(available_nll)),
        "per_class_recall": recall, "per_class_nll": nll,
        "confusion_true_by_predicted": confusion.tolist(),
    }


def evaluate(model, mixer, centroids, streams, scale, config, device):
    rows = []
    for stream in streams:
        logits = _stream_logits(model, mixer, centroids, stream, scale, config, device)
        probability = _softmax(logits)
        row = {
            "record_id": stream.record_id, "role": stream.role,
            "second_count": len(stream.labels), "packet_count": int(stream.packet_counts.sum()),
            "second_metrics": _metrics(probability, stream.labels, np.ones(len(stream.labels))),
            "packet_weighted_metrics": _metrics(probability, stream.labels, stream.packet_counts),
        }
        changes = np.flatnonzero(np.diff(stream.labels) != 0) + 1
        row["transition_indices"] = changes.tolist()
        if len(changes):
            transition = int(changes[0]); label = int(stream.labels[transition])
            hits = np.flatnonzero(probability[transition:].argmax(axis=1) == label)
            row["first_correct_after_transition_seconds"] = int(hits[0]) if len(hits) else None
            row["transition_trace"] = [
                {
                    "age": age, "true_class_probability": float(probability[transition + age, label]),
                    "prediction": int(probability[transition + age].argmax()),
                }
                for age in range(min(30, len(probability) - transition))
            ]
        rows.append(row)
        shown_accuracy = row["second_metrics"]["balanced_accuracy"]
        shown_nll = row["second_metrics"]["class_balanced_nll"]
        if shown_accuracy is None:
            shown_accuracy = row["second_metrics"]["available_class_mean_accuracy"]
            shown_nll = row["second_metrics"]["available_class_mean_nll"]
        print(
            f"EVAL {stream.record_id} {stream.role} "
            f"accuracy={shown_accuracy:.6f} nll={shown_nll:.6f}", flush=True,
        )
    mixed = next(row for row in rows if row["role"] == "mixed_evaluation")
    references = [row for row in rows if row["role"] == "background_reference"]
    return {
        "mixed_evaluation": mixed,
        "background_reference_mean_false_positive_rate": float(np.mean([
            1.0 - row["second_metrics"]["per_class_recall"]["BACKGROUND"]
            for row in references
        ])),
        "records": rows,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
