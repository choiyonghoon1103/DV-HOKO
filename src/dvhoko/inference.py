"""Pretrained HUST and Data_final inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dvhoko.adapters import (
    coordinate_paths_and_state, final_coordinates, hust_uniform_envelopes,
)
from dvhoko.model import build_mixer, build_model
from dvhoko.readouts import (
    AttentiveStateViewDecoder, QueryAdaptiveOperationMetric, prefix_sum,
)


def _load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_hust(weight, config, device):
    payload = torch.load(weight, map_location="cpu", weights_only=False)
    model, mixer = build_model(config, device), build_mixer(config, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    health_config = config["readouts"]["health"]
    health = AttentiveStateViewDecoder(**health_config).to(device)
    health.load_state_dict(payload["health_state_dict"], strict=True)
    operation_config = config["readouts"]["operation"]
    operation = QueryAdaptiveOperationMetric(**operation_config).to(device)
    operation.load_state_dict(payload["operation_state_dict"], strict=True)
    return model.eval(), mixer.eval(), health.eval(), operation.eval(), payload


def _load_final(weight, config, device):
    payload = torch.load(weight, map_location="cpu", weights_only=False)
    if payload.get("config", {}).get("method") != "DV-HOKO":
        raise ValueError("not a released Final DV-HOKO checkpoint")
    model, mixer = build_model(config, device), build_mixer(config, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    return model.eval(), mixer.eval(), payload


def _views(model, mixer, coordinates, config):
    paths, state, modes = coordinate_paths_and_state(mixer, coordinates, config)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=paths.device)
    field = model._field(paths, valid, modes)
    base = model.state_base(paths, valid, modes, state)
    return field, F.normalize(model.state_view(base), dim=-1)


def predict_hust_record(
    waveform: np.ndarray, shaft_frequency_hz: float,
    weight: str | Path, config_path: str | Path,
    *, device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    device = torch.device(device); config = _load_config(config_path)
    model, mixer, health_decoder, operation_metric, payload = _load_hust(
        weight, config, device
    )
    envelopes = hust_uniform_envelopes(
        waveform, shaft_frequency_hz=float(shaft_frequency_hz), config=config,
    )
    with torch.inference_mode():
        coordinates = torch.from_numpy(envelopes).to(device)
        paths, state, modes = coordinate_paths_and_state(mixer, coordinates, config)
        valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
        field = model._field(paths, valid, modes)
        state_modes = model.state_mode_base(paths, valid, modes, state)
        health_local = health_decoder(state_modes) @ payload["health_centroids"].to(device).T
        health = prefix_sum(health_local)
        support = tuple(
            tuple(value.to(device) for value in environment)
            for environment in payload["operation_support_fields"]
        )
        operation = prefix_sum(operation_metric(support, field, modes))
        health_probability = torch.softmax(health, dim=-1)
        operation_probability = torch.softmax(operation, dim=-1)
        probability = torch.cat((
            health_probability[:, :1], health_probability[:, 1:] * operation_probability,
        ), dim=-1)
    return {
        "probability": probability.cpu().numpy(),
        "prediction": probability.argmax(-1).cpu().numpy(),
    }


def predict_final_stream(
    observations: np.ndarray, timestamps: np.ndarray,
    weight: str | Path, config_path: str | Path,
    *, device: str | torch.device = "cpu",
) -> dict[str, np.ndarray]:
    device = torch.device(device); config = _load_config(config_path)
    model, mixer, payload = _load_final(weight, config, device)
    centroids = payload["health_centroids"].to(device)
    scale = float(payload["source_history"]["source_amplitude_scale"])
    local = []
    with torch.inference_mode():
        for start in range(0, len(observations), 2048):
            spectra = torch.from_numpy(np.asarray(observations[start : start + 2048])).to(device)
            coordinates = final_coordinates(spectra, scale, config)
            _, state = _views(model, mixer, coordinates, config)
            local.append((state @ centroids.T).cpu().numpy())
    local = np.concatenate(local)
    horizon = int(config["inference"]["evidence_window_seconds"])
    logits = np.empty_like(local); running = np.zeros(2, dtype=np.float64); left = 0
    for index, timestamp in enumerate(timestamps):
        running += local[index]
        while int(timestamp) - int(timestamps[left]) >= horizon:
            running -= local[left]; left += 1
        logits[index] = running
    shifted = logits - logits.max(1, keepdims=True)
    probability = np.exp(shifted); probability /= probability.sum(1, keepdims=True)
    return {"probability": probability, "prediction": probability.argmax(1)}
