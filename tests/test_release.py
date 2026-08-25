from __future__ import annotations

import json
from pathlib import Path

import torch

from dvhoko.adapters import coordinate_paths_and_state, final_coordinates, mode_grid
from dvhoko.model import DualViewKoopmanMoriField, build_mixer, build_model
from dvhoko.readouts import AttentiveStateViewDecoder, QueryAdaptiveOperationMetric


ROOT = Path(__file__).resolve().parents[1]


def _config(name):
    return json.loads((ROOT / "configs" / name).read_text())


def test_final_and_hust_use_the_same_model_class():
    hust = build_model(_config("hust.json"), torch.device("cpu"))
    final = build_model(_config("final.json"), torch.device("cpu"))
    assert type(hust) is type(final) is DualViewKoopmanMoriField


def test_released_weights_load_strictly():
    for bearing in ("6205", "6206", "6207", "6208"):
        config = _config("hust.json")
        model, mixer = build_model(config, torch.device("cpu")), build_mixer(config, torch.device("cpu"))
        payload = torch.load(ROOT / "weights/hust" / f"heldout_{bearing}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
        health = AttentiveStateViewDecoder(**config["readouts"]["health"])
        health.load_state_dict(payload["health_state_dict"], strict=True)
        operation = QueryAdaptiveOperationMetric(**config["readouts"]["operation"])
        operation.load_state_dict(payload["operation_state_dict"], strict=True)
    config = _config("final.json")
    model, mixer = build_model(config, torch.device("cpu")), build_mixer(config, torch.device("cpu"))
    payload = torch.load(ROOT / "weights/final/model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)


def test_final_adapter_shape():
    config = _config("final.json"); device = torch.device("cpu")
    mixer = build_mixer(config, device)
    coordinates = final_coordinates(torch.rand(2, 320), 0.01, config)
    paths, state, modes = coordinate_paths_and_state(mixer, coordinates, config)
    assert paths.shape == (2, 5, 4, 15, 2)
    assert state.shape == (2, 8) and len(modes) == 15


def test_hust_full_nyquist_and_source_only_payloads():
    config = _config("hust.json")
    modes = mode_grid(config, torch.device("cpu"))
    assert len(modes) == 128
    assert float(modes[0]) == 0.25 and float(modes[-1]) == 32.0
    for held in ("6205", "6206", "6207", "6208"):
        payload = torch.load(
            ROOT / "weights/hust" / f"heldout_{held}.pt",
            map_location="cpu", weights_only=False,
        )
        assert payload["schema"] == "dv-hoko-hust-full-nyquist-attentive-v2"
        assert payload["target_resources_used_for_fit"] == 0
        assert len(payload["source_resources"]) == 36
        held_digit = held[-1]
        assert all(record_id[1] != held_digit for record_id in payload["source_resources"])
        support = payload["operation_support_fields"]
        assert len(support) == 3 and all(len(environment) == 3 for environment in support)
        assert all(
            value.ndim == 3 and value.shape[1:] == (128, 12)
            for environment in support for value in environment
        )
