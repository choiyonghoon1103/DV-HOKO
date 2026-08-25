import json
from pathlib import Path

import torch

from hoko.final.adapter import build_mixer, build_model


ROOT = Path(__file__).resolve().parents[2]


def test_final_training_builds_exact_released_model_type_and_shape():
    config = json.loads((ROOT / "training/configs/final/v1.json").read_text())
    payload = torch.load(
        ROOT / "weights/final/model.pt", map_location="cpu", weights_only=False
    )
    model = build_model(config, torch.device("cpu"))
    mixer = build_mixer(config, torch.device("cpu"))
    model.load_state_dict(payload["state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_count += sum(parameter.numel() for parameter in mixer.parameters())
    assert parameter_count == payload["source_history"]["parameter_count"] == 42_401


def test_final_training_cli_seals_source_before_target_loader_call():
    text = (ROOT / "training/scripts/train_final.py").read_text()
    fit = text.index("fit_source(episodes")
    seal = text.index("torch.save(")
    target = text.index("load_external_streams(args.target)")
    assert fit < seal < target
