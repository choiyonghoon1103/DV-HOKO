#!/usr/bin/env python3
"""Fit the shared DV-HOKO model on Final 2024 and replay Final 2026."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hoko.final.data import load_external_streams, load_source_episodes
from hoko.final.adapter import build_mixer, build_model
from hoko.final.experiment import atomic_json, evaluate, fit_source, sha256


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final/v1.json")
    parser.add_argument("--source", type=Path, default=ROOT.parent / "NoRDiM-Live/data/Data_final/2024")
    parser.add_argument("--target", type=Path, default=ROOT.parent / "NoRDiM-Live/data/Data_final/2026")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/final_v1")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    checkpoint = output / "model.pt"
    source_path = output / "source.json"
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("method") != "DV-HOKO" or payload.get("config") != config:
            raise ValueError("sealed checkpoint does not match the Final DV-HOKO contract")
        source_result = json.loads(source_path.read_text(encoding="utf-8"))
        if source_result["checkpoint_sha256"] != sha256(checkpoint):
            raise ValueError("sealed checkpoint hash changed")
        model, mixer = build_model(config, device), build_mixer(config, device)
        model.load_state_dict(payload["state_dict"], strict=True)
        mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
        model.eval(); mixer.eval()
        centroids = payload["health_centroids"].to(device)
        history = payload["source_history"]
        print(f"RESUME sealed source model sha256={source_result['checkpoint_sha256']}", flush=True)
    else:
        print("SOURCE phase: opening Data_final 2024 only", flush=True)
        episodes = load_source_episodes(args.source)
        model, mixer, centroids, history = fit_source(episodes, config, device=device)
        torch.save(
            {
                "schema_version": 1, "method": "DV-HOKO", "source_year": 2024,
                "class_names": ["BACKGROUND", "LEAK"], "config": config,
                "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                "mixer_state_dict": {name: value.cpu() for name, value in mixer.state_dict().items()},
                "health_centroids": centroids.cpu(),
                "source_history": history,
            },
            checkpoint,
        )
        source_result = {
            "method": "DV-HOKO", "source_year": 2024,
            "target_opened_during_fit": False, "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint), "fit": history,
        }
        atomic_json(source_result, source_path)
        print(f"SEALED {checkpoint} sha256={source_result['checkpoint_sha256']}", flush=True)

    print("TARGET phase: frozen replay of Data_final 2026", flush=True)
    streams = load_external_streams(args.target)
    target_result = evaluate(
        model, mixer, centroids, streams, history["source_amplitude_scale"], config, device
    )
    result = {
        "method": "DV-HOKO", "evaluation_mode": "retrospective_pragmatic",
        "source_year": 2024, "target_year": 2026,
        "target_adaptation": False, "mixed_stream_reset": "file_start_only",
        "same_second_packets": "mean_into_one_observation",
        "checkpoint_sha256": source_result["checkpoint_sha256"],
        "source": source_result, "target": target_result,
    }
    atomic_json(result, output / "result.json")
    print(f"WROTE {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
