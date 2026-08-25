#!/usr/bin/env python3
"""Package the frozen attentive-health HUST folds for the standalone release."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from hoko.common import sha256
from hoko.data.hust import discover_hust_records, load_subband_cache
from hoko.dynamics.refine import load_refined_fold
from hoko.dynamics.train import record_field
from hoko.health.attention_state import AttentiveStateViewDecoder, record_state_modes
from hoko.memory.train import load_fold as load_metric


ROOT = Path(__file__).resolve().parents[1]


def health_centroids(decoder, rows, sources, device):
    grouped = defaultdict(list)
    with torch.inference_mode():
        for row in rows:
            if str(row["bearing"]) not in sources:
                continue
            values = torch.from_numpy(
                np.asarray(row["frozen_state_modes"], dtype=np.float32)
            ).to(device)
            grouped[int(int(row["class_index"]) != 0)].append(decoder(values))
        return F.normalize(
            torch.stack([torch.cat(grouped[index]).mean(0) for index in (0, 1)]),
            dim=-1,
        ).cpu()


def support_fields(rows, sources):
    output = []
    for bearing in sources:
        classes = []
        for label in (1, 2, 3):
            values = [
                torch.from_numpy(np.asarray(row["frozen_field"], dtype=np.float32))
                for row in rows
                if str(row["bearing"]) == bearing and int(row["class_index"]) == label
            ]
            classes.append(torch.cat(values))
        output.append(tuple(classes))
    return tuple(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument(
        "--health-config",
        type=Path,
        default=ROOT / "configs/hust/model_full_nyquist_attentive_health.json",
    )
    parser.add_argument(
        "--health-artifact-root",
        type=Path,
        default=ROOT / "artifacts/hust_attentive_health_v1",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    release = args.release_root.resolve()
    health_config_path = args.health_config.resolve()
    health_config = json.loads(health_config_path.read_text())
    dynamics = json.loads((ROOT / health_config["dynamics_config"]).read_text())
    dynamics = copy.deepcopy(dynamics)
    dynamics["observation"]["fourier_backend"] = "explicit_selected_dft"
    refinement = json.loads((ROOT / health_config["refinement_config"]).read_text())
    metric_config = json.loads((ROOT / health_config["metric_config"]).read_text())
    bearings = tuple(health_config["bearings"])
    rows = discover_hust_records(args.data_root, bearings)
    load_subband_cache(rows, ROOT / health_config["subband_cache_root"])

    output_root = release / "weights/hust"
    output_root.mkdir(parents=True, exist_ok=True)
    for held in bearings:
        sources = tuple(value for value in bearings if value != held)
        dynamics_path = (
            ROOT / health_config["refined_dynamics_root"] / "dynamics"
            / f"heldout_{held}" / "dynamics_terminal.pt"
        )
        model, mixer, dynamics_payload = load_refined_fold(
            held, bearings, dynamics, refinement, device, dynamics_path
        )
        for row in rows:
            row["frozen_field"] = record_field(model, mixer, row, dynamics, device)
            row["frozen_state_modes"] = record_state_modes(
                model, mixer, row, dynamics, device
            )
        health_path = (
            args.health_artifact_root.resolve() / "state"
            / f"heldout_{held}" / "state_terminal.pt"
        )
        health_payload = torch.load(health_path, map_location=device, weights_only=False)
        input_width = int(rows[0]["frozen_state_modes"].shape[-1])
        section = health_config["state_training"]
        decoder = AttentiveStateViewDecoder(
            input_width, int(section["hidden_width"]), int(section["output_width"]),
            int(section["attention_heads"]),
        ).to(device)
        decoder.load_state_dict(health_payload["state_dict"], strict=True)
        metric_path = (
            ROOT / health_config["operation_metric_root"]
            / f"heldout_{held}" / "metric_terminal.pt"
        )
        metric, metric_payload = load_metric(metric_config, device, metric_path)
        payload = {
            "schema": "dv-hoko-hust-full-nyquist-attentive-v2",
            "method": "DV-HOKO",
            "held_bearing": held,
            "source_bearings": sources,
            "model_state_dict": dynamics_payload["model_state_dict"],
            "mixer_state_dict": dynamics_payload["mixer_state_dict"],
            "health_state_dict": health_payload["state_dict"],
            "health_centroids": health_centroids(decoder.eval(), rows, sources, device),
            "operation_state_dict": metric_payload["state_dict"],
            "operation_support_fields": support_fields(rows, sources),
            "source_resources": {
                row["record_id"]: row["resource_sha256"]
                for row in rows if str(row["bearing"]) in sources
            },
            "dynamics_checkpoint_sha256": sha256(dynamics_path),
            "health_checkpoint_sha256": sha256(health_path),
            "operation_checkpoint_sha256": sha256(metric_path),
            "health_config_sha256": sha256(health_config_path),
            "target_resources_used_for_fit": 0,
            "shaft_clock_contract": "record-observed fs; no target-fitted statistic",
        }
        target = output_root / f"heldout_{held}.pt"
        torch.save(payload, target)
        print(f"exported {target} ({target.stat().st_size / 1024**2:.2f} MiB)")


if __name__ == "__main__":
    main()
