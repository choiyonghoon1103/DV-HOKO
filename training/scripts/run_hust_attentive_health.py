#!/usr/bin/env python3
"""Evaluate learned full-Nyquist mode attention for health state."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from hoko.common import order_grid, sha256
from hoko.data.hust import discover_hust_records, load_subband_cache
from hoko.dynamics.refine import load_refined_fold
from hoko.dynamics.train import record_field
from hoko.evaluation.metrics import summarize_health, summarize_joint, summarize_operation
from hoko.health.attention_state import (
    attentive_health_streams,
    fit_attentive_state_view,
    record_state_modes,
)
from hoko.memory.train import load_fold as load_metric, operation_streams


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--held", choices=("6205", "6206", "6207", "6208"), required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    dynamics_config = json.loads((ROOT / config["dynamics_config"]).read_text())
    dynamics_config = copy.deepcopy(dynamics_config)
    dynamics_config["observation"]["fourier_backend"] = config["runtime"]["fourier_backend"]
    refinement_config = json.loads((ROOT / config["refinement_config"]).read_text())
    metric_config = json.loads((ROOT / config["metric_config"]).read_text())
    bearings = tuple(config["bearings"])
    held = str(args.held)
    sources = tuple(value for value in bearings if value != held)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)
    rows = discover_hust_records(args.data_root, bearings)
    load_subband_cache(rows, ROOT / config["subband_cache_root"])
    dynamics_checkpoint = (
        ROOT / config["refined_dynamics_root"] / "dynamics"
        / f"heldout_{held}" / "dynamics_terminal.pt"
    )
    model, mixer, _ = load_refined_fold(
        held, bearings, dynamics_config, refinement_config, device, dynamics_checkpoint
    )
    for row in rows:
        row["frozen_field"] = record_field(model, mixer, row, dynamics_config, device)
        row["frozen_state_modes"] = record_state_modes(
            model, mixer, row, dynamics_config, device
        )
    state_checkpoint = args.artifact_root / "state" / f"heldout_{held}" / "state_terminal.pt"
    decoder, trace = fit_attentive_state_view(
        rows, held, bearings, config, device, state_checkpoint
    )
    health = attentive_health_streams(decoder, rows, held, sources, device)
    metric_checkpoint = (
        ROOT / config["operation_metric_root"] / f"heldout_{held}" / "metric_terminal.pt"
    )
    metric, _ = load_metric(metric_config, device, metric_checkpoint)
    orders = torch.from_numpy(order_grid(dynamics_config)).to(device)
    operation, _ = operation_streams(metric, rows, held, sources, orders, device)
    query = [row for row in rows if str(row["bearing"]) == held]
    result = {
        "schema": config["schema"],
        "status": "complete",
        "held_bearing": held,
        "source_bearings": list(sources),
        "health_binary": summarize_health(query, health),
        "operation_conditional": summarize_operation(query, operation),
        "joint_four_class": summarize_joint(query, health, operation),
        "state_training_terminal": trace[-1],
        "dynamics_checkpoint_sha256": sha256(dynamics_checkpoint),
        "operation_metric_checkpoint_sha256": sha256(metric_checkpoint),
        "state_checkpoint_sha256": sha256(state_checkpoint),
        "shared_trunk_and_operation_metric_frozen": True,
        "held_bearing_resources_used_for_fit": 0,
    }
    output = args.artifact_root / f"heldout_{held}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    joint = result["joint_four_class"]
    print(
        f"ATTN-HEALTH held={held} final={joint['final_balanced_accuracy']:.6f} "
        f"prefix={joint['balanced_prefix_accuracy']:.6f} "
        f"nll={joint['hierarchical_prequential_nll']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
