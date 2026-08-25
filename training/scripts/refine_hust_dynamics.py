#!/usr/bin/env python3
"""Refine a HUST dynamical field through a frozen source-trained class memory."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from hoko.common import order_grid, sha256
from hoko.data.hust import discover_hust_records, load_subband_cache
from hoko.dynamics.refine import refine_fold
from hoko.dynamics.train import load_fold as load_dynamics, record_field
from hoko.evaluation.metrics import summarize_operation
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

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dynamics_path = ROOT / config["dynamics_config"]
    metric_path = ROOT / config["metric_config"]
    dynamics_config = json.loads(dynamics_path.read_text(encoding="utf-8"))
    runtime_dynamics_config = copy.deepcopy(dynamics_config)
    runtime_dynamics_config["observation"]["fourier_backend"] = str(
        config.get("runtime", {}).get("fourier_backend", "fft")
    )
    metric_config = json.loads(metric_path.read_text(encoding="utf-8"))
    bearings = tuple(str(value) for value in config["bearings"])
    held = str(args.held)
    sources = tuple(value for value in bearings if value != held)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)

    rows = discover_hust_records(args.data_root, bearings)
    load_subband_cache(rows, ROOT / config["subband_cache_root"])
    held_override = config.get("dynamics_checkpoint_by_held", {}).get(held)
    source_dynamics = (
        ROOT / held_override
        if held_override is not None
        else ROOT
        / config["dynamics_checkpoint_root"]
        / f"heldout_{held}"
        / "dynamics_terminal.pt"
    )
    source_metric = ROOT / config["metric_checkpoint_root"] / f"heldout_{held}" / "metric_terminal.pt"
    model, mixer, _ = load_dynamics(
        held, bearings, dynamics_config, device, source_dynamics
    )
    metric, _ = load_metric(metric_config, device, source_metric)

    checkpoint = args.artifact_root / "dynamics" / f"heldout_{held}" / "dynamics_terminal.pt"
    model, mixer, trace = refine_fold(
        rows,
        held,
        bearings,
        model,
        mixer,
        metric,
        runtime_dynamics_config,
        config,
        device,
        checkpoint,
    )
    mode_bank = getattr(model, "angular_mode_bank", None)
    orders = (
        mode_bank.orders().detach()
        if mode_bank is not None
        else torch.from_numpy(order_grid(runtime_dynamics_config)).to(device)
    )
    for row in rows:
        row["frozen_field"] = record_field(
            model, mixer, row, runtime_dynamics_config, device
        )
    logits, weights = operation_streams(
        metric, rows, held, sources, orders, device
    )
    query = [row for row in rows if str(row["bearing"]) == held]
    result = {
        "schema": config["schema"],
        "held_bearing": held,
        "source_bearings": list(sources),
        "source_dynamics_checkpoint_sha256": sha256(source_dynamics),
        "source_metric_checkpoint_sha256": sha256(source_metric),
        "refined_dynamics_checkpoint_sha256": sha256(checkpoint),
        "training_terminal": trace[-1],
        "meta_metric": summarize_operation(query, logits),
        "diagnostic_order_weights": weights.tolist(),
        "metric_parameters_updated": False,
        "held_bearing_resources_used_for_fit": 0,
    }
    output = args.artifact_root / f"heldout_{held}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary = result["meta_metric"]
    print(
        f"HOKO-META-DYNAMICS held={held} "
        f"final={summary['final_balanced_accuracy']:.6f} "
        f"prefix={summary['balanced_prefix_accuracy']:.6f} "
        f"nll={summary['hierarchical_prequential_nll']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
