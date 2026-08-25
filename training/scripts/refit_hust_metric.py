#!/usr/bin/env python3
"""Fit a new support metric on an immutable HUST dynamics checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from hoko.common import order_grid, sha256
from hoko.data.hust import discover_hust_records, load_subband_cache
from hoko.dynamics.train import load_fold as load_dynamics, record_field
from hoko.evaluation.metrics import summarize_operation
from hoko.evaluation.transfer import causal_centroid_logits
from hoko.memory.train import fit_fold as fit_metric, operation_streams


ROOT = Path(__file__).resolve().parents[1]


def _identity_logits(rows: list[dict], held: str, sources: tuple[str, ...]) -> dict:
    source_fields, source_labels, query_fields, query_rows = [], [], [], []
    for row in rows:
        label = int(row["class_index"])
        if label not in (1, 2, 3):
            continue
        if str(row["bearing"]) in sources:
            source_fields.append(row["frozen_field"])
            source_labels.append(label)
        elif str(row["bearing"]) == held:
            query_fields.append(row["frozen_field"])
            query_rows.append(row)
    values = causal_centroid_logits(source_fields, source_labels, query_fields)
    return {
        str(row["record_id"]): torch.from_numpy(value).float()
        for row, value in zip(query_rows, values, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dynamics-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--held", choices=("6205", "6206", "6207", "6208"), required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
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
    dynamics_checkpoint = args.dynamics_root / f"heldout_{held}" / "dynamics_terminal.pt"
    model, mixer, dynamics_trace = load_dynamics(
        held, bearings, base, device, dynamics_checkpoint
    )
    mode_bank = getattr(model, "angular_mode_bank", None)
    orders = (
        mode_bank.orders().detach()
        if mode_bank is not None
        else torch.from_numpy(order_grid(base)).to(device)
    )
    for row in rows:
        row["frozen_field"] = record_field(model, mixer, row, base, device)

    metric_checkpoint = args.artifact_root / "metric" / f"heldout_{held}" / "metric_terminal.pt"
    metric, metric_trace, weights = fit_metric(
        rows, held, bearings, orders, config, device, metric_checkpoint
    )
    learned, _ = operation_streams(metric, rows, held, sources, orders, device)
    identity = _identity_logits(rows, held, sources)
    query = [row for row in rows if str(row["bearing"]) == held]
    result = {
        "schema": config["schema"],
        "held_bearing": held,
        "source_bearings": list(sources),
        "dynamics_checkpoint_sha256": sha256(dynamics_checkpoint),
        "metric_checkpoint_sha256": sha256(metric_checkpoint),
        "dynamics_training_terminal": dynamics_trace[-1],
        "metric_training_terminal": metric_trace[-1],
        "identity_scm": summarize_operation(query, identity),
        "meta_metric": summarize_operation(query, learned),
        "order_weights": weights.cpu().tolist(),
        "active_order_count": int((weights > 0).sum()),
        "held_bearing_resources_used_for_fit": 0,
    }
    output = args.artifact_root / f"heldout_{held}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result: {output}")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"HOKO-SPARSE held={held} active={result['active_order_count']} "
        f"final={result['meta_metric']['final_balanced_accuracy']:.6f} "
        f"prefix={result['meta_metric']['balanced_prefix_accuracy']:.6f} "
        f"nll={result['meta_metric']['hierarchical_prequential_nll']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
