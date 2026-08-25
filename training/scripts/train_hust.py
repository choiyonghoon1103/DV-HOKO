#!/usr/bin/env python3
"""Train the frozen-dynamics + learned-field-geometry HUST LODO operation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hoko.evaluation.metrics import aggregate_folds, summarize_operation
from hoko.evaluation.transfer import causal_centroid_logits
from hoko.data.hust import discover_hust_records, load_subband_cache
from hoko.common import order_grid, sha256
from hoko.dynamics.train import fit_fold as fit_dynamics
from hoko.dynamics.train import record_field
from hoko.memory.train import fit_fold as fit_metric
from hoko.memory.train import operation_streams


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
    values = causal_centroid_logits(
        source_fields, np.asarray(source_labels, dtype=np.int64), query_fields
    )
    return {
        str(row["record_id"]): torch.from_numpy(value).float()
        for row, value in zip(query_rows, values, strict=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hust/model.json")
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/hust")
    parser.add_argument("--only-held", choices=("6205", "6206", "6207", "6208"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    base_path = ROOT / config["base_config"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    bearings = tuple(str(value) for value in config["bearings"])
    selected = (args.only_held,) if args.only_held else bearings
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    torch.use_deterministic_algorithms(True)

    rows = discover_hust_records(args.data_root, bearings)
    load_subband_cache(rows, ROOT / config["subband_cache_root"])
    fixed_orders = (
        None
        if base["observation"].get("path_operator") == "learned_causal_angular_modes"
        else torch.from_numpy(order_grid(base)).to(device)
    )
    folds = {}
    for held in selected:
        sources = tuple(value for value in bearings if value != held)
        dynamics_checkpoint = (
            args.artifact_root / "dynamics" / f"heldout_{held}" / "dynamics_terminal.pt"
        )
        metric_checkpoint = (
            args.artifact_root / "metric" / f"heldout_{held}" / "metric_terminal.pt"
        )
        model, mixer, dynamics_trace = fit_dynamics(
            rows, held, bearings, base, device, dynamics_checkpoint
        )
        mode_bank = getattr(model, "angular_mode_bank", None)
        orders = (
            mode_bank.orders().detach()
            if mode_bank is not None
            else fixed_orders
        )
        if orders is None:
            raise RuntimeError("dynamics model did not expose an order coordinate")
        for row in rows:
            row["frozen_field"] = record_field(model, mixer, row, base, device)
        metric, metric_trace, _ = fit_metric(
            rows, held, bearings, orders, config, device, metric_checkpoint
        )
        learned, weights = operation_streams(metric, rows, held, sources, orders, device)
        identity = _identity_logits(rows, held, sources)
        query = [row for row in rows if str(row["bearing"]) == held]
        fold = {
            "held_bearing": held,
            "source_bearings": list(sources),
            "dynamics_checkpoint_sha256": sha256(dynamics_checkpoint),
            "metric_checkpoint_sha256": sha256(metric_checkpoint),
            "dynamics_training_terminal": dynamics_trace[-1],
            "metric_training_terminal": metric_trace[-1],
            "identity_scm": summarize_operation(query, identity),
            "meta_metric": summarize_operation(query, learned),
            "order_weights": weights.tolist(),
            "held_bearing_resources_used_for_fit": 0,
        }
        folds[held] = fold
        output = args.artifact_root / f"heldout_{held}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"refusing to overwrite result: {output}")
        output.write_text(json.dumps(fold, indent=2, sort_keys=True) + "\n")
        print(
            f"HOKO held={held} final={fold['meta_metric']['final_balanced_accuracy']:.6f} "
            f"prefix={fold['meta_metric']['balanced_prefix_accuracy']:.6f} "
            f"nll={fold['meta_metric']['hierarchical_prequential_nll']:.6f}",
            flush=True,
        )
    result = {
        "schema": config["schema"],
        "status": "complete" if len(folds) == len(bearings) else "partial",
        "config_sha256": sha256(args.config.resolve()),
        "base_config_sha256": sha256(base_path),
        "folds": folds,
        "aggregate": aggregate_folds(folds),
    }
    summary = args.artifact_root / (
        "operation_result.json" if len(folds) == len(bearings) else f"operation_{selected[0]}.json"
    )
    if summary.exists():
        raise FileExistsError(f"refusing to overwrite summary: {summary}")
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
