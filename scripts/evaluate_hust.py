#!/usr/bin/env python3
"""Evaluate one released HUST held-bearing checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dvhoko.data import HUST_CLASSES, load_hust_target
from dvhoko.inference import predict_hust_record


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="directory containing N500.mat etc.")
    parser.add_argument("--bearing", choices=("6205", "6206", "6207", "6208"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    weight = ROOT / "weights/hust" / f"heldout_{args.bearing}.pt"
    config = ROOT / "configs/hust.json"
    rows = []
    for record in load_hust_target(args.data, args.bearing):
        output = predict_hust_record(
            record["waveform"], record["shaft_frequency_hz"],
            weight, config, device=args.device
        )
        label = int(record["class_index"]); probability = output["probability"]
        prediction = output["prediction"]
        rows.append({
            "record_id": record["record_id"], "class": HUST_CLASSES[label],
            "load": record["load"], "final_correct": bool(prediction[-1] == label),
            "prefix_accuracy": float(np.mean(prediction == label)),
            "nll": float(np.mean(-np.log(probability[:, label].clip(1e-12)))),
        })
    per_class = {}
    for name in HUST_CLASSES:
        selected = [row for row in rows if row["class"] == name]
        per_class[name] = {
            "final_recall": float(np.mean([row["final_correct"] for row in selected])),
            "prefix_accuracy": float(np.mean([row["prefix_accuracy"] for row in selected])),
            "nll": float(np.mean([row["nll"] for row in selected])),
        }
    result = {
        "bearing": args.bearing,
        "balanced_accuracy": float(np.mean([value["final_recall"] for value in per_class.values()])),
        "prefix_balanced_accuracy": float(np.mean([value["prefix_accuracy"] for value in per_class.values()])),
        "class_balanced_nll": float(np.mean([value["nll"] for value in per_class.values()])),
        "per_class": per_class, "records": rows,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
