#!/usr/bin/env python3
"""Evaluate the released Data_final checkpoint on the original 2026 files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dvhoko.data import FINAL_CLASSES, load_final_streams
from dvhoko.inference import predict_final_stream


ROOT = Path(__file__).resolve().parents[1]


def metrics(probability, labels, weights):
    prediction = probability.argmax(1); rows = {}
    for label, name in enumerate(FINAL_CLASSES):
        selected = labels == label
        if not np.any(selected):
            rows[name] = None
        else:
            rows[name] = {
                "recall": float(np.average(prediction[selected] == label, weights=weights[selected])),
                "nll": float(np.average(-np.log(probability[selected, label].clip(1e-12)), weights=weights[selected])),
            }
    available = [value for value in rows.values() if value is not None]
    return {
        "balanced_accuracy": float(np.mean([value["recall"] for value in available])),
        "class_balanced_nll": float(np.mean([value["nll"] for value in available])),
        "per_class": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="prepared Data_final/2026 directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    weight, config = ROOT / "weights/final/model.pt", ROOT / "configs/final.json"
    rows = []
    for stream in load_final_streams(args.data):
        output = predict_final_stream(
            stream.observations, stream.timestamps, weight, config, device=args.device
        )
        probability = output["probability"]
        rows.append({
            "record_id": stream.record_id, "role": stream.role,
            "second_metrics": metrics(probability, stream.labels, np.ones(len(stream.labels))),
            "packet_metrics": metrics(probability, stream.labels, stream.packet_counts),
            "transitions": (np.flatnonzero(np.diff(stream.labels) != 0) + 1).tolist(),
        })
    result = {
        "mixed_evaluation": next(row for row in rows if row["role"] == "mixed_evaluation"),
        "records": rows,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
