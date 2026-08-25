"""Source-meta-trained health readout for the shared Koopman--Mori trunk."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hoko.common import deterministic_prefix_sum
from hoko.dynamics.train import angular_paths_and_state, _observation_orders


class StateViewDecoder(nn.Module):
    """Map the preserved equilibrium state to a normalized health view."""

    def __init__(self, input_width: int, hidden_width: int, output_width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, values: Tensor) -> Tensor:
        decoded = self.network(values)
        reference = torch.ones_like(decoded[..., :1])
        return F.normalize(torch.cat((decoded, reference), dim=-1), dim=-1)


def record_state_base(model, mixer, row, config: dict, device: torch.device) -> np.ndarray:
    """Materialize one record's causal state base without target fitting."""

    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths, state = angular_paths_and_state(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        return model.state_base(paths, valid, orders, state).cpu().numpy()


def _cells(rows: list[dict], bearings: tuple[str, ...], device: torch.device) -> dict:
    cells: dict[tuple[str, int], list[Tensor]] = defaultdict(list)
    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing in bearings and label in (0, 1, 2, 3):
            cells[(bearing, label)].append(
                torch.from_numpy(np.asarray(row["frozen_state_base"], dtype=np.float32)).to(device)
            )
    expected = {(bearing, label) for bearing in bearings for label in (0, 1, 2, 3)}
    if set(cells) != expected:
        raise ValueError("dual-view health requires every bearing/class cell")
    return dict(cells)


def _health_centroids(
    decoder: StateViewDecoder,
    cells: dict,
    bearings: tuple[str, ...],
) -> Tensor:
    by_health: dict[int, list[Tensor]] = defaultdict(list)
    for bearing in bearings:
        for label in (0, 1, 2, 3):
            health = int(label != 0)
            for record in cells[(bearing, label)]:
                by_health[health].append(decoder(record))
    return F.normalize(
        torch.stack([torch.cat(by_health[index]).mean(dim=0) for index in (0, 1)]),
        dim=-1,
    )


def _pseudoheld_group_risks(
    decoder: StateViewDecoder,
    cells: dict,
    sources: tuple[str, ...],
) -> Tensor:
    group_risks = []
    for pseudoheld in sources:
        support = tuple(value for value in sources if value != pseudoheld)
        centroids = _health_centroids(decoder, cells, support)
        class_risks = []
        for label in (0, 1, 2, 3):
            record_risks = []
            for record in cells[(pseudoheld, label)]:
                logits = deterministic_prefix_sum(decoder(record) @ centroids.T)
                target = torch.full(
                    (len(logits),), int(label != 0), dtype=torch.long, device=logits.device
                )
                record_risks.append(F.cross_entropy(logits, target))
            class_risks.append(torch.stack(record_risks).mean())
        group_risks.append(torch.stack(class_risks))
    return torch.stack(group_risks)


def fit_state_view(
    rows: list[dict],
    held: str,
    bearings: tuple[str, ...],
    config: dict,
    device: torch.device,
    checkpoint: Path,
):
    """Fit only the state readout through source pseudo-held bearings."""

    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite state view: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    section = config["state_training"]
    torch.manual_seed(int(section["seed"]) + bearings.index(held))
    cells = _cells(rows, sources, device)
    input_width = int(next(iter(cells.values()))[0].shape[-1])
    decoder = StateViewDecoder(
        input_width,
        int(section["hidden_width"]),
        int(section["output_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    objective = str(section.get("objective", "mean_nll"))
    if objective not in {"mean_nll", "maximum_group_nll"}:
        raise ValueError("unknown state-view health objective")
    trace = []
    for update in range(1, int(section["updates"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        group_risks = _pseudoheld_group_risks(decoder, cells, sources)
        loss = (
            group_risks.mean()
            if objective == "mean_nll"
            else group_risks.reshape(-1).max()
        )
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), float(section["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % int(section["log_every"]) == 0 or update == int(section["updates"]):
            item = {
                "update": update,
                "objective": objective,
                "objective_value": float(loss.detach()),
                "source_pseudoheld_mean_health_nll": float(group_risks.mean().detach()),
                "source_pseudoheld_maximum_group_nll": float(group_risks.max().detach()),
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"STATE-VIEW held={held} update={update}/{section['updates']} "
                f"mean={item['source_pseudoheld_mean_health_nll']:.6f} "
                f"max={item['source_pseudoheld_maximum_group_nll']:.6f}",
                flush=True,
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": config["schema"],
            "held_bearing": held,
            "source_bearings": sources,
            "state_dict": {key: value.detach().cpu() for key, value in decoder.state_dict().items()},
            "trace": trace,
            "parameter_count": sum(parameter.numel() for parameter in decoder.parameters()),
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return decoder.eval(), trace


def load_state_view(config: dict, device: torch.device, checkpoint: Path):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    input_width = int(payload["state_dict"]["network.0.weight"].numel())
    hidden_width = int(payload["state_dict"]["network.1.weight"].shape[0])
    output_width = int(payload["state_dict"]["network.3.weight"].shape[0])
    decoder = StateViewDecoder(input_width, hidden_width, output_width).to(device)
    decoder.load_state_dict(payload["state_dict"], strict=True)
    return decoder.eval(), payload["trace"]


def health_streams(
    decoder: StateViewDecoder,
    rows: list[dict],
    held: str,
    sources: tuple[str, ...],
    device: torch.device,
) -> dict[str, Tensor]:
    cells = _cells(rows, sources, device)
    centroids = _health_centroids(decoder, cells, sources)
    streams = {}
    with torch.inference_mode():
        for row in rows:
            if str(row["bearing"]) != held:
                continue
            base = torch.from_numpy(
                np.asarray(row["frozen_state_base"], dtype=np.float32)
            ).to(device)
            logits = deterministic_prefix_sum(decoder(base) @ centroids.T)
            streams[str(row["record_id"])] = (logits[:, 1] - logits[:, 0]).cpu()
    return streams


__all__ = [
    "StateViewDecoder",
    "fit_state_view",
    "health_streams",
    "load_state_view",
    "record_state_base",
]
