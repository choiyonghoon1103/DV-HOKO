"""Attention readout over frozen order-resolved Koopman--Mori state modes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hoko.common import deterministic_prefix_sum
from hoko.dynamics.train import angular_paths_and_state, _observation_orders


class AttentiveStateViewDecoder(nn.Module):
    """Learn which full-Nyquist modes carry transferable health state evidence."""

    def __init__(
        self,
        input_width: int,
        hidden_width: int,
        output_width: int,
        attention_heads: int,
    ) -> None:
        super().__init__()
        if hidden_width % attention_heads:
            raise ValueError("health attention width must divide its head count")
        self.token = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
        )
        self.query = nn.Parameter(torch.empty(1, 1, hidden_width))
        nn.init.normal_(self.query, std=hidden_width**-0.5)
        self.attention = nn.MultiheadAttention(
            hidden_width, attention_heads, dropout=0.0, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, output_width),
        )

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 3:
            raise ValueError("attentive health state requires [time, mode, feature]")
        tokens = self.token(values)
        query = self.query.expand(len(tokens), -1, -1)
        pooled, _ = self.attention(query, tokens, tokens, need_weights=False)
        decoded = self.output(pooled[:, 0])
        reference = torch.ones_like(decoded[..., :1])
        return F.normalize(torch.cat((decoded, reference), dim=-1), dim=-1)


def record_state_modes(model, mixer, row, config: dict, device: torch.device) -> np.ndarray:
    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths, state = angular_paths_and_state(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        return model.state_mode_base(paths, valid, orders, state).cpu().numpy()


def _cells(rows, bearings, device):
    cells = defaultdict(list)
    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing in bearings and label in (0, 1, 2, 3):
            cells[(bearing, label)].append(
                torch.from_numpy(
                    np.asarray(row["frozen_state_modes"], dtype=np.float32)
                ).to(device)
            )
    expected = {(bearing, label) for bearing in bearings for label in (0, 1, 2, 3)}
    if set(cells) != expected:
        raise ValueError("attentive health state cells are incomplete")
    return dict(cells)


def _centroids(decoder, cells, bearings):
    by_health = defaultdict(list)
    for bearing in bearings:
        for label in (0, 1, 2, 3):
            for record in cells[(bearing, label)]:
                by_health[int(label != 0)].append(decoder(record))
    return F.normalize(
        torch.stack([torch.cat(by_health[index]).mean(dim=0) for index in (0, 1)]),
        dim=-1,
    )


def _pseudoheld_risks(decoder, cells, sources):
    risks = []
    for pseudoheld in sources:
        support = tuple(value for value in sources if value != pseudoheld)
        centroids = _centroids(decoder, cells, support)
        classes = []
        for label in (0, 1, 2, 3):
            records = []
            for record in cells[(pseudoheld, label)]:
                logits = deterministic_prefix_sum(decoder(record) @ centroids.T)
                target = torch.full(
                    (len(logits),), int(label != 0), dtype=torch.long, device=logits.device
                )
                records.append(F.cross_entropy(logits, target))
            classes.append(torch.stack(records).mean())
        risks.append(torch.stack(classes))
    return torch.stack(risks)


def fit_attentive_state_view(rows, held, bearings, config, device, checkpoint: Path):
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite attentive state view: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    section = config["state_training"]
    torch.manual_seed(int(section["seed"]) + bearings.index(held))
    cells = _cells(rows, sources, device)
    input_width = int(next(iter(cells.values()))[0].shape[-1])
    decoder = AttentiveStateViewDecoder(
        input_width,
        int(section["hidden_width"]),
        int(section["output_width"]),
        int(section["attention_heads"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
    )
    trace = []
    for update in range(1, int(section["updates"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        risks = _pseudoheld_risks(decoder, cells, sources)
        loss = risks.mean()
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            decoder.parameters(), float(section["gradient_clip_norm"])
        )
        optimizer.step()
        if update == 1 or update % int(section["log_every"]) == 0 or update == int(section["updates"]):
            item = {
                "stage": "full_nyquist_mode_attention_health_view",
                "update": update,
                "source_pseudoheld_mean_health_nll": float(loss.detach()),
                "source_pseudoheld_maximum_group_nll": float(risks.max().detach()),
                "gradient_norm_before_clip": float(gradient),
            }
            trace.append(item)
            print(
                f"ATTN-HEALTH held={held} update={update}/{section['updates']} "
                f"nll={item['source_pseudoheld_mean_health_nll']:.6f}",
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
            "parameter_count": sum(p.numel() for p in decoder.parameters()),
            "held_bearing_resources_used_for_fit": 0,
        },
        checkpoint,
    )
    return decoder.eval(), trace


def attentive_health_streams(decoder, rows, held, sources, device):
    cells = _cells(rows, sources, device)
    centroids = _centroids(decoder, cells, sources)
    streams = {}
    with torch.inference_mode():
        for row in rows:
            if str(row["bearing"]) != held:
                continue
            values = torch.from_numpy(
                np.asarray(row["frozen_state_modes"], dtype=np.float32)
            ).to(device)
            logits = deterministic_prefix_sum(decoder(values) @ centroids.T)
            streams[str(row["record_id"])] = (logits[:, 1] - logits[:, 0]).cpu()
    return streams


__all__ = [
    "AttentiveStateViewDecoder",
    "attentive_health_streams",
    "fit_attentive_state_view",
    "record_state_modes",
]
