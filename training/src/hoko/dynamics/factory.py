"""Configuration-driven construction of the dynamics model."""

from __future__ import annotations

import torch

from hoko.dynamics.angular import LearnedCausalAngularModeBank
from hoko.dynamics.model import KoopmanMoriSpectralField


def build_model(config: dict, device: torch.device) -> KoopmanMoriSpectralField:
    observation, architecture = config["observation"], config["architecture"]
    model = KoopmanMoriSpectralField(
        carrier_bands=int(observation["carrier_bands"]),
        field_width=int(architecture["field_width"]),
        embedding_width=int(architecture["embedding_width"]),
        operation_count=int(architecture["operation_count"]),
        attention_heads=int(architecture["attention_heads"]),
        temporal_layers=int(architecture["temporal_layers"]),
        feedforward_width=int(architecture["feedforward_width"]),
        mixture_count=int(architecture["mixture_count"]),
        minimum_query_width=float(architecture["minimum_query_width_orders"]),
        maximum_query_width=float(architecture["maximum_query_width_orders"]),
        hop_revolutions=float(observation["hop_revolutions"]),
        operation_reader="self_contextual_mixture",
        forecast_horizons=int(architecture["forecast_horizons"]),
        forecast_distribution=str(
            architecture.get("forecast_distribution", "deterministic")
        ),
        order_interaction_layers=int(architecture.get("order_interaction_layers", 0)),
        factorized_operation_closure=bool(
            architecture.get("factorized_operation_closure", False)
        ),
        operation_closure_type=str(
            architecture.get("operation_closure_type", "linear")
        ),
        transport_type=str(architecture.get("transport_type", "fixed_rotation")),
        transport_hidden_width=int(
            architecture.get("transport_hidden_width", architecture["embedding_width"])
        ),
        transport_coordinate_harmonics=int(
            architecture.get("transport_coordinate_harmonics", 4)
        ),
        transport_initial_decay=float(architecture.get("transport_initial_decay", 1e-3)),
        operator_loss_weight=float(architecture.get("operator_loss_weight", 0.0)),
    ).to(device)
    if observation.get("path_operator") == "learned_causal_angular_modes":
        model.angular_mode_bank = LearnedCausalAngularModeBank(
            mode_count=int(observation["learned_mode_count"]),
            samples_per_revolution=int(observation["samples_per_revolution"]),
            initial_memory_revolutions=float(
                observation.get("initial_memory_revolutions", 1.0)
            ),
        ).to(device)
    return model


__all__ = ["build_model"]
