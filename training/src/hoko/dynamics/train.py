"""Efficient class-free learned-subband Koopman--Mori training."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from hoko.common import order_grid
from hoko.dynamics.environment import CenteredEnvironmentClosure
from hoko.dynamics.factory import build_model
from hoko.dynamics.filterbank import BalancedSubbandAttentionMixer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_external_temporal_initialization(model, config: dict) -> str | None:
    """Load only external causal temporal weights before HOKO source fitting."""

    section = config.get("external_temporal_initialization")
    if section is None:
        return None
    checkpoint = Path(section["checkpoint"])
    expected = str(section["sha256"])
    observed = _sha256(checkpoint)
    if observed != expected:
        raise ValueError("external temporal checkpoint hash differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_labels = bool(section.get("class_labels_loaded_or_used", False))
    if payload.get("class_labels_loaded_or_used") is not expected_labels:
        raise ValueError("external initialization label-use contract differs")
    architecture = payload.get("architecture", {})
    if (
        int(architecture.get("embedding_width", -1)) != model.embedding_width
        or int(architecture.get("temporal_layers", -1))
        != len(model.temporal_encoder.layers)
    ):
        raise ValueError("external temporal architecture differs from HOKO")
    model.temporal_encoder.load_state_dict(
        payload["temporal_encoder_state_dict"], strict=True
    )
    return observed


def materialize_uniform_subband_envelopes(rows, data_root: Path, config: dict) -> None:
    section, observation = config["learned_filterbank"], config["observation"]
    sample_count = int(section["sample_count"])
    atom_count = int(section["uniform_subband_atoms"])
    samples_per_revolution = int(observation["samples_per_revolution"])
    bins = sample_count // 2 + 1
    edges = np.linspace(0, bins, atom_count + 1, dtype=np.int64)
    for row in rows:
        archive = loadmat(data_root / f"{row['record_id']}.mat", variable_names=("data", "fs"))
        waveform = np.asarray(archive["data"], dtype=np.float64).reshape(-1)
        shaft = float(np.asarray(archive["fs"]).squeeze())
        count = len(waveform) // sample_count
        blocks = waveform[: count * sample_count].reshape(count, sample_count)
        blocks -= blocks.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(blocks, axis=-1)
        analytic = np.zeros((count, atom_count, sample_count), dtype=np.complex64)
        for atom, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
            local = np.zeros((count, sample_count), dtype=np.complex64)
            local[:, left:right] = spectrum[:, left:right]
            lo = max(left, 1)
            hi = min(right, bins - 1)
            local[:, lo:hi] *= 2.0
            analytic[:, atom] = np.fft.ifft(local, axis=-1)
        envelope = np.abs(analytic).astype(np.float32)
        revolutions = int(np.floor((sample_count - 1) * shaft / sample_count))
        angular_count = revolutions * samples_per_revolution
        positions = np.arange(angular_count) * sample_count / (samples_per_revolution * shaft)
        left = np.floor(positions).astype(np.int64)
        right = np.minimum(left + 1, sample_count - 1)
        fraction = (positions - left).astype(np.float32)
        row["uniform_subband_envelopes"] = (
            envelope[..., left] * (1.0 - fraction) + envelope[..., right] * fraction
        )


def build_mixer(config: dict, device: torch.device) -> BalancedSubbandAttentionMixer:
    section = config["learned_filterbank"]
    return BalancedSubbandAttentionMixer(
        atom_count=int(section["uniform_subband_atoms"]),
        band_count=int(config["observation"]["carrier_bands"]),
        coordinate_harmonics=int(section["coordinate_harmonics"]),
        sinkhorn_iterations=int(section["sinkhorn_iterations"]),
    ).to(device)


def angular_paths_and_state(
    mixer, envelopes: torch.Tensor, config: dict, mode_bank=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centered dynamical paths and the state discarded by centering.

    The Koopman--Mori path is intentionally invariant to the per-second log
    envelope level.  Health is not: equilibrium level and spread can separate
    a normal regime from a fault even when their centered dynamics are close.
    Both views are computed by the same mixer and therefore remain one shared
    observation model rather than two independently trained encoders.
    """

    observation = config["observation"]
    samples_per_revolution = int(observation["samples_per_revolution"])
    angular = torch.log1p(mixer(envelopes))
    level = angular.mean(dim=-1)
    angular = angular - level[..., None]
    spread = angular.square().mean(dim=-1).add(1e-8).sqrt()
    state = torch.cat((level, spread), dim=-1)
    if mode_bank is not None:
        return mode_bank(angular), state
    window = int(round(float(observation["window_revolutions"]) * samples_per_revolution))
    hop = int(round(float(observation["hop_revolutions"]) * samples_per_revolution))
    local = angular.unfold(-1, window, hop)
    taper = torch.hann_window(window, periodic=False, device=angular.device, dtype=angular.dtype)
    orders = order_grid(config)
    indices = np.rint(orders * window / samples_per_revolution).astype(np.int64)
    backend = str(observation.get("fourier_backend", "fft"))
    if backend == "fft":
        coefficient = (
            torch.fft.rfft(local * taper, dim=-1)
            / taper.square().sum().sqrt()
        )
        selected = coefficient[..., torch.from_numpy(indices).to(angular.device)]
        paths = torch.view_as_real(selected.permute(0, 2, 1, 3).contiguous())
        return paths, state
    if backend != "explicit_selected_dft":
        raise ValueError("unknown selected Fourier backend")
    sample = torch.arange(window, device=angular.device, dtype=angular.dtype)
    selected_index = torch.from_numpy(indices).to(
        device=angular.device, dtype=angular.dtype
    )
    phase = 2.0 * torch.pi * selected_index[:, None] * sample[None] / window
    tapered = local * taper
    denominator = taper.square().sum().sqrt()
    real = torch.einsum("...n,kn->...k", tapered, torch.cos(phase)) / denominator
    imaginary = -torch.einsum("...n,kn->...k", tapered, torch.sin(phase)) / denominator
    selected = torch.stack((real, imaginary), dim=-1)
    paths = selected.permute(0, 2, 1, 3, 4).contiguous()
    return paths, state


def angular_paths(mixer, envelopes: torch.Tensor, config: dict, mode_bank=None) -> torch.Tensor:
    return angular_paths_and_state(mixer, envelopes, config, mode_bank)[0]


def _observation_orders(config: dict, device: torch.device, mode_bank=None):
    if mode_bank is not None:
        return mode_bank.orders()
    return torch.from_numpy(order_grid(config)).to(device)


def _dynamics_labels(config: dict) -> tuple[int, ...]:
    regime = str(config["architecture"].get("dynamics_regimes", "fault_only"))
    if regime == "fault_only":
        return (1, 2, 3)
    if regime == "all_known_regimes":
        return (0, 1, 2, 3)
    raise ValueError(f"unknown dynamics regime universe: {regime}")


def _cells(rows, bearings, device, labels=(1, 2, 3)):
    cells = defaultdict(list)
    for row in rows:
        bearing, label = str(row["bearing"]), int(row["class_index"])
        if bearing in bearings and label in labels:
            cells[(bearing, label)].append(
                {
                    "envelopes": torch.from_numpy(row["uniform_subband_envelopes"]).to(device),
                    "record_id": str(row["record_id"]),
                }
            )
    if set(cells) != {(bearing, label) for bearing in bearings for label in labels}:
        raise ValueError("uniform-subband training cells are incomplete")
    return cells


def _bearing_batch(cells, bearing, mixer, config, labels=(1, 2, 3), mode_bank=None):
    paths = [
        angular_paths(mixer, record["envelopes"], config, mode_bank)
        for label in labels
        for record in cells[(bearing, label)]
    ]
    maximum = max(value.shape[1] for value in paths)
    exemplar = paths[0]
    packed = exemplar.new_zeros((sum(len(value) for value in paths), maximum, *exemplar.shape[2:]))
    valid = torch.zeros(packed.shape[:2], dtype=torch.bool, device=packed.device)
    cursor = 0
    for value in paths:
        packed[cursor : cursor + len(value), : value.shape[1]] = value
        valid[cursor : cursor + len(value), : value.shape[1]] = True
        cursor += len(value)
    return packed, valid


def _cell_batch(cells, bearing, label, mixer, config, mode_bank=None):
    paths = [
        angular_paths(mixer, record["envelopes"], config, mode_bank)
        for record in cells[(bearing, label)]
    ]
    maximum = max(value.shape[1] for value in paths)
    exemplar = paths[0]
    packed = exemplar.new_zeros((sum(len(value) for value in paths), maximum, *exemplar.shape[2:]))
    valid = torch.zeros(packed.shape[:2], dtype=torch.bool, device=packed.device)
    cursor = 0
    for value in paths:
        packed[cursor : cursor + len(value), : value.shape[1]] = value
        valid[cursor : cursor + len(value), : value.shape[1]] = True
        cursor += len(value)
    return packed, valid


def fit_fold(rows, held, bearings, config, device, checkpoint):
    if checkpoint.exists():
        raise FileExistsError(f"refusing to resume learned-subband KM fit: {checkpoint}")
    sources = tuple(value for value in bearings if value != held)
    seed = int(config["optimization"]["field_seed"]) + bearings.index(held)
    torch.manual_seed(seed)
    model, mixer = build_model(config, device), build_mixer(config, device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    external_initialization_sha256 = load_external_temporal_initialization(model, config)
    use_environment_closure = bool(
        config["architecture"].get("centered_environment_closure", False)
    )
    environment_closure = None
    if use_environment_closure:
        forecast_components = (
            2 if model.forecast_distribution == "deterministic" else 3
        )
        environment_closure = CenteredEnvironmentClosure(
            environment_count=len(sources),
            input_width=model.field_width,
            output_width=(
                model.forecast_horizons
                * forecast_components
                * model.carrier_bands
            ),
        ).to(device)
    parameters = list(model.field_parameters()) + list(mixer.parameters())
    if mode_bank is not None:
        parameters += list(mode_bank.parameters())
    operation_dynamics_parameters = list(model.operation_dynamics_parameters())
    parameters += operation_dynamics_parameters
    if environment_closure is not None:
        parameters += list(environment_closure.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["optimization"]["learning_rate"]),
        weight_decay=float(config["optimization"]["weight_decay"]),
    )
    dynamics_labels = _dynamics_labels(config)
    cells = _cells(rows, sources, device, dynamics_labels)
    updates = int(config["optimization"]["field_pretrain_updates"])
    environment_objective = str(
        config["optimization"].get("field_environment_objective", "mean")
    )
    if environment_objective not in {"mean", "maximum"}:
        raise ValueError("unknown source-environment dynamics objective")
    trace = []
    model.train(); mixer.train()
    for update in range(1, updates + 1):
        optimizer.zero_grad(set_to_none=True)
        selected_environment = None
        if model.factorized_operation_closure:
            if environment_objective != "mean":
                raise ValueError("factorized operation dynamics require the crossed mean objective")
            risks, cell_risks = [], {}
            for environment_index, bearing in enumerate(sources):
                for label in (1, 2, 3):
                    paths, valid = _cell_batch(
                        cells, bearing, label, mixer, config, mode_bank
                    )
                    risk = model.dynamics_loss(
                        paths,
                        valid,
                        _observation_orders(config, device, mode_bank),
                        environment_residual_head=environment_closure,
                        environment_index=(
                            environment_index if environment_closure is not None else None
                        ),
                        operation_index=label - 1,
                    )
                    (risk / (len(sources) * 3)).backward()
                    risks.append(float(risk.detach()))
                    cell_risks[f"{bearing}:{label}"] = float(risk.detach())
            objective_risk = float(np.mean(risks))
            environment_risks = {
                bearing: float(
                    np.mean([cell_risks[f"{bearing}:{label}"] for label in (1, 2, 3)])
                )
                for bearing in sources
            }
        elif environment_objective == "mean":
            risks = []
            for environment_index, bearing in enumerate(sources):
                if bool(config["optimization"].get("regime_balanced_dynamics", False)):
                    local_risks = []
                    for label in dynamics_labels:
                        paths, valid = _cell_batch(
                            cells, bearing, label, mixer, config, mode_bank
                        )
                        local_risks.append(
                            model.dynamics_loss(
                                paths,
                                valid,
                                _observation_orders(config, device, mode_bank),
                                environment_residual_head=environment_closure,
                                environment_index=(
                                    environment_index
                                    if environment_closure is not None
                                    else None
                                ),
                            )
                        )
                    risk = torch.stack(local_risks).mean()
                else:
                    paths, valid = _bearing_batch(
                        cells, bearing, mixer, config, dynamics_labels, mode_bank
                    )
                    risk = model.dynamics_loss(
                        paths,
                        valid,
                        _observation_orders(config, device, mode_bank),
                        environment_residual_head=environment_closure,
                        environment_index=(
                            environment_index if environment_closure is not None else None
                        ),
                    )
                (risk / len(sources)).backward()
                risks.append(float(risk.detach()))
            objective_risk = float(np.mean(risks))
            environment_risks = dict(zip(sources, risks))
        else:
            # The encoder is deterministic (dropout=0 and no batch statistics), so
            # this first pass selects the exact active group for a subgradient of
            # the empirical maximum.  Recompute only that graph to retain the
            # original bounded-memory training contract.
            with torch.no_grad():
                risks = []
                for environment_index, bearing in enumerate(sources):
                    paths, valid = _bearing_batch(
                        cells, bearing, mixer, config, dynamics_labels, mode_bank
                    )
                    risks.append(
                        float(
                            model.dynamics_loss(
                                paths,
                                valid,
                                _observation_orders(config, device, mode_bank),
                                environment_residual_head=environment_closure,
                                environment_index=(
                                    environment_index
                                    if environment_closure is not None
                                    else None
                                ),
                            )
                        )
                    )
            worst_index = int(np.argmax(risks))
            selected_environment = sources[worst_index]
            paths, valid = _bearing_batch(
                cells, selected_environment, mixer, config, dynamics_labels, mode_bank
            )
            robust_risk = model.dynamics_loss(
                paths,
                valid,
                _observation_orders(config, device, mode_bank),
                environment_residual_head=environment_closure,
                environment_index=(
                    worst_index if environment_closure is not None else None
                ),
            )
            robust_risk.backward()
            risks[worst_index] = float(robust_risk.detach())
            objective_risk = float(risks[worst_index])
            environment_risks = dict(zip(sources, risks))
        gradient = torch.nn.utils.clip_grad_norm_(parameters, float(config["optimization"]["gradient_clip_norm"]))
        optimizer.step()
        if update == 1 or update % int(config["optimization"]["log_every"]) == 0 or update == updates:
            item = {
                "stage": "class_free_uniform_subband_attention_koopman_mori",
                "update": update,
                "source_environment_mean_dynamics_risk": float(np.mean(risks)),
                "optimized_environment_objective": environment_objective,
                "optimized_environment_risk": objective_risk,
                "selected_worst_environment": selected_environment,
                "centered_environment_closure": use_environment_closure,
                "per_source_environment_risk": environment_risks,
                "per_source_system_operation_risk": (
                    cell_risks if model.factorized_operation_closure else None
                ),
                "gradient_norm_before_clip": float(gradient),
                "learned_angular_mode_bank": mode_bank is not None,
            }
            trace.append(item)
            print(
                f"LSB-KM held={held} update={update}/{updates} "
                f"mean={item['source_environment_mean_dynamics_risk']:.6f} "
                f"objective={objective_risk:.6f}",
                flush=True,
            )
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema": config["schema"], "held_bearing": held, "source_bearings": sources,
            "seed": seed,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "mixer_state_dict": {k: v.detach().cpu() for k, v in mixer.state_dict().items()},
            "environment_closure_state_dict": (
                None
                if environment_closure is None
                else {
                    k: v.detach().cpu()
                    for k, v in environment_closure.state_dict().items()
                }
            ),
            "trace": trace,
            "external_temporal_initialization_sha256": external_initialization_sha256,
            "parameter_count": sum(p.numel() for p in parameters),
            "class_loss_gradient_reached_filterbank": False,
            "fixed_carrier_band_boundaries_used": False,
            "uniform_subband_atoms": int(config["learned_filterbank"]["uniform_subband_atoms"]),
            "learned_angular_orders": (
                None
                if mode_bank is None
                else mode_bank.orders().detach().cpu().tolist()
            ),
            "learned_angular_memory_revolutions": (
                None
                if mode_bank is None
                else mode_bank.memory_scales().detach().cpu().tolist()
            ),
            "training_only_centered_environment_closure": use_environment_closure,
            "factorized_operation_closure": model.factorized_operation_closure,
            "dynamics_regimes": list(dynamics_labels),
            "regime_balanced_dynamics": bool(
                config["optimization"].get("regime_balanced_dynamics", False)
            ),
            "operation_closure_type": model.operation_closure_type,
            "source_fault_labels_used_only_to_select_dynamics_operation": (
                model.factorized_operation_closure
            ),
            "dense_teacher_materialized": False, "held_bearing_resources_used_for_fit": 0,
        }, checkpoint,
    )
    return model.eval(), mixer.eval(), trace


def load_fold(held, bearings, config, device, checkpoint):
    """Load an immutable terminal dynamics checkpoint for evaluation only."""

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    sources = tuple(value for value in bearings if value != held)
    if (
        payload.get("schema") != config["schema"]
        or str(payload.get("held_bearing")) != held
        or tuple(payload.get("source_bearings", ())) != sources
    ):
        raise ValueError("terminal dynamics checkpoint contract differs")
    model, mixer = build_model(config, device), build_mixer(config, device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    mixer.load_state_dict(payload["mixer_state_dict"], strict=True)
    if payload.get("environment_closure_state_dict") is not None:
        forecast_components = 2 if model.forecast_distribution == "deterministic" else 3
        closure = CenteredEnvironmentClosure(
            environment_count=len(sources),
            input_width=model.field_width,
            output_width=model.forecast_horizons * forecast_components * model.carrier_bands,
        ).to(device)
        closure.load_state_dict(payload["environment_closure_state_dict"], strict=True)
        model.source_environment_closure = closure.eval()
    return model.eval(), mixer.eval(), payload["trace"]


def record_field(model, mixer, row, config, device):
    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths = angular_paths(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        return model._field(paths, valid, orders).cpu().numpy()


def record_innovation_field(model, mixer, row, config, device):
    """Frozen order-resolved energy/correlation signature of Mori innovations."""

    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths = angular_paths(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        return model.innovation_field(paths, valid, orders).cpu().numpy()


def record_operation_logits(model, mixer, row, config, device):
    """Cumulative direct evidence from operation-conditioned forecast NLL."""

    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths = angular_paths(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        local_nll = model.operation_forecast_nll(paths, valid, orders)
        # CUDA cumsum has no deterministic-algorithm implementation.  The
        # evidence stream is tiny, so transfer the already-computed local NLL
        # and perform the exact ordered accumulation on CPU.
        return (-torch.cumsum(local_nll.cpu(), dim=0)).numpy()


def record_marginal_operation_logits(model, mixer, row, config, device):
    """Cumulative operation evidence marginalized over source-system closures."""

    closure = getattr(model, "source_environment_closure", None)
    if closure is None:
        raise ValueError("source-system marginalization requires a loaded closure bank")
    envelopes = torch.from_numpy(row["uniform_subband_envelopes"]).to(device)
    mode_bank = getattr(model, "angular_mode_bank", None)
    paths = angular_paths(mixer, envelopes, config, mode_bank)
    valid = torch.ones(paths.shape[:2], dtype=torch.bool, device=device)
    orders = _observation_orders(config, device, mode_bank)
    with torch.inference_mode():
        local_nll = model.operation_forecast_nll(
            paths,
            valid,
            orders,
            environment_residual_head=closure,
            environment_count=closure.environment_count,
        )
        return (-torch.cumsum(local_nll.cpu(), dim=0)).numpy()
