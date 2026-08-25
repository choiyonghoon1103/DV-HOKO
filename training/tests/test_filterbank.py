import hashlib
from types import SimpleNamespace

import torch

from hoko.dynamics.filterbank import BalancedSubbandAttentionMixer
from hoko.dynamics.train import (
    _dynamics_labels,
    angular_paths,
    load_external_temporal_initialization,
)


def test_dynamics_regime_universe_is_explicit():
    assert _dynamics_labels({"architecture": {}}) == (1, 2, 3)
    assert _dynamics_labels(
        {"architecture": {"dynamics_regimes": "all_known_regimes"}}
    ) == (0, 1, 2, 3)


def test_subband_mixer_reaches_order_paths_and_is_balanced():
    torch.manual_seed(5)
    mixer = BalancedSubbandAttentionMixer(
        atom_count=16, band_count=4, coordinate_harmonics=4, sinkhorn_iterations=30
    )
    masks = mixer.masks()
    assert torch.allclose(masks.sum(0), torch.ones(16), atol=1e-6)
    assert torch.allclose(masks.sum(1), torch.full((4,), 4.0), atol=2e-3)
    envelopes = torch.rand(2, 16, 128)
    config = {
        "observation": {
            "carrier_bands": 4, "samples_per_revolution": 16,
            "window_revolutions": 2.0, "hop_revolutions": 0.5,
            "minimum_order": 0.5, "maximum_order": 6.0, "order_step": 0.5,
        }
    }
    paths = angular_paths(mixer, envelopes, config)
    assert paths.shape[0] == 2 and paths.shape[2:] == (4, 12, 2)
    paths.square().mean().backward()
    assert mixer.band_queries.grad is not None


def test_external_initialization_loads_only_label_free_temporal_weights(tmp_path):
    layer = torch.nn.TransformerEncoderLayer(
        d_model=24,
        nhead=4,
        dim_feedforward=48,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    )
    source = torch.nn.TransformerEncoder(layer, num_layers=1)
    target = torch.nn.TransformerEncoder(layer, num_layers=1)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
        for parameter in target.parameters():
            parameter.zero_()
    checkpoint = tmp_path / "external.pt"
    torch.save(
        {
            "class_labels_loaded_or_used": False,
            "architecture": {"embedding_width": 24, "temporal_layers": 1},
            "temporal_encoder_state_dict": source.state_dict(),
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model = SimpleNamespace(embedding_width=24, temporal_encoder=target)
    observed = load_external_temporal_initialization(
        model,
        {"external_temporal_initialization": {"checkpoint": str(checkpoint), "sha256": digest}},
    )
    assert observed == digest
    for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
        assert torch.equal(source_parameter, target_parameter)
