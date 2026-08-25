import torch

from hoko.health.attention_state import AttentiveStateViewDecoder


def test_attentive_state_view_preserves_time_and_returns_unit_vectors():
    torch.manual_seed(71)
    decoder = AttentiveStateViewDecoder(16, 24, 8, 4)
    values = torch.randn(5, 11, 16)
    output = decoder(values)
    assert output.shape == (5, 9)
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-6)
