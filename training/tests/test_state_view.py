import torch

from hoko.health.state import StateViewDecoder


def test_state_view_decoder_returns_unit_vectors_with_reference_coordinate():
    decoder = StateViewDecoder(12, 8, 4)
    values = decoder(torch.randn(5, 12))
    assert values.shape == (5, 5)
    torch.testing.assert_close(values.norm(dim=-1), torch.ones(5))
    assert torch.all(values[:, -1] > 0)
