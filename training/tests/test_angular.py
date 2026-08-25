import math

import torch
from torch import nn

from hoko.dynamics.angular import LearnedCausalAngularModeBank
from hoko.dynamics.train import angular_paths


def test_learned_angular_modes_are_ordered_full_range_and_differentiable():
    torch.manual_seed(81)
    model = LearnedCausalAngularModeBank(mode_count=8, samples_per_revolution=16)
    angular = torch.randn(3, 4, 80, requires_grad=True)
    paths = model(angular)
    orders = model.orders()
    assert paths.shape == (3, 5, 4, 8, 2)
    assert torch.all(orders[1:] > orders[:-1])
    assert torch.allclose(orders[-1], torch.tensor(8.0), atol=1e-6)
    paths.square().mean().backward()
    assert angular.grad is not None and angular.grad.abs().sum() > 0
    assert model.raw_frequency_increments.grad is not None
    assert model.raw_memory_scales.grad is not None


def test_learned_angular_modes_are_strictly_causal():
    torch.manual_seed(83)
    model = LearnedCausalAngularModeBank(mode_count=8, samples_per_revolution=16)
    angular = torch.randn(2, 3, 80)
    changed = angular.clone()
    changed[..., 48:] = 100.0 * torch.randn_like(changed[..., 48:])
    first = model(angular)
    second = model(changed)
    # Endpoints after one, two, and three revolutions cannot see revolution 4+.
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-6)


def test_learned_mode_phase_advances_by_its_order_per_revolution():
    model = LearnedCausalAngularModeBank(mode_count=4, samples_per_revolution=32)
    with torch.no_grad():
        # Use a short memory to make the endpoint-local phase relation sharp.
        model.raw_memory_scales.fill_(math.log(math.expm1(0.15)))
    order = float(model.orders()[0])
    index = torch.arange(6 * 32, dtype=torch.float32)
    signal = torch.cos(2.0 * math.pi * order * index / 32.0)[None, None]
    coefficient = torch.view_as_complex(model(signal).contiguous())[0, :, 0, 0]
    phase_step = torch.angle(coefficient[1:] * coefficient[:-1].conj())
    expected = torch.full_like(phase_step, 2.0 * math.pi * order)
    wrapped_error = torch.angle(torch.exp(1j * (phase_step - expected)))
    assert wrapped_error.abs().mean() < 0.12


def test_learned_mode_bank_replaces_fixed_window_grid_in_angular_paths():
    torch.manual_seed(89)
    bank = LearnedCausalAngularModeBank(mode_count=8, samples_per_revolution=16)
    envelopes = torch.rand(4, 3, 80)
    config = {"observation": {"samples_per_revolution": 16}}
    paths = angular_paths(nn.Identity(), envelopes, config, bank)
    assert paths.shape == (4, 5, 3, 8, 2)
    paths.square().mean().backward()
    assert bank.raw_frequency_increments.grad is not None
    assert bank.raw_memory_scales.grad is not None


def test_learned_order_recurrence_has_deterministic_cuda_backward():
    if not torch.cuda.is_available():
        return
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        bank = LearnedCausalAngularModeBank(
            mode_count=8, samples_per_revolution=16
        ).cuda()
        bank.orders().square().mean().backward()
        assert bank.raw_frequency_increments.grad is not None
    finally:
        torch.use_deterministic_algorithms(previous)
