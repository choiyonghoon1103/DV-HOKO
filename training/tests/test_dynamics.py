import torch
import pytest

from hoko.dynamics.closure import CenteredConditionalOperationClosure, CenteredFactorClosure
from hoko.dynamics.environment import CenteredEnvironmentClosure
from hoko.dynamics.hypothesis import install_operation_hypotheses
from hoko.dynamics.model import KoopmanMoriSpectralField
from hoko.dynamics.transport import (
    FixedRotationTransport,
    EquivariantResidualTransport,
    LearnedStableTransport,
    LearnedUnitaryTransport,
)


def _model(
    forecast_distribution="deterministic",
    order_interaction_layers=0,
    factorized_operation_closure=False,
    operation_closure_type="linear",
    transport_type="fixed_rotation",
    operator_loss_weight=0.0,
):
    return KoopmanMoriSpectralField(
        carrier_bands=3,
        field_width=8,
        embedding_width=24,
        operation_count=3,
        attention_heads=4,
        temporal_layers=1,
        feedforward_width=48,
        mixture_count=2,
        minimum_query_width=0.15,
        maximum_query_width=1.5,
        hop_revolutions=0.5,
        operation_reader="self_contextual_mixture",
        forecast_horizons=2,
        forecast_distribution=forecast_distribution,
        order_interaction_layers=order_interaction_layers,
        factorized_operation_closure=factorized_operation_closure,
        operation_closure_type=operation_closure_type,
        transport_type=transport_type,
        operator_loss_weight=operator_loss_weight,
    )


def test_fixed_transport_reproduces_analytic_rotation_and_semigroup():
    transport = FixedRotationTransport(0.5)
    values = torch.randn(2, 5, 3, 7, 2)
    orders = torch.linspace(0.5, 3.5, 7)
    angle = 2.0 * torch.pi * orders * 0.5
    expected = torch.stack(
        (
            torch.cos(angle)[None, None, None] * values[..., 0]
            - torch.sin(angle)[None, None, None] * values[..., 1],
            torch.sin(angle)[None, None, None] * values[..., 0]
            + torch.cos(angle)[None, None, None] * values[..., 1],
        ),
        dim=-1,
    )
    assert torch.allclose(transport(values, orders), expected, atol=1e-6, rtol=1e-6)
    twice = transport(transport(values, orders), orders)
    assert torch.allclose(twice, transport(values, orders, 2), atol=1e-6, rtol=1e-6)


def test_learned_transport_is_stable_semigroup_and_receives_source_gradient():
    torch.manual_seed(29)
    transport = LearnedStableTransport(carrier_bands=3, hidden_width=16)
    values = torch.randn(2, 5, 3, 7, 2)
    orders = torch.linspace(0.5, 3.5, 7)
    one = transport(values, orders)
    two = transport(values, orders, 2)
    assert torch.allclose(transport(one, orders), two, atol=1e-6, rtol=1e-6)
    source_norm = values.square().sum(dim=-1).sqrt()
    transported_norm = one.square().sum(dim=-1).sqrt()
    assert torch.all(transported_norm <= source_norm + 1e-6)

    model = _model(transport_type="learned_stable", operator_loss_weight=1.0)
    paths = torch.randn(3, 7, 3, 11, 2)
    valid = torch.ones(3, 7, dtype=torch.bool)
    model.dynamics_loss(paths, valid, torch.linspace(0.5, 3.0, 11)).backward()
    gradients = [parameter.grad for parameter in model.transport.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_learned_unitary_transport_cannot_use_zero_shrinkage_shortcut():
    torch.manual_seed(30)
    transport = LearnedUnitaryTransport(carrier_bands=3, hidden_width=16)
    values = torch.randn(2, 5, 3, 7, 2)
    orders = torch.linspace(0.5, 3.5, 7)
    one = transport(values, orders)
    two = transport(values, orders, 2)
    assert torch.allclose(transport(one, orders), two, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        one.square().sum(dim=-1), values.square().sum(dim=-1), atol=1e-5, rtol=1e-5
    )
    model = _model(transport_type="learned_unitary", operator_loss_weight=1.0)
    paths = torch.randn(3, 7, 3, 11, 2)
    valid = torch.ones(3, 7, dtype=torch.bool)
    model.dynamics_loss(paths, valid, torch.linspace(0.5, 3.0, 11)).backward()
    assert sum(
        float(parameter.grad.abs().sum()) for parameter in model.transport.parameters()
    ) > 0.0


def test_equivariant_residual_contains_fixed_rotation_and_learns_detuning():
    torch.manual_seed(32)
    fixed = FixedRotationTransport(0.5)
    residual = EquivariantResidualTransport(
        carrier_bands=3, hop_revolutions=0.5, hidden_width=16
    )
    values = torch.randn(2, 5, 3, 7, 2)
    orders = torch.linspace(0.5, 3.5, 7)
    assert torch.allclose(
        residual(values, orders), fixed(values, orders), atol=1e-6, rtol=1e-6
    )
    target = torch.roll(fixed(values, orders), shifts=1, dims=1)
    (residual(values, orders) - target).square().mean().backward()
    gradients = [parameter.grad for parameter in residual.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_teacher_free_koopman_mori_shapes_and_parameter_firewall():
    torch.manual_seed(31)
    model = _model()
    paths = torch.randn(4, 7, 3, 11, 2)
    valid = torch.ones(4, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    output = model(paths, valid, orders)
    assert output.field.shape == (4, 11, 8)
    assert output.logits.shape == (4, 3)
    assert output.class_attention.shape == (4, 3, 11)
    assert output.forecast_loss.ndim == 0
    output.forecast_loss.backward()
    assert all(parameter.grad is not None for parameter in model.field_parameters())
    assert all(parameter.grad is None for parameter in model.operation_parameters())


def test_internal_causal_state_does_not_change_when_only_future_changes():
    torch.manual_seed(37)
    model = _model().eval()
    paths = torch.randn(2, 7, 3, 11, 2)
    changed = paths.clone()
    changed[:, 5:] = 20.0 * torch.randn_like(changed[:, 5:])
    valid = torch.ones(2, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    with torch.no_grad():
        normalized_a, states_a = model._causal_states(paths, valid, orders)
        normalized_b, states_b = model._causal_states(changed, valid, orders)
    assert torch.allclose(normalized_a[:, :5], normalized_b[:, :5], atol=1e-6, rtol=1e-6)
    assert torch.allclose(states_a[:, :, :5], states_b[:, :, :5], atol=1e-5, rtol=1e-5)


def test_cross_order_operator_is_class_free_causal_field_parameter():
    torch.manual_seed(39)
    model = _model(order_interaction_layers=1).eval()
    paths = torch.randn(2, 7, 3, 11, 2)
    changed = paths.clone()
    changed[:, 5:] = 20.0 * torch.randn_like(changed[:, 5:])
    valid = torch.ones(2, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    normalized_a, encoded_a = model._causal_states(paths, valid, orders)
    normalized_b, encoded_b = model._causal_states(changed, valid, orders)
    joint_a = model._joint_order_states(encoded_a, orders)
    joint_b = model._joint_order_states(encoded_b, orders)
    assert torch.allclose(normalized_a[:, :5], normalized_b[:, :5], atol=1e-6, rtol=1e-6)
    assert torch.allclose(joint_a[:, :, :5], joint_b[:, :, :5], atol=1e-5, rtol=1e-5)

    loss = model.dynamics_loss(paths, valid, orders)
    loss.backward()
    assert model.order_interaction_encoder is not None
    interaction_parameters = list(model.order_interaction_encoder.parameters())
    assert interaction_parameters
    assert all(parameter.grad is not None for parameter in interaction_parameters)
    assert all(parameter.grad is None for parameter in model.operation_parameters())


def test_padding_does_not_change_teacher_free_field():
    torch.manual_seed(41)
    model = _model().eval()
    paths = torch.randn(2, 6, 3, 11, 2)
    padded = torch.cat((paths, torch.zeros(2, 2, 3, 11, 2)), dim=1)
    valid = torch.zeros(2, 8, dtype=torch.bool)
    valid[:, :6] = True
    orders = torch.linspace(0.5, 3.0, 11)
    with torch.no_grad():
        reference = model(paths, torch.ones(2, 6, dtype=torch.bool), orders)
        observed = model(padded, valid, orders)
    assert torch.allclose(reference.field, observed.field, atol=1e-5, rtol=1e-5)
    assert torch.allclose(reference.logits, observed.logits, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regression")
def test_cuda_dynamics_backward_is_deterministic_algorithm_compatible():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(43)
        model = _model().cuda()
        paths = torch.randn(2, 7, 3, 11, 2, device="cuda")
        valid = torch.ones(2, 7, dtype=torch.bool, device="cuda")
        orders = torch.linspace(0.5, 3.0, 11, device="cuda")
        loss = model.dynamics_loss(paths, valid, orders)
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_stochastic_koopman_mori_learns_mean_and_noise_intensity():
    torch.manual_seed(47)
    model = _model("isotropic_gaussian")
    paths = torch.randn(3, 7, 3, 11, 2)
    valid = torch.ones(3, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    loss = model.dynamics_loss(paths, valid, orders)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    final = model.memory_forecast_head[-1]
    assert final.weight.grad is not None
    assert final.weight.grad.abs().sum() > 0


def test_mori_innovation_field_is_finite_and_padding_invariant():
    torch.manual_seed(49)
    model = _model().eval()
    paths = torch.randn(2, 7, 3, 11, 2)
    padded = torch.cat((paths, torch.zeros(2, 2, 3, 11, 2)), dim=1)
    valid = torch.zeros(2, 9, dtype=torch.bool)
    valid[:, :7] = True
    orders = torch.linspace(0.5, 3.0, 11)
    with torch.inference_mode():
        reference = model.innovation_field(
            paths, torch.ones(2, 7, dtype=torch.bool), orders
        )
        observed = model.innovation_field(padded, valid, orders)
    assert reference.shape == (2, 11, 13)
    assert torch.isfinite(reference).all()
    assert torch.allclose(reference, observed, atol=1e-5, rtol=1e-5)


def test_centered_environment_closure_is_zero_mean_and_training_only():
    torch.manual_seed(53)
    closure = CenteredEnvironmentClosure(3, 8, 12)
    with torch.no_grad():
        closure.weight.normal_()
        closure.bias.normal_()
    state = torch.randn(2, 5, 8)
    outputs = torch.stack([closure(state, index) for index in range(3)])
    assert torch.allclose(outputs.mean(dim=0), torch.zeros_like(outputs[0]), atol=1e-6)

    model = _model()
    paths = torch.randn(2, 7, 3, 11, 2)
    valid = torch.ones(2, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    residual = CenteredEnvironmentClosure(3, model.field_width, 12)
    loss = model.dynamics_loss(paths, valid, orders, residual, 1)
    loss.backward()
    assert residual.weight.grad is not None and residual.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.operation_parameters())

    with pytest.raises(ValueError, match="requires a source index"):
        model.dynamics_loss(paths, valid, orders, residual)


def test_centered_factor_closure_supports_batched_factor_selection():
    torch.manual_seed(59)
    closure = CenteredFactorClosure(3, 8, 12)
    with torch.no_grad():
        closure.weight.normal_()
        closure.bias.normal_()
    state = torch.randn(5, 4, 7, 8)
    indices = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    batched = closure(state, indices)
    reference = torch.stack([closure(value, int(index)) for value, index in zip(state, indices)])
    assert torch.allclose(batched, reference, atol=1e-6)
    all_factors = torch.stack([closure(state, index) for index in range(3)])
    assert torch.allclose(
        all_factors.mean(dim=0), torch.zeros_like(all_factors[0]), atol=1e-6
    )


def test_factorized_operation_closure_trains_by_forecast_and_scores_hypotheses():
    torch.manual_seed(61)
    model = _model("isotropic_gaussian", factorized_operation_closure=True)
    paths = torch.randn(6, 7, 3, 11, 2)
    valid = torch.ones(6, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    loss = model.dynamics_loss(paths, valid, orders, operation_index=labels)
    loss.backward()
    operation_parameters = list(model.operation_dynamics_parameters())
    assert operation_parameters
    assert all(parameter.grad is not None for parameter in operation_parameters)
    assert all(parameter.grad is None for parameter in model.operation_parameters())
    with torch.no_grad():
        nll = model.operation_forecast_nll(paths, valid, orders)
        shared_field = model._field(paths, valid, orders)
    assert nll.shape == (6, 3) and torch.isfinite(nll).all()
    assert shared_field.shape == (6, 11, 8)
    environment = CenteredEnvironmentClosure(
        3,
        model.field_width,
        model.forecast_horizons * 3 * model.carrier_bands,
    )
    with torch.no_grad():
        marginalized = model.operation_forecast_nll(
            paths, valid, orders, environment, environment.environment_count
        )
        pseudoheld = model.operation_forecast_nll(
            paths,
            valid,
            orders,
            environment,
            environment_indices=(0, 2),
        )
    assert marginalized.shape == (6, 3) and torch.isfinite(marginalized).all()
    assert pseudoheld.shape == (6, 3) and torch.isfinite(pseudoheld).all()


def test_conditional_operation_generator_is_state_conditioned_and_centered():
    torch.manual_seed(67)
    closure = CenteredConditionalOperationClosure(3, 8, 24, 12)
    state = torch.randn(5, 4, 7, 8)
    all_operations = closure.all_operations(state)
    assert all_operations.shape == (5, 4, 7, 3, 12)
    assert torch.allclose(
        all_operations.mean(dim=-2), torch.zeros_like(all_operations[..., 0, :]), atol=1e-6
    )
    indices = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    selected = closure(state, indices)
    reference = torch.stack([closure(value, int(index)) for value, index in zip(state, indices)])
    assert torch.allclose(selected, reference, atol=1e-6)

    model = _model(
        "isotropic_gaussian",
        factorized_operation_closure=True,
        operation_closure_type="conditional_mlp",
    )
    paths = torch.randn(5, 7, 3, 11, 2)
    valid = torch.ones(5, 7, dtype=torch.bool)
    orders = torch.linspace(0.5, 3.0, 11)
    loss = model.dynamics_loss(paths, valid, orders, operation_index=indices)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.operation_dynamics_parameters())


def test_operation_hypothesis_installation_freezes_validated_trunk():
    model = _model(factorized_operation_closure=False)
    environment = install_operation_hypotheses(
        model,
        environment_count=3,
        closure_type="conditional_mlp",
        device=torch.device("cpu"),
    )
    operation_ids = {id(parameter) for parameter in model.operation_dynamics_parameters()}
    assert operation_ids
    for parameter in model.parameters():
        assert parameter.requires_grad is (id(parameter) in operation_ids)
    assert all(parameter.requires_grad for parameter in environment.parameters())
    assert model.factorized_operation_closure
