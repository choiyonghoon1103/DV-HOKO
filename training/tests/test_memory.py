import torch

from hoko.memory.model import (
    AttentiveSupportConditionedFieldMetric,
    AttentiveSupportConditionedMahalanobisMetric,
    NeuralSupportConditionedFieldMetric,
    QueryAdaptiveSupportConditionedMahalanobisMetric,
    SupportConditionedFieldMetric,
)


def _support(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        tuple(torch.randn(7, 6, 4, generator=generator) for _ in range(3))
        for _ in range(2)
    )


def test_zero_initialization_is_exact_identity_cosine_metric():
    model = SupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    support = _support()
    orders = torch.linspace(0.5, 2.0, 6)
    query = torch.randn(5, 6, 4)
    output = model(support, query, orders)
    assert torch.equal(output.order_weights, torch.ones_like(output.order_weights))

    source = []
    for class_index in range(3):
        source.append(torch.cat([env[class_index] for env in support]))
    expected_centroids = []
    for values in source:
        flat = values.reshape(len(values), -1)
        flat = flat / flat.norm(dim=1, keepdim=True)
        centroid = flat.mean(dim=0)
        expected_centroids.append(centroid / centroid.norm())
    expected_query = query.reshape(len(query), -1)
    expected_query = expected_query / expected_query.norm(dim=1, keepdim=True)
    expected = expected_query @ torch.stack(expected_centroids).T
    assert torch.allclose(output.logits, expected, atol=1e-6, rtol=1e-6)


def test_class_permutation_only_permutes_candidate_logits():
    torch.manual_seed(3)
    model = SupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    support = _support()
    orders = torch.linspace(0.5, 2.0, 6)
    query = torch.randn(5, 6, 4)
    original = model(support, query, orders)
    permutation = (2, 0, 1)
    permuted_support = tuple(
        tuple(environment[index] for index in permutation) for environment in support
    )
    permuted = model(permuted_support, query, orders)
    assert torch.allclose(original.order_weights, permuted.order_weights, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        original.logits[:, permutation], permuted.logits, atol=1e-6, rtol=1e-6
    )


def test_meta_query_loss_reaches_reliability_network():
    torch.manual_seed(5)
    model = SupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    output = model(_support(), torch.randn(5, 6, 4), torch.linspace(0.5, 2.0, 6))
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 0, 1])).backward()
    assert model.reliability[-1].weight.grad is not None
    assert model.reliability[-1].weight.grad.abs().sum() > 0


def test_explicit_identity_centroids_match_initialized_induction():
    model = SupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    support = _support()
    orders = torch.linspace(0.5, 2.0, 6)
    learned_centroids, learned_weights = model.induce(support, orders)
    explicit = model.induce_with_weights(support, torch.ones(6))
    assert torch.equal(learned_weights, torch.ones_like(learned_weights))
    assert torch.allclose(learned_centroids, explicit, atol=1e-7, rtol=1e-7)


def test_neural_raw_support_metric_is_environment_and_class_permutation_invariant():
    torch.manual_seed(7)
    model = NeuralSupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    support = _support()
    query = torch.randn(5, 6, 4)
    orders = torch.linspace(0.5, 2.0, 6)
    original = model(support, query, orders)
    permutation = (2, 0, 1)
    changed = tuple(
        tuple(environment[index] for index in permutation)
        for environment in reversed(support)
    )
    permuted = model(changed, query, orders)
    assert torch.allclose(original.order_weights, permuted.order_weights, atol=1e-6, rtol=1e-6)
    assert torch.allclose(original.logits[:, permutation], permuted.logits, atol=1e-6, rtol=1e-6)


def test_neural_raw_support_metric_starts_at_identity_and_receives_gradient():
    torch.manual_seed(11)
    model = NeuralSupportConditionedFieldMetric(statistic_width=4, hidden_width=12)
    output = model(_support(), torch.randn(5, 6, 4), torch.linspace(0.5, 2.0, 6))
    assert torch.equal(output.order_weights, torch.ones_like(output.order_weights))
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 0, 1])).backward()
    assert model.raw_reliability[-1].weight.grad is not None
    assert model.raw_reliability[-1].weight.grad.abs().sum() > 0


def test_attentive_raw_support_metric_is_nested_set_permutation_invariant():
    torch.manual_seed(13)
    model = AttentiveSupportConditionedFieldMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    support = _support()
    query = torch.randn(5, 6, 4)
    orders = torch.linspace(0.5, 2.0, 6)
    original = model(support, query, orders)
    permutation = (2, 0, 1)
    changed = tuple(
        tuple(environment[index].flip(0) for index in permutation)
        for environment in reversed(support)
    )
    permuted = model(changed, query, orders)
    assert torch.allclose(original.order_weights, permuted.order_weights, atol=1e-6, rtol=1e-6)
    assert torch.allclose(original.logits[:, permutation], permuted.logits, atol=1e-6, rtol=1e-6)


def test_attentive_raw_support_metric_starts_at_identity_and_receives_gradient():
    torch.manual_seed(17)
    model = AttentiveSupportConditionedFieldMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    output = model(_support(), torch.randn(5, 6, 4), torch.linspace(0.5, 2.0, 6))
    assert torch.equal(output.order_weights, torch.ones_like(output.order_weights))
    torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 2, 0, 1])).backward()
    assert model.raw_reliability[-1].weight.grad is not None
    assert model.raw_reliability[-1].weight.grad.abs().sum() > 0


def test_attentive_mahalanobis_metric_starts_exactly_at_v17_geometry():
    torch.manual_seed(19)
    baseline = AttentiveSupportConditionedFieldMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    learned = AttentiveSupportConditionedMahalanobisMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    learned.load_state_dict(
        {key: value for key, value in baseline.state_dict().items()}, strict=False
    )
    support = _support()
    query = torch.randn(5, 6, 4)
    orders = torch.linspace(0.5, 2.0, 6)
    expected = baseline(support, query, orders)
    actual = learned(support, query, orders)
    assert torch.allclose(actual.logits, expected.logits, atol=1e-7, rtol=1e-7)
    assert torch.equal(
        learned.feature_metric.weight, torch.eye(learned.statistic_width)
    )


def test_attentive_mahalanobis_metric_is_equivariant_and_learns_geometry():
    torch.manual_seed(23)
    model = AttentiveSupportConditionedMahalanobisMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    support = _support()
    query = torch.randn(5, 6, 4)
    orders = torch.linspace(0.5, 2.0, 6)
    original = model(support, query, orders)
    permutation = (2, 0, 1)
    changed = tuple(
        tuple(environment[index].flip(0) for index in permutation)
        for environment in reversed(support)
    )
    permuted = model(changed, query, orders)
    assert torch.allclose(
        original.logits[:, permutation], permuted.logits, atol=1e-6, rtol=1e-6
    )
    torch.nn.functional.cross_entropy(
        original.logits, torch.tensor([0, 1, 2, 0, 1])
    ).backward()
    assert model.feature_metric.weight.grad is not None
    assert model.feature_metric.weight.grad.abs().sum() > 0


def test_sparsemax_starts_at_identity_and_can_remove_orders():
    model = SupportConditionedFieldMetric(
        statistic_width=4,
        hidden_width=12,
        order_weight_normalization="sparsemax",
    )
    initial = model(_support(), torch.randn(5, 6, 4), torch.linspace(0.5, 2.0, 6))
    assert torch.equal(initial.order_weights, torch.ones_like(initial.order_weights))

    logits = torch.tensor([0.7, 0.4, -0.2, -2.0, -3.0, -4.0], requires_grad=True)
    weights = model._normalize_order_logits(logits)
    assert torch.all(weights >= 0)
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert torch.count_nonzero(weights == 0) >= 1
    weights.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_sparse_order_metric_accepts_exact_zero_weights():
    model = AttentiveSupportConditionedMahalanobisMetric(
        statistic_width=4,
        hidden_width=12,
        attention_heads=3,
        order_weight_normalization="sparsemax",
    )
    weights = torch.tensor(
        [3.0, 2.0, 1.0, 0.0, 0.0, 0.0], requires_grad=True
    )
    centroids = model.induce_with_weights(_support(), weights)
    output = model.score_queries(centroids, weights, torch.randn(5, 6, 4))
    assert torch.isfinite(centroids).all()
    assert torch.isfinite(output.logits).all()
    torch.nn.functional.cross_entropy(
        output.logits, torch.tensor([0, 1, 2, 0, 1])
    ).backward()
    assert weights.grad is not None
    assert torch.isfinite(weights.grad).all()


def test_query_adaptive_metric_starts_at_global_metric_and_learns_attention():
    torch.manual_seed(29)
    baseline = AttentiveSupportConditionedMahalanobisMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    adaptive = QueryAdaptiveSupportConditionedMahalanobisMetric(
        statistic_width=4, hidden_width=12, attention_heads=3
    )
    adaptive.load_state_dict(baseline.state_dict(), strict=False)
    support = _support()
    query = torch.randn(5, 6, 4)
    orders = torch.linspace(0.5, 2.0, 6)
    expected = baseline(support, query, orders)
    actual = adaptive(support, query, orders)
    assert torch.allclose(actual.logits, expected.logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual.order_weights,
        expected.order_weights[None].expand(len(query), -1),
        atol=1e-7,
        rtol=1e-7,
    )
    torch.nn.functional.cross_entropy(
        actual.logits, torch.tensor([0, 1, 2, 0, 1])
    ).backward()
    assert adaptive.query_reliability[-1].weight.grad is not None
    assert adaptive.query_reliability[-1].weight.grad.abs().sum() > 0
