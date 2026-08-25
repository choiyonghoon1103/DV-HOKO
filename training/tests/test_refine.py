import torch

from hoko.dynamics.refine import meta_query_risk, normalized_refinement_objective
from hoko.dynamics.train import angular_paths
from hoko.memory.model import QueryAdaptiveSupportConditionedMahalanobisMetric


class _IdentityMixer(torch.nn.Module):
    def forward(self, values):
        return values


def test_explicit_selected_dft_matches_fft_backend():
    torch.manual_seed(2)
    values = torch.rand(2, 3, 64)
    base = {
        "observation": {
            "samples_per_revolution": 16,
            "window_revolutions": 2.0,
            "hop_revolutions": 0.5,
            "minimum_order": 0.5,
            "maximum_order": 8.0,
            "order_step": 0.5,
        }
    }
    expected = angular_paths(_IdentityMixer(), values, base)
    explicit = {"observation": dict(base["observation"])}
    explicit["observation"]["fourier_backend"] = "explicit_selected_dft"
    observed = angular_paths(_IdentityMixer(), values, explicit)
    assert torch.allclose(observed, expected, atol=2e-5, rtol=2e-5)


def test_frozen_metric_meta_risk_reaches_source_and_query_fields():
    torch.manual_seed(3)
    bearings = ("a", "b", "c")
    fields = {}
    leaves = []
    for bearing in bearings:
        for label in (1, 2, 3):
            value = torch.randn(2, 4, 3, requires_grad=True)
            fields[(bearing, label)] = [value]
            leaves.append(value)
    metric = QueryAdaptiveSupportConditionedMahalanobisMetric(
        statistic_width=3,
        hidden_width=8,
        attention_heads=2,
    ).requires_grad_(False)
    risk, per_bearing = meta_query_risk(
        fields, bearings, metric, torch.arange(1, 5, dtype=torch.float32)
    )
    risk.backward()
    assert set(per_bearing) == set(bearings)
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in leaves)
    assert all(parameter.grad is None for parameter in metric.parameters())


def test_refinement_objective_preserves_equal_weight_default():
    observed = normalized_refinement_objective(
        torch.tensor(3.0),
        torch.tensor(10.0),
        torch.tensor(2.0),
        torch.tensor(4.0),
        {},
    )
    assert torch.allclose(observed, torch.tensor(2.0))


def test_refinement_objective_can_isolate_meta_classification_loss():
    meta = torch.tensor(3.0, requires_grad=True)
    forecast = torch.tensor(10.0, requires_grad=True)
    objective = normalized_refinement_objective(
        meta,
        forecast,
        torch.tensor(2.0),
        torch.tensor(4.0),
        {"meta_query_weight": 1.0, "forecast_retention_weight": 0.0},
    )
    objective.backward()
    assert torch.allclose(objective, torch.tensor(1.5))
    assert torch.allclose(meta.grad, torch.tensor(0.5))
    assert torch.allclose(forecast.grad, torch.tensor(0.0))
