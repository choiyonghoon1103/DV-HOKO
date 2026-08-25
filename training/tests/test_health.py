import math

import torch

from hoko.evaluation.metrics import summarize_health, summarize_joint
from hoko.health.model import binary_health_logits


def test_binary_health_logits_are_exchangeable_over_fault_locations():
    scores = torch.tensor([[2.0, 1.0, 3.0, 5.0]])
    permuted = scores[:, [0, 3, 1, 2]]
    assert torch.allclose(binary_health_logits(scores), binary_health_logits(permuted))
    expected_fault = torch.logsumexp(scores[:, 1:], dim=-1) - math.log(3.0)
    assert torch.allclose(binary_health_logits(scores)[:, 1], expected_fault)


def test_health_and_joint_summaries_use_causal_streams():
    rows = [
        {"record_id": "N", "class_index": 0},
        {"record_id": "I", "class_index": 1},
        {"record_id": "O", "class_index": 2},
        {"record_id": "B", "class_index": 3},
    ]
    health = {
        "N": torch.tensor([-2.0, -4.0]),
        "I": torch.tensor([2.0, 4.0]),
        "O": torch.tensor([2.0, 4.0]),
        "B": torch.tensor([2.0, 4.0]),
    }
    operation = {
        "N": torch.zeros(2, 3),
        "I": torch.tensor([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
        "O": torch.tensor([[0.0, 2.0, 0.0], [0.0, 4.0, 0.0]]),
        "B": torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 4.0]]),
    }
    binary = summarize_health(rows, health)
    joint = summarize_joint(rows, health, operation)
    assert binary["final_balanced_accuracy"] == 1.0
    assert joint["final_balanced_accuracy"] == 1.0
