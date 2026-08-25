"""Training-only centered environment residual for a shared Mori closure."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CenteredEnvironmentClosure(nn.Module):
    """A minimal source-environment output map with an exact zero-mean gauge.

    For every common latent state, the residual outputs sum to zero across source
    environments.  The shared closure is therefore the arithmetic center of the
    fitted source closures instead of being exchangeable with an arbitrary common
    offset.  This module is a training nuisance absorber and is never deployed.
    """

    def __init__(self, environment_count: int, input_width: int, output_width: int) -> None:
        super().__init__()
        if min(environment_count, input_width, output_width) < 1:
            raise ValueError("invalid centered environment closure dimensions")
        self.environment_count = int(environment_count)
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.weight = nn.Parameter(
            torch.zeros(environment_count, output_width, input_width)
        )
        self.bias = nn.Parameter(torch.zeros(environment_count, output_width))

    def centered_parameters(self) -> tuple[Tensor, Tensor]:
        return self.weight - self.weight.mean(dim=0), self.bias - self.bias.mean(dim=0)

    def forward(self, normalized_state: Tensor, environment_index: int) -> Tensor:
        if not 0 <= int(environment_index) < self.environment_count:
            raise ValueError("environment index is outside the fitted source set")
        weight, bias = self.centered_parameters()
        return F.linear(normalized_state, weight[int(environment_index)], bias[int(environment_index)])

