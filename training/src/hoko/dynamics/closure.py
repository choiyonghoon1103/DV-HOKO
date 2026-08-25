"""Gauge-identified factor effects for a shared neural dynamics closure."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CenteredFactorClosure(nn.Module):
    """Linear factor-specific closure with an exact zero-mean gauge.

    The factor may be a source system (training nuisance) or a known operation
    (deployable hypothesis).  Centering the parameters makes the shared closure
    the arithmetic centre of the fitted factor family for every latent state.
    """

    def __init__(self, factor_count: int, input_width: int, output_width: int) -> None:
        super().__init__()
        if min(factor_count, input_width, output_width) < 1:
            raise ValueError("invalid centered factor closure dimensions")
        self.factor_count = int(factor_count)
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.weight = nn.Parameter(torch.zeros(factor_count, output_width, input_width))
        self.bias = nn.Parameter(torch.zeros(factor_count, output_width))

    def centered_parameters(self) -> tuple[Tensor, Tensor]:
        return self.weight - self.weight.mean(dim=0), self.bias - self.bias.mean(dim=0)

    def forward(self, normalized_state: Tensor, factor_index: int | Tensor) -> Tensor:
        weight, bias = self.centered_parameters()
        if isinstance(factor_index, int):
            if not 0 <= factor_index < self.factor_count:
                raise ValueError("factor index is outside the fitted factor set")
            return F.linear(normalized_state, weight[factor_index], bias[factor_index])
        if factor_index.ndim != 1 or len(factor_index) != len(normalized_state):
            raise ValueError("batched factor index must match the leading state dimension")
        if factor_index.dtype != torch.long:
            raise ValueError("batched factor index must be torch.long")
        if bool(((factor_index < 0) | (factor_index >= self.factor_count)).any()):
            raise ValueError("factor index is outside the fitted factor set")
        selected_weight = weight[factor_index]
        selected_bias = bias[factor_index]
        output = torch.einsum("b...i,boi->b...o", normalized_state, selected_weight)
        bias_shape = (len(selected_bias),) + (1,) * (normalized_state.ndim - 2) + (
            self.output_width,
        )
        return output + selected_bias.reshape(bias_shape)


class CenteredConditionalOperationClosure(nn.Module):
    """Nonlinear operation generator conditioned on the observed latent state.

    A shared neural operator combines a state feature with one learned operation
    query.  All candidate outputs are evaluated and centered at every state, so
    operation effects remain gauge-identified while their expression may change
    with an unseen system's amortized state context.
    """

    def __init__(
        self,
        operation_count: int,
        input_width: int,
        hidden_width: int,
        output_width: int,
    ) -> None:
        super().__init__()
        if min(operation_count, input_width, hidden_width, output_width) < 1:
            raise ValueError("invalid conditional operation closure dimensions")
        self.factor_count = int(operation_count)
        self.input_width = int(input_width)
        self.hidden_width = int(hidden_width)
        self.output_width = int(output_width)
        self.state = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden_width),
            nn.GELU(),
        )
        self.operation_query = nn.Parameter(torch.empty(operation_count, hidden_width))
        nn.init.normal_(self.operation_query, std=hidden_width**-0.5)
        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_width),
            nn.Linear(hidden_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, output_width),
        )

    def all_operations(self, normalized_state: Tensor) -> Tensor:
        state = self.state(normalized_state)
        raw = self.generator(state.unsqueeze(-2) + self.operation_query)
        return raw - raw.mean(dim=-2, keepdim=True)

    def forward(self, normalized_state: Tensor, operation_index: int | Tensor) -> Tensor:
        centered = self.all_operations(normalized_state)
        if isinstance(operation_index, int):
            if not 0 <= operation_index < self.factor_count:
                raise ValueError("operation index is outside the fitted operation set")
            return centered[..., operation_index, :]
        if operation_index.ndim != 1 or len(operation_index) != len(normalized_state):
            raise ValueError("batched operation index must match the leading state dimension")
        if operation_index.dtype != torch.long:
            raise ValueError("batched operation index must be torch.long")
        if bool(((operation_index < 0) | (operation_index >= self.factor_count)).any()):
            raise ValueError("operation index is outside the fitted operation set")
        view_shape = (len(operation_index),) + (1,) * (normalized_state.ndim - 2) + (1, 1)
        gather_index = operation_index.reshape(view_shape).expand(
            *centered.shape[:-2], 1, self.output_width
        )
        return centered.gather(dim=-2, index=gather_index).squeeze(-2)
