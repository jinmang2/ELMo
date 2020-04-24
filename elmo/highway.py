# *~ coding convention ~*
from overrides import overrides
from typing import Callable

import torch
import torch.nn as nn


# residual connection
class Highway(nn.Module):
    def __init__(self,
                 input_dim: int,
                 num_layers: int = 1,
                 activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu) -> None:
        super(Highway, self).__init__()
        self._input_dim = input_dim
        self._layers = nn.ModuleList(
            [nn.Linear(input_dim, input_dim * 2)
             for _ in range(num_layers)])
        self._activation = activation
        for layer in self._layers:
            # 왜 1일까 논문에선 -1, -2, -3인데... Hmm...
            layer.bias[input_dim:].data.fill_(1)

    @overrides
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part = projected_input[:, (0 * self._input_dim):(1 * self._input_dim)]
            gate = projected_input[:, (1 * self._input_dim):(2 * self._input_dim)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
