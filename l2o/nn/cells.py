"""Custom recurrent cells used by learned optimizers."""

from __future__ import annotations

import numpy as np
import torch


class LSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.input_weight = torch.nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.hidden_weight = torch.nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):
        batch_size = input.size(0)

        if hx is None:
            h = input.new_zeros(batch_size, self.hidden_size)
            hx = (h, h)

        h, c = hx

        gates = self.input_weight(input) + self.hidden_weight(h)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c = c * f_t + i_t * g_t
        h = o_t * torch.tanh(c)
        return h, c


class ScalarLSTMCell(torch.nn.Module):
    def __init__(self, input_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.bias = bias

        small = 1e-6

        self.input_weight = torch.nn.Parameter(torch.randn(input_size, 4) * small)
        self.hidden_weight = torch.nn.Parameter(torch.randn(input_size, 4) * small)

        self.input_bias = torch.nn.Parameter(torch.randn(input_size, 4) * small) if bias else None
        self.hidden_bias = torch.nn.Parameter(torch.randn(input_size, 4) * small) if bias else None

    def forward(self, input, hx=None):
        batch_size = input.size(0)

        if hx is None:
            h = input.new_zeros(batch_size, self.input_size)
            hx = (h, h)

        h, c = hx

        gates = input.unsqueeze(-1) * self.input_weight.unsqueeze(0)  # (batch_size, input_size, 4)
        if self.bias:
            gates += self.input_bias
        gates += h.unsqueeze(-1) * self.hidden_weight.unsqueeze(0)
        if self.bias:
            gates += self.hidden_bias

        input_gate, forget_gate, cell_gate, output_gate = [gates[..., i] for i in range(4)]

        i_t = torch.sigmoid(input_gate)
        f_t = torch.sigmoid(forget_gate)
        g_t = torch.tanh(cell_gate)
        o_t = torch.sigmoid(output_gate)

        c = c * f_t + i_t * g_t
        h = o_t * torch.tanh(c)
        return h, c

