"""Flat recurrent core (S=0), protocol §5.2: a single dense GRU.

    h_t = GRU(z_t, h_{t-1}),  h_t in R^{H_flat} (default 256)

No spatial mask, no manager -- this is the architecture-side null for
knob S. Used directly by cells M000/M001/M010/M011.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FlatGRUCore(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, z_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return self.cell(z_t, h_prev)

    def readout_state(self, h_t: torch.Tensor) -> torch.Tensor:
        """h*_t used by the shared output heads (protocol §5.3)."""
        return h_t

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
