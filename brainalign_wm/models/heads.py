"""Output heads: a policy head (fixation/yes/no) and a value head.

The readout scale gamma parameterizes the aligned-versus-oblique dynamical
regime (Schuessler et al. 2024): a small gamma favors an oblique
initialization and a large gamma favors an aligned one. It is held at a
neutral default (gamma=1.0) in the core experimental design and swept in a
planned aligned/oblique sub-experiment, in which training can optionally be
regularized toward a target readout norm via `readout_norm()`.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_ACTIONS = 3  # 0 = fixation/no-response, 1 = yes (in-set), 2 = no (not-in-set)


class Heads(nn.Module):
    def __init__(self, input_dim: int, readout_scale: float = 1.0, n_actions: int = N_ACTIONS):
        super().__init__()
        self.pi = nn.Linear(input_dim, n_actions)
        self.value = nn.Linear(input_dim, 1)
        self.readout_scale = readout_scale
        with torch.no_grad():
            self.pi.weight.mul_(readout_scale)
            self.value.weight.mul_(readout_scale)

    def forward(self, h_star: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.pi(h_star)
        policy = torch.softmax(logits, dim=-1)
        value = self.value(h_star).squeeze(-1)
        return policy, value, logits

    def readout_norm(self) -> torch.Tensor:
        """||W_pi||_F^2 + ||w_v||_F^2 -- for an optional readout-scale
        regularizer in the aligned/oblique sweep, keeping gamma near its
        target throughout training rather than only at initialization."""
        return self.pi.weight.pow(2).sum() + self.value.weight.pow(2).sum()
