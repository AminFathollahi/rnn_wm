"""Output heads: policy (fixation/yes/no) and value (protocol §5.3).

Readout scale gamma parameterizes the aligned/oblique regime (Schuessler et al.
2024, §4.3/H5): small gamma -> oblique-favoring init, large gamma -> aligned-
favoring init. Neutral (gamma=1.0) in Core; swept in the Extended aligned/
oblique sub-experiment (§4.3, M9), optionally regularized toward a target norm
via `readout_norm()` during that sweep's training.
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
        """||W_pi||_F^2 + ||w_v||_F^2 -- for an optional readout-scale regularizer
        during the Extended aligned/oblique sweep (§4.3), keeping gamma near its
        target through training rather than only at init."""
        return self.pi.weight.pow(2).sum() + self.value.weight.pow(2).sum()
