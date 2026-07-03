"""A from-scratch GRU cell exposing two hooks that `nn.GRUCell` does not
provide:

1. an optional fixed elementwise mask on the recurrent weight matrix, used
   by the spatially-constrained worker population; and
2. an optional external additive bias into the update-gate pre-activation,
   used by the reflective gate's modulation of the manager's update:
   `u_t = sigmoid(... + beta*R_t)`.

The gate convention matches the standard GRU: the update gate `u_t`
multiplies the new candidate state, and `(1-u_t)` retains the previous
state -- so no relabeling is needed between this implementation and the
mechanism/logging/analysis code that reads `u_t` downstream.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MaskedGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_dim))
        self.reset_parameters()
        if mask is not None:
            if mask.shape != (hidden_dim, hidden_dim):
                raise ValueError(f"mask must be [{hidden_dim},{hidden_dim}], got {tuple(mask.shape)}")
            # same locality mask applied to all three gate blocks (reset/update/candidate)
            self.register_buffer("mask", mask.repeat(3, 1))
        else:
            self.mask = None

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_dim ** 0.5)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        extra_update_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_t, u_t). `u_t` (the update gate) is returned for logging
        and analysis: a high reflection value drives u_t toward 1, causing the
        manager to overwrite its state, while a low value keeps the gate
        closed and the state persistent."""
        w_hh = self.weight_hh if self.mask is None else self.weight_hh * self.mask
        gi = x_t @ self.weight_ih.t() + self.bias_ih
        gh = h_prev @ w_hh.t() + self.bias_hh
        i_r, i_u, i_n = gi.chunk(3, dim=-1)
        h_r, h_u, h_n = gh.chunk(3, dim=-1)
        r_t = torch.sigmoid(i_r + h_r)
        u_pre = i_u + h_u
        if extra_update_bias is not None:
            u_pre = u_pre + extra_update_bias
        u_t = torch.sigmoid(u_pre)
        n_t = torch.tanh(i_n + r_t * h_n)
        h_t = (1 - u_t) * h_prev + u_t * n_t
        return h_t, u_t


def make_locality_mask(
    grid: tuple[int, int], density: float, seed: int = 0, kernel: str = "exponential"
) -> torch.Tensor:
    """Spatial locality mask: worker units are arranged on a `grid` (default
    14x14); Mask[i,j] = 1 with probability p(d_ij) decaying with grid
    distance, calibrated so the achieved density matches the target
    `density` (default 4%, following the spatially-constrained sparse RNN
    construction of Khona & Chandra 2023). The mask is fixed at
    initialization and never re-sampled during training.
    """
    import numpy as np

    gh, gw = grid
    n = gh * gw
    ys, xs = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    coords = np.stack([ys.ravel(), xs.ravel()], axis=1).astype(np.float64)  # [n, 2]
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))  # [n, n]

    rng = np.random.RandomState(seed)
    # calibrate the kernel length-scale by bisection so mean(prob) ~= density
    lo, hi = 1e-3, max(gh, gw) * 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if kernel == "gaussian":
            p = np.exp(-(d ** 2) / (2 * mid ** 2))
        else:  # exponential
            p = np.exp(-d / mid)
        np.fill_diagonal(p, 0.0)  # no self-connections
        if p.mean() > density:
            hi = mid
        else:
            lo = mid
    mask = (rng.uniform(size=(n, n)) < p).astype(np.float32)
    np.fill_diagonal(mask, 0.0)
    return torch.from_numpy(mask)
