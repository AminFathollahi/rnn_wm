"""Shared front end: frozen encoder features and task-context vector are
concatenated, layer-normalized, and projected through a noisy bottleneck.
This module is identical across all eight factorial cells -- a controlled
element held fixed so that only the downstream recurrent core varies.

    a_t = LayerNorm( W_in . concat(v_t, c_t) ),  W_in in R^{d_h x (feature_dim+task_vec_dim)}
    z_t = W_bottleneck . a_t + eps_t,  eps_t ~ N(0, sigma_in^2 I),  z_t in R^{bottleneck_dim}

The pre-bottleneck width `d_h` defaults to the bottleneck width itself (two
linear layers of matched width around one LayerNorm) unless overridden.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FrontEnd(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        task_vec_dim: int = 10,
        bottleneck_dim: int = 128,
        input_noise_sigma: float = 0.05,
        pre_bottleneck_dim: Optional[int] = None,
    ):
        super().__init__()
        d_h = pre_bottleneck_dim or bottleneck_dim
        self.w_in = nn.Linear(feature_dim + task_vec_dim, d_h)
        self.layernorm = nn.LayerNorm(d_h)
        self.w_bottleneck = nn.Linear(d_h, bottleneck_dim)
        self.input_noise_sigma = input_noise_sigma
        self.bottleneck_dim = bottleneck_dim

    def forward(self, v_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        a_t = self.layernorm(self.w_in(torch.cat([v_t, c_t], dim=-1)))
        z_t = self.w_bottleneck(a_t)
        if self.input_noise_sigma > 0:
            z_t = z_t + torch.randn_like(z_t) * self.input_noise_sigma
        return z_t
