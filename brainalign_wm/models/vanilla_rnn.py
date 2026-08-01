"""Vanilla tanh RNN cell -- the Stage 1 substrate (comments.txt §4, Phase 1
item 1.4): the training-regime-mapping factorial runs on this cheap
substrate before any compute goes into the GRU bio-plausibility battery.

    h_t = tanh(W_ih x_t + W_hh_masked h_{t-1} + b)

Same `mask`/`extra_update_bias` constructor and forward shape as
`MaskedGRUCell` (single-gate, so `forward` returns just `h_t`, not a
(h_t, u_t) pair) so `_step_core` can dispatch on `model.substrate` without a
substrate-specific branch for the M=0,P=0 case.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        mask: Optional[torch.Tensor] = None,
        recurrent_init_spectral_radius: Optional[float] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.reset_parameters()
        if mask is not None:
            if mask.shape != (hidden_dim, hidden_dim):
                raise ValueError(f"mask must be [{hidden_dim},{hidden_dim}], got {tuple(mask.shape)}")
            self.register_buffer("mask", mask)
        else:
            self.mask = None
        # `None` (the default) leaves the uniform(-1/sqrt(H), 1/sqrt(H)) draw
        # above untouched -- every existing checkpoint, test, and result
        # depends on that being bit-identical. A target radius rescales
        # `weight_hh` in place, computed on the EFFECTIVE (mask-applied)
        # matrix, since that is what the forward/backward pass actually use.
        if recurrent_init_spectral_radius is not None:
            self._rescale_recurrent_spectral_radius(recurrent_init_spectral_radius)

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_dim ** 0.5)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)

    def _rescale_recurrent_spectral_radius(self, target: float) -> None:
        with torch.no_grad():
            effective = self.weight_hh if self.mask is None else self.weight_hh * self.mask
            radius = torch.linalg.eigvals(effective).abs().max().item()
            if radius > 1e-12:
                self.weight_hh.mul_(target / radius)

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        """Same interface/return shape as `_GatedFlatCore.init_state` --
        `training/train.py::_init_state`'s S=0 branch dispatches on this
        method existing, not on cell type."""
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def readout_state(self, h_t: torch.Tensor) -> torch.Tensor:
        """Identity, same as `_GatedFlatCore.readout_state` -- single
        recurrent state IS the readout for a flat core, gated or not."""
        return h_t

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, extra_update_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w_hh = self.weight_hh if self.mask is None else self.weight_hh * self.mask
        pre = x_t @ self.weight_ih.t() + h_prev @ w_hh.t() + self.bias
        if extra_update_bias is not None:
            pre = pre + extra_update_bias
        return torch.tanh(pre)

    def n_units(self) -> int:
        return self.hidden_dim

    def effective_param_count(self) -> int:
        """Structural synapse count, same convention as
        `MaskedGRUCell.effective_param_count` (Khona & Chandra 2023):
        `weight_ih` dense, `weight_hh` counted only at unmasked entries."""
        n_hh = self.weight_hh.numel() if self.mask is None else int(self.mask.sum().item())
        return self.weight_ih.numel() + n_hh
