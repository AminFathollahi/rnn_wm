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

    def n_units(self) -> int:
        return self.hidden_dim

    def effective_param_count(self) -> int:
        """Structural synapse count (Khona & Chandra 2023 node-vs-synapse
        convention): `weight_ih` (dense, always) plus `weight_hh` counted
        only at entries the `mask` leaves alive -- a masked-out entry is not
        a physical connection and must not inflate the reported budget
        (A2/B2). Biases are per-unit, not per-synapse, so excluded here;
        they still appear in the model's raw `n_params_total`."""
        n_hh = self.weight_hh.numel() if self.mask is None else int(self.mask.sum().item())
        return self.weight_ih.numel() + n_hh


class PlasticGRUCell(MaskedGRUCell):
    """`MaskedGRUCell` plus differentiable Hebbian fast weights (Miconi,
    Clune & Stanley 2018; protocol §6.2, Knob P=1): alongside the slow
    BPTT-trained `weight_hh`, each recurrent synapse gets a fast per-trial
    trace `hebb` and a BPTT-trained mixing coefficient `alpha`,
    `W_eff(t) = weight_hh + alpha * hebb(t-1)`, used in place of `weight_hh`
    for that tick's recurrent update; `hebb` is then updated from this
    tick's own pre/post-synaptic activity. `hebb` is per-trial state (reset
    every trial via `init_hebb`, batched), not a learned parameter.

    Applied identically to all three GRU gate blocks (reset/update/
    candidate), like the locality mask -- there is no principled way to
    give a GRU a single "the" recurrent weight the way a plain RNN has one,
    so each gate's [hidden, hidden] block gets its own trace using the
    same pre-synaptic (`h_prev`) and post-synaptic (`h_t`) signal.

    `effective_param_count()`/`n_units()` are inherited unchanged from
    `MaskedGRUCell`: `alpha` is a per-synapse modulation coefficient on an
    EXISTING `weight_hh` entry, not a new connection, so it does not add to
    the structural synapse count (it does still count in the raw
    `n_params_total` any caller computes via `.parameters()`)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, mask: Optional[torch.Tensor] = None,
        eta_decay: float = 0.9, eta_hebb: float = 0.05, hebb_clip: float = 2.0,
    ):
        super().__init__(input_dim, hidden_dim, mask)
        self.alpha = nn.Parameter(torch.zeros(3 * hidden_dim, hidden_dim))
        self.eta_decay = eta_decay
        self.eta_hebb = eta_hebb
        self.hebb_clip = hebb_clip

    def init_hebb(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, 3 * self.hidden_dim, self.hidden_dim, device=device)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        hebb_prev: torch.Tensor,
        extra_update_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_t, u_t, hebb_t)."""
        w_hh = self.weight_hh if self.mask is None else self.weight_hh * self.mask
        alpha = self.alpha if self.mask is None else self.alpha * self.mask
        w_eff = w_hh.unsqueeze(0) + alpha.unsqueeze(0) * hebb_prev  # [B, 3H, H]
        gi = x_t @ self.weight_ih.t() + self.bias_ih
        gh = torch.einsum("boh,bh->bo", w_eff, h_prev) + self.bias_hh
        i_r, i_u, i_n = gi.chunk(3, dim=-1)
        h_r, h_u, h_n = gh.chunk(3, dim=-1)
        r_t = torch.sigmoid(i_r + h_r)
        u_pre = i_u + h_u
        if extra_update_bias is not None:
            u_pre = u_pre + extra_update_bias
        u_t = torch.sigmoid(u_pre)
        n_t = torch.tanh(i_n + r_t * h_n)
        h_t = (1 - u_t) * h_prev + u_t * n_t

        post = torch.cat([h_t, h_t, h_t], dim=-1).unsqueeze(-1)  # [B, 3H, 1]
        pre = h_prev.unsqueeze(-2)  # [B, 1, H]
        hebb_t = (self.eta_decay * hebb_prev + self.eta_hebb * post * pre).clamp(-self.hebb_clip, self.hebb_clip)
        return h_t, u_t, hebb_t


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


class PBWMManagerCell(nn.Module):
    """Bio-plausible ablation-battery arm `M111_pbwm` (§4.4, §6.1): replaces
    the manager's single GRU update gate with three explicit gates (input/
    forget/output -- O'Reilly & Frank 2006 PBWM), structurally an LSTM
    cell. Forget and input gates are driven by the reflection signal R_t
    (additive `beta*R_t` bias, like the GRU update gate they replace); the
    output gate is NOT R-driven ("reads out normally" per spec -- only
    what gets written/kept is under reflective control, not readout).
    Cell state `c_t` persists across manager ticks exactly like an LSTM's,
    replacing the GRU manager's single hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int, beta: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / (self.hidden_dim ** 0.5)
        nn.init.uniform_(self.weight_ih, -std, std)
        nn.init.uniform_(self.weight_hh, -std, std)

    def init_cell(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def n_units(self) -> int:
        return self.hidden_dim

    def effective_param_count(self) -> int:
        """Never masked (the manager is always dense), so this is just the
        raw weight_ih/weight_hh synapse count."""
        return self.weight_ih.numel() + self.weight_hh.numel()

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor, R_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_t, c_t, o_t) -- `o_t` (output gate) is returned in
        place of the GRU manager's `u_t`, for logging parity."""
        gates = x_t @ self.weight_ih.t() + self.bias_ih + h_prev @ self.weight_hh.t() + self.bias_hh
        i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        i_t = torch.sigmoid(i_pre + self.beta * R_t)
        f_t = torch.sigmoid(f_pre + self.beta * R_t)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)  # ungated by R_t -- readout is not under reflective control
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t, o_t
