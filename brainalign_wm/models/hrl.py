"""Hierarchical, spatially-sparse recurrent core (structure factor S=1): a
manager-worker pair in which the worker is a locality-masked recurrent
network (an LM-RNN, following Khona & Chandra 2023).

    Worker (LM-RNN), h^w in R^Nw (default 196, 14x14 grid):
        W_rec_eff = W_rec (.) Mask   -- fixed spatial "lottery ticket" at init
        h^w_t = GRU([z_t ; g_t], h^w_{t-1})           -- every step

    Manager, h^m in R^Nm (default 128):
        s_t = pool(h^w_t)                              -- bottom-up summary
        h^m_t = GRU(s_t, h^m_{t-1})                     -- manager ticks only
        g_t = W_g . h^m_t                               -- top-down context

Manager update timing (modulation factor M; the resolution of this design
choice below):
  M=0 (no reflective gate): a hard periodic clock. The manager GRU runs
      only every `manager_period` steps; on other steps h^m_t = h^m_{t-1}
      exactly (state held, with no gradient path through a no-op step).
  M=1 (reflective gate): the manager runs its GRU cell on every step, but
      its update-gate pre-activation receives an additive `beta*R_t` bias
      from `mechanisms.reflective_gate.ReflectiveGate`. This makes the
      effective update rate continuous and reflection-driven rather than a
      fixed clock: a low R_t keeps u_t near 0 (state held, soft
      persistence), while a run of surprising events (high load, a lure, or
      an error) raises R_t, driving u_t toward 1 so the manager overwrites
      its state and propagates a new top-down signal g_t to the worker.
      This gives the manager an event-gated update rule without requiring a
      separate discrete tick/no-tick decision.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainalign_wm.models.gru_cell import MaskedGRUCell, PBWMManagerCell, PlasticGRUCell, make_locality_mask


class HRLCore(nn.Module):
    def __init__(
        self,
        input_dim: int,
        worker_units: int = 196,
        manager_units: int = 128,
        grid: tuple[int, int] = (14, 14),
        density: float = 0.04,
        manager_period: int = 5,
        g_dim: int = 128,
        pool_block: int = 2,
        reflective: bool = False,
        plastic: bool = False,
        hebb_kwargs: Optional[dict] = None,
        pbwm_gate: bool = False,
        reflection_beta: float = 1.0,
        mask_seed: int = 0,
    ):
        super().__init__()
        gh, gw = grid
        if gh * gw != worker_units:
            raise ValueError(f"grid {grid} has {gh*gw} cells != worker_units={worker_units}")
        if pbwm_gate and not reflective:
            raise ValueError("pbwm_gate requires reflective=True (its gates are R_t-driven)")
        self.grid = grid
        self.worker_units = worker_units
        self.manager_units = manager_units
        self.manager_period = manager_period
        self.g_dim = g_dim
        self.pool_block = pool_block
        self.reflective = reflective
        self.plastic = plastic
        self.pbwm_gate = pbwm_gate

        mask = make_locality_mask(grid, density, seed=mask_seed)
        self.register_buffer("worker_mask", mask)
        # Knob P (§6.2) adds Hebbian fast weights to the WORKER only, per
        # protocol §4.1 ("worker for S=1, the flat GRU for S=0") -- the
        # manager stays a plain MaskedGRUCell regardless of P.
        if plastic:
            self.worker = PlasticGRUCell(input_dim + g_dim, worker_units, mask=mask, **(hebb_kwargs or {}))
        else:
            self.worker = MaskedGRUCell(input_dim + g_dim, worker_units, mask=mask)

        s_dim = self._pooled_dim()
        # Ablation-battery arm M111_pbwm (§4.4/§6.1): 3-gate LSTM-style
        # manager instead of the single GRU update gate. Mutually exclusive
        # with the standard reflective manager below.
        if pbwm_gate:
            self.manager = PBWMManagerCell(s_dim, manager_units, beta=reflection_beta)
        else:
            self.manager = MaskedGRUCell(s_dim, manager_units, mask=None)
        self.g_proj = nn.Linear(manager_units, g_dim)

    def _pooled_dim(self) -> int:
        gh, gw = self.grid
        b = self.pool_block
        return ((gh + b - 1) // b) * ((gw + b - 1) // b)

    def pool_worker(self, h_w: torch.Tensor) -> torch.Tensor:
        """s_t = pool(h^w_t): non-overlapping spatial mean-pool over the
        worker's 14x14 grid. [batch, Nw] -> [batch, s_dim]."""
        gh, gw = self.grid
        b = self.pool_block
        batch = h_w.shape[0]
        grid_act = h_w.view(batch, 1, gh, gw)
        pooled = F.avg_pool2d(grid_act, kernel_size=b, stride=b, ceil_mode=True)
        return pooled.reshape(batch, -1)

    def init_state(self, batch_size: int, device=None) -> dict[str, torch.Tensor]:
        state = {
            "h_worker": torch.zeros(batch_size, self.worker_units, device=device),
            "h_manager": torch.zeros(batch_size, self.manager_units, device=device),
            "g": torch.zeros(batch_size, self.g_dim, device=device),
        }
        if self.plastic:
            state["hebb_worker"] = self.worker.init_hebb(batch_size, device)
        if self.pbwm_gate:
            state["c_manager"] = self.manager.init_cell(batch_size, device)
        return state

    def forward(
        self,
        z_t: torch.Tensor,
        state: dict[str, torch.Tensor],
        t: int,
        gate_bias: Optional[torch.Tensor] = None,
        R_t: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """One step. `gate_bias` (the reflection-derived term beta*R_t) is
        required when `self.reflective` is True (M=1) and must be None
        otherwise -- EXCEPT when `self.pbwm_gate` is also True, which needs
        the RAW `R_t` instead (its three gates apply their own beta
        internally, per gate). Returns (new_state, manager_update/output_
        gate); exposed for logging and for the causal diagnostics on the
        reflective gate's effect."""
        h_w_prev, h_m_prev, g_prev = state["h_worker"], state["h_manager"], state["g"]

        worker_in = torch.cat([z_t, g_prev], dim=-1)
        new_state: dict[str, torch.Tensor] = {}
        if self.plastic:
            h_w_t, _, hebb_w_t = self.worker(worker_in, h_w_prev, state["hebb_worker"])
            new_state["hebb_worker"] = hebb_w_t
        else:
            h_w_t, _ = self.worker(worker_in, h_w_prev)

        s_t = self.pool_worker(h_w_t)
        if self.pbwm_gate:
            if R_t is None:
                raise ValueError("pbwm_gate=True requires R_t (raw reflection signal) every step")
            h_m_t, c_m_t, u_t = self.manager(s_t, h_m_prev, state["c_manager"], R_t)
            new_state["c_manager"] = c_m_t
        elif self.reflective:
            if gate_bias is None:
                raise ValueError("reflective=True requires gate_bias (beta*R_t) every step")
            h_m_t, u_t = self.manager(s_t, h_m_prev, extra_update_bias=gate_bias)
        else:
            is_tick = (t % self.manager_period) == 0
            if is_tick:
                h_m_t, u_t = self.manager(s_t, h_m_prev)
            else:
                h_m_t = h_m_prev
                u_t = torch.zeros(h_m_prev.shape[0], self.manager_units, device=h_m_prev.device)

        g_t = self.g_proj(h_m_t)
        new_state.update({"h_worker": h_w_t, "h_manager": h_m_t, "g": g_t})
        return new_state, u_t

    def readout_state(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """h*_t = [h^w_t ; h^m_t] for the shared output heads."""
        return torch.cat([state["h_worker"], state["h_manager"]], dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
