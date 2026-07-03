"""Hierarchical, spatially-sparse recurrent core (S=1), protocol §5.2: a
manager-worker pair with a locality-masked worker (LM-RNN).

    Worker (LM-RNN), h^w in R^Nw (default 196, 14x14 grid):
        W_rec_eff = W_rec (.) Mask   -- fixed spatial "lottery ticket" at init
        h^w_t = GRU([z_t ; g_t], h^w_{t-1})           -- every step

    Manager, h^m in R^Nm (default 128):
        s_t = pool(h^w_t)                              -- bottom-up summary
        h^m_t = GRU(s_t, h^m_{t-1})                     -- manager ticks only
        g_t = W_g . h^m_t                               -- top-down context

Manager ticking (M knob, resolved design choice -- see DECISIONS.md):
  M=0 (no reflective gate): hard periodic clock. The manager GRU only runs
      every `manager_period` steps; on other steps h^m_t = h^m_{t-1} exactly
      (state held, no gradient path through a no-op step).
  M=1 (reflective gate, protocol §6.1): the manager runs its GRU cell every
      step, but its update-gate pre-activation gets an additive
      `beta*R_t` bias from `mechanisms.reflective_gate.ReflectiveGate`. This
      makes ticking *continuous and reflection-driven* rather than a fixed
      clock: low R_t -> u_t~0 -> state held (soft persistence); a run of
      surprises (high load / lure / error) -> R_t rises -> u_t->1 -> the
      manager overwrites its state and pushes a new g_t down to the worker.
      This directly implements the "event-gated" language in §5.2/§6.1
      without needing a separate discrete tick/no-tick decision.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from brainalign_wm.models.gru_cell import MaskedGRUCell, make_locality_mask


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
        mask_seed: int = 0,
    ):
        super().__init__()
        gh, gw = grid
        if gh * gw != worker_units:
            raise ValueError(f"grid {grid} has {gh*gw} cells != worker_units={worker_units}")
        self.grid = grid
        self.worker_units = worker_units
        self.manager_units = manager_units
        self.manager_period = manager_period
        self.g_dim = g_dim
        self.pool_block = pool_block
        self.reflective = reflective

        mask = make_locality_mask(grid, density, seed=mask_seed)
        self.register_buffer("worker_mask", mask)
        self.worker = MaskedGRUCell(input_dim + g_dim, worker_units, mask=mask)

        s_dim = self._pooled_dim()
        self.manager = MaskedGRUCell(s_dim, manager_units, mask=None)
        self.g_proj = nn.Linear(manager_units, g_dim)

    def _pooled_dim(self) -> int:
        gh, gw = self.grid
        b = self.pool_block
        return ((gh + b - 1) // b) * ((gw + b - 1) // b)

    def pool_worker(self, h_w: torch.Tensor) -> torch.Tensor:
        """s_t = pool(h^w_t): non-overlapping spatial mean-pool over the
        worker's 14x14 grid (protocol §5.2). [batch, Nw] -> [batch, s_dim]."""
        gh, gw = self.grid
        b = self.pool_block
        batch = h_w.shape[0]
        grid_act = h_w.view(batch, 1, gh, gw)
        pooled = F.avg_pool2d(grid_act, kernel_size=b, stride=b, ceil_mode=True)
        return pooled.reshape(batch, -1)

    def init_state(self, batch_size: int, device=None) -> dict[str, torch.Tensor]:
        return {
            "h_worker": torch.zeros(batch_size, self.worker_units, device=device),
            "h_manager": torch.zeros(batch_size, self.manager_units, device=device),
            "g": torch.zeros(batch_size, self.g_dim, device=device),
        }

    def forward(
        self,
        z_t: torch.Tensor,
        state: dict[str, torch.Tensor],
        t: int,
        gate_bias: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """One step. `gate_bias` (protocol §6.1 beta*R_t) is required iff
        `self.reflective` is True (M=1); ignored/must be None otherwise.
        Returns (new_state, manager_update_gate_u_t) -- u_t is exposed for
        logging/analysis (§6.1 causal-control diagnostics)."""
        h_w_prev, h_m_prev, g_prev = state["h_worker"], state["h_manager"], state["g"]

        worker_in = torch.cat([z_t, g_prev], dim=-1)
        h_w_t, _ = self.worker(worker_in, h_w_prev)

        s_t = self.pool_worker(h_w_t)
        if self.reflective:
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
        new_state = {"h_worker": h_w_t, "h_manager": h_m_t, "g": g_t}
        return new_state, u_t

    def readout_state(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """h*_t = [h^w_t ; h^m_t] for the shared output heads (protocol §5.3)."""
        return torch.cat([state["h_worker"], state["h_manager"]], dim=-1)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
