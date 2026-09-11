"""Leaky rate recurrent cores: three ungated alternatives to the GRU flat
core, each isolating one candidate explanation for a representational
mismatch that the gated substrate shows.

    h_t = (1 - alpha) h_{t-1} + alpha * phi(W_in x_t + W_rec s_{t-1} + b)

`alpha = 1 / time_constant_ticks` is the single-timescale leak; the
`(1 - alpha)` term is the only path that carries state across a tick, so
unlike a plain tanh RNN the delay period does not have to be bridged by the
recurrent weights alone.

The three cells differ in exactly one factor each:

`ExcitatoryInhibitoryRateCell`   hard sign separation (a unit's outgoing
                                 weights are all excitatory or all
                                 inhibitory, enforced structurally) and a
                                 non-negative firing rate. Varies the sign
                                 constraint and the activity scale.
`DynamicSynapseRateCell`         Tsodyks-Markram short-term facilitation
                                 and depression per presynaptic unit, so
                                 part of the trial's memory lives in
                                 synaptic state rather than in the
                                 observable firing rates. Distinct from the
                                 Hebbian fast weights in `gru_cell.py`: the
                                 state here is per-unit, not per-synapse, is
                                 not learned or mixed by a trained
                                 coefficient, and follows fixed
                                 release/recovery kinetics.
`LowRankRateCell`                recurrent connectivity constrained to rank
                                 R, so the population's recurrent dynamics
                                 are confined to an R-dimensional subspace.

All three expose the flat-core interface (`init_state`, `forward` returning
`(h_t, u_t)`, `readout_state`, `n_units`, `effective_param_count`) and are
single-gate, so `effective_param_count` matches a GRU of the same synapse
budget at roughly sqrt(3) times its width.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LeakyRateCell(nn.Module):
    """Shared leak, input weights, bias and interface. Subclasses supply the
    recurrent weight parameterization and the rate nonlinearity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        time_constant_ticks: float = 2.0,
        recurrent_init_spectral_radius: Optional[float] = None,
    ):
        super().__init__()
        if time_constant_ticks < 1.0:
            raise ValueError(f"time_constant_ticks must be >= 1 (got {time_constant_ticks})")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = 1.0 / float(time_constant_ticks)
        self.weight_ih = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        std = 1.0 / (hidden_dim ** 0.5)
        nn.init.uniform_(self.weight_ih, -std, std)
        self._init_recurrent()
        if recurrent_init_spectral_radius is not None:
            self._set_spectral_radius(float(recurrent_init_spectral_radius))

    # -- subclass hooks ---------------------------------------------------

    def _init_recurrent(self) -> None:
        raise NotImplementedError

    def recurrent_weight(self) -> torch.Tensor:
        """The effective [H, H] recurrent matrix, row = postsynaptic unit,
        column = presynaptic unit. For a cell whose synapses carry state,
        this is the matrix at the initial synaptic state."""
        raise NotImplementedError

    def _scale_recurrent(self, factor: float) -> None:
        """Multiply the effective recurrent matrix by `factor` in place."""
        raise NotImplementedError

    def activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    # -- shared -----------------------------------------------------------

    def _set_spectral_radius(self, target: float) -> None:
        with torch.no_grad():
            radius = torch.linalg.eigvals(self.recurrent_weight()).abs().max().item()
            if radius > 1e-12:
                self._scale_recurrent(target / radius)

    def _leak(self, h_prev: torch.Tensor, pre_activation: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.alpha) * h_prev + self.alpha * self.activation(pre_activation)

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def readout_state(self, h_t: torch.Tensor) -> torch.Tensor:
        return h_t

    def n_units(self) -> int:
        return self.hidden_dim

    def effective_param_count(self) -> int:
        """Structural synapse count, same convention as the GRU and vanilla
        cells: every realized connection, counted once. The recurrent matrix
        is dense in all three cells -- a low-rank factorization constrains
        the values a connection can take, it does not remove the connection
        -- so it contributes `hidden_dim ** 2` regardless of how the weights
        are parameterized. Per-unit quantities (bias, time constants,
        release parameters) are not synapses and are excluded, as biases are
        elsewhere."""
        return self.weight_ih.numel() + self.hidden_dim ** 2

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, extra_update_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None]:
        """Returns `(h_t, None)`: the second slot is the gate activation the
        gated cores report, and these cores have no gate."""
        pre = x_t @ self.weight_ih.t() + h_prev @ self.recurrent_weight().t() + self.bias
        if extra_update_bias is not None:
            pre = pre + extra_update_bias
        return self._leak(h_prev, pre), None


class ExcitatoryInhibitoryRateCell(_LeakyRateCell):
    """Hard excitatory/inhibitory separation: each unit is assigned a fixed
    sign at construction and `W_rec = relu(weight_magnitude) * sign`, so a
    unit's outgoing weights can never change sign under any optimizer step
    -- a structural constraint, unlike the soft sign penalty applied to the
    gated cells' recurrent weights during training. The rate nonlinearity is
    softplus, so unit activity is non-negative and unbounded above rather
    than the gated cores' bounded, signed activity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        excitatory_fraction: float = 0.8,
        time_constant_ticks: float = 2.0,
        recurrent_init_spectral_radius: Optional[float] = None,
    ):
        if not 0.0 < excitatory_fraction < 1.0:
            raise ValueError(f"excitatory_fraction must be in (0, 1), got {excitatory_fraction}")
        self.excitatory_fraction = excitatory_fraction
        super().__init__(input_dim, hidden_dim, time_constant_ticks, recurrent_init_spectral_radius)

    def _init_recurrent(self) -> None:
        n_excitatory = int(round(self.excitatory_fraction * self.hidden_dim))
        sign = torch.full((self.hidden_dim,), -1.0)
        sign[:n_excitatory] = 1.0
        self.register_buffer("sign", sign)
        self.n_excitatory = n_excitatory
        # Non-negative magnitudes, so no synapse starts on the flat side of
        # the rectifier where it would receive no gradient.
        self.weight_magnitude = nn.Parameter(
            torch.empty(self.hidden_dim, self.hidden_dim).uniform_(0.0, 2.0 / self.hidden_dim ** 0.5)
        )

    def recurrent_weight(self) -> torch.Tensor:
        return F.relu(self.weight_magnitude) * self.sign

    def _scale_recurrent(self, factor: float) -> None:
        self.weight_magnitude.mul_(factor)

    def activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return F.softplus(pre_activation)


class DynamicSynapseRateCell(_LeakyRateCell):
    """Recurrent synapses with short-term facilitation and depression
    (Tsodyks-Markram kinetics, one facilitation variable `u` and one
    available-resources variable `x` per PRESYNAPTIC unit):

        transmitted_j = u_j * x_j * h_j
        u_j <- u_j + (tick_ms / tau_facilitation_j) (U_j - u_j) + U_j (1 - u_j) r_j
        x_j <- x_j + (tick_ms / tau_depression_j) (1 - x_j) - u_j x_j r_j

    with `r_j = max(h_j, 0)` the release-driving rate (only positive
    activity drives release), expressed in release events per tick -- so the
    recovery terms carry the tick-to-time-constant ratio and the release
    terms do not. Half the units carry facilitation-dominant kinetics and
    half depression-dominant, so both directions are present in the
    population. Both variables are clamped to [0, 1]; the clamp is a guard
    against a forward Euler overshoot, not a mechanism, and `tick_ms` is
    required to be below every time constant.

    The synaptic state is per-trial, batched, and threaded through the
    unroll as its own state tensor -- it is never a learned parameter and
    never enters `effective_param_count`."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        tick_ms: float = 100.0,
        facilitating_fraction: float = 0.5,
        facilitating: Optional[dict] = None,
        depressing: Optional[dict] = None,
        time_constant_ticks: float = 2.0,
        recurrent_init_spectral_radius: Optional[float] = None,
    ):
        self.tick_ms = float(tick_ms)
        self.facilitating_fraction = float(facilitating_fraction)
        self._facilitating = facilitating or {
            "utilization": 0.15, "tau_facilitation_ms": 1500.0, "tau_depression_ms": 200.0,
        }
        self._depressing = depressing or {
            "utilization": 0.45, "tau_facilitation_ms": 200.0, "tau_depression_ms": 1500.0,
        }
        super().__init__(input_dim, hidden_dim, time_constant_ticks, recurrent_init_spectral_radius)

    def _init_recurrent(self) -> None:
        std = 1.0 / (self.hidden_dim ** 0.5)
        self.weight_hh = nn.Parameter(torch.empty(self.hidden_dim, self.hidden_dim).uniform_(-std, std))
        n_facilitating = int(round(self.facilitating_fraction * self.hidden_dim))
        for name, key in (
            ("utilization", "utilization"),
            ("decay_facilitation", "tau_facilitation_ms"),
            ("decay_depression", "tau_depression_ms"),
        ):
            values = torch.empty(self.hidden_dim)
            values[:n_facilitating] = self._facilitating[key]
            values[n_facilitating:] = self._depressing[key]
            if key != "utilization":
                values = self.tick_ms / values  # per-tick recovery rate
                if bool((values >= 1.0).any()):
                    raise ValueError(
                        f"tick_ms={self.tick_ms} is not below every {key} "
                        "-- the forward Euler synaptic update would be unstable"
                    )
            self.register_buffer(name, values)
        self.n_facilitating = n_facilitating

    def recurrent_weight(self) -> torch.Tensor:
        """The effective matrix at the initial synaptic state (`u = U`,
        `x = 1`), which is what the initial spectral radius should describe."""
        return self.weight_hh * self.utilization

    def _scale_recurrent(self, factor: float) -> None:
        self.weight_hh.mul_(factor)

    def activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_activation)

    def init_synaptic_state(self, batch_size: int, device=None) -> torch.Tensor:
        """[B, 2, H] stacking (facilitation, available resources) at rest."""
        facilitation = self.utilization.to(device).expand(batch_size, self.hidden_dim)
        resources = torch.ones(batch_size, self.hidden_dim, device=device)
        return torch.stack([facilitation, resources], dim=1).contiguous()

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        synaptic_prev: torch.Tensor,
        extra_update_bias: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        """Returns `(h_t, None, synaptic_t)`."""
        facilitation, resources = synaptic_prev.unbind(dim=1)
        transmitted = facilitation * resources * h_prev
        pre = x_t @ self.weight_ih.t() + transmitted @ self.weight_hh.t() + self.bias
        if extra_update_bias is not None:
            pre = pre + extra_update_bias
        h_t = self._leak(h_prev, pre)

        release_rate = h_prev.clamp_min(0.0)
        facilitation_t = (
            facilitation
            + self.decay_facilitation * (self.utilization - facilitation)
            + self.utilization * (1.0 - facilitation) * release_rate
        ).clamp(0.0, 1.0)
        resources_t = (
            resources
            + self.decay_depression * (1.0 - resources)
            - facilitation_t * resources * release_rate
        ).clamp(0.0, 1.0)
        return h_t, None, torch.stack([facilitation_t, resources_t], dim=1)


class LowRankRateCell(_LeakyRateCell):
    """Recurrent connectivity constrained to rank R:
    `W_rec = left @ right.T / hidden_dim`, so the recurrent contribution to
    every unit is a linear combination of R population-level modes."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rank: int = 16,
        time_constant_ticks: float = 2.0,
        recurrent_init_spectral_radius: Optional[float] = None,
    ):
        if not 1 <= rank <= hidden_dim:
            raise ValueError(f"rank must be in [1, {hidden_dim}], got {rank}")
        self.rank = int(rank)
        super().__init__(input_dim, hidden_dim, time_constant_ticks, recurrent_init_spectral_radius)

    def _init_recurrent(self) -> None:
        self.left = nn.Parameter(torch.randn(self.hidden_dim, self.rank))
        self.right = nn.Parameter(torch.randn(self.hidden_dim, self.rank))

    def recurrent_weight(self) -> torch.Tensor:
        return self.left @ self.right.t() / self.hidden_dim

    def _scale_recurrent(self, factor: float) -> None:
        self.left.mul_(factor)

    def activation(self, pre_activation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_activation)


RATE_SUBSTRATES = {
    "excitatory_inhibitory": ExcitatoryInhibitoryRateCell,
    "dynamic_synapse": DynamicSynapseRateCell,
    "low_rank": LowRankRateCell,
}


def build_rate_cell(substrate: str, input_dim: int, cfg: dict) -> _LeakyRateCell:
    """Constructs one of `RATE_SUBSTRATES` from the `model.rate_rnn` config
    block, passing only the keys that substrate takes."""
    shared = {
        "hidden_dim": int(cfg["units"]),
        "time_constant_ticks": float(cfg["time_constant_ticks"]),
        "recurrent_init_spectral_radius": cfg.get("recurrent_init_spectral_radius"),
    }
    if substrate == "excitatory_inhibitory":
        extra = {"excitatory_fraction": float(cfg["excitatory_fraction"])}
    elif substrate == "dynamic_synapse":
        extra = {
            "tick_ms": float(cfg["tick_ms"]),
            "facilitating_fraction": float(cfg["facilitating_fraction"]),
            "facilitating": dict(cfg["facilitating"]),
            "depressing": dict(cfg["depressing"]),
        }
    elif substrate == "low_rank":
        extra = {"rank": int(cfg["rank"])}
    else:
        raise ValueError(f"unknown rate substrate {substrate!r}; expected one of {sorted(RATE_SUBSTRATES)}")
    return RATE_SUBSTRATES[substrate](input_dim=input_dim, **shared, **extra)
