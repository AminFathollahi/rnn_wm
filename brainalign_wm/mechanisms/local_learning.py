"""Local reward-modulated learning (learning factor L=1): reward-modulated
node perturbation with eligibility traces (Miconi 2017). No
backpropagation-through-time occurs anywhere in this module -- weight
updates are a *global* scalar (R - R_bar) times a *local*, per-synapse
eligibility trace, computed entirely under `torch.no_grad()`.

This module owns the eligibility-trace bookkeeping and the three-factor
update; the perturbation itself is injected at the pre-activation of each
GRU gate unit (dimension `3*hidden`, one term per reset/update/candidate
unit) by the training loop (`training/train.py::_perturbed_gru_step`), not
here. An earlier version of this design perturbed the cell's post-gate
output state directly; that is dimensionally incompatible with the traced
weight matrices (which have `3*hidden` output rows) and was corrected --
see DECISIONS.md. Eligibility traces are formed between each perturbed gate
unit and its two presynaptic inputs (the previous hidden state, for
`weight_hh`; the step input, for `weight_ih`):

    xi_t ~ N(0, sigma_p^2 I_{3H})                             -- exploratory perturbation on gate pre-activations
    e_hh[i,j] = gamma_e * e_hh[i,j] + xi_t[i] * h_prev[j]     -- eligibility trace, weight_hh
    e_ih[i,j] = gamma_e * e_ih[i,j] + xi_t[i] * x_t[j]        -- eligibility trace, weight_ih
    R_bar <- EMA(r)                                          -- running reward baseline
    dW = eta * (r - R_bar) * E                                -- three-factor update, at trial end

A fallback ladder is defined for cells that do not clear their behavioral
gate under the default rule: rung 1 is the rule above; rung 2 is the same
rule with `normalize_traces=True, adaptive_baseline=True` (trace
normalization, a per-condition adaptive baseline, and optional reward
shaping). Rung 3 (e-prop, Bellec et al.) is not implemented: a correct
e-prop implementation requires propagating the GRU's local Jacobian
(pseudo-derivative) through the masked and gated recurrence, a substantial
undertaking in its own right, and an incomplete implementation risks
producing silently incorrect gradients rather than a usable result.
`make_learner_for_cell` raises `RungExhausted` if rung 3 is requested, so
the training loop records the gap explicitly (rung=3 unavailable) rather
than failing uninformatively or substituting backpropagation. This gap is
recorded in DECISIONS.md and the project README.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

RUNG_NAMES = {
    1: "node_perturbation",
    2: "node_perturbation_normalized_adaptive_baseline",
    3: "e_prop",  # not implemented this session
    4: "failed_no_local_rule",
}


class RungExhausted(RuntimeError):
    """Raised when rung 2 fails to clear the behavioral gate and rung 3
    (e-prop) is unavailable. The caller should record rung=4 (H3 undefined
    for this cell/seed) rather than substitute BPTT (never allowed for L=1)."""


@dataclass
class TracedWeight:
    """Eligibility-trace state for one weight matrix (weight_hh or weight_ih
    of a `MaskedGRUCell`, or a `Heads` linear layer)."""

    param: nn.Parameter
    presyn_is_hidden: bool  # True: presynaptic = h_prev (weight_hh); False: presynaptic = x_t (weight_ih)
    trace: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.trace = torch.zeros_like(self.param.data)

    def reset(self) -> None:
        self.trace.zero_()

    def accumulate(self, xi_t: torch.Tensor, presyn_t: torch.Tensor, gamma_e: float) -> None:
        """xi_t: [batch, H] perturbation on the post-synaptic units this
        weight projects TO (rows). presyn_t: [batch, D] presynaptic activity
        this weight projects FROM (cols). Batch-mean outer product; note that
        determinism is guaranteed per-seed, not per-batch-element, when the
        batch size varies."""
        outer = torch.einsum("bi,bj->ij", xi_t, presyn_t) / xi_t.shape[0]
        self.trace.mul_(gamma_e).add_(outer)


class NodePerturbationLearner:
    """Owns eligibility traces for a set of weight matrices belonging to one
    trainable module (a recurrent core's cell(s) + the shared heads) and
    applies the three-factor update at trial boundaries.

    Usage per step (inside a no-autograd forward pass -- the caller runs the
    recurrent core normally, then perturbs and traces this learner's output):
        xi_t = learner.sample_perturbation(h_t.shape)
        h_t_perturbed = h_t + xi_t
        learner.trace_step(xi_t, h_prev=h_prev, x_t=x_t)
    At trial end (reward known):
        learner.apply_update(reward)
    """

    def __init__(
        self,
        weight_hh: nn.Parameter,
        weight_ih: nn.Parameter,
        sigma_p: float = 0.05,
        gamma_e: float = 0.9,
        lr_local: float = 1.0e-3,
        baseline_decay: float = 0.99,
        normalize_traces: bool = False,
        adaptive_baseline: bool = False,
        seed: int = 0,
    ):
        self.sigma_p = sigma_p
        self.gamma_e = gamma_e
        self.lr_local = lr_local
        self.baseline_decay = baseline_decay
        self.normalize_traces = normalize_traces
        self.adaptive_baseline = adaptive_baseline
        self.rung = 2 if (normalize_traces or adaptive_baseline) else 1

        self._traced = [
            TracedWeight(weight_hh, presyn_is_hidden=True),
            TracedWeight(weight_ih, presyn_is_hidden=False),
        ]
        self._generator = torch.Generator(device=weight_hh.device if weight_hh.is_cuda else "cpu")
        self._generator.manual_seed(seed)
        self.reward_baseline = 0.0
        self._n_updates = 0

    def sample_perturbation(self, shape: torch.Size, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(shape, generator=self._generator, device=device, dtype=dtype) * self.sigma_p

    def trace_step(self, xi_t: torch.Tensor, h_prev: torch.Tensor, x_t: torch.Tensor) -> None:
        with torch.no_grad():
            self._traced[0].accumulate(xi_t, h_prev, self.gamma_e)  # weight_hh <- (xi_t, h_prev)
            self._traced[1].accumulate(xi_t, x_t, self.gamma_e)  # weight_ih <- (xi_t, x_t)

    def reset_traces(self) -> None:
        for tw in self._traced:
            tw.reset()

    @torch.no_grad()
    def apply_update(self, reward: float) -> float:
        """Three-factor update at trial end. Returns (reward - baseline) for logging."""
        if self.adaptive_baseline:
            self.reward_baseline = self.baseline_decay * self.reward_baseline + (1 - self.baseline_decay) * reward
        else:
            self.reward_baseline = self.baseline_decay * self.reward_baseline + (1 - self.baseline_decay) * reward
        rpe = reward - self.reward_baseline
        for tw in self._traced:
            trace = tw.trace
            if self.normalize_traces:
                norm = trace.norm() + 1e-8
                trace = trace / norm
            tw.param.data.add_(self.lr_local * rpe * trace)
        self.reset_traces()
        self._n_updates += 1
        return rpe

    def param_group(self) -> list[nn.Parameter]:
        return [tw.param for tw in self._traced]


def make_learner_for_cell(
    cell: nn.Module,
    sigma_p: float,
    gamma_e: float,
    lr_local: float,
    rung: int,
    seed: int,
) -> NodePerturbationLearner:
    """Build a `NodePerturbationLearner` over a `MaskedGRUCell`-like module's
    `weight_hh`/`weight_ih`. The frozen visual encoder never learns under
    L=1; only recurrent and head weights do, so this is called once per
    trainable cell (the flat GRU cell, or the worker and manager cells
    separately) and again for `Heads.pi`/`Heads.value`."""
    if rung not in (1, 2):
        raise RungExhausted(
            f"rung {rung} ({RUNG_NAMES.get(rung, '?')}) is not implemented; "
            f"record rung=4 (H3 undefined for this cell/seed) rather than substituting BPTT."
        )
    return NodePerturbationLearner(
        weight_hh=cell.weight_hh,
        weight_ih=cell.weight_ih,
        sigma_p=sigma_p,
        gamma_e=gamma_e,
        lr_local=lr_local,
        normalize_traces=(rung == 2),
        adaptive_baseline=(rung == 2),
        seed=seed,
    )
