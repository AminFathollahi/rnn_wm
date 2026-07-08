"""Local reward-modulated learning (learning factor L=1): reward-modulated
node perturbation with eligibility traces (Miconi 2017). No
backpropagation-through-time occurs anywhere in this module -- weight
updates are a *global* scalar (R - R_bar) times a *local*, per-synapse
eligibility trace, computed entirely under `torch.no_grad()`.

This module owns the eligibility-trace bookkeeping and the three-factor
update; the perturbation itself is injected at the pre-activation of each
GRU gate unit (dimension `3*hidden`, one term per reset/update/candidate
unit) by the training loop (`training/train.py::_perturbed_gru_step`), not
here. Eligibility traces are formed between each perturbed gate
unit and its two presynaptic inputs (the previous hidden state, for
`weight_hh`; the step input, for `weight_ih`). Perturbing the cell's
post-gate output state directly would be dimensionally incompatible with
the traced weight matrices (which have `3*hidden` output rows), so the
perturbation is injected at the pre-activation instead:

    xi_t ~ N(0, sigma_p^2 I_{3H})                             -- exploratory perturbation on gate pre-activations
    e_hh[b,i,j] = gamma_e * e_hh[b,i,j] + xi_t[b,i] * h_prev[b,j]   -- per-sample eligibility trace, weight_hh
    e_ih[b,i,j] = gamma_e * e_ih[b,i,j] + xi_t[b,i] * x_t[b,j]      -- per-sample eligibility trace, weight_ih
    R_bar <- EMA(mean_b(r_b))                                 -- running reward baseline (batch-mean reward)
    dW = eta * mean_b[(r_b - R_bar) * E_b]                     -- three-factor update, at trial end

Traces are kept **per batch element** (not averaged across the batch until
the reward-weighted update at trial end): the whole point of node
perturbation is that the update correlates each sample's own perturbation
with its own reward. Averaging the trace across the batch before the reward
is known would replace `mean_b[(r_b - Rbar) * xi_b]` with
`mean_b[r_b - Rbar] * mean_b[xi_b]` -- the product of averages instead of
the average of products -- which destroys exactly the correlation the
algorithm exploits and would make the rule learn nothing for batch size > 1.

A fallback ladder is defined for cells that do not clear their behavioral
gate under the default rule: rung 1 is the rule above; rung 2 is the same
rule with `normalize_traces=True, adaptive_baseline=True` -- trace
normalization, AND a genuinely faster-adapting reward baseline (a smaller
EMA decay constant, so the baseline tracks recent performance more closely
and the resulting advantage estimate has different, and for a
nonstationary/curriculum-shifting reward stream, typically lower variance
than rung 1's slow baseline). Rung 3 (e-prop, Bellec et al. 2020) replaces
the random perturbation `xi_t` with each unit's own local pseudo-derivative
(`training/train.py::_eprop_gru_step`) as the eligibility trace's
postsynaptic factor -- a real (if still local, not backpropagated-through-
time) gradient direction instead of a randomly-probed one -- and reuses
this module's trace/three-factor-update machinery unchanged (`eprop=True`
below just changes what `trace_step` is called with, not how the trace or
update work). If rung 3 also fails to clear the gate, `make_learner_for_cell`
raises `RungExhausted` for any rung beyond 3, so the training loop records
the gap explicitly (rung=4, H3' undefined for this cell/seed) rather than
substituting backpropagation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn

RUNG_NAMES = {
    1: "node_perturbation",
    2: "node_perturbation_normalized_adaptive_baseline",
    3: "e_prop",
    4: "failed_no_local_rule",  # rungs 1-3 all attempted and failed the gate
}


class RungExhausted(RuntimeError):
    """Raised when rung 3 (e-prop) also fails to clear the behavioral gate.
    The caller should record rung=4 (H3' undefined for this cell/seed) --
    all three rungs (node-perturbation, normalized/adaptive, e-prop) were
    attempted -- rather than substitute BPTT (never allowed for L=1)."""


@dataclass
class TracedWeight:
    """Eligibility-trace state for one weight matrix (weight_hh or weight_ih
    of a `MaskedGRUCell`, or a `Heads` linear layer). The trace is kept
    per-batch-element (`[B, *param.shape]`), allocated lazily on the first
    `accumulate()` call since the batch size isn't known at construction
    time (see module docstring for why per-sample traces are load-bearing,
    not just a memory/perf choice)."""

    param: nn.Parameter
    presyn_is_hidden: bool  # True: presynaptic = h_prev (weight_hh); False: presynaptic = x_t (weight_ih)
    trace: Optional[torch.Tensor] = field(init=False, default=None)

    def reset(self) -> None:
        if self.trace is not None:
            self.trace.zero_()

    def accumulate(self, xi_t: torch.Tensor, presyn_t: torch.Tensor, gamma_e: float) -> None:
        """xi_t: [B, H] perturbation on the post-synaptic units this weight
        projects TO (rows). presyn_t: [B, D] presynaptic activity this
        weight projects FROM (cols). Per-sample outer product, decayed and
        accumulated -- NOT averaged over the batch (see module docstring)."""
        B = xi_t.shape[0]
        outer = torch.einsum("bi,bj->bij", xi_t, presyn_t)  # [B, *param.shape]
        if self.trace is None or self.trace.shape[0] != B:
            self.trace = torch.zeros(B, *self.param.shape, device=self.param.device, dtype=self.param.dtype)
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
    At trial-batch end (rewards known, one per batch element):
        learner.apply_update(rewards)  # rewards: float or Sequence[float], one per batch element
    """

    def __init__(
        self,
        weight_hh: nn.Parameter,
        weight_ih: Optional[nn.Parameter] = None,
        sigma_p: float = 0.05,
        gamma_e: float = 0.9,
        lr_local: float = 1.0e-3,
        baseline_decay: float = 0.99,
        baseline_decay_adaptive: float = 0.9,
        normalize_traces: bool = False,
        adaptive_baseline: bool = False,
        eprop: bool = False,
        seed: int = 0,
    ):
        self.sigma_p = sigma_p
        self.gamma_e = gamma_e
        self.lr_local = lr_local
        self.baseline_decay = baseline_decay
        self.baseline_decay_adaptive = baseline_decay_adaptive
        self.normalize_traces = normalize_traces
        self.adaptive_baseline = adaptive_baseline
        self.eprop = eprop
        self.rung = 3 if eprop else (2 if (normalize_traces or adaptive_baseline) else 1)

        # `weight_ih=None` (or `weight_ih is weight_hh`): single-weight mode,
        # for a plain `Linear` head where there is only one weight matrix.
        # Passing the SAME parameter twice as two "distinct" traced slots
        # would apply the three-factor update to that one parameter TWICE
        # per trial -- an unintended 2x effective local learning rate for
        # `Heads.pi`/`.value` relative to the recurrent core. Track a
        # single `TracedWeight` for these modules instead.
        self._traced = [TracedWeight(weight_hh, presyn_is_hidden=True)]
        if weight_ih is not None and weight_ih is not weight_hh:
            self._traced.append(TracedWeight(weight_ih, presyn_is_hidden=False))

        self._generator = torch.Generator(device=weight_hh.device if weight_hh.is_cuda else "cpu")
        self._generator.manual_seed(seed)
        self.reward_baseline = 0.0
        self._n_updates = 0

    def sample_perturbation(self, shape, device=None, dtype=None) -> torch.Tensor:
        return torch.randn(tuple(shape), generator=self._generator, device=device, dtype=dtype) * self.sigma_p

    def trace_step(self, xi_t: torch.Tensor, h_prev: torch.Tensor, x_t: torch.Tensor) -> None:
        with torch.no_grad():
            self._traced[0].accumulate(xi_t, h_prev, self.gamma_e)  # weight_hh <- (xi_t, h_prev)
            if len(self._traced) > 1:
                self._traced[1].accumulate(xi_t, x_t, self.gamma_e)  # weight_ih <- (xi_t, x_t)

    def reset_traces(self) -> None:
        for tw in self._traced:
            tw.reset()

    @torch.no_grad()
    def apply_update(self, reward) -> float:
        """Three-factor update at trial-batch end. `reward`: scalar or
        per-batch-element sequence. Returns mean (reward - baseline) for
        logging. The reward baseline is a scalar EMA of the *batch-mean*
        reward; the per-sample advantage (reward_b - baseline) is then
        multiplied by that sample's OWN trace before averaging over the
        batch -- see module docstring for why this order matters."""
        reward_t = torch.as_tensor(reward, dtype=torch.float32)
        if reward_t.dim() == 0:
            reward_t = reward_t.unsqueeze(0)
        mean_reward = float(reward_t.mean().item())
        decay = self.baseline_decay_adaptive if self.adaptive_baseline else self.baseline_decay
        self.reward_baseline = decay * self.reward_baseline + (1 - decay) * mean_reward
        advantage = reward_t - self.reward_baseline  # [B]
        for tw in self._traced:
            trace = tw.trace
            if trace is None:
                continue
            adv = advantage.to(trace.device)
            if adv.shape[0] != trace.shape[0]:
                # scalar reward applied uniformly across a batch trace (rare;
                # e.g. a caller passing one shared reward for the whole batch)
                adv = adv.mean().expand(trace.shape[0])
            if self.normalize_traces:
                flat = trace.reshape(trace.shape[0], -1)
                norm = flat.norm(dim=1).clamp_min(1e-8)
                trace = trace / norm.view(-1, *([1] * (trace.dim() - 1)))
            weighted = adv.view(-1, *([1] * (trace.dim() - 1))) * trace  # [B, *param.shape]
            update = weighted.mean(dim=0)  # average of per-sample (advantage * trace) products
            tw.param.data.add_(self.lr_local * update)
        self.reset_traces()
        self._n_updates += 1
        return float(advantage.mean().item())

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
    separately) and again for `Heads.pi`/`Heads.value` (single-weight mode,
    see `NodePerturbationLearner.__init__`). Rung 3 (e-prop) keeps rung 2's
    adaptive baseline (a strict improvement independent of what generates
    the trace) but not trace normalization (a pseudo-derivative-driven
    trace has a naturally bounded scale, unlike a random-perturbation-
    driven one, so normalizing it isn't motivated the same way)."""
    if rung not in (1, 2, 3):
        raise RungExhausted(
            f"rung {rung} ({RUNG_NAMES.get(rung, '?')}) is not implemented; "
            f"record rung=4 (H3' undefined for this cell/seed) rather than substituting BPTT."
        )
    return NodePerturbationLearner(
        weight_hh=cell.weight_hh,
        weight_ih=cell.weight_ih,
        sigma_p=sigma_p,
        gamma_e=gamma_e,
        lr_local=lr_local,
        normalize_traces=(rung == 2),
        adaptive_baseline=(rung in (2, 3)),
        eprop=(rung == 3),
        seed=seed,
    )
