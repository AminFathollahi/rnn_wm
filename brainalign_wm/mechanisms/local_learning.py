"""Local reward-modulated learning (knob L=1), protocol §6.2.

Default rule: reward-modulated node perturbation with eligibility traces
(Miconi 2017). No BPTT anywhere in this module -- weight updates are a
*global* scalar (R - R_bar) times a *local*, per-synapse eligibility trace,
computed entirely under `torch.no_grad()`.

**Simplification, logged (protocol §0 rule 7 -- ambiguity resolved by
precedent/tractability):** Miconi's original rule perturbs a simple
continuous-time RNN's single pre-activation per unit. Our recurrent cores are
GRUs (3 internal gates per unit), for which "the" pre-activation to perturb
is not literally specified by the protocol. We perturb the cell's **output
activity** h_t directly (`h_t_perturbed = h_t + xi_t`, "activity/node
perturbation" -- a standard, simpler variant of node perturbation that sidesteps
picking one of the 3 GRU gate pre-activations, is agnostic to cell internals,
and composes cleanly with the masked worker / gated manager). Eligibility
traces are then formed between each perturbed unit and its two presynaptic
inputs (previous hidden state, for `weight_hh`; step input, for `weight_ih`).

    xi_t ~ N(0, sigma_p^2 I_H)                              -- exploratory node perturbation
    e_hh[i,j] = gamma_e * e_hh[i,j] + xi_t[i] * h_prev[j]     -- eligibility trace, weight_hh
    e_ih[i,j] = gamma_e * e_ih[i,j] + xi_t[i] * x_t[j]        -- eligibility trace, weight_ih
    R_bar <- EMA(r)                                          -- running reward baseline
    dW = eta * (r - R_bar) * E                                -- three-factor update, at trial end

Fallback ladder (§6.2): rung 1 = this rule as-is. Rung 2 = this rule with
`normalize_traces=True, adaptive_baseline=True` (trace normalization +
per-condition adaptive baseline + optional reward shaping). Rung 3 (e-prop,
Bellec et al.) is **not implemented this session** -- a real e-prop
implementation requires propagating the GRU's actual local Jacobian
(pseudo-derivative) through the masked/gated recurrence, which is a
substantial separate effort; attempting a rushed version risked silently
wrong gradients, which protocol §0 rule 6 explicitly forbids ("never
fabricate a gate pass"). `walk_fallback_ladder` raises `RungExhausted` after
rung 2 so the trainer (M7) can log rung=3-not-available honestly rather than
crash uninformatively. This gap is logged in DECISIONS.md and README.
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
        this weight projects FROM (cols). Batch-mean outer product (batched
        training; §11.4 determinism holds per-seed, not per-batch-element)."""
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
    `weight_hh`/`weight_ih` (protocol §6.2's "only recurrent + head weights
    learn" -- call once per trainable cell, e.g. flat GRU cell, worker cell,
    manager cell, and separately for `Heads.pi`/`Heads.value`)."""
    if rung not in (1, 2):
        raise RungExhausted(
            f"rung {rung} ({RUNG_NAMES.get(rung, '?')}) is not implemented this session; "
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
