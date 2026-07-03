"""Reflective gating (knob M=1), protocol §6.1.

Computes the reflection variable R_t (a leaky accumulator of rectified
surprise) and the additive bias it contributes to the manager's GRU
update-gate pre-activation. The manager cell itself (models/hrl.py,
`MaskedGRUCell`) is what actually consumes `gate_bias(R_t)` -- this module
owns only the surprise -> R_t -> bias computation, so it can be swapped out
(e.g. reflection-shuffle causal control, §6.1/F4) without touching the model.

    delta_t = surprise at step t:
        feedback steps: r_t - V_{t-1}            (reward-prediction error)
        non-feedback:   1 - p_t(chosen action)     (self-generated surprise)
    R_t = lambda_R * R_{t-1} + (1 - lambda_R) * phi(|delta_t|),  phi = softplus
    bias_t = beta * R_t   -->  added to u^m_t pre-activation (models/hrl.py)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveGate(nn.Module):
    def __init__(self, lambda_R: float = 0.9, beta: float = 1.0):
        super().__init__()
        self.lambda_R = lambda_R
        self.beta = beta

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, 1, device=device)

    def surprise(
        self,
        is_feedback: torch.Tensor,
        reward: torch.Tensor,
        value_prev: torch.Tensor,
        action_logp_chosen: torch.Tensor,
    ) -> torch.Tensor:
        """delta_t, protocol §6.1. All args are [batch] or [batch,1] tensors.
        `is_feedback`: 1.0 on feedback steps, else 0.0 (float mask, not bool,
        so this is differentiable-shape-compatible and branch-free)."""
        rpe = reward - value_prev
        self_surprise = 1.0 - torch.exp(action_logp_chosen)  # 1 - p_t(chosen)
        return is_feedback * rpe + (1.0 - is_feedback) * self_surprise

    def step(self, delta_t: torch.Tensor, R_prev: torch.Tensor) -> torch.Tensor:
        """R_t update. `delta_t`/`R_prev`: [batch, 1]."""
        phi = F.softplus(delta_t.abs())
        return self.lambda_R * R_prev + (1 - self.lambda_R) * phi

    def gate_bias(self, R_t: torch.Tensor) -> torch.Tensor:
        return self.beta * R_t


def shuffle_reflection(R_sequence: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Causal control (§6.1/F4): time-shuffle R_t within a trial to destroy its
    temporal alignment with epochs/load/lures, while preserving its marginal
    distribution. `R_sequence`: [T, batch, 1] (time-major)."""
    T = R_sequence.shape[0]
    perm = torch.randperm(T, generator=generator, device=R_sequence.device)
    return R_sequence[perm]
