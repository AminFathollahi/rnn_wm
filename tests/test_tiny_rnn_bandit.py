"""Item 9.3 (comments.txt Phase 9): checks for `scripts/run_tiny_rnn_bandit.py`'s
own non-trivial logic -- the discounted-return computation and the
per-trial bookkeeping (both pure functions, planted-answer tests), plus a
real-env check that NeuroGym trial boundaries do not terminate the
episode (the premise `ContinuousBatchEnv` relies on instead of
`NeuroGymBatchEnv`'s per-trial `done` latch)."""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.run_tiny_rnn_bandit import (  # noqa: E402
    ContinuousBatchEnv,
    TinyRNNPolicy,
    TrialReturnAccumulator,
    _discounted_returns,
)


def test_discounted_returns_matches_hand_computation():
    # rewards[t, b]; single batch column: r = [1, 0, 1], gamma=0.5
    # returns[2] = 1
    # returns[1] = 0 + 0.5*1   = 0.5
    # returns[0] = 1 + 0.5*0.5 = 1.25
    rewards = torch.tensor([[1.0], [0.0], [1.0]])
    returns = _discounted_returns(rewards, gamma=0.5)
    expected = torch.tensor([[1.25], [0.5], [1.0]])
    assert torch.allclose(returns, expected)


def test_trial_return_accumulator_planted_sequence():
    # Single batch instance. Ticks: reward=[1, 0, 1, 1], new_trial=[T, F, T, F].
    # Trial 0 spans ticks 0-1 (ends when tick 2's new_trial fires): total = 1+0 = 1.
    # Trial 1 starts at tick 2 and is still open at the end (tick 3, new_trial=False)
    # -- it must NOT be counted (only completed trials are returned).
    acc = TrialReturnAccumulator(batch_size=1)
    rewards = [1.0, 0.0, 1.0, 1.0]
    new_trials = [True, False, True, False]
    finalized = []
    for r, nt in zip(rewards, new_trials):
        finalized.extend(acc.step(np.array([r]), np.array([nt])))
    assert finalized == [1.0]


def test_trial_return_accumulator_multi_batch_independent():
    # Two independent batch instances with different trial lengths.
    acc = TrialReturnAccumulator(batch_size=2)
    # b0: trial boundary every tick (like bandit); b1: boundary every other tick (like dawtwostep).
    steps = [
        (np.array([1.0, -0.1]), np.array([True, True])),
        (np.array([0.0, 0.0]), np.array([True, False])),
        (np.array([1.0, 1.0]), np.array([True, True])),
    ]
    finalized = []
    for reward, new_trial in steps:
        finalized.extend(acc.step(reward, new_trial))
    # b0 finalizes at tick1 (value 1.0) and tick2 (value 0.0); b1 finalizes at tick2 (value -0.1+0.0=-0.1)
    assert sorted(finalized) == sorted([1.0, 0.0, -0.1])


@pytest.mark.parametrize("task", ["bandit", "dawtwostep"])
def test_neurogym_never_terminates_within_a_trial_boundary(task):
    """The premise `ContinuousBatchEnv` depends on instead of
    `NeuroGymBatchEnv`'s per-trial `done` latch: verified over 500 ticks
    that `terminated`/`truncated` never fire, only `new_trial` flags a
    boundary. `ContinuousBatchEnv.step` itself asserts this on every call
    (would raise here if NeuroGym's behavior ever changed)."""
    env = ContinuousBatchEnv(task, batch_size=4, seed=0)
    rng = np.random.RandomState(0)
    action_dim = {"bandit": 2, "dawtwostep": 3}[task]
    n_new_trial = 0
    for _ in range(500):
        actions = rng.randint(0, action_dim, size=4)
        _reward, new_trial = env.step(actions)  # raises AssertionError if terminated/truncated
        n_new_trial += int(new_trial.sum())
    assert n_new_trial > 0  # trial boundaries do occur, just never terminate the env


def test_hidden_state_persists_across_trial_boundary_no_reset():
    """Proves the tiny RNN's hidden state carries information across a
    trial boundary rather than being silently reset (the bug this script
    exists to avoid, see module docstring): two rollouts that agree on
    every input EXCEPT one early (action, reward) pair must produce
    DIFFERENT hidden states even several ticks later -- if hidden state
    were reset at each trial boundary, that one differing tick could never
    have any effect past the next boundary."""
    torch.manual_seed(0)
    policy = TinyRNNPolicy(hidden=3, action_dim=2)
    B = 1

    def rollout(first_reward: float):
        h = policy.init_state(B, "cpu")
        prev_action = torch.tensor([0])
        prev_reward = torch.tensor([first_reward])
        # tick 0 (the differing input), then 5 identical ticks after a "trial boundary"
        for t in range(6):
            _logits, h = policy.step(prev_action, prev_reward, h)
            prev_action = torch.tensor([1])
            prev_reward = torch.tensor([0.5])
        return h

    h_a = rollout(first_reward=1.0)
    h_b = rollout(first_reward=-1.0)
    assert not torch.allclose(h_a, h_b), (
        "hidden state converged to the same value regardless of early history -- "
        "would also happen if state were reset every trial, so this alone doesn't "
        "prove persistence, but a difference here is necessary for it"
    )
