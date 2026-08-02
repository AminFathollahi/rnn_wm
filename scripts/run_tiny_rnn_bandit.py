#!/usr/bin/env python3
"""Item 9.3 (comments.txt Phase 9): [JI-AN25]'s regime -- tiny (1-4 unit)
GRU-cell RNNs trained via REINFORCE directly on Phase 5's Bandit-v0 and
DawTwoStep-v0, with inputs restricted to (a_{t-1}, r_{t-1}) only (no
stimulus -- comments.txt's own explicit simplification vs. the paper,
which also conditions on s_{t-1}).

Does NOT reuse `NeuroGymBatchEnv` (`brainalign_wm/tasks/multitask.py`):
that wrapper marks an instance `done` the first time `new_trial` fires and
never steps it again (`if self.done[b]: continue`) -- correct for the
existing multi-task-diet training (one trial per batch instance) but wrong
here, since item 9.3 needs the tiny RNN's hidden state to persist across
MANY consecutive trials to learn cross-trial value tracking from
(a_{t-1}, r_{t-1}) alone. Verified empirically
(`scripts/verify_neurogym_semantics.py`) that Bandit-v0/DawTwoStep-v0 never
set terminated/truncated over 5000 ticks -- `new_trial=True` only flags a
trial boundary, the env auto-advances internally. `ContinuousBatchEnv`
below is a ~15-line stripped analog of `NeuroGymBatchEnv` that never
latches `done`.

Training: REINFORCE with discounted reward-to-go (gamma=0.95) over each
truncated-BPTT chunk, and a per-timestep batch-mean baseline for variance
reduction. (ponytail: short-horizon reward-to-go substitutes for exact
per-trial return bookkeeping across ragged trial boundaries during
training -- fine here since trials are ~1-2 ticks long, so gamma^{1,2}
barely discounts. Exact trial-boundary bookkeeping IS used for the
EVALUATION metric, `eval_trial_returns`, to stay comparable to
`run_multitask_diet.py::chance_reward`'s per-trial convention.)

DawTwoStep-v0 caveat: dropping s_{t-1} (per item 9.3's own spec) means the
tiny RNN cannot observe which second-stage state a trial is in when it
must pick a second-stage action -- a structural information gap, not a
capacity limit (same category as Sternberg's SCOPE LIMIT note). Expect
this to cap DawTwoStep performance well below what a stimulus-aware policy
could reach; that is not a tiny-RNN failure finding, see executor.md.

Usage:
  python scripts/run_tiny_rnn_bandit.py --hidden 1 2 3 4 --ticks 200000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # editable install's finder maps to a stale pre-rename path; see executor.md Phase 9b

from brainalign_wm.tasks.multitask import make_env  # noqa: E402
ACTION_DIM = {"bandit": 2, "dawtwostep": 3}


class ContinuousBatchEnv:
    """B independent NeuroGym instances of `task`, stepped forever -- no
    `done` latch; a trial boundary (`new_trial`) does not stop or reset
    anything (see module docstring)."""

    def __init__(self, task: str, batch_size: int, seed: int):
        self.task = task
        self.B = batch_size
        self.envs = [make_env(task) for _ in range(batch_size)]
        for b, env in enumerate(self.envs):
            env.reset(seed=seed + b)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reward = np.zeros(self.B, dtype=np.float32)
        new_trial = np.zeros(self.B, dtype=bool)
        for b, env in enumerate(self.envs):
            _obs, r, terminated, truncated, info = env.step(int(actions[b]))
            assert not terminated and not truncated, (
                f"{self.task} instance {b} terminated/truncated -- ContinuousBatchEnv "
                "assumes NeuroGym trials auto-advance forever (verified empirically, see "
                "verify_neurogym_semantics.py); an actual episode end here would silently "
                "break hidden-state continuity."
            )
            reward[b] = float(r)
            new_trial[b] = bool(info.get("new_trial", False))
        return reward, new_trial


class TinyRNNPolicy(nn.Module):
    """GRU-cell policy: input = one-hot(a_{t-1}) ++ [r_{t-1}], hidden = H
    units, output = logits over the task's native action space (no shared
    3-dim head space -- this is a standalone [JI-AN25] reproduction, not
    part of the multi-task net)."""

    def __init__(self, hidden: int, action_dim: int):
        super().__init__()
        self.hidden = hidden
        self.action_dim = action_dim
        self.cell = nn.GRUCell(action_dim + 1, hidden)
        self.readout = nn.Linear(hidden, action_dim)

    def init_state(self, batch_size: int, device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden, device=device)

    def step(self, prev_action: torch.Tensor, prev_reward: torch.Tensor, h: torch.Tensor):
        onehot = torch.zeros(prev_action.shape[0], self.action_dim, device=h.device)
        valid = prev_action >= 0
        if valid.any():
            onehot[valid] = F.one_hot(prev_action[valid], self.action_dim).float()
        x = torch.cat([onehot, prev_reward.unsqueeze(-1)], dim=-1)
        h = self.cell(x, h)
        return self.readout(h), h


class TrialReturnAccumulator:
    """Sums reward per batch instance between `new_trial` boundaries,
    yielding a trial's total the tick its NEXT trial starts (the
    in-progress trial at the end of a rollout is discarded, not counted --
    same convention as `run_multitask_diet.py::chance_reward`'s per-trial
    mean, extended to a non-terminating rollout)."""

    def __init__(self, batch_size: int):
        self.trial_return = np.zeros(batch_size, dtype=np.float64)
        self.started = np.zeros(batch_size, dtype=bool)

    def step(self, reward: np.ndarray, new_trial: np.ndarray) -> list[float]:
        finalized: list[float] = []
        for b in range(len(reward)):
            if new_trial[b] and self.started[b]:
                finalized.append(float(self.trial_return[b]))
                self.trial_return[b] = 0.0
            self.trial_return[b] += reward[b]
            self.started[b] = True
        return finalized


def _discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """rewards: [T, B] -> returns[t] = sum_{t'>=t} gamma^{t'-t} * rewards[t']."""
    returns = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train_tiny_rnn(task: str, hidden: int, n_ticks: int, batch_size: int, seed: int, chunk: int = 20,
                    gamma: float = 0.95, lr: float = 3e-3, device: str = "cpu") -> tuple[nn.Module, list[float]]:
    torch.manual_seed(seed)
    action_dim = ACTION_DIM[task]
    policy = TinyRNNPolicy(hidden, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    env = ContinuousBatchEnv(task, batch_size, seed)
    h = policy.init_state(batch_size, device)
    prev_action = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    prev_reward = torch.zeros(batch_size, device=device)
    tick_rewards: list[float] = []

    for _ in range(n_ticks // chunk):
        logps, rewards = [], []
        for _t in range(chunk):
            logits, h = policy.step(prev_action, prev_reward, h)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logps.append(dist.log_prob(action))
            reward, _new_trial = env.step(action.cpu().numpy())
            reward_t = torch.tensor(reward, device=device)
            rewards.append(reward_t)
            tick_rewards.append(float(reward_t.mean()))
            prev_action, prev_reward = action.detach(), reward_t.detach()
        logps_t = torch.stack(logps)
        rewards_t = torch.stack(rewards)
        returns = _discounted_returns(rewards_t, gamma)
        advantage = returns - returns.mean(dim=1, keepdim=True)
        loss = -(logps_t * advantage.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        h = h.detach()
    return policy, tick_rewards


@torch.no_grad()
def eval_trial_returns(policy: nn.Module | None, task: str, n_ticks: int, batch_size: int, seed: int,
                        device: str = "cpu") -> list[float]:
    """Mean-per-trial total reward, using `new_trial` boundaries to sum
    each trial's ticks -- same convention as
    `run_multitask_diet.py::chance_reward` (mean reward per trial),
    extended to a continuous (non-terminating) rollout. `policy=None` runs
    a uniform-random policy (the chance baseline)."""
    env = ContinuousBatchEnv(task, batch_size, seed)
    action_dim = ACTION_DIM[task]
    h = policy.init_state(batch_size, device) if policy is not None else None
    prev_action = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    prev_reward = torch.zeros(batch_size, device=device)
    rng = np.random.RandomState(seed + 777)
    acc = TrialReturnAccumulator(batch_size)
    completed_returns: list[float] = []

    for _ in range(n_ticks):
        if policy is None:
            action_np = rng.randint(0, action_dim, size=batch_size)
        else:
            logits, h = policy.step(prev_action, prev_reward, h)
            action_np = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        reward, new_trial = env.step(action_np)
        completed_returns.extend(acc.step(reward, new_trial))
        prev_action = torch.tensor(action_np, dtype=torch.long, device=device)
        prev_reward = torch.tensor(reward, dtype=torch.float32, device=device)
    return completed_returns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tasks", type=str, nargs="+", default=["bandit", "dawtwostep"])
    ap.add_argument("--ticks", type=int, default=200_000, help="training ticks per (task, H)")
    ap.add_argument("--eval-ticks", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for task in args.tasks:
        chance_returns = eval_trial_returns(None, task, args.eval_ticks, args.batch_size, args.seed, device)
        chance_mean = float(np.mean(chance_returns))
        for H in args.hidden:
            t0 = time.time()
            policy, tick_rewards = train_tiny_rnn(task, H, args.ticks, args.batch_size, args.seed, device=device)
            train_s = time.time() - t0
            trained_returns = eval_trial_returns(policy, task, args.eval_ticks, args.batch_size, args.seed + 1, device)
            trained_mean = float(np.mean(trained_returns))
            rec = {
                "task": task, "hidden": H, "n_train_ticks": args.ticks,
                "n_eval_trials": len(trained_returns), "trained_mean_trial_reward": trained_mean,
                "chance_mean_trial_reward": chance_mean, "train_wall_s": round(train_s, 1),
                "last_1000_tick_mean_reward": float(np.mean(tick_rewards[-1000:])),
            }
            results.append(rec)
            print(f"[tiny_rnn_bandit] task={task:10s} H={H} trained_mean_trial_reward={trained_mean:+.4f} "
                  f"chance={chance_mean:+.4f} wall_s={train_s:.1f}", flush=True)

    out_path = ROOT / "results" / "tiny_rnn_bandit.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[tiny_rnn_bandit] wrote {out_path}")
    print("\n=== task, H, trained_mean_trial_reward, chance_mean_trial_reward ===")
    for r in results:
        print(f"{r['task']:10s} H={r['hidden']} trained={r['trained_mean_trial_reward']:+.4f} "
              f"chance={r['chance_mean_trial_reward']:+.4f}")


if __name__ == "__main__":
    main()
