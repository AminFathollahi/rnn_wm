"""Multi-task diet (Phase 5, comments.txt items 5.1-5.5): the "trained on
general cognitive tasks vs working memory only" arm. Wraps 5 NeuroGym envs
plus the existing Sternberg generator behind one task-selection interface.

Unlike Sternberg (whose trials are fully pre-scripted -- the epoch/image
sequence is fixed in advance and the model's action only affects the
recorded outcome, never what's shown next), a NeuroGym trial is genuinely
INTERACTIVE: the agent's action can end the trial early (e.g. GoNogo ends
the instant a non-fixate action fires during the decision period) or leave
it running to a timeout ("miss"). So there is no `sample_batch() -> list[
list[Step]]` to pre-generate the way `TaskGenerator` does for Sternberg --
`NeuroGymBatchEnv` instead steps the model's own action against the
underlying env(s) tick by tick, which is why the training loop
(`training/train.py::run_multitask_neurogym_trial`) rolls out and computes
the loss in the same loop, rather than consuming a pre-built batch.

Gated entirely behind `task.multitask_diet` (default False, unset in every
existing config) -- nothing in Phases 0-4's Sternberg-only pipeline
imports this module or is affected by its presence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# Task order fixes the one-hot index below -- do not reorder without also
# bumping every checkpoint that used the old order (none exist yet, this
# arm is new in this phase).
DIET_TASKS = ["sternberg", "bandit", "dawtwostep", "delaymatchsample", "gonogo", "contextdecisionmaking"]
NEUROGYM_TASKS = DIET_TASKS[1:]

NEUROGYM_ENV_IDS = {
    "bandit": "Bandit-v0",
    "dawtwostep": "DawTwoStep-v0",
    "delaymatchsample": "DelayMatchSample-v0",
    "gonogo": "GoNogo-v0",
    "contextdecisionmaking": "ContextDecisionMaking-v0",
}
# Bandit-v0's default p=(0.5, 0.5) makes both arms equally rewarding -- there
# is nothing to learn (any policy gets the same expected reward), which
# would make "learns above chance" untestable. Skew it so the diet has an
# actual best arm.
NEUROGYM_ENV_KWARGS: dict[str, dict] = {
    "bandit": {"n": 2, "p": (0.1, 0.9)},
}

# The shared policy head is {0: no-action/fixate, 1: yes/A, 2: no/B}
# (item 5.5, action_dim=3, see configs/config.yaml). Each task's native
# Discrete(N) action space maps onto it; derived by reading each task's
# actual source in site-packages/neurogym/envs/native/*.py, not guessed:
#   dawtwostep            {0: fixate, 1: action1, 2: action2}      -- already 1:1
#   delaymatchsample      {0: fixation, 1: match, 2: non-match}    -- already 1:1
#   contextdecisionmaking {0: fixation, 1: choice1, 2: choice2}    -- already 1:1 (dim_ring=2)
#   gonogo                {0: fixation/no-go, 1: go}               -- only 2 native actions;
#                                                                     head 2 has no natural target, folds into 0
#   bandit                {0: arm0, 1: arm1}, no fixate action     -- head 0 (no-action) has no natural
#                                                                     target either; folds into arm0
_IDENTITY3 = {0: 0, 1: 1, 2: 2}
HEAD_TO_ENV = {
    "bandit": {0: 0, 1: 0, 2: 1},
    "dawtwostep": _IDENTITY3,
    "delaymatchsample": _IDENTITY3,
    "gonogo": {0: 0, 1: 1, 2: 0},
    "contextdecisionmaking": _IDENTITY3,
}
# Translates info["gt"] (env action space) into head-action space, for the
# 3 tasks that provide a gt at all.
ENV_GT_TO_HEAD = {
    "delaymatchsample": _IDENTITY3,
    "gonogo": {0: 0, 1: 1},
    "contextdecisionmaking": _IDENTITY3,
}
# Bandit/DawTwoStep are bandit-style (reward-only): NeuroGym gives no gt for
# them (verified empirically -- info["gt"] is always None), so there is no
# imitation target; trained via REINFORCE on the native reward instead.
HAS_GT = {"bandit": False, "dawtwostep": False, "delaymatchsample": True, "gonogo": True, "contextdecisionmaking": True}

# Phase 3/5 item 5.4: c_t currently (Sternberg-only, C_DIM=10) wastes 3 dims
# -- c_t[0] (WM_family) is always 1, c_t[8]/c_t[9] are permanently 0 (post
# F1 leak fix). Reclaimed here for a 6-task one-hot: 10 - 3 + 6 = 13. This
# is a SEPARATE schema from Sternberg's own C_DIM=10 (see
# `task_context_vector`'s docstring) -- Phases 0-4's tested single-task
# pipeline is untouched.
C_DIM_MULTITASK = 13
TASK_ACTION_DIM = 3  # matches models.heads.N_ACTIONS; every task maps onto this, no per-task head


def task_one_hot(task_name: str) -> list[float]:
    v = [0.0] * len(DIET_TASKS)
    v[DIET_TASKS.index(task_name)] = 1.0
    return v


def task_context_vector(task_name: str, sternberg_c: Optional[list[float]] = None) -> list[float]:
    """13-dim multi-task cue. dims[0:7] = Sternberg's own 10-dim
    `context_vector` with the 3 wasted bits (WM_family, the two reserved
    zeros) dropped: `[aux_family, load1, load2, load3, epoch_encode,
    epoch_maintain, epoch_probe]`. dims[7:13] = the task one-hot (item 5.4).
    A NeuroGym trial has no encode/maintain/probe epoch structure, so its
    first 7 dims are always zero -- only the task one-hot carries
    information for it. `sternberg_c` is the trial's own `TrialStep.c_t`
    (10-dim); pass `None` for a NeuroGym tick."""
    if sternberg_c is not None:
        base = [sternberg_c[1], sternberg_c[2], sternberg_c[3], sternberg_c[4], sternberg_c[5], sternberg_c[6], sternberg_c[7]]
    else:
        base = [0.0] * 7
    return base + task_one_hot(task_name)


class NeuroGymAdapter(nn.Module):
    """Per-task small learned linear embedding straight to bottleneck width
    (item 5.3) -- NOT routed through `models.front_end.FrontEnd` (that stays
    the Sternberg-only frozen-ResNet path; "the frozen-ResNet path is used
    only by Sternberg"). Mirrors `FrontEnd.forward`'s `(v_t, c_t) -> z_t`
    call shape so the training loop's dispatch on task family is exactly
    one branch (which adapter/front_end to call), not scattered per-task
    logic downstream -- the recurrent core and heads are unmodified and
    shared across every task (item 5.3)."""

    def __init__(self, obs_dim: int, c_dim: int, bottleneck_dim: int):
        super().__init__()
        self.w = nn.Linear(obs_dim + c_dim, bottleneck_dim)

    def forward(self, obs_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        return self.w(torch.cat([obs_t, c_t], dim=-1))


def make_env(task_name: str):
    import neurogym as ngym

    return ngym.make(NEUROGYM_ENV_IDS[task_name], **NEUROGYM_ENV_KWARGS.get(task_name, {}))


def obs_dim_for(task_name: str) -> int:
    return int(make_env(task_name).observation_space.shape[0])


class NeuroGymBatchEnv:
    """B independent instances of one NeuroGym task, stepped together.
    NeuroGym has no native vectorized-env support in the version pinned
    here, so this is a plain Python loop over B env objects per tick --
    the cost is inherent to single-instance-stepping, not something
    Phase 2's throughput fix (Sternberg-only) could have addressed.

    A trial's natural length varies per instance and depends on the
    model's OWN actions (see module docstring), so this does not
    pre-generate a fixed-length batch: `step()` takes the model's
    head-space actions for THIS tick and returns the next observation,
    each instance's native reward, its `gt` in head-space (only for tasks
    in `HAS_GT`), and which instances just finished their trial. Once an
    instance is done, further `step()` calls hold its last observation and
    return zero reward for it (masked out by the caller's `active` flag,
    same convention either way)."""

    def __init__(self, task_name: str, batch_size: int, seed: int):
        self.task = task_name
        self.B = batch_size
        self.envs = [make_env(task_name) for _ in range(batch_size)]
        self.done = np.zeros(batch_size, dtype=bool)
        self.obs = np.zeros((batch_size, obs_dim_for(task_name)), dtype=np.float32)
        for b, env in enumerate(self.envs):
            obs, _info = env.reset(seed=seed + b)
            self.obs[b] = obs

    def step(self, head_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
        """`head_actions`: [B] int array in {0,1,2}. Returns (obs [B,obs_dim],
        reward [B] float32, gt_head [B] int or None if this task has no gt,
        newly_done [B] bool -- True for instances whose trial ends on THIS
        tick)."""
        head_to_env = HEAD_TO_ENV[self.task]
        env_gt_to_head = ENV_GT_TO_HEAD.get(self.task)
        reward = np.zeros(self.B, dtype=np.float32)
        gt_head = np.zeros(self.B, dtype=np.int64) if env_gt_to_head is not None else None
        newly_done = np.zeros(self.B, dtype=bool)
        for b, env in enumerate(self.envs):
            if self.done[b]:
                continue
            env_action = head_to_env[int(head_actions[b])]
            obs, r, terminated, truncated, info = env.step(env_action)
            self.obs[b] = obs
            reward[b] = float(r)
            if env_gt_to_head is not None:
                gt_head[b] = env_gt_to_head[int(info["gt"])]
            if info.get("new_trial", False) or terminated or truncated:
                self.done[b] = True
                newly_done[b] = True
        return self.obs.copy(), reward, gt_head, newly_done


def pick_task(seed: int, step_idx: int, tasks: list[str] = DIET_TASKS) -> str:
    """Uniform draw over `tasks`, deterministic given (seed, step_idx) --
    same determinism contract as `TaskGenerator.sample_trial`/`sample_batch`."""
    seed_state = np.random.SeedSequence([seed, step_idx, 0xD1E7]).generate_state(4)
    rng = np.random.RandomState(seed_state)
    return str(rng.choice(tasks))
