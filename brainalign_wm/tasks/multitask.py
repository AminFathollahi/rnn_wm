"""The six-task training diet: the image Sternberg generator plus five
NeuroGym environments, behind one task-selection interface.

Unlike Sternberg (whose trials are fully pre-scripted -- the epoch/image
sequence is fixed in advance and the model's action only affects the
recorded outcome, never what is shown next), a NeuroGym trial is
INTERACTIVE: the agent's action can end the trial early (GoNogo ends the
instant a non-fixate action fires during the decision period) or leave it
running to a timeout. So there is no `sample_batch() -> list[list[Step]]`
to pre-generate the way `TaskGenerator` does for Sternberg --
`NeuroGymBatchEnv` instead steps the model's own action against the
underlying environments tick by tick, which is why
`training/train.py::run_multitask_neurogym_trial` rolls out and computes
the loss in the same loop rather than consuming a pre-built batch.

Gated behind `task.multitask_diet` (default False): nothing in the
Sternberg-only pipeline imports this module or is affected by its presence.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

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
# (`model.action_dim` = 3). Each task's native Discrete(N) action space
# maps onto it; derived by reading each task's actual source in
# site-packages/neurogym/envs/native/*.py, not guessed:
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
# 3 tasks that provide a gt at all. NeuroGym sets info["gt"] from the
# within-trial time index BEFORE advancing it, so the target returned by
# `step(a)` belongs to the observation `a` was chosen from -- verified by
# scripts/verify_neurogym_semantics.py.
ENV_GT_TO_HEAD = {
    "delaymatchsample": _IDENTITY3,
    "gonogo": {0: 0, 1: 1},
    "contextdecisionmaking": _IDENTITY3,
}
# Bandit/DawTwoStep are bandit-style (reward-only): NeuroGym gives no gt for
# them (verified empirically -- info["gt"] is always None), so there is no
# imitation target; trained via REINFORCE on the native reward instead.
HAS_GT = {"bandit": False, "dawtwostep": False, "delaymatchsample": True, "gonogo": True, "contextdecisionmaking": True}

TASK_CUE_SCHEMA = "task_code_10_v1"
LEGACY_TASK_CUE_SCHEMA = "task_one_hot_13_v1"
C_DIM_MULTITASK = 10
TASK_CODES = {
    "sternberg": (1.0, 0.0, 0.0),
    "bandit": (0.0, 1.0, 0.0),
    "dawtwostep": (0.0, 0.0, 1.0),
    "delaymatchsample": (1.0, 1.0, 0.0),
    "gonogo": (1.0, 0.0, 1.0),
    "contextdecisionmaking": (0.0, 1.0, 1.0),
}
TASK_ACTION_DIM = 3  # matches models.heads.N_ACTIONS; every task maps onto this, no per-task head


def task_one_hot(task_name: str) -> list[float]:
    v = [0.0] * len(DIET_TASKS)
    v[DIET_TASKS.index(task_name)] = 1.0
    return v


def task_context_vector(
    task_name: str,
    sternberg_c: Optional[list[float]] = None,
    schema: str = TASK_CUE_SCHEMA,
) -> list[float]:
    """Encode task identity without exposing an action or future outcome."""
    if schema == LEGACY_TASK_CUE_SCHEMA:
        base = [sternberg_c[i] for i in range(1, 8)] if sternberg_c is not None else [0.0] * 7
        return base + task_one_hot(task_name)
    if schema != TASK_CUE_SCHEMA:
        raise ValueError(f"unknown task cue schema {schema!r}")
    if task_name == "sternberg" and sternberg_c is not None:
        return list(sternberg_c)
    cue = [0.0] * C_DIM_MULTITASK
    cue[0], cue[8], cue[9] = TASK_CODES[task_name]
    return cue


def task_cue_metadata(schema: str = TASK_CUE_SCHEMA) -> dict:
    """Describe the cue schema stored with a continuation checkpoint."""
    return {"schema": schema, "tasks": list(DIET_TASKS), "codes": {name: list(code) for name, code in TASK_CODES.items()}}


class NeuroGymAdapter(nn.Module):
    """Per-task learned linear embedding of (observation, task cue) straight
    to bottleneck width -- NOT routed through `models.front_end.FrontEnd`,
    which stays the Sternberg-only frozen-encoder path. Mirrors
    `FrontEnd.forward`'s `(v_t, c_t) -> z_t` call shape so the training
    loop's dispatch on task family is exactly one branch (which adapter to
    call) and nothing downstream is per-task: the recurrent core and the
    heads are shared, unmodified, across every task."""

    def __init__(self, obs_dim: int, c_dim: int, bottleneck_dim: int):
        super().__init__()
        self.w = nn.Linear(obs_dim + c_dim, bottleneck_dim)

    def forward(self, obs_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        return self.w(torch.cat([obs_t, c_t], dim=-1))


def make_env(task_name: str):
    import neurogym as ngym

    return ngym.make(NEUROGYM_ENV_IDS[task_name], **NEUROGYM_ENV_KWARGS.get(task_name, {}))


def reset_env(env, seed: int, fixate_action: int = 0, max_advance: int = 200):
    """Reset a NeuroGym environment reproducibly and at a trial boundary.

    Two corrections to a bare `env.reset(seed=...)`:

    Seeding. `reset` seeds only the Gymnasium-side generator; NeuroGym
    draws trial timing, stimulus noise and stochastic reward from its own
    `RandomState`, which is constructed unseeded and left untouched by
    `reset`. Without the explicit `seed()` below, two runs at the same seed
    see different trials (measured: stimulus noise in DelayMatchSample,
    reward draws in Bandit), so nothing built on these environments is
    reproducible or resumable. Seeding must precede `reset`, which already
    generates the first trial.

    Trial boundary. `reset` itself takes one step with a sampled action, so
    it returns the trial's SECOND observation and the caller starts one
    tick in. For DawTwoStep that is fatal rather than cosmetic: its trial
    is two ticks, so the sampled action makes the first-stage choice and
    banks its reward, leaving the caller a second stage whose best
    available outcome is zero. Stepping the fixation action to the next
    trial boundary hands back the first observation of a fresh trial."""
    seed %= 2 ** 32  # the task generators seed a 32-bit RNG
    env.unwrapped.seed(seed)
    obs, info = env.reset(seed=seed)
    for _ in range(max_advance):
        if env.unwrapped.t_ind == 0:
            return obs, info
        obs, _reward, _terminated, _truncated, info = env.step(fixate_action)
    raise RuntimeError(f"{env} did not reach a trial boundary within {max_advance} ticks")


def obs_dim_for(task_name: str) -> int:
    return int(make_env(task_name).observation_space.shape[0])


class NeuroGymBatchEnv:
    """B independent instances of one NeuroGym task, stepped together.
    NeuroGym has no native vectorized-env support in the version pinned
    here, so this is a plain Python loop over B env objects per tick.

    Instance `b` is seeded `seed + b`, so consecutive callers must advance
    `seed` by at least `batch_size` to draw disjoint trials.

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
            obs, _info = reset_env(env, seed + b)
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


# Yang's 20-task suite via `neurogym.envs.collections.yang19`. Unlike the
# 6-task diet above, every yang19 env shares the SAME native action space
# (verified empirically:
# `Discrete(17)`, obs_dim 33 -- one fixation action plus a 16-direction
# response ring, and every task provides a `gt` in that same space, so
# there is no HEAD_TO_ENV/ENV_GT_TO_HEAD table to hand-derive per task the
# way the 6-task diet needed -- the identity mapping below is exact, not a
# compromise). This is dimensionally incompatible with the 6-task diet's
# shared `N_ACTIONS=3` head (`models/heads.py`), so Yang-19 runs use their
# own `Heads(n_actions=YANG19_ACTION_DIM)` instance -- the recurrent CORE
# under test is still exactly the same module, only the input
# adapter/output head (per-task-family plumbing, same convention the
# 6-task diet already uses) differ.
YANG19_TASKS = [
    "anti", "ctxdlydm1", "ctxdlydm2", "ctxdm1", "ctxdm2", "dlyanti", "dlydm1", "dlydm2",
    "dlygo", "dm1", "dm2", "dmc", "dms", "dnmc", "dnms", "go", "multidlydm", "multidm",
    "rtanti", "rtgo",
]
YANG19_ENV_IDS = {t: f"yang19.{t}-v0" for t in YANG19_TASKS}
YANG19_OBS_DIM = 33
YANG19_ACTION_DIM = 17


def make_yang19_env(task_name: str):
    import neurogym as ngym

    return ngym.make(YANG19_ENV_IDS[task_name])


class Yang19BatchEnv:
    """Same per-tick stepping contract as `NeuroGymBatchEnv`, but with an
    identity head<->env action mapping (see module comment above) -- no
    per-task lookup table, since every yang19 task already shares one
    native action space."""

    def __init__(self, task_name: str, batch_size: int, seed: int):
        self.task = task_name
        self.B = batch_size
        self.envs = [make_yang19_env(task_name) for _ in range(batch_size)]
        self.done = np.zeros(batch_size, dtype=bool)
        self.obs = np.zeros((batch_size, YANG19_OBS_DIM), dtype=np.float32)
        for b, env in enumerate(self.envs):
            obs, _info = reset_env(env, seed + b)
            self.obs[b] = obs

    def step(self, head_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """`head_actions`: [B] int in [0,17). Returns (obs, reward, gt_head,
        newly_done) -- gt_head is never None here (every yang19 task has a
        gt, unlike the bandit-style tasks in the 6-task diet)."""
        reward = np.zeros(self.B, dtype=np.float32)
        gt_head = np.zeros(self.B, dtype=np.int64)
        newly_done = np.zeros(self.B, dtype=bool)
        for b, env in enumerate(self.envs):
            if self.done[b]:
                continue
            obs, r, terminated, truncated, info = env.step(int(head_actions[b]))
            self.obs[b] = obs
            reward[b] = float(r)
            gt_head[b] = int(info["gt"])
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
