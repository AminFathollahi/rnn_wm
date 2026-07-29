"""Item 10.1 Tier 1 (comments.txt §5): `multitask.py`'s Yang-19 wiring
(20-task suite, identity head<->env action mapping) and
`scripts/run_yang19_baseline.py`'s task-onehot/rollout helpers. Skips
entirely if `neurogym` is not installed, same convention as
`test_multitask.py`."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neurogym")

from brainalign_wm.tasks.multitask import (
    YANG19_ACTION_DIM,
    YANG19_ENV_IDS,
    YANG19_OBS_DIM,
    YANG19_TASKS,
    Yang19BatchEnv,
    make_yang19_env,
    pick_task,
)

BATCH = 4


def test_yang19_has_twenty_tasks_no_duplicates():
    assert len(YANG19_TASKS) == 20
    assert len(set(YANG19_TASKS)) == 20
    assert set(YANG19_ENV_IDS) == set(YANG19_TASKS)


@pytest.mark.parametrize("task", YANG19_TASKS)
def test_every_yang19_task_shares_the_same_action_and_obs_space(task):
    """The identity head<->env mapping this module relies on (no per-task
    lookup table, unlike the 6-task diet) is only valid if every yang19
    task really does share one native action/obs space -- verified here,
    not assumed."""
    env = make_yang19_env(task)
    assert env.observation_space.shape == (YANG19_OBS_DIM,)
    assert env.action_space.n == YANG19_ACTION_DIM


@pytest.mark.parametrize("task", YANG19_TASKS)
def test_yang19_batch_env_step_shapes_and_gt_always_present(task):
    batch_env = Yang19BatchEnv(task, batch_size=BATCH, seed=0)
    assert batch_env.obs.shape == (BATCH, YANG19_OBS_DIM)
    actions = np.zeros(BATCH, dtype=np.int64)  # fixate
    obs, reward, gt_head, newly_done = batch_env.step(actions)
    assert obs.shape == (BATCH, YANG19_OBS_DIM)
    assert reward.shape == (BATCH,)
    assert gt_head is not None and gt_head.shape == (BATCH,)
    assert newly_done.shape == (BATCH,)
    assert ((gt_head >= 0) & (gt_head < YANG19_ACTION_DIM)).all()


def test_yang19_batch_env_holds_done_instances():
    """Same per-trial done-latch convention as `NeuroGymBatchEnv` (correct
    here: `run_yang19_trial` masks by `active`, one trial per rollout, not
    a cross-trial-persistence design like Phase 9b's `ContinuousBatchEnv`)."""
    batch_env = Yang19BatchEnv("go", batch_size=BATCH, seed=0)
    for _ in range(300):
        if batch_env.done.all():
            break
        batch_env.step(np.zeros(BATCH, dtype=np.int64))
    assert batch_env.done.any(), "expected at least one instance to finish within 300 fixate-only ticks"
    obs_before = batch_env.obs.copy()
    batch_env.step(np.zeros(BATCH, dtype=np.int64))
    done_mask = batch_env.done
    assert np.allclose(batch_env.obs[done_mask], obs_before[done_mask]), (
        "a done instance's observation must not change on further step() calls"
    )


def test_pick_task_over_yang19_is_deterministic_and_in_range():
    seen = set()
    for step in range(200):
        t = pick_task(seed=0, step_idx=step, tasks=YANG19_TASKS)
        assert t in YANG19_TASKS
        seen.add(t)
    assert pick_task(seed=0, step_idx=5, tasks=YANG19_TASKS) == pick_task(seed=0, step_idx=5, tasks=YANG19_TASKS)
    assert len(seen) > 1  # actually varies across steps, not stuck on one task


def test_task_onehot_vec_matches_run_yang19_baseline_convention():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.run_yang19_baseline import task_onehot_vec

    for task in YANG19_TASKS:
        v = task_onehot_vec(task)
        assert len(v) == len(YANG19_TASKS) == 20
        assert sum(v) == 1.0
        assert v[YANG19_TASKS.index(task)] == 1.0
