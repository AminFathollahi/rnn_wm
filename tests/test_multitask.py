"""Phase 5 (comments.txt §5, multi-task diet): one test per task in the
6-task diet confirming the adapter/wrapper yields the expected observation
and action shapes (item 5.1's explicit acceptance ask), plus the extended
13-dim context vector and the multitask training loop's mechanics. Skips
entirely if `neurogym` is not installed -- the Sternberg-only pipeline
(Phases 0-4) never imports this module."""
import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("neurogym")

from brainalign_wm.tasks.multitask import (
    C_DIM_MULTITASK,
    DIET_TASKS,
    ENV_GT_TO_HEAD,
    HAS_GT,
    HEAD_TO_ENV,
    NEUROGYM_TASKS,
    NeuroGymAdapter,
    NeuroGymBatchEnv,
    make_env,
    obs_dim_for,
    task_context_vector,
    task_one_hot,
)

ROOT = __file__.rsplit("/tests/", 1)[0]
FULL_CFG = yaml.safe_load(open(f"{ROOT}/configs/config.yaml"))
BOTTLENECK = FULL_CFG["model"]["bottleneck"]
BATCH = 4


def test_diet_has_six_tasks_sternberg_anchor_first():
    assert len(DIET_TASKS) == 6
    assert DIET_TASKS[0] == "sternberg"
    assert len(NEUROGYM_TASKS) == 5


def test_context_vector_is_13_dim_and_one_hot_matches_task_order():
    for task in DIET_TASKS:
        c = task_context_vector(task)
        assert len(c) == C_DIM_MULTITASK == 13
        assert c[7:] == task_one_hot(task)
        assert sum(c[7:]) == 1.0


def test_sternberg_context_vector_drops_wasted_dims_keeps_the_rest():
    from brainalign_wm.tasks.sternberg import context_vector

    base = context_vector(epoch="encode", encoded_count=2)
    mc = task_context_vector("sternberg", sternberg_c=base)
    assert mc[:7] == [base[1], base[2], base[3], base[4], base[5], base[6], base[7]]
    assert mc[7:] == task_one_hot("sternberg")


@pytest.mark.parametrize("task", NEUROGYM_TASKS)
def test_neurogym_adapter_yields_expected_observation_and_action_shapes(task):
    """Item 5.1's explicit acceptance ask, per task."""
    obs_dim = obs_dim_for(task)
    env = make_env(task)
    assert env.observation_space.shape == (obs_dim,)
    assert env.action_space.n >= 2

    batch_env = NeuroGymBatchEnv(task, batch_size=BATCH, seed=0)
    assert batch_env.obs.shape == (BATCH, obs_dim)

    adapter = NeuroGymAdapter(obs_dim, C_DIM_MULTITASK, BOTTLENECK)
    obs_t = torch.as_tensor(batch_env.obs, dtype=torch.float32)
    c_t = torch.as_tensor([task_context_vector(task)] * BATCH, dtype=torch.float32)
    z_t = adapter(obs_t, c_t)
    assert z_t.shape == (BATCH, BOTTLENECK)

    # Every head action (0,1,2) must map to a valid index in this task's
    # native action space (item 5.5: shared 3-action head, no per-task head).
    head_to_env = HEAD_TO_ENV[task]
    assert set(head_to_env.keys()) == {0, 1, 2}
    for env_action in head_to_env.values():
        assert env.action_space.contains(env_action)

    head_actions = np.array([1] * BATCH)
    obs2, reward, gt_head, newly_done = batch_env.step(head_actions)
    assert obs2.shape == (BATCH, obs_dim)
    assert reward.shape == (BATCH,)
    assert newly_done.shape == (BATCH,)
    if HAS_GT[task]:
        assert gt_head is not None
        assert gt_head.shape == (BATCH,)
        assert set(ENV_GT_TO_HEAD[task].values()) <= {0, 1, 2}
    else:
        assert gt_head is None


@pytest.mark.parametrize("task", NEUROGYM_TASKS)
def test_multitask_trial_runs_and_produces_finite_grad(task):
    """One 40-tick rollout per task: loss is finite, and (for the 3 tasks
    with a supervised target) backprop reaches the adapter's weights."""
    from brainalign_wm.training.train import _build_model, run_multitask_neurogym_trial

    device = torch.device("cpu")
    _front_end, core, heads = _build_model(FULL_CFG, S=0, M=0, P=0, device=device)
    obs_dim = obs_dim_for(task)
    adapter = NeuroGymAdapter(obs_dim, C_DIM_MULTITASK, BOTTLENECK).to(device)
    batch_env = NeuroGymBatchEnv(task, batch_size=BATCH, seed=1)

    loss, reward_per_trial = run_multitask_neurogym_trial(
        adapter, core, heads, S=0, M=0, P=0, task_name=task, batch_env=batch_env, B=BATCH,
        max_ticks=FULL_CFG["task"]["multitask_max_ticks"], device=device, mode="bptt",
    )
    assert loss is not None
    assert torch.isfinite(loss)
    assert len(reward_per_trial) == BATCH

    loss.backward()
    assert adapter.w.weight.grad is not None
    assert torch.isfinite(adapter.w.weight.grad).all()
