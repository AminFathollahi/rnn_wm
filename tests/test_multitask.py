"""Multi-task diet: one test per task in the 6-task diet confirming the
adapter/wrapper yields the expected observation and action shapes, plus the
shared task cue and the multitask training loop's mechanics. Skips entirely
if `neurogym` is not installed -- the Sternberg-only pipeline never imports
this module."""
import numpy as np
import pytest
import yaml

from brainalign_wm.config import load_config

torch = pytest.importorskip("torch")
pytest.importorskip("neurogym")

from brainalign_wm.tasks.multitask import (
    C_DIM_MULTITASK,
    DIET_TASKS,
    LEGACY_TASK_CUE_SCHEMA,
    ENV_GT_TO_HEAD,
    HAS_GT,
    HEAD_TO_ENV,
    NEUROGYM_TASKS,
    NeuroGymAdapter,
    NeuroGymBatchEnv,
    TASK_CODES,
    TASK_CUE_SCHEMA,
    make_env,
    obs_dim_for,
    task_cue_metadata,
    task_context_vector,
    task_one_hot,
)

ROOT = __file__.rsplit("/tests/", 1)[0]
FULL_CFG = load_config()
BOTTLENECK = FULL_CFG["model"]["bottleneck"]
BATCH = 4


def test_diet_has_six_tasks_sternberg_anchor_first():
    assert len(DIET_TASKS) == 6
    assert DIET_TASKS[0] == "sternberg"
    assert len(NEUROGYM_TASKS) == 5


# The three entries the Sternberg cue never varies: index 0 is its constant
# working-memory family bit and indices 8 and 9 are permanently zero, so a
# task code placed in them costs the working-memory schema nothing.
TASK_CODE_INDICES = (0, 8, 9)


def test_task_cue_is_ten_dim_and_every_task_has_a_distinct_code():
    seen = {}
    for task in DIET_TASKS:
        c = task_context_vector(task)
        assert len(c) == C_DIM_MULTITASK == 10
        assert tuple(c[i] for i in TASK_CODE_INDICES) == TASK_CODES[task]
        assert all(c[i] == 0.0 for i in range(10) if i not in TASK_CODE_INDICES)
        seen.setdefault(TASK_CODES[task], task)
        assert seen[TASK_CODES[task]] == task, "two tasks share one code"


def test_sternberg_cue_is_the_working_memory_vector_unchanged():
    """The working-memory cue passes through verbatim -- that is what lets a
    Sternberg-trained checkpoint keep reading its own input -- and its code
    entries already spell the Sternberg code."""
    from brainalign_wm.tasks.sternberg import context_vector

    base = context_vector(epoch="encode", encoded_count=2)
    mc = task_context_vector("sternberg", sternberg_c=base)
    assert mc == base
    assert tuple(mc[i] for i in TASK_CODE_INDICES) == TASK_CODES["sternberg"]


def test_legacy_schema_still_decodes_the_thirteen_entry_cue():
    """Checkpoints trained on the wider cue must stay readable, so the old
    schema remains addressable by name."""
    from brainalign_wm.tasks.sternberg import context_vector

    for task in DIET_TASKS:
        c = task_context_vector(task, schema=LEGACY_TASK_CUE_SCHEMA)
        assert len(c) == 13
        assert c[:7] == [0.0] * 7
        assert c[7:] == task_one_hot(task)
        assert sum(c[7:]) == 1.0

    base = context_vector(epoch="encode", encoded_count=2)
    mc = task_context_vector("sternberg", sternberg_c=base, schema=LEGACY_TASK_CUE_SCHEMA)
    assert mc[:7] == [base[1], base[2], base[3], base[4], base[5], base[6], base[7]]
    assert mc[7:] == task_one_hot("sternberg")


def test_cue_metadata_records_the_schema_a_checkpoint_was_trained_on():
    meta = task_cue_metadata()
    assert meta["schema"] == TASK_CUE_SCHEMA
    assert meta["tasks"] == DIET_TASKS
    assert meta["codes"] == {name: list(code) for name, code in TASK_CODES.items()}


def test_unknown_cue_schema_is_rejected():
    with pytest.raises(ValueError):
        task_context_vector("sternberg", schema="no_such_schema")


@pytest.mark.parametrize("task", NEUROGYM_TASKS)
def test_neurogym_adapter_yields_expected_observation_and_action_shapes(task):
    """Observation and action shapes the adapter must produce, per task."""
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
    # native action space: one shared 3-action head, no per-task head.
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
