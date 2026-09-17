import pytest
import torch

import scripts.run_continuations as continuations
from brainalign_wm.config import load_config
from brainalign_wm.tasks.multitask import DIET_TASKS


def _run(diet="multitask", updates=12):
    return continuations.enumerate_continuations(
        seeds=[3], models=("M10101",), diets=(diet,), updates=updates
    )[0]


def test_continuation_arms_share_initialization_and_update_budget():
    wm, multitask = continuations.enumerate_continuations(
        seeds=[3], models=("M10101",), updates=19
    )

    for key in (
        "S", "M", "P", "T", "D", "source_run_id", "seed", "continuation_seed",
        "additional_updates", "optimizer_restart",
    ):
        assert wm[key] == multitask[key]
    assert wm["diet"] == "wm_only"
    assert multitask["diet"] == "multitask"
    assert wm["run_id"] != multitask["run_id"]


def test_continuation_arms_build_identical_initial_states(tmp_path, monkeypatch):
    pytest.importorskip("neurogym")
    cfg = load_config()
    wm, multitask = continuations.enumerate_continuations(
        seeds=[3], models=("M00000",), updates=2
    )
    front_end, core, heads = continuations._build_model(cfg, 0, 0, 0, torch.device("cpu"))
    source_dir = tmp_path / "checkpoints" / wm["source_run_id"]
    source_dir.mkdir(parents=True)
    torch.save(
        {
            "front_end": front_end.state_dict(),
            "core": core.state_dict(),
            "heads": heads.state_dict(),
        },
        source_dir / "ckpt.pt",
    )
    monkeypatch.setattr(continuations, "RESULTS", tmp_path)

    assert continuations.verify_source(wm, cfg, torch.device("cpu")) == (
        continuations.verify_source(multitask, cfg, torch.device("cpu"))
    )


def test_exposure_plan_uses_the_deterministic_branch_schedule(monkeypatch):
    monkeypatch.setattr(
        continuations,
        "pick_task",
        lambda seed, step: DIET_TASKS[step % len(DIET_TASKS)],
    )
    run = _run(updates=14)

    assert continuations._expected_exposures(run, 8) == {
        "sternberg": 2,
        "bandit": 2,
        "dawtwostep": 1,
        "delaymatchsample": 1,
        "gonogo": 1,
        "contextdecisionmaking": 1,
    }
    planned = continuations._expected_exposures(run)
    assert sum(planned.values()) == run["additional_updates"]
    assert continuations._wm_curriculum_position(
        continuations._expected_exposures(run, 8), planned
    ) == (2, 3)


def test_wm_only_exposure_plan_matches_every_optimizer_update():
    run = _run(diet="wm_only", updates=9)

    assert continuations._expected_exposures(run) == {
        task: 9 if task == "sternberg" else 0 for task in DIET_TASKS
    }


def test_resume_requires_exact_metadata_and_exposure_prefix(monkeypatch):
    monkeypatch.setattr(
        continuations,
        "pick_task",
        lambda seed, step: DIET_TASKS[step % len(DIET_TASKS)],
    )
    run = _run(updates=14)
    saved = {
        "step": 8,
        "continuation": {
            "source_run_id": run["source_run_id"],
            "additional_updates": run["additional_updates"],
            "optimizer_restart": run["optimizer_restart"],
            "exposures": continuations._expected_exposures(run, 8),
        },
    }

    assert continuations._resume_progress(saved, run) == (
        8,
        continuations._expected_exposures(run, 8),
    )
    saved["continuation"]["exposures"]["sternberg"] += 1
    with pytest.raises(RuntimeError, match="exposure counts"):
        continuations._resume_progress(saved, run)

    saved["continuation"]["exposures"] = continuations._expected_exposures(run, 8)
    saved["continuation"]["completed_updates"] = 7
    with pytest.raises(RuntimeError, match="completed update counts"):
        continuations._resume_progress(saved, run)


def test_resume_rejects_a_different_source_branch():
    run = _run(diet="wm_only", updates=4)
    saved = {
        "step": 0,
        "continuation": {
            "source_run_id": "M00000_SUP_s3",
            "additional_updates": 4,
            "optimizer_restart": "adam_fresh",
            "exposures": {task: 0 for task in DIET_TASKS},
        },
    }

    with pytest.raises(RuntimeError, match="source_run_id"):
        continuations._resume_progress(saved, run)


def test_checkpoint_save_is_atomic(tmp_path):
    cfg = load_config()
    run = _run(diet="wm_only", updates=1)
    modules = continuations._build_model(cfg, run["S"], run["M"], run["P"], torch.device("cpu"))
    path = tmp_path / "ckpt.pt"
    exposures = {task: int(task == "sternberg") for task in DIET_TASKS}
    continuations._save(path, 1, modules, {}, None, exposures, run)
    assert path.exists()
    assert not path.with_suffix(".pt.partial").exists()
    assert torch.load(path, weights_only=False)["continuation"]["completed_updates"] == 1
