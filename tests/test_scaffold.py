"""Tests for the grid orchestrator (run_grid): lock in the resume,
breadth-first ordering, wall-clock budget, and reporting behavior the
training grid execution depends on."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_grid", ROOT / "run_grid.py")
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def test_parse_budget():
    assert rg.parse_budget("10h") == 36000
    assert rg.parse_budget("30m") == 1800
    assert rg.parse_budget("600s") == 600
    assert rg.parse_budget("42") == 42


def test_ablation_battery_cells_with_correct_bits():
    """The 5-arm ablation battery: 15 cells (baseline, full reference, 5
    knock-one-out, 5 add-one, 3 two-arm interaction probes), all trained
    by BPTT (S/M/P/T/D, not L)."""
    assert len(rg.CELLS) == 15
    for c in rg.CELLS:
        assert c["model_id"] == f"M{c['S']}{c['M']}{c['P']}{c['T']}{c['D']}"
        assert "L" not in c
    ids = {c["model_id"] for c in rg.CELLS}
    assert ids == {
        "M00000", "M11111",
        "M01111", "M10111", "M11011", "M11101", "M11110",
        "M10000", "M01000", "M00100", "M00010", "M00001",
        "M10010", "M00011", "M10001",
    }


def test_four_local_learning_cells_with_correct_bits():
    assert len(rg.LOCAL_LEARNING_CELLS) == 4
    ids = {c["model_id"] for c in rg.LOCAL_LEARNING_CELLS}
    assert ids == {f"M{s}{m}L" for s in (0, 1) for m in (0, 1)}
    for c in rg.LOCAL_LEARNING_CELLS:
        assert c["model_id"] == f"M{c['S']}{c['M']}L"
        assert c["L"] == 1
        assert "P" not in c


def test_enumerate_is_breadth_first():
    runs = rg.enumerate_runs([0, 1])
    n_cells = len(rg.CELLS)
    # seed-major: all cells at seed 0 come before any seed-1 run
    first_batch = runs[:n_cells]
    assert all(r["seed"] == 0 for r in first_batch)
    assert {r["model_id"] for r in first_batch} == {c["model_id"] for c in rg.CELLS}
    assert runs[n_cells]["seed"] == 1
    assert len(runs) == 2 * n_cells


def test_enumerate_includes_local_learning_cells_when_requested():
    runs = rg.enumerate_runs([0], include_local_learning=True)
    assert len(runs) == len(rg.CELLS) + 4  # ablation battery + 4 local-learning
    ll_ids = {r["model_id"] for r in runs} & {c["model_id"] for c in rg.LOCAL_LEARNING_CELLS}
    assert ll_ids == {c["model_id"] for c in rg.LOCAL_LEARNING_CELLS}


def test_enumerate_runs_carries_explicit_supervision():
    """comments.txt §16 item 16.4 / advisor.md D24: every enumerated battery
    run dict must carry `supervision` explicitly so it reaches `train_one`
    and `build_resolved_config` from the run, not from config.yaml's
    `legacy` default (the same fall-through defect class as F1/D21)."""
    runs = rg.enumerate_runs([0, 1], supervision="RL")
    assert all(r["supervision"] == "RL" for r in runs)
    assert len(runs) == 2 * len(rg.CELLS)


def test_main_requires_supervision_flag(capsys):
    """The launcher must fail rather than silently defaulting to `legacy`
    when `--supervision` is omitted -- this is the actual defect §16.4
    closes, not just enumerate_runs's plumbing."""
    import pytest

    with pytest.raises(SystemExit) as exc_info:
        rg.main(["--scaffold", "--seeds", "1", "--budget", "1s"])
    assert exc_info.value.code != 0
    assert "--supervision" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc_info:
        rg.main(["--scaffold", "--seeds", "1", "--budget", "1s", "--supervision", "legacy"])
    assert exc_info.value.code != 0
    assert "--supervision" in capsys.readouterr().err


def test_resume_skips_completed(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_id": "M000_s0", "status": "completed"}) + "\n"
        + json.dumps({"run_id": "M001_s0", "status": "error"}) + "\n"
    )
    done = rg.load_completed(manifest)
    assert done == {"M000_s0"}  # errored run is NOT skipped -> will retry


def test_scaffold_train_one_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "RESULTS", tmp_path)
    run = {"run_id": "M111_s0", "model_id": "M111", "S": 1, "M": 1, "P": 1, "seed": 0}
    r1 = rg._scaffold_train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
    r2 = rg._scaffold_train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
    assert r1["accuracy"] == r2["accuracy"]  # deterministic per run_id
    assert set(r1["gates"]) == {"load1>=0.95", "load3>=0.80"}
    assert (tmp_path / "checkpoints" / "M111_s0" / "ckpt.json").exists()


def test_scaffold_train_one_local_learning_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "RESULTS", tmp_path)
    run = {"run_id": "M11L_s0", "model_id": "M11L", "S": 1, "M": 1, "L": 1, "seed": 0}
    r = rg._scaffold_train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
    assert r["rung"] == 1  # local-learning cells start at rung 1


def test_write_report(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_id": "M000_s0", "S": 0, "M": 0, "P": 0, "status": "completed",
                    "gates": {"load1>=0.95": True}, "accuracy": {"load1": 0.99},
                    "rung": 0, "wall_clock_s": 1.2}) + "\n"
    )
    report = tmp_path / "RUN_REPORT.md"
    rg.write_report(manifest, report, budget_s=36000, elapsed_s=1.2)
    text = report.read_text()
    assert "Training Grid Report" in text and "M000_s0" in text and "completed=1" in text
