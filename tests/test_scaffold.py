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


def test_eight_cells_with_correct_bits():
    assert len(rg.CELLS) == 8
    ids = {c["model_id"] for c in rg.CELLS}
    assert ids == {f"M{s}{m}{l}" for s in (0, 1) for m in (0, 1) for l in (0, 1)}
    for c in rg.CELLS:
        assert c["model_id"] == f"M{c['S']}{c['M']}{c['L']}"


def test_enumerate_is_breadth_first():
    runs = rg.enumerate_runs([0, 1])
    # seed-major: all 8 cells at seed 0 come before any seed-1 run
    first8 = runs[:8]
    assert all(r["seed"] == 0 for r in first8)
    assert {r["model_id"] for r in first8} == {c["model_id"] for c in rg.CELLS}
    assert runs[8]["seed"] == 1
    assert len(runs) == 16


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
    run = {"run_id": "M111_s0", "model_id": "M111", "S": 1, "M": 1, "L": 1, "seed": 0}
    r1 = rg._scaffold_train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
    r2 = rg._scaffold_train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
    assert r1["accuracy"] == r2["accuracy"]  # deterministic per run_id
    assert set(r1["gates"]) == {"load1>=0.95", "load3>=0.80"}
    assert (tmp_path / "checkpoints" / "M111_s0" / "ckpt.json").exists()


def test_write_report(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_id": "M000_s0", "S": 0, "M": 0, "L": 0, "status": "completed",
                    "gates": {"load1>=0.95": True}, "accuracy": {"load1": 0.99},
                    "rung": 0, "wall_clock_s": 1.2}) + "\n"
    )
    report = tmp_path / "RUN_REPORT.md"
    rg.write_report(manifest, report, budget_s=36000, elapsed_s=1.2)
    text = report.read_text()
    assert "Training Grid Report" in text and "M000_s0" in text and "completed=1" in text
