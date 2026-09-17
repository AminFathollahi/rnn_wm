"""`run_multitask_from_init.py` trains the six-task diet from a random
initialization -- enumeration, collision detection, and the resumable
--execute loop, none of which need real training or GPU."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_multitask_from_init as rmfi


def test_enumerate_runs_is_seed_major_and_states_diet_and_substrate():
    runs = rmfi.enumerate_runs([0, 1], ("M00000", "M10000"), "SUP")
    assert [r["run_id"] for r in runs] == [
        "M00000_multitask_SUP_s0", "M10000_multitask_SUP_s0",
        "M00000_multitask_SUP_s1", "M10000_multitask_SUP_s1",
    ]
    assert all(r["model_id"] in ("M00000_multitask", "M10000_multitask") for r in runs)
    assert all(r["diet"] == "multitask" for r in runs)
    assert all(r["substrate"] == "gru" and r["recurrent_init_spectral_radius"] is None for r in runs)
    assert runs[1]["S"] == 1 and runs[1]["M"] == 0


def test_enumerate_runs_rejects_an_unrecognized_model_id():
    with pytest.raises(ValueError):
        rmfi.enumerate_runs([0], ("M1",), "SUP")


def test_default_enumeration_is_4_cells_by_8_seeds_and_unique():
    runs = rmfi.enumerate_runs(range(8), rmfi.SELECTED_MODELS, "SUP")
    assert len(runs) == 32
    assert len({r["run_id"] for r in runs}) == 32


def test_report_collisions_ignores_the_arms_own_prior_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_id": "M00000_multitask_SUP_s0", "diet": "multitask", "status": "completed"}) + "\n"
        + json.dumps({"run_id": "M10000_multitask_SUP_s0", "diet": "wm_only", "status": "completed"}) + "\n"
    )
    runs = rmfi.enumerate_runs([0], ("M00000", "M10000"), "SUP")
    assert rmfi.report_collisions(runs, manifest) == ["M10000_multitask_SUP_s0"]


def test_report_collisions_empty_manifest(tmp_path):
    runs = rmfi.enumerate_runs([0], ("M00000",), "SUP")
    assert rmfi.report_collisions(runs, tmp_path / "no_such_file.jsonl") == []


def test_execute_reports_nothing_left_and_never_trains_when_all_done(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    report = tmp_path / "RUN_REPORT.md"
    runs = rmfi.enumerate_runs([0], ("M00000",), "SUP")
    manifest.write_text(json.dumps({**runs[0], "status": "completed", "tier": "full"}) + "\n")

    monkeypatch.setattr(rmfi, "MANIFEST", manifest)
    monkeypatch.setattr(rmfi, "REPORT", report)
    monkeypatch.setattr(rmfi, "resolve_train_fn", lambda force_scaffold: pytest.fail("must not train"))

    rc = rmfi.main(["--supervision", "SUP", "--models", "M00000", "--seeds", "1", "--execute"])
    assert rc == 0


def test_execute_resumes_only_the_pending_run(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.jsonl"
    report = tmp_path / "RUN_REPORT.md"
    runs = rmfi.enumerate_runs([0, 1], ("M00000",), "SUP")
    manifest.write_text(json.dumps({**runs[0], "status": "completed", "tier": "full"}) + "\n")

    calls = []

    def fake_train_one(run, cfg):
        calls.append(run["run_id"])
        return {"status": "completed", "accuracy": {}}

    monkeypatch.setattr(rmfi, "MANIFEST", manifest)
    monkeypatch.setattr(rmfi, "REPORT", report)
    monkeypatch.setattr(rmfi, "resolve_train_fn", lambda force_scaffold: fake_train_one)

    rc = rmfi.main(["--supervision", "SUP", "--models", "M00000", "--seeds", "2", "--execute"])
    assert rc == 0
    assert calls == ["M00000_multitask_SUP_s1"]

    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert sum(1 for r in rows if r["run_id"] == "M00000_multitask_SUP_s1" and r["status"] == "completed") == 1
