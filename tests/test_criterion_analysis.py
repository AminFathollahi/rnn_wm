import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from brainalign_wm.analysis.run_all import _is_ablation_or_catch_variant, _load_completed_runs
from brainalign_wm.training.generate_activity_logs import _parse_run_id, _run_id_extras, activity_log_path
from scripts import merge_analysis_shards, run_perf_matched_baselines


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "alignment_by_session",
    "alignment_results",
    "alignment_probe_by_region",
    "alignment_by_population",
)


def test_performance_controls_are_signal_qualified_and_resumable(tmp_path, monkeypatch):
    results = tmp_path / "results"
    manifest = results / "manifest.jsonl"
    results.mkdir()
    manifest.write_text(json.dumps({
        "run_id": "M00000_2x_SUP_s0", "model_id": "M00000_2x", "supervision": "SUP",
        "tier": "smoke", "status": "completed",
    }) + "\n")
    trained = []

    def train_one(run, cfg):
        trained.append((run, cfg))
        checkpoint_dir = results / "checkpoints" / run["run_id"]
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "ckpt.pt").write_bytes(b"final")
        (checkpoint_dir / "ckpt_at_criterion.pt").write_bytes(b"criterion")
        return {"status": "completed", "accuracy": {"load1": 0.9}}

    monkeypatch.setattr(run_perf_matched_baselines, "RESULTS", results)
    monkeypatch.setattr(run_perf_matched_baselines, "MANIFEST", manifest)
    monkeypatch.setattr(run_perf_matched_baselines, "REPORT", results / "RUN_REPORT.md")
    monkeypatch.setattr(run_perf_matched_baselines, "resolve_train_fn", lambda force_scaffold: train_one)
    monkeypatch.setattr(run_perf_matched_baselines, "resolved_config_path", lambda name: results / f"resolved_config_{name}.yaml")
    monkeypatch.setattr(run_perf_matched_baselines, "write_report", lambda *args: None)
    monkeypatch.setattr(run_perf_matched_baselines, "git_commit", lambda: "test")
    monkeypatch.setattr(run_perf_matched_baselines.signal, "signal", lambda *args: None)
    monkeypatch.setattr(run_perf_matched_baselines, "_STOP", False)

    common = ["--seeds", "1", "--tier", "full", "--budget", "1h", "--config", str(ROOT / "configs/config.yaml")]
    for supervision in ("SUP", "RL"):
        assert run_perf_matched_baselines.main([*common, "--supervision", supervision]) == 0

    assert len(trained) == 6
    assert all(cfg == {"steps": 24000, "max_steps_if_criterion_unmet": 48000} for _, cfg in trained)
    run_ids = {run["run_id"] for run, _ in trained}
    assert run_ids == {
        f"M00000_{arm}_{supervision}_s0"
        for supervision in ("SUP", "RL")
        for arm in ("2x", "l1", "dropout")
    }
    assert all((results / "checkpoints" / run_id / "ckpt_at_criterion.pt").is_file() for run_id in run_ids)

    for supervision in ("SUP", "RL"):
        assert run_perf_matched_baselines.main([*common, "--supervision", supervision]) == 0
    assert len(trained) == 6

    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    full_records = [record for record in records if record.get("tier") == "full"]
    assert len(full_records) == 6
    assert len({record["config_hash"] for record in full_records}) == 6
    assert {record["run_id"] for record in _load_completed_runs(manifest)} == run_ids
    assert len(list(results.glob("resolved_config_perf_*.yaml"))) == 6

    for record in full_records:
        model_id, *_, seed = _parse_run_id(record["run_id"])
        assert seed == 0
        assert record["supervision"] in model_id.split("_")
        assert _run_id_extras(model_id)[2] == (2 if "2x" in model_id.split("_") else 1)
        assert _is_ablation_or_catch_variant(record)
        assert activity_log_path(record["run_id"], "ckpt_at_criterion.pt", tmp_path).name.endswith(
            "_at_criterion.parquet"
        )


def test_suffix_merge_is_complete_atomic_and_separate(tmp_path, monkeypatch):
    suffix = "_at_criterion"
    shards = tmp_path / "shards"
    results = tmp_path / "results"
    run_ids = ["M00000_SUP_s0", "M00000_RL_s0"]
    run_list = tmp_path / "runs.txt"
    run_list.write_text("\n".join(run_ids) + "\n")
    for index, run_id in enumerate(run_ids):
        shard = shards / run_id
        shard.mkdir(parents=True)
        (shard / ".complete").touch()
        for table in REQUIRED:
            pd.DataFrame({"run_id": [run_id], "session": [f"s{1 - index}"], "value": [index]}).to_csv(
                shard / f"{table}{suffix}.csv", index=False
            )
    sentinel = results / "alignment_results.csv"
    results.mkdir()
    sentinel.write_text("default\n")
    monkeypatch.setattr(merge_analysis_shards, "get_path", lambda name: results)

    assert merge_analysis_shards.main([
        "--checkpoint-dir", str(shards), "--run-list", str(run_list), "--suffix", suffix,
    ]) == 0
    assert sentinel.read_text() == "default\n"
    merged = pd.read_csv(results / f"alignment_results{suffix}.csv")
    assert merged["run_id"].tolist() == sorted(run_ids)
    assert not list(results.glob(".alignment_results_at_criterion.*"))

    (shards / run_ids[0] / f"alignment_results{suffix}.csv").write_text("")
    with pytest.raises(SystemExit, match="alignment_results"):
        merge_analysis_shards.main([
            "--checkpoint-dir", str(shards), "--run-list", str(run_list), "--suffix", suffix,
        ])


def test_suffix_merge_rejects_empty_run_list(tmp_path):
    run_list = tmp_path / "runs.txt"
    run_list.write_text("")
    with pytest.raises(SystemExit, match="empty analysis run list"):
        merge_analysis_shards.main([
            "--checkpoint-dir", str(tmp_path / "shards"),
            "--run-list", str(run_list),
            "--suffix", "_at_criterion",
        ])


def test_criterion_analysis_runner_resumes_completed_shards(tmp_path):
    runner = tmp_path / "analysis.sh"
    shutil.copy2(ROOT / "analysis.sh", runner)
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        """#!/usr/bin/env python3
import json
import sys
from pathlib import Path

root = Path.cwd()
args = sys.argv[1:]
with (root / "calls.jsonl").open("a") as handle:
    handle.write(json.dumps(args) + "\\n")
if args == ["scripts/list_active_analysis_runs.py"]:
    print("M00000_SUP_s0")
    print("M00000_RL_s0")
    print("M11111_SUP_s0")
elif "brainalign_wm.analysis.run_all" in args:
    run_id = args[args.index("--runs") + 1]
    checkpoint = args[args.index("--checkpoint") + 1]
    out_dir = Path(args[args.index("--out-dir") + 1])
    suffix = "" if checkpoint == "ckpt.pt" else "_" + Path(checkpoint).stem.removeprefix("ckpt_")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("alignment_by_session", "alignment_results", "alignment_probe_by_region", "alignment_by_population"):
        (out_dir / f"{name}{suffix}.csv").write_text(f"run_id,value\\n{run_id},1\\n")
elif args and args[0] == "scripts/merge_analysis_shards.py":
    if args[args.index("--suffix") + 1] != "_at_criterion":
        raise SystemExit(3)
else:
    raise SystemExit(2)
"""
    )
    fake_python.chmod(0o755)
    for run_id in ("M00000_SUP_s0", "M00000_RL_s0"):
        checkpoint_dir = tmp_path / "results" / "checkpoints" / run_id
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "ckpt_at_criterion.pt").write_bytes(b"checkpoint")

    env = {
        **os.environ,
        "ANALYSIS_PY": str(fake_python),
        "ANALYSIS_CHECKPOINT": "ckpt_at_criterion.pt",
    }
    first = subprocess.run([str(runner)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    second = subprocess.run([str(runner)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert "M11111_SUP_s0: no ckpt_at_criterion.pt" in first.stderr
    assert "checkpoint=ckpt_at_criterion.pt eligible=2 active=3 omitted=1" in first.stdout
    checkpoint_root = tmp_path / "results" / "analysis_checkpoints_at_criterion"
    assert (checkpoint_root / "active_core_runs.txt").read_text().splitlines() == [
        "M00000_SUP_s0", "M00000_RL_s0",
    ]
    assert all((checkpoint_root / run_id / ".complete").is_file() for run_id in ("M00000_SUP_s0", "M00000_RL_s0"))
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    analysis_calls = [args for args in calls if "brainalign_wm.analysis.run_all" in args]
    merge_calls = [args for args in calls if args and args[0] == "scripts/merge_analysis_shards.py"]
    assert len(analysis_calls) == 2
    assert len(merge_calls) == 2
    assert all(args[args.index("--suffix") + 1] == "_at_criterion" for args in merge_calls)
    assert second.stdout.count("checkpoint found; skipping") == 2
