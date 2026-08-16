"""Regression coverage for lossless legacy-run archival."""
import csv
import json

from scripts import archive_runs


def test_archive_runs_is_dry_by_default_and_rewrites_every_manifest_row(tmp_path, monkeypatch):
    results = tmp_path / "results"
    manifest = results / "manifest.jsonl"
    metrics = results / "metrics"
    checkpoints = results / "checkpoints"
    metrics.mkdir(parents=True)
    checkpoints.mkdir()
    run_id = "M00000_SUP_s0"
    archived_id = f"{run_id}_budget80000"
    rows = [
        {"run_id": run_id, "tier": "full", "status": "error"},
        {"run_id": run_id, "tier": "full", "status": "completed"},
        {"run_id": "UNRELATED", "tier": "full", "status": "completed"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with (metrics / f"{run_id}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step"])
        writer.writeheader()
        writer.writerow({"step": 80000})
    run_checkpoint = checkpoints / run_id
    run_checkpoint.mkdir()
    (run_checkpoint / "ckpt.pt").write_bytes(b"checkpoint")
    resolved = results / f"resolved_config_{run_id.lower()}.yaml"
    resolved.write_text("gates:\n  max_steps: 80000\n")

    monkeypatch.setattr(archive_runs, "RESULTS", results)
    monkeypatch.setattr(archive_runs, "MANIFEST", manifest)
    monkeypatch.setattr(archive_runs, "ROOT", tmp_path)

    assert archive_runs.main([]) == 0
    assert (metrics / f"{run_id}.csv").exists()
    assert run_checkpoint.exists()
    assert any(json.loads(line)["run_id"] == run_id for line in manifest.read_text().splitlines())

    assert archive_runs.main(["--apply"]) == 0
    assert not (metrics / f"{run_id}.csv").exists()
    assert (metrics / f"{archived_id}.csv").exists()
    assert not run_checkpoint.exists()
    assert (checkpoints / archived_id / "ckpt.pt").read_bytes() == b"checkpoint"
    assert not resolved.exists()
    assert (results / f"resolved_config_{archived_id.lower()}.yaml").exists()
    rewritten_ids = [json.loads(line)["run_id"] for line in manifest.read_text().splitlines()]
    assert rewritten_ids == [archived_id, archived_id, "UNRELATED"]
