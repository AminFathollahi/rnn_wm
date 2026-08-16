"""Archived checkpoints must be invisible to every standard analysis driver."""
import json

import brainalign_wm.analysis.run_all as run_all
import brainalign_wm.figures.make_all as make_all
import scripts.analyze_attractors as analyze_attractors
import scripts.analyze_network_properties as analyze_network_properties
import scripts.run_geometry as run_geometry


def test_standard_analysis_loaders_exclude_archived_runs(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"run_id": "LIVE", "status": "completed"},
        {"run_id": "OLD", "status": "completed", "archived": True},
        {"run_id": "FAILED", "status": "error"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert [row["run_id"] for row in run_all._load_completed_runs(manifest)] == ["LIVE"]
    assert [row["run_id"] for row in run_geometry._completed_records(manifest)] == ["LIVE"]
    assert make_all._load_manifest(manifest)["run_id"].tolist() == ["LIVE", "FAILED"]

    monkeypatch.setattr(analyze_network_properties, "MANIFEST", manifest)
    monkeypatch.setattr(analyze_attractors, "MANIFEST", manifest)
    assert analyze_network_properties._completed_run_ids() == ["LIVE"]
    assert analyze_attractors._completed_run_ids() == ["LIVE"]
