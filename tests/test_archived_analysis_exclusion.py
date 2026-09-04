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


def test_campaign_only_analysis_loaders_exclude_noncampaign_runs(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.jsonl"
    campaign = {
        "run_id": "M00000_SUP_s0", "model_id": "M00000", "supervision": "SUP",
        "S": 0, "M": 0, "P": 0, "T": 0, "D": 0, "seed": 0, "status": "completed",
    }
    local_learning = {
        "run_id": "M00L_RL_s0", "model_id": "M00L", "supervision": "RL",
        "S": 0, "M": 0, "L": 1, "seed": 0, "status": "completed",
    }
    pilot = {"run_id": "M00000_H2_s0", "model_id": "M00000_H2", "status": "completed"}
    manifest.write_text("".join(json.dumps(row) + "\n" for row in (campaign, local_learning, pilot)))

    monkeypatch.setattr(analyze_network_properties, "MANIFEST", manifest)
    monkeypatch.setattr(analyze_attractors, "MANIFEST", manifest)
    assert analyze_network_properties._completed_run_ids(campaign_only=True) == ["M00000_SUP_s0"]
    assert analyze_attractors._completed_run_ids(campaign_only=True) == ["M00000_SUP_s0"]
