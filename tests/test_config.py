from pathlib import Path

import yaml

from brainalign_wm.config import PROJECT_ROOT, get_path, load_config, load_paths


def test_scientific_config_contains_no_machine_paths():
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "config.yaml").read_text())
    assert "paths" not in raw


def test_relative_paths_are_project_anchored(tmp_path):
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(
        "paths:\n"
        "  data_root: test-data\n"
        "  stimuli: test-stimuli\n"
        "  results: test-results\n"
    )
    paths = load_paths(paths_file)
    assert paths["data_root"] == PROJECT_ROOT / "test-data"
    assert paths["feature_cache"] == PROJECT_ROOT / "test-results" / "feat_cache"
    assert paths["activity_logs"] == PROJECT_ROOT / "test-results" / "activity_logs"


def test_environment_override_wins(monkeypatch, tmp_path):
    override = tmp_path / "portable-data"
    monkeypatch.setenv("BRAINALIGN_WM_DATA_ROOT", str(override))
    assert get_path("data_root") == override


def test_load_config_injects_effective_paths(tmp_path):
    paths_file = tmp_path / "paths.yaml"
    paths_file.write_text(
        "paths:\n"
        "  data_root: data\n"
        "  stimuli: stimuli\n"
        "  results: results\n"
    )
    cfg = load_config(paths_config=paths_file)
    assert Path(cfg["paths"]["data_root"]) == PROJECT_ROOT / "data"
    assert Path(cfg["paths"]["results"]) == PROJECT_ROOT / "results"
