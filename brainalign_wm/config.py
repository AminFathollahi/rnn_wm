"""Central configuration and runtime-path resolution.

Scientific settings live in ``configs/config.yaml``.  Machine-specific
storage locations live in ``configs/paths.yaml`` and are injected whenever
an experiment config is loaded.  This keeps absolute mount points out of
training and analysis code while ensuring resolved configs still record the
effective paths used for a run.

The paths file can be replaced with ``BRAINALIGN_WM_PATHS_CONFIG``.  Each
individual path can also be overridden with the environment variables in
``PATH_ENV_VARS``; command-line jobs and schedulers can therefore select a
storage layout without modifying tracked files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_PATHS_CONFIG_PATH = PROJECT_ROOT / "configs" / "paths.yaml"

PATH_ENV_VARS = {
    "data_root": "BRAINALIGN_WM_DATA_ROOT",
    "stimuli": "BRAINALIGN_WM_STIMULI_ROOT",
    "results": "BRAINALIGN_WM_RESULTS_ROOT",
    "feature_cache": "BRAINALIGN_WM_FEATURE_CACHE_ROOT",
    "activity_logs": "BRAINALIGN_WM_ACTIVITY_LOGS_ROOT",
}


def _project_path(path: str | os.PathLike[str]) -> Path:
    """Expand a path and anchor relative values at the repository root."""
    expanded = Path(os.path.expandvars(os.path.expanduser(os.fspath(path))))
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve(strict=False)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"expected a YAML mapping in {path}, got {type(value).__name__}")
    return value


def paths_config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the selected path-config file.

    Explicit arguments take precedence over ``BRAINALIGN_WM_PATHS_CONFIG``;
    both may be relative to the project root.
    """
    selected = path or os.environ.get("BRAINALIGN_WM_PATHS_CONFIG") or DEFAULT_PATHS_CONFIG_PATH
    return _project_path(selected)


def load_paths(path: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Load, override, derive, and resolve all runtime paths."""
    source = paths_config_path(path)
    raw = _yaml_mapping(source)
    values = raw.get("paths", raw)
    if not isinstance(values, dict):
        raise TypeError(f"expected a 'paths' mapping in {source}")

    selected: dict[str, str | os.PathLike[str]] = {}
    for key, env_name in PATH_ENV_VARS.items():
        value = os.environ.get(env_name, values.get(key))
        if value not in (None, ""):
            selected[key] = value

    for required in ("data_root", "stimuli", "results"):
        if required not in selected:
            raise KeyError(f"missing required path '{required}' in {source}")

    resolved = {key: _project_path(value) for key, value in selected.items()}
    resolved.setdefault("feature_cache", resolved["results"] / "feat_cache")
    resolved.setdefault("activity_logs", resolved["results"] / "activity_logs")
    return resolved


def get_path(name: str, paths_config: str | os.PathLike[str] | None = None) -> Path:
    """Return one resolved runtime path by configuration key."""
    paths = load_paths(paths_config)
    try:
        return paths[name]
    except KeyError as exc:
        known = ", ".join(sorted(paths))
        raise KeyError(f"unknown runtime path {name!r}; configured keys: {known}") from exc


def config_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the experiment config, honoring ``BRAINALIGN_WM_CONFIG``."""
    selected = path or os.environ.get("BRAINALIGN_WM_CONFIG") or DEFAULT_CONFIG_PATH
    return _project_path(selected)


def apply_runtime_paths(
    config: Mapping[str, Any], paths_config: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Copy a config and replace its path block with effective runtime paths."""
    merged = dict(config)
    merged["paths"] = {key: str(value) for key, value in load_paths(paths_config).items()}
    return merged


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    paths_config: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load an experiment/resolved config with centralized runtime paths."""
    return apply_runtime_paths(_yaml_mapping(config_path(path)), paths_config)


def main() -> int:
    """Print the effective path configuration for preflight/provenance."""
    payload = {
        "experiment_config": str(config_path()),
        "paths_config": str(paths_config_path()),
        "paths": {key: str(value) for key, value in load_paths().items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
