"""`build_resolved_config` must record every run-dict key that picks the
model class, not just substrate/supervision/init -- a flat (S=0) and a
hierarchical (S=1) vanilla run once resolved to the same config and the
same config_hash because the S/M/P/T/D cell selector was absent."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_grid import build_resolved_config, config_hash

FULL_CFG = {"model": {"substrate": "gru"}, "train": {"supervision": "legacy"}, "tiers": {}}


def test_flat_and_hierarchical_runs_resolve_to_different_config_hashes():
    run_flat = {"S": 0, "M": 0, "P": 0, "T": 0, "D": 0, "substrate": "vanilla",
                "supervision": "legacy", "recurrent_init_spectral_radius": 1.0}
    run_hier = {**run_flat, "S": 1}

    resolved_flat = build_resolved_config(FULL_CFG, {}, "full", run=run_flat)
    resolved_hier = build_resolved_config(FULL_CFG, {}, "full", run=run_hier)

    assert resolved_flat["model"]["cell"]["S"] == 0
    assert resolved_hier["model"]["cell"]["S"] == 1
    assert config_hash(resolved_flat) != config_hash(resolved_hier)


def test_run_none_omits_cell_key_and_reproduces_prior_output():
    resolved = build_resolved_config(FULL_CFG, {}, "full")
    assert "cell" not in resolved["model"]
