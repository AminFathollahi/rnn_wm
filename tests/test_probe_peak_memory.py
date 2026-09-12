"""`scripts/probe_peak_memory.py` measures memory against `run_grid._run_mib`'s
prediction. If the probe's load-3 trial were a different length than the
scheduler predicts for, or if it built a tagged variant's core at the
config's default width, it would silently agree with a broken memory
estimate -- these tests are the guard against that."""
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_grid as rg  # noqa: E402
import scripts.probe_peak_memory as probe  # noqa: E402
from brainalign_wm.training.train import DEFAULT_FLAT_GRID, _run_trial  # noqa: E402
from brainalign_wm.training.model_variants import FLAT_CONTROLS  # noqa: E402
from scripts.probe_peak_memory import _bits, build_load3_batch, load_full_cfg  # noqa: E402

STIMULI_READY = (ROOT / "stimuli").exists()
pytestmark = pytest.mark.skipif(not STIMULI_READY, reason="stimuli/ pool not built yet")


def _make_task_gen(cfg):
    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    image_bank = ImageTokenBank(
        stimuli_root=ROOT / cfg["paths"]["stimuli"], categories=cfg["task"]["categories"],
        feature_cache_path=ROOT / cfg["paths"]["feature_cache"] / "image_token_bank.npy", seed=0,
    )
    return TaskGenerator(cfg, image_bank, seed=0)


def test_load3_trial_length_matches_run_grid_trial_ticks_prediction():
    cfg = load_full_cfg()
    task_gen = _make_task_gen(cfg)
    batch = build_load3_batch(task_gen, cfg, batch_size=4, seed=0)
    assert len(batch[0]) == rg._trial_ticks(cfg)


def test_bits_parses_model_id():
    assert _bits("M11111") == (1, 1, 1, 1, 1)
    assert _bits("M00000") == (0, 0, 0, 0, 0)
    assert _bits("M11011") == (1, 1, 0, 1, 1)
    assert _bits("M00010_dense289") == (0, 0, 0, 1, 0)
    with pytest.raises(ValueError):
        _bits("M111")
    with pytest.raises(ValueError):
        _bits("M000001_dense289")


@pytest.mark.parametrize("tag", sorted(FLAT_CONTROLS))
def test_tagged_model_id_is_probed_at_the_core_it_would_train(tag):
    """A crossed flat control is measured at its own width and connectivity,
    not at the resolved config's default flat core."""
    cfg = load_full_cfg(f"M00000_{tag}")["model"]
    for key, value in FLAT_CONTROLS[tag].items():
        assert cfg[key] == value
    assert load_full_cfg("M00000")["model"] == load_full_cfg()["model"]


def test_flat_grid_defaults_match_model_configuration():
    expected = tuple(load_full_cfg()["model"]["flat_grid"])
    assert DEFAULT_FLAT_GRID == expected
    assert probe.DEFAULT_FLAT_GRID == expected
    assert inspect.signature(_run_trial).parameters["flat_grid"].default == expected
