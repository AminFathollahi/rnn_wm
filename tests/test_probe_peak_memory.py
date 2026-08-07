"""`scripts/probe_peak_memory.py` measures memory against `run_grid._run_mib`'s
prediction (comments.txt §21.1, D49). If the probe's load-3 trial were a
different length than the scheduler predicts for, it would silently agree
with a broken memory estimate -- this test is the guard against that."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_grid as rg  # noqa: E402
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
    with pytest.raises(ValueError):
        _bits("M111")
