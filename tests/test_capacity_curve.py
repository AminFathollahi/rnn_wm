"""Item 9.1 (comments.txt §5): checks for `scripts/analyze_capacity_curve.py`'s
own non-trivial logic -- the knee heuristic (pure function, planted-answer
test) and the eval-rollout capture (real-model smoke test, skipped if the
stimuli pool isn't built, same convention as `tests/test_training.py`)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from brainalign_wm.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config()
STIMULI_READY = (ROOT / "stimuli" / "faces").exists()

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.analyze_capacity_curve import _find_knee, _rollout_hidden_states  # noqa: E402


def test_find_knee_detects_early_plateau():
    H_vals = [2, 4, 8, 16, 32, 64, 128, 256]
    acc = [0.4, 0.6, 0.90, 0.905, 0.91, 0.908, 0.912, 0.91]  # plateaus at H=8
    assert _find_knee(H_vals, acc) == 8


def test_find_knee_falls_back_to_largest_h_when_monotonic():
    H_vals = [2, 4, 8, 16, 32, 64, 128, 256]
    acc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]  # never plateaus
    assert _find_knee(H_vals, acc) == 256


pytestmark = pytest.mark.skipif(not STIMULI_READY, reason="stimuli/ pool not built yet")
torch = pytest.importorskip("torch")


def test_rollout_hidden_states_shape_and_finite():
    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.training.train import ROOT as MODEL_ROOT, _build_model
    from brainalign_wm.utils.device import get_device

    H = 4
    cfg_H = {**CFG, "model": {**CFG["model"], "flat_units": H}}
    device = get_device()
    front_end, core, heads = _build_model(cfg_H, S=0, M=0, P=0, device=device)
    image_bank = ImageTokenBank(
        stimuli_root=MODEL_ROOT / cfg_H["paths"]["stimuli"], categories=cfg_H["task"]["categories"],
        feature_cache_path=MODEL_ROOT / cfg_H["paths"]["feature_cache"] / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(cfg_H, image_bank, seed=0)

    Z = _rollout_hidden_states(front_end, core, heads, task_gen, image_bank, cfg_H, device, load=3, n_trials=6, eval_seed=1)

    assert Z.ndim == 3
    assert Z.shape[0] == 6
    assert Z.shape[2] == H
    assert Z.shape[1] > 0  # load-3 Sternberg has a non-empty maintain epoch
    assert np.all(np.isfinite(Z))
