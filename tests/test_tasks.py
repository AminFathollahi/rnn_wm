"""M4 gates (protocol §11.5, §14): trial epoch sequence/step-counts correct,
load matches held-set size, c_t one-hot consistent, lure rate matches
config, curriculum phase behavior, determinism."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from brainalign_wm.tasks.curriculum import CurriculumSchedule, phase_at
from brainalign_wm.tasks.sternberg import SternbergGenerator, context_vector
from brainalign_wm.tasks.generator import TaskGenerator

ROOT = Path(__file__).resolve().parents[1]
FULL_CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
STIMULI_READY = (ROOT / "stimuli" / "faces").exists()

pytestmark_needs_stimuli = pytest.mark.skipif(
    not STIMULI_READY, reason="stimuli/ pool not built yet -- run scripts/build_stimuli_pool.py"
)


def _make_bank():
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    return ImageTokenBank(
        stimuli_root=ROOT / "stimuli",
        categories=FULL_CFG["task"]["categories"],
        feature_cache_path=ROOT / "results" / "feat_cache" / "test_image_token_bank.npy",
        seed=0,
    )


# ---------------- context_vector (no ImageTokenBank needed) ----------------

def test_context_vector_one_hot_consistency():
    c = context_vector(load=2, epoch="encode", lure_flag=False)
    assert len(c) == 10
    assert c[0] == 1.0 and c[1] == 0.0  # WM_family, aux_family
    assert c[2] == 0.0 and c[3] == 1.0 and c[4] == 0.0  # load1,load2,load3 one-hot -> load2
    assert c[5] == 1.0 and c[6] == 0.0 and c[7] == 0.0  # epoch_encode
    assert c[8] == 0.0  # lure_flag only live at probe
    assert c[9] == 0.0  # rule_flag reserved


def test_context_vector_lure_only_at_probe():
    c_probe_lure = context_vector(load=1, epoch="probe", lure_flag=True)
    assert c_probe_lure[8] == 1.0
    c_maintain_lure = context_vector(load=1, epoch="maintain", lure_flag=True)
    assert c_maintain_lure[8] == 0.0  # lure_flag not asserted outside probe


# ---------------- curriculum (no ImageTokenBank needed) ----------------

def test_phase_boundaries():
    assert phase_at(0, 1000, warmup_frac=0.2, ramp_frac=0.6) == "warmup"
    assert phase_at(199, 1000, warmup_frac=0.2, ramp_frac=0.6) == "warmup"
    assert phase_at(500, 1000, warmup_frac=0.2, ramp_frac=0.6) == "ramp"
    assert phase_at(999, 1000, warmup_frac=0.2, ramp_frac=0.6) == "target"


def test_curriculum_warmup_load1_no_lures():
    sched = CurriculumSchedule.from_config(FULL_CFG)
    p = sched.params_for(0, 1000)
    assert p["phase"] == "warmup"
    assert p["loads"] == [1]
    assert p["lure_fraction"] == 0.0
    assert p["maintain_steps"] < FULL_CFG["task"]["maintain_steps"]


def test_curriculum_ramp_lure_interpolates():
    sched = CurriculumSchedule.from_config(FULL_CFG)
    total = 1000
    warmup_end = int(sched.warmup_frac * total)
    ramp_end = int((sched.warmup_frac + sched.ramp_frac) * total)
    p_start = sched.params_for(warmup_end + 1, total)
    p_end = sched.params_for(ramp_end - 1, total)
    assert p_start["lure_fraction"] < 0.1 * sched.full_lure_fraction
    assert p_end["lure_fraction"] > 0.8 * sched.full_lure_fraction
    assert p_start["loads"] == sorted(FULL_CFG["task"]["loads"])


def test_curriculum_target_matches_full_config():
    sched = CurriculumSchedule.from_config(FULL_CFG)
    p = sched.params_for(999, 1000)
    assert p["phase"] == "target"
    assert p["lure_fraction"] == FULL_CFG["task"]["lure_fraction"]
    assert p["maintain_steps"] == FULL_CFG["task"]["maintain_steps"]


# ---------------- SternbergGenerator (needs a real ImageTokenBank) ----------------

@pytestmark_needs_stimuli
@pytest.mark.parametrize("load", [1, 2, 3])
def test_trial_epoch_sequence_and_load(load):
    bank = _make_bank()
    gen = SternbergGenerator(FULL_CFG, bank)
    rng = np.random.RandomState(0)
    steps = gen.generate_trial(
        rng, loads=[load], lure_fraction=0.3, maintain_steps=FULL_CFG["task"]["maintain_steps"],
        trial_id=0,
    )
    epochs_seen = [s.epoch for s in steps]
    assert epochs_seen[0] == "fixation"
    assert epochs_seen[-1] == "iti"
    for s in steps:
        assert s.load == load
        assert len(s.held_items) == load
    encode_count = sum(1 for e in epochs_seen if e == "encode")
    assert encode_count == load * FULL_CFG["task"]["encode_steps"]
    maintain_count = sum(1 for e in epochs_seen if e == "maintain")
    assert maintain_count == FULL_CFG["task"]["maintain_steps"]


@pytestmark_needs_stimuli
def test_lure_fraction_matches_config_in_aggregate():
    bank = _make_bank()
    gen = SternbergGenerator(FULL_CFG, bank)
    rng = np.random.RandomState(1)
    n_trials = 300
    lure_fraction = 0.4
    n_not_in_set = 0
    n_lures = 0
    for i in range(n_trials):
        steps = gen.generate_trial(rng, loads=[1], lure_fraction=lure_fraction, maintain_steps=5, trial_id=i)
        probe_step = next(s for s in steps if s.epoch == "probe")
        if not probe_step.in_set:
            n_not_in_set += 1
            if probe_step.lure_flag:
                n_lures += 1
    observed = n_lures / n_not_in_set
    assert abs(observed - lure_fraction) < 0.12, f"observed lure rate {observed:.2f} vs configured {lure_fraction}"


@pytestmark_needs_stimuli
def test_determinism_same_seed_identical_trial():
    bank = _make_bank()
    gen = TaskGenerator(FULL_CFG, bank, seed=42)
    steps_a = gen.sample_trial(step_idx=7, total_steps=1000)
    steps_b = gen.sample_trial(step_idx=7, total_steps=1000)
    assert [s.image_id for s in steps_a] == [s.image_id for s in steps_b]
    assert [s.epoch for s in steps_a] == [s.epoch for s in steps_b]
    assert steps_a[0].held_items == steps_b[0].held_items
