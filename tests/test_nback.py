"""N-back task-generator tests (Phase 6, comments.txt §5): sequence length
and per-position field consistency, match rate matches configuration in
aggregate, and trial generation is deterministic. Mirrors tests/test_tasks.py."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from brainalign_wm.tasks.nback import NBackGenerator

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


@pytestmark_needs_stimuli
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("feature", ["identity", "category"])
def test_sequence_length_and_step_consistency(n, feature):
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    rng = np.random.RandomState(0)
    seq_len = FULL_CFG["nback"]["sequence_length"]
    steps = gen.generate_trial(
        rng, n=n, feature=feature, sequence_length=seq_len,
        match_fraction=FULL_CFG["nback"]["match_fraction"], trial_id=0,
    )
    assert len(steps) == seq_len
    for i, s in enumerate(steps):
        assert s.t_in_trial == i
        assert s.n == n
        assert s.feature == feature
        if i < n:
            assert s.is_match is None
        else:
            assert isinstance(s.is_match, bool)
            back = steps[i - n]
            if feature == "identity":
                assert (s.image_id == back.image_id) == s.is_match
            else:
                assert (s.category == back.category) == s.is_match


@pytestmark_needs_stimuli
@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("feature", ["identity", "category"])
def test_match_fraction_matches_config_in_aggregate(n, feature):
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    rng = np.random.RandomState(1)
    seq_len = FULL_CFG["nback"]["sequence_length"]
    match_fraction = 0.4
    n_trials = 150
    n_scored, n_matched = 0, 0
    for i in range(n_trials):
        steps = gen.generate_trial(
            rng, n=n, feature=feature, sequence_length=seq_len,
            match_fraction=match_fraction, trial_id=i,
        )
        for s in steps:
            if s.is_match is None:  # i < n: not a scoreable position
                continue
            n_scored += 1
            n_matched += int(s.is_match)
    observed = n_matched / n_scored
    assert abs(observed - match_fraction) < 0.08, f"observed match rate {observed:.2f} vs configured {match_fraction}"


@pytestmark_needs_stimuli
def test_determinism_same_seed_identical_trial():
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    steps_a = gen.generate_trial(
        np.random.RandomState(42), n=2, feature="category",
        sequence_length=FULL_CFG["nback"]["sequence_length"],
        match_fraction=FULL_CFG["nback"]["match_fraction"], trial_id=7,
    )
    steps_b = gen.generate_trial(
        np.random.RandomState(42), n=2, feature="category",
        sequence_length=FULL_CFG["nback"]["sequence_length"],
        match_fraction=FULL_CFG["nback"]["match_fraction"], trial_id=7,
    )
    assert [s.image_id for s in steps_a] == [s.image_id for s in steps_b]
    assert [s.is_match for s in steps_a] == [s.is_match for s in steps_b]


@pytestmark_needs_stimuli
def test_sequence_length_must_exceed_n():
    bank = _make_bank()
    gen = NBackGenerator(FULL_CFG, bank)
    with pytest.raises(ValueError):
        gen.generate_trial(
            np.random.RandomState(0), n=3, feature="identity",
            sequence_length=3, match_fraction=0.4, trial_id=0,
        )
