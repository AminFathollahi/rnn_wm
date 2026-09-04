"""SimulatedBrain tests: satisfies the NeuralDataset contract, and the
scrambled control is distinguishable from the real data."""
import numpy as np
import pytest
import yaml

from brainalign_wm.config import load_config

from brainalign_wm.neural.dataset_contract import validate_dataset
from brainalign_wm.neural.sim_brain.spiking_generator import SimulatedBrain, make_scrambled

ROOT = __file__.rsplit("/tests/", 1)[0]
SIM_CFG = load_config()["sim_brain"]

_SMALL_CFG = {**SIM_CFG, "n_sessions": 2, "n_units_per_session": 6, "n_patients": 2, "trials_per_condition": 4}


def test_satisfies_neural_dataset_contract():
    brain = SimulatedBrain(_SMALL_CFG, seed=0)
    validate_dataset(brain, region=None)
    assert set(brain.regions()) == {"sim"}
    assert len(brain.sessions()) == _SMALL_CFG["n_sessions"]
    assert brain.patient_of(brain.sessions()[0]) in brain._patients


def test_spike_statistics_sane():
    brain = SimulatedBrain(_SMALL_CFG, seed=0)
    unit = brain.units()[0]
    spikes = brain.spike_times(unit)
    assert len(spikes) > 0
    assert (np.diff(np.sort(spikes)) >= 0).all()
    # refractory period is enforced WITHIN a trial's own spike train (each
    # trial's times are trial-relative, so concatenating across trials before
    # checking ISIs would spuriously "violate" it across unrelated trials).
    trials = brain.trials()
    checked_any_multi_spike_trial = False
    for tid in trials.trial_id.values:
        trial_spikes = brain._spikes.get((unit, tid))
        if trial_spikes is None or len(trial_spikes) < 2:
            continue
        checked_any_multi_spike_trial = True
        isi = np.diff(np.sort(trial_spikes))
        assert (isi >= brain.refractory_s - 1e-9).all()
    assert checked_any_multi_spike_trial, "no trial had >=2 spikes to check refractory ISI on"


def test_rates_shape_and_nonnegative():
    brain = SimulatedBrain(_SMALL_CFG, seed=0)
    rates = brain.rates(None, brain.bin_ms, ["maintain"])
    assert rates.ndim == 3
    assert rates.shape[0] == len(brain.units())
    assert rates.shape[1] == len(brain.trials())
    assert (rates >= 0).all()


def test_scrambled_destroys_condition_locked_structure():
    """A direct, cheap correlate of the full recovery gate: per-unit
    condition-mean rate variance (across the coarse conditions) should be
    much larger for the real dataset than for the scrambled control, since
    scrambling reassigns which trial's latents drive which spike train."""
    brain = SimulatedBrain(_SMALL_CFG, seed=0)
    scrambled = make_scrambled(_SMALL_CFG, seed=0)
    patt_real = brain.response_patterns(None, "maintain")  # [n_cond, n_units]
    patt_scr = scrambled.response_patterns(None, "maintain")
    var_real = patt_real.var(axis=0).mean()
    var_scr = patt_scr.var(axis=0).mean()
    assert var_real > var_scr


def test_noise_ceiling_bounds_sane():
    brain = SimulatedBrain(_SMALL_CFG, seed=0)
    lower, upper = brain.noise_ceiling(None, "maintain")
    assert 0.0 <= lower <= upper <= 1.0 + 1e-6
