"""Label-only reference RDMs and the RDM-level semipartial estimator."""
from __future__ import annotations

import numpy as np

from brainalign_wm.analysis.task_structure import (
    label_field_rdm,
    normalized_hamming_rdm,
    out_of_sample_semipartial,
    probe_task_structure_rdms,
    rdm_vectors,
    semipartial_correlation,
)

CONDITIONS = [(load, in_set, correct) for load in (1, 2, 3) for in_set in (False, True) for correct in (False, True)]


def test_field_rdm_is_zero_only_within_a_field_value():
    rdm = label_field_rdm(CONDITIONS, 0)
    assert rdm.shape == (12, 12)
    assert np.all(np.diag(rdm) == 0.0)
    # conditions 0-3 all have load 1, condition 4 has load 2
    assert rdm[0, 3] == 0.0
    assert rdm[0, 4] == 1.0


def test_normalized_hamming_is_the_mean_of_the_field_rdms():
    combined = normalized_hamming_rdm(CONDITIONS)
    fields = np.mean([label_field_rdm(CONDITIONS, f) for f in range(3)], axis=0)
    assert np.allclose(combined, fields)
    assert set(np.unique(combined)) <= {0.0, 1 / 3, 2 / 3, 1.0}


def test_probe_rdms_cover_every_field_plus_the_combination():
    rdms = probe_task_structure_rdms(CONDITIONS)
    assert set(rdms) == {"load_only", "response_only", "correctness_only", "combined_task_structure"}


def test_semipartial_is_zero_when_the_model_is_the_task_rdm_itself():
    """A model RDM that is exactly the task structure has nothing left
    after the task structure is removed."""
    rng = np.random.RandomState(0)
    task = normalized_hamming_rdm(CONDITIONS)
    neural = task + rng.randn(12, 12) * 0.05
    neural = (neural + neural.T) / 2
    nv, mv, tv = rdm_vectors(neural, task, task)
    assert abs(semipartial_correlation(nv, mv, tv)) < 1e-6


def test_semipartial_recovers_structure_the_task_rdm_does_not_carry():
    """Neural and model share a component orthogonal to the task labels;
    the raw correlation and the semipartial should both be clearly
    positive, and removing the task structure must not destroy it."""
    rng = np.random.RandomState(1)
    task = normalized_hamming_rdm(CONDITIONS)
    extra = rng.randn(12, 12)
    extra = (extra + extra.T) / 2
    np.fill_diagonal(extra, 0.0)
    neural = task + extra
    model = task + extra + rng.randn(12, 12) * 0.1
    model = (model + model.T) / 2
    nv, mv, tv = rdm_vectors(neural, model, task)
    assert semipartial_correlation(nv, mv, tv) > 0.5


def test_out_of_sample_semipartial_uses_the_fitted_coefficients():
    """Fitting on one sample and evaluating on an independent one drawn
    from the same process gives a similar answer, not a degenerate one."""
    rng = np.random.RandomState(2)

    def sample():
        task = normalized_hamming_rdm(CONDITIONS)
        extra = rng.randn(12, 12)
        extra = (extra + extra.T) / 2
        np.fill_diagonal(extra, 0.0)
        return rdm_vectors(task + extra, task + extra + rng.randn(12, 12) * 0.1, task)

    a, b = sample(), sample()
    held_out = out_of_sample_semipartial(a, b)
    assert np.isfinite(held_out)
    assert held_out > 0.3


def test_nan_entries_are_dropped_rather_than_poisoning_the_estimate():
    """Stratified RDMs leave cross-stratum entries NaN by construction."""
    rng = np.random.RandomState(3)
    task = normalized_hamming_rdm(CONDITIONS)
    extra = rng.randn(12, 12)
    extra = (extra + extra.T) / 2
    np.fill_diagonal(extra, 0.0)
    neural, model = task + extra, task + extra
    nv, mv, tv = rdm_vectors(neural, model, task)
    nv = nv.copy()
    nv[:5] = np.nan
    value = semipartial_correlation(nv, mv, tv)
    assert np.isfinite(value) and value > 0.5
