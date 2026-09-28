"""Interpretability guards for the two delay-period dynamics measures.

The cross-temporal stability ratio is a ratio of two accuracies, so it
tends to 1 when both are at chance -- numerically indistinguishable from a
perfectly stable code. The persistent-activity index divides by
`|baseline| + 1`, an absolute-rate regularizer that makes the index depend
on the units the activity happens to be measured in.
"""
from __future__ import annotations

import numpy as np

from brainalign_wm.analysis.cross_temporal import cross_temporal_stability, stability_index


def _no_information_data(seed: int = 0):
    rng = np.random.RandomState(seed)
    X = rng.randn(90, 5, 8)
    y = rng.randint(0, 3, size=90)
    return X, y


def _stable_code_data(seed: int = 0):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, size=90)
    direction = rng.randn(8)
    X = rng.randn(90, 5, 8) * 0.5
    X += y[:, None, None] * direction[None, None, :] * 4.0
    return X, y


def test_labelless_data_gives_chance_diagonal_and_undefined_ratio():
    X, y = _no_information_data()
    result = cross_temporal_stability(X, y, n_folds=3, seed=0, n_permutations=40)

    assert abs(result["diagonal_accuracy"] - result["permutation_null_mean"]) < 0.1
    assert result["permutation_p_value"] > 0.05
    assert not result["diagonal_above_null"]
    assert np.isnan(result["stability_ratio"])
    # The bare ratio is the defect this guards: it reports a near-perfectly
    # stable code for data whose labels carry no information at all.
    assert result["stability_index"] > 0.9


def test_decodable_stable_code_gives_defined_ratio_above_null():
    X, y = _stable_code_data()
    result = cross_temporal_stability(X, y, n_folds=3, seed=0, n_permutations=40)

    assert result["diagonal_accuracy"] > result["chance_accuracy"] + 0.2
    assert result["permutation_p_value"] < 0.05
    assert result["diagonal_above_null"]
    assert result["stability_ratio"] == stability_index(result["generalization_matrix"])
    assert result["stability_ratio"] > 0.8


def test_diagonal_only_decoding_matches_full_diagonal():
    from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding

    X, y = _stable_code_data(seed=3)
    full = cross_temporal_decoding(X, y, n_folds=3, seed=0)
    diag_only = cross_temporal_decoding(X, y, n_folds=3, seed=0, diagonal_only=True)
    assert np.allclose(np.diag(full), np.diag(diag_only))
    assert np.isnan(diag_only[0, 1])


def _persistence_data(seed: int = 0, scale: float = 1.0):
    rng = np.random.RandomState(seed)
    n_units, n_trials = 12, 60
    labels = np.repeat([1, 2, 3], n_trials // 3)
    baseline = rng.rand(n_units) * 0.2
    maintain = rng.randn(n_units, n_trials) * 0.1 + baseline[:, None]
    # Half the units are elevated during maintenance at condition 3.
    maintain[: n_units // 2, labels == 3] += 0.8
    return maintain * scale, baseline * scale, labels


def test_standardized_persistence_is_invariant_to_activity_rescaling():
    from brainalign_wm.analysis.persistence import (
        persistent_activity_index,
        standardized_persistence_index,
    )

    maintain, baseline, labels = _persistence_data()
    maintain_scaled, baseline_scaled, _ = _persistence_data(scale=10.0)

    standardized = standardized_persistence_index(maintain, baseline, labels, seed=0)
    standardized_scaled = standardized_persistence_index(maintain_scaled, baseline_scaled, labels, seed=0)
    assert np.allclose(standardized, standardized_scaled)

    # The retained absolute-rate index moves under the same rescaling --
    # the units dependence this estimator exists to remove.
    conditions = np.unique(labels)
    by_condition = np.stack([maintain[:, labels == c].mean(axis=1) for c in conditions], axis=1)
    by_condition_scaled = np.stack([maintain_scaled[:, labels == c].mean(axis=1) for c in conditions], axis=1)
    old = persistent_activity_index(by_condition, baseline[:, None], conditions)
    old_scaled = persistent_activity_index(by_condition_scaled, baseline_scaled[:, None], conditions)
    assert not np.allclose(old, old_scaled)


def test_standardized_persistence_separates_elevated_from_flat_units():
    from brainalign_wm.analysis.persistence import standardized_persistence_index

    maintain, baseline, labels = _persistence_data()
    index = standardized_persistence_index(maintain, baseline, labels, seed=0)
    assert index[:6].min() > index[6:].max()


def test_standardized_persistence_bounds_saturated_units():
    """Saturated units sit at a fixed point during maintenance with an
    across-trial SD orders of magnitude below the rest of the population.
    Dividing by that SD is what put the population mean in the hundreds,
    so the exactly-constant case, the near-constant case (SD just above any
    validity threshold) and the population mean are all pinned here."""
    from brainalign_wm.analysis.persistence import standardized_persistence_index

    maintain, baseline, labels = _persistence_data(seed=1)
    rng = np.random.RandomState(2)

    exactly_constant = maintain.copy()
    exactly_constant[-1, :] = 0.5
    index_exact = standardized_persistence_index(exactly_constant, baseline, labels, seed=0)
    assert np.isfinite(index_exact[-1])
    assert abs(index_exact[-1]) < 50

    # A whole saturated sub-population, the regime a trained recurrent core
    # is actually in: every unit clamped at the activation bound well away
    # from its own fixation baseline.
    saturated = maintain.copy()
    saturated_baseline = baseline.copy()
    saturated[6:, :] = 1.0 + rng.randn(6, maintain.shape[1]) * 1e-6
    saturated_baseline[6:] = -1.0
    index = standardized_persistence_index(saturated, saturated_baseline, labels, seed=0)
    unfloored = (
        saturated.mean(axis=1) - saturated_baseline
    ) / saturated.std(axis=1, ddof=1)
    assert abs(unfloored).max() > 1e5
    assert np.isfinite(index).all()
    assert abs(index).max() < 50
    assert abs(np.mean(index)) < 50

    # Scale invariance must survive the floor: the denominator rescales
    # with the data, so a 10x rescaling leaves every index unchanged.
    rescaled = standardized_persistence_index(
        saturated * 10.0, saturated_baseline * 10.0, labels, seed=0
    )
    assert np.allclose(index, rescaled)


def test_standardized_persistence_selection_is_cross_validated():
    from brainalign_wm.analysis.persistence import standardized_persistence_index

    # Pure noise over baseline: an uncross-validated maximum over
    # conditions is positively biased, a cross-validated one is not.
    rng = np.random.RandomState(1)
    n_units, n_trials = 40, 90
    labels = np.repeat([1, 2, 3], n_trials // 3)
    baseline = np.zeros(n_units)
    maintain = rng.randn(n_units, n_trials)

    index = standardized_persistence_index(maintain, baseline, labels, seed=0)
    uncrossvalidated = np.stack(
        [maintain[:, labels == c].mean(axis=1) for c in np.unique(labels)], axis=1
    ).max(axis=1) / maintain.std(axis=1, ddof=1)
    assert abs(index.mean()) < uncrossvalidated.mean()
    assert abs(index.mean()) < 0.15
