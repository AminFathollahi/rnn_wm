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
