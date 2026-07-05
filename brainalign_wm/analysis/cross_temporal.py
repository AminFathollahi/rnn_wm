"""Cross-temporal (temporal generalization) decoding (King & Dehaene 2014;
Stokes 2015; Spaak et al. 2017). A decoder is trained at each timebin and
tested at every other timebin, producing an [n_train_bins, n_test_bins]
generalization matrix. Stable coding is indicated by high off-diagonal
generalization; dynamic coding by a high diagonal with low off-diagonal
values.

Also implements the Libby & Buschman memory/sensation cross-decode: same
machinery, decoding "previously-held item" vs. "currently-probed item"
labels across the delay instead of a single condition variable.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def cross_temporal_decoding(
    X: np.ndarray, y: np.ndarray, n_folds: int = 4, seed: int = 0
) -> np.ndarray:
    """X: [n_trials, n_timebins, n_units]; y: [n_trials] labels.
    Returns [n_timebins, n_timebins] cross-validated accuracy matrix
    (train at row-time, test at column-time). Features (raw firing
    rates/hidden-unit activations, unbounded and often large-magnitude) are
    standardized per training fold before fitting `LogisticRegression` --
    both for L2 regularization to treat units comparably regardless of
    scale, and because unscaled features were causing `lbfgs` to routinely
    fail to converge within `max_iter`, silently returning a poorly-fit
    classifier rather than erroring (audit-fix wiring, found while
    profiling H5's wall-clock cost across a full grid)."""
    n_trials, n_time, n_units = X.shape
    classes = np.unique(y)
    if len(classes) < 2:
        return np.full((n_time, n_time), np.nan)
    skf = StratifiedKFold(n_splits=min(n_folds, np.min(np.bincount(_encode(y)))), shuffle=True, random_state=seed)
    y_enc = _encode(y)
    mat = np.zeros((n_time, n_time))
    n_splits_run = 0
    for train_idx, test_idx in skf.split(X[:, 0, :], y_enc):
        n_splits_run += 1
        for t_train in range(n_time):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx, t_train, :])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_enc[train_idx])
            for t_test in range(n_time):
                X_test = scaler.transform(X[test_idx, t_test, :])
                acc = clf.score(X_test, y_enc[test_idx])
                mat[t_train, t_test] += acc
    mat /= max(n_splits_run, 1)
    return mat


def _encode(y: np.ndarray) -> np.ndarray:
    classes, inv = np.unique(y, return_inverse=True)
    return inv


def stability_index(gen_matrix: np.ndarray) -> float:
    """Off-diagonal / diagonal generalization ratio -- high => stable code,
    low => dynamic code (protocol H5)."""
    n = gen_matrix.shape[0]
    diag = np.diag(gen_matrix)
    off_mask = ~np.eye(n, dtype=bool)
    off = gen_matrix[off_mask]
    diag_mean = np.nanmean(diag)
    off_mean = np.nanmean(off)
    if diag_mean <= 0:
        return 0.0
    return float(off_mean / diag_mean)


def memory_sensation_cross_decode(
    X: np.ndarray, y_prev_item: np.ndarray, y_curr_item: np.ndarray, n_folds: int = 4, seed: int = 0
) -> dict:
    """Libby & Buschman-style analysis: decode previously-held versus
    currently-probed item identity across the same delay activity,
    quantifying subspace separation between memory and sensation."""
    mat_prev = cross_temporal_decoding(X, y_prev_item, n_folds=n_folds, seed=seed)
    mat_curr = cross_temporal_decoding(X, y_curr_item, n_folds=n_folds, seed=seed)
    return {
        "prev_item_matrix": mat_prev,
        "curr_item_matrix": mat_curr,
        "prev_item_stability": stability_index(mat_prev),
        "curr_item_stability": stability_index(mat_curr),
    }
