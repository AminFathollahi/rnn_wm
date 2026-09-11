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
    X: np.ndarray, y: np.ndarray, n_folds: int = 4, seed: int = 0, diagonal_only: bool = False
) -> np.ndarray:
    """X: [n_trials, n_timebins, n_units]; y: [n_trials] labels.
    Returns [n_timebins, n_timebins] cross-validated accuracy matrix
    (train at row-time, test at column-time). Features (raw firing
    rates/hidden-unit activations, unbounded and often large-magnitude) are
    standardized per training fold before fitting `LogisticRegression` --
    both for L2 regularization to treat units comparably regardless of
    scale, and because unscaled features made `lbfgs` routinely fail to
    converge within `max_iter`, silently returning a poorly-fit classifier
    rather than erroring.

    `diagonal_only` scores each classifier at its own training timebin and
    leaves every off-diagonal entry NaN, skipping the O(n_timebins^2)
    scoring loop. Intended for label-permutation nulls of the diagonal,
    where the off-diagonal is never read."""
    n_trials, n_time, n_units = X.shape
    classes = np.unique(y)
    if len(classes) < 2:
        return np.full((n_time, n_time), np.nan)
    skf = StratifiedKFold(n_splits=min(n_folds, np.min(np.bincount(_encode(y)))), shuffle=True, random_state=seed)
    y_enc = _encode(y)
    mat = np.full((n_time, n_time), np.nan) if diagonal_only else np.zeros((n_time, n_time))
    if diagonal_only:
        np.fill_diagonal(mat, 0.0)
    n_splits_run = 0
    for train_idx, test_idx in skf.split(X[:, 0, :], y_enc):
        n_splits_run += 1
        for t_train in range(n_time):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx, t_train, :])
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_enc[train_idx])
            for t_test in [t_train] if diagonal_only else range(n_time):
                X_test = scaler.transform(X[test_idx, t_test, :])
                acc = clf.score(X_test, y_enc[test_idx])
                mat[t_train, t_test] += acc
    mat /= max(n_splits_run, 1)
    return mat


def _encode(y: np.ndarray) -> np.ndarray:
    classes, inv = np.unique(y, return_inverse=True)
    return inv


def stability_index(gen_matrix: np.ndarray) -> float:
    """Bare off-diagonal / diagonal generalization ratio -- high => stable
    code, low => dynamic code.

    This ratio is only interpretable where the diagonal itself carries
    decodable signal: when diagonal and off-diagonal are both at chance the
    ratio tends to 1, which is indistinguishable from a perfectly stable
    code. `cross_temporal_stability` returns this ratio together with the
    absolute diagonal accuracy and its label-permutation null, and leaves
    the ratio undefined where the diagonal is not above that null. Prefer
    it for any comparison; this function remains for callers that only need
    the raw ratio."""
    n = gen_matrix.shape[0]
    diag = np.diag(gen_matrix)
    off_mask = ~np.eye(n, dtype=bool)
    off = gen_matrix[off_mask]
    diag_mean = np.nanmean(diag)
    off_mean = np.nanmean(off)
    if diag_mean <= 0:
        return 0.0
    return float(off_mean / diag_mean)


def majority_class_accuracy(y: np.ndarray) -> float:
    """Accuracy of the best constant predictor for this label distribution
    -- the empirical chance level a decoder must beat, which equals 1/n
    classes only when the classes are balanced."""
    counts = np.bincount(_encode(np.asarray(y)))
    return float(counts.max() / counts.sum())


def cross_temporal_stability(
    X: np.ndarray, y: np.ndarray, n_folds: int = 4, seed: int = 0,
    n_permutations: int = 100, alpha: float = 0.05,
) -> dict:
    """Cross-temporal generalization with the evidence needed to read its
    stability ratio: the absolute diagonal accuracy, the empirical chance
    level for the actual class distribution, and a label-permutation null
    for the diagonal.

    `stability_ratio` is NaN (undefined, not 1.0 and not 0.0) wherever the
    diagonal does not exceed its permutation null at `alpha`, because the
    ratio of two chance-level quantities carries no information about
    coding stability. With `n_permutations=0` no null is built and the
    diagonal is instead required to exceed `majority_class_accuracy`.

    Returns the generalization matrix, `diagonal_accuracy`,
    `off_diagonal_accuracy`, `chance_accuracy`, `permutation_null_mean`,
    `permutation_null_sd`, `permutation_p_value`, `n_permutations`,
    `diagonal_above_null`, `stability_ratio`, and the bare
    `stability_index`."""
    mat = cross_temporal_decoding(X, y, n_folds=n_folds, seed=seed)
    n = mat.shape[0]
    diagonal = float(np.nanmean(np.diag(mat)))
    off_diagonal = float(np.nanmean(mat[~np.eye(n, dtype=bool)]))
    chance = majority_class_accuracy(y)

    null = np.empty(n_permutations)
    rng = np.random.RandomState(seed + 1)
    for i in range(n_permutations):
        permuted = rng.permutation(np.asarray(y))
        null_mat = cross_temporal_decoding(X, permuted, n_folds=n_folds, seed=seed, diagonal_only=True)
        null[i] = float(np.nanmean(np.diag(null_mat)))

    if n_permutations > 0:
        # (exceedances + 1) / (draws + 1): the observed value is itself one
        # draw from the null under the null hypothesis, so p is never 0.
        p_value = float((np.sum(null >= diagonal) + 1) / (n_permutations + 1))
        above_null = bool(p_value < alpha)
        null_mean, null_sd = float(null.mean()), float(null.std(ddof=1)) if n_permutations > 1 else float("nan")
    else:
        p_value, null_mean, null_sd = float("nan"), float("nan"), float("nan")
        above_null = bool(diagonal > chance)

    return {
        "generalization_matrix": mat,
        "diagonal_accuracy": diagonal,
        "off_diagonal_accuracy": off_diagonal,
        "chance_accuracy": chance,
        "permutation_null_mean": null_mean,
        "permutation_null_sd": null_sd,
        "permutation_p_value": p_value,
        "n_permutations": int(n_permutations),
        "diagonal_above_null": above_null,
        "stability_ratio": stability_index(mat) if above_null else float("nan"),
        "stability_index": stability_index(mat),
    }


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
