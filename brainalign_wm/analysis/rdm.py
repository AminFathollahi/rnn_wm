"""Cross-validated Mahalanobis (crossnobis) RDMs (protocol §9.4).

Unlike plain squared-Euclidean or correlation distance, crossnobis has
expectation ~0 under the null (no true condition difference): each distance
is a dot product of *independent* cross-validation folds' mean differences,
whitened by an estimated (shrinkage-regularized) noise covariance, so
positive-bias-inducing self-products never appear. This unbiasedness is
exactly what makes it safe to compare raw model/brain RDM magnitudes without
a systematic inflation confound (verified by `test_analysis.py`'s null-data
test, protocol §11.5).

Works agnostically over real per-trial neural data (`NeuralDataset.rates`)
or model-side per-trial unit activations -- the function only needs
`[n_trials, n_units]` + per-trial condition labels.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def _group_indices(labels: Sequence[Hashable]) -> dict:
    """Pure-Python grouping (label -> list of positions). Deliberately avoids
    numpy `==`/`unique` on object arrays of tuples: `arr == a_tuple` tries to
    broadcast the tuple as if it were itself an array axis (silently wrong
    shape errors), rather than doing the elementwise Python `==` you'd want
    for opaque object-dtype elements. Condition labels are tuples/strings
    here (protocol §8.2 `ConditionLabel.coarse_key()`/`.key()`), so this
    matters -- not just a style choice."""
    groups: dict = {}
    for i, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(i)
    return groups


def _assign_folds(labels: Sequence[Hashable], n_folds: int, rng: np.random.RandomState) -> np.ndarray:
    """Stratified fold assignment: each condition's trials are spread as
    evenly as possible across folds."""
    folds = np.full(len(labels), -1, dtype=int)
    for idxs in _group_indices(labels).values():
        idx = np.array(idxs)
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_folds
    return folds


def crossnobis_rdm(
    data: np.ndarray,
    condition_labels: Sequence[Hashable],
    n_folds: int = 4,
    shrinkage: float = 0.1,
    seed: int = 0,
) -> tuple[np.ndarray, list]:
    """data: [n_trials, n_units]. Returns (rdm [n_cond, n_cond], cond_order).

    RDM entries are the cross-validated, noise-whitened *squared* Mahalanobis
    distance between condition-mean patterns, averaged over all pairs of
    distinct folds (unbiased: expectation 0 under the null).
    """
    data = np.asarray(data, dtype=float)
    labels = list(condition_labels)  # kept as a plain list; see `_group_indices`
    n_trials, n_units = data.shape
    label_groups = _group_indices(labels)
    conds = sorted(label_groups.keys(), key=lambda c: str(c))
    n_cond = len(conds)
    rng = np.random.RandomState(seed)
    folds = _assign_folds(labels, n_folds, rng)

    fold_means = np.full((n_folds, n_cond, n_units), np.nan)
    for f in range(n_folds):
        for ci, c in enumerate(conds):
            idx = np.array([i for i in label_groups[c] if folds[i] == f])
            if len(idx) > 0:
                fold_means[f, ci] = data[idx].mean(axis=0)

    # noise covariance from within-condition residuals (pooled across all trials)
    residuals = []
    for c in conds:
        idx = np.array(label_groups[c])
        if len(idx) > 1:
            residuals.append(data[idx] - data[idx].mean(axis=0))
    residuals = np.concatenate(residuals, axis=0) if residuals else data - data.mean(0)
    cov = np.cov(residuals, rowvar=False) if n_units > 1 else np.array([[residuals.var()]])
    cov = np.atleast_2d(cov)
    shrink_target = np.eye(n_units) * (np.trace(cov) / max(n_units, 1))
    cov_shrunk = (1 - shrinkage) * cov + shrinkage * shrink_target
    cov_inv = np.linalg.pinv(cov_shrunk)

    rdm = np.zeros((n_cond, n_cond))
    counts = np.zeros((n_cond, n_cond))
    for f1 in range(n_folds):
        for f2 in range(n_folds):
            if f1 == f2:
                continue
            for i in range(n_cond):
                for j in range(i + 1, n_cond):
                    m1i, m1j = fold_means[f1, i], fold_means[f1, j]
                    m2i, m2j = fold_means[f2, i], fold_means[f2, j]
                    if np.any(np.isnan(m1i)) or np.any(np.isnan(m2i)):
                        continue
                    d1 = m1i - m1j
                    d2 = m2i - m2j
                    val = float(d1 @ cov_inv @ d2)
                    rdm[i, j] += val
                    rdm[j, i] += val
                    counts[i, j] += 1
                    counts[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rdm = np.where(counts > 0, rdm / np.maximum(counts, 1), 0.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm, conds


def vectorize_upper(rdm: np.ndarray) -> np.ndarray:
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    return rdm[iu]
