"""Demixed principal component analysis (Kobak & Machens 2016), implemented
from scratch. Decomposes condition-mean, time-resolved activity into
condition-independent, per-factor (e.g. item/category, load), and
interaction marginalizations, each then reduced by ordinary PCA.

Marginalization via the standard inclusion-exclusion / ANOVA-style
decomposition: for a subset S of factors (which may include `"time"` as a
factor), phi_S = <R>_{not S} minus every lower-order phi_{S'} for S' subset
S. This recursively strips out lower-order structure so each marginalization
contains only the variance genuinely explained by exactly that combination
of factors -- e.g. the "load" marginalization contains no item- or
time-driven variance.

`R` is a condition-mean tensor `[n_units, level(f1), level(f2), ..., n_time]`
-- the caller (recovery gate / real analysis) builds this by averaging
per-trial rates over trials within each (f1, f2, ..., time-bin) cell.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def _powerset(items: Sequence[str]):
    for r in range(len(items) + 1):
        yield from combinations(items, r)


def marginalize(R: np.ndarray, factor_names: Sequence[str]) -> dict[tuple, np.ndarray]:
    """R: [n_units, *factor_levels] with `factor_names[i]` labeling axis i+1.
    Returns {subset_tuple: marginalized_tensor} for every subset of
    `factor_names` (including the empty tuple = the grand mean / removed
    entirely, and singleton "condition-independent" style terms)."""
    n_factors = len(factor_names)
    axes = list(range(1, n_factors + 1))  # axis 0 = units
    phi: dict[tuple, np.ndarray] = {}
    for subset in _powerset(factor_names):
        not_in_subset_axes = [axes[i] for i, f in enumerate(factor_names) if f not in subset]
        marg = R.mean(axis=tuple(not_in_subset_axes), keepdims=True) if not_in_subset_axes else R.copy()
        marg = np.broadcast_to(marg, R.shape).copy()
        for r in range(len(subset)):
            for sub_subset in combinations(subset, r):
                marg -= phi[sub_subset]
        phi[subset] = marg
    return phi


def dpca_components(
    R: np.ndarray, factor_names: Sequence[str], n_components: int = 3
) -> dict[tuple, dict]:
    """Runs marginalize() then PCA (via SVD) within each non-empty
    marginalization. Returns per-subset {"explained_variance_ratio": [...],
    "components": [n_components, n_units], "total_variance": float}."""
    phi = marginalize(R, factor_names)
    total_var = sum(float((v**2).sum()) for k, v in phi.items() if len(k) > 0)
    out = {}
    n_units = R.shape[0]
    for subset, marg in phi.items():
        if len(subset) == 0:
            continue
        flat = marg.reshape(n_units, -1)  # [n_units, samples]
        flat_centered = flat - flat.mean(axis=1, keepdims=True)
        var_this = float((marg**2).sum())
        try:
            u, s, vt = np.linalg.svd(flat_centered, full_matrices=False)
        except np.linalg.LinAlgError:
            out[subset] = {"explained_variance_ratio": np.zeros(n_components), "components": None,
                            "fraction_of_total_variance": var_this / total_var if total_var > 0 else 0.0}
            continue
        ev = s**2
        evr = ev / ev.sum() if ev.sum() > 0 else np.zeros_like(ev)
        k = min(n_components, len(evr))
        out[subset] = {
            "explained_variance_ratio": evr[:k],
            "components": u[:, :k].T,  # [k, n_units]
            "fraction_of_total_variance": var_this / total_var if total_var > 0 else 0.0,
        }
    return out
