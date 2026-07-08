"""Valid cross-session pseudopopulation construction for coarse,
cross-session-poolable conditions.

Single units are recorded per-session, never simultaneously across the
whole population. A condition-level pseudopopulation must therefore never
average a unit's response over trials it did not itself record --
`dandi_nwb.rates(None, ...)` (and `SimulatedBrain.rates`) zero-fill exactly
those out-of-session entries when naively averaged over ALL trials, which
silently dilutes every condition mean (see their docstrings). This module
builds the pseudopopulation correctly: for each unit, only that unit's own
session's trials contribute to its condition means.

Because different units may have different (possibly zero) trial counts
for a given condition, the resulting per-condition, per-unit matrix can
have missing (NaN) entries. `crossnobis_rdm` (analysis/rdm.py) assumes a
dense, shared per-trial data matrix and Mahalanobis-whitens with a single
shared noise covariance -- not well-posed here (a shared covariance across
units that never co-occur in a trial isn't meaningfully estimable this
way). This module instead uses the simpler cross-validated (squared)
Euclidean distance (Walther et al. 2016): still unbiased under the null
(expectation 0, same fold-based cross-validation logic as crossnobis) but
without noise-covariance whitening. This is a deliberate scoping decision
-- the per-session identity-based path (`analysis/rsa.py::
_session_condition_rdm`) keeps full crossnobis, since a proper trial-level
noise covariance IS estimable within one session.
"""
from __future__ import annotations

from typing import Callable, Hashable, Sequence

import numpy as np


def _unit_session(uid: str) -> str:
    return uid.split("#")[0]


def build_condition_fold_means(
    ds, region, bin_ms: int, epoch: str, condition_fn: Callable, n_folds: int = 4, seed: int = 0
) -> tuple[np.ndarray, list]:
    """Returns (fold_means [n_folds, n_cond, n_units], conds). For each unit,
    its OWN session's trials matching a condition are stratified-assigned to
    folds and averaged within each fold; a (unit, condition, fold) with no
    matching trials for that unit's session is left NaN. `condition_fn(row)`
    is applied to each row of `ds.trials()` (a `namedtuple` via `itertuples`)
    to produce a hashable condition label."""
    units = ds.units(region)
    trials = ds.trials()
    rates = ds.rates(region, bin_ms, [epoch])  # [n_units, n_trials, n_bins]
    mean_rate = rates.mean(axis=2)  # [n_units, n_trials]
    labels = [condition_fn(row) for row in trials.itertuples()]
    conds = sorted(set(labels), key=str)
    cond_idx = {c: i for i, c in enumerate(conds)}
    n_cond = len(conds)
    n_units = len(units)
    fold_means = np.full((n_folds, n_cond, n_units), np.nan)
    rng = np.random.RandomState(seed)

    session_col = trials.session.values
    for ui, uid in enumerate(units):
        own_session = _unit_session(uid)
        own_idx = np.where(session_col == own_session)[0]
        by_cond: dict = {}
        for ti in own_idx:
            by_cond.setdefault(labels[ti], []).append(ti)
        for c, idxs in by_cond.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            fold_assign = np.arange(len(idxs)) % n_folds
            ci = cond_idx[c]
            for f in range(n_folds):
                sel = idxs[fold_assign == f]
                if len(sel) > 0:
                    fold_means[f, ci, ui] = mean_rate[ui, sel].mean()
    return fold_means, conds


def cv_euclidean_rdm(fold_means: np.ndarray, conds: list) -> np.ndarray:
    """Cross-validated (squared) Euclidean distance RDM from `fold_means`
    (`[n_folds, n_cond, n_units]`, NaN where a unit's own session had no
    matching trials for that condition/fold). NaN-aware: a (fold-pair,
    condition-pair) contribution only includes units with valid data in
    BOTH conditions being compared, in both folds of the pair."""
    n_folds, n_cond, n_units = fold_means.shape
    rdm = np.zeros((n_cond, n_cond))
    counts = np.zeros((n_cond, n_cond))
    for f1 in range(n_folds):
        for f2 in range(n_folds):
            if f1 == f2:
                continue
            for i in range(n_cond):
                for j in range(i + 1, n_cond):
                    d1 = fold_means[f1, i] - fold_means[f1, j]
                    d2 = fold_means[f2, i] - fold_means[f2, j]
                    valid = ~(np.isnan(d1) | np.isnan(d2))
                    if not valid.any():
                        continue
                    val = float(np.dot(d1[valid], d2[valid]) / valid.sum() * n_units)
                    rdm[i, j] += val
                    rdm[j, i] += val
                    counts[i, j] += 1
                    counts[j, i] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rdm = np.where(counts > 0, rdm / np.maximum(counts, 1), 0.0)
    np.fill_diagonal(rdm, 0.0)
    return rdm


def pooled_condition_rdm(
    ds, region, bin_ms: int, epoch: str, condition_fn: Callable, n_folds: int = 4, seed: int = 0
) -> tuple[np.ndarray, list]:
    """The pooled, cross-session-valid RDM used for `raw_alignment` (A2c: the
    representation the ceiling below must match)."""
    fold_means, conds = build_condition_fold_means(ds, region, bin_ms, epoch, condition_fn, n_folds, seed)
    return cv_euclidean_rdm(fold_means, conds), conds


def pooled_noise_ceiling(
    ds, region, bin_ms: int, epoch: str, condition_fn: Callable, n_resamples: int = 30, seed: int = 0,
) -> tuple[float, float]:
    """(lower, upper) reliability of the SAME pooled representation
    `pooled_condition_rdm` builds (raw and ceiling on the
    same representation). Each resample draws an independent 2-fold split
    of every unit's own-session trials (`build_condition_fold_means` with
    `n_folds=2`); each of the two folds gives one independent "half" RDM
    (via `cv_euclidean_rdm` on a single-fold-pair -- degenerate case n_folds=2
    still cross-validates between the two halves). Aggregation mirrors
    `analysis.rsa.noise_ceiling_from_dataset`'s split-half/LOSO logic:
    upper = mean corr(half, grand mean-of-halves); lower = mean corr(half,
    mean of all OTHER halves)."""
    from brainalign_wm.analysis.rsa import compare_rdms

    half_rdms = []
    modal_shape = None
    for r in range(n_resamples):
        # 4 independent folds per resample, split into two pairs of folds;
        # each pair cross-validates internally (`cv_euclidean_rdm` needs
        # >=2 folds) to give one unbiased "half" RDM measurement.
        fold_means, conds = build_condition_fold_means(
            ds, region, bin_ms, epoch, condition_fn, n_folds=4, seed=seed + r
        )
        for half in (0, 1):
            sub = fold_means[2 * half : 2 * half + 2]
            rdm = cv_euclidean_rdm(sub, conds)
            if modal_shape is None:
                modal_shape = rdm.shape
            if rdm.shape == modal_shape:
                half_rdms.append(rdm)
    if len(half_rdms) < 2:
        return 0.0, 0.0
    stacked = np.stack(half_rdms)
    grand_mean = stacked.mean(axis=0)
    uppers, lowers = [], []
    for i in range(len(half_rdms)):
        uppers.append(compare_rdms(half_rdms[i], grand_mean))
        others_mean = np.mean([half_rdms[j] for j in range(len(half_rdms)) if j != i], axis=0)
        lowers.append(compare_rdms(half_rdms[i], others_mean))
    upper = float(np.clip(np.nanmean(uppers), 0.0, 1.0))
    lower = float(np.clip(np.nanmean(lowers), 0.0, upper))
    return lower, upper
