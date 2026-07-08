"""Cross-validated Mahalanobis (crossnobis) representational dissimilarity
matrices.

Unlike plain squared-Euclidean or correlation distance, crossnobis has
expectation ~0 under the null (no true condition difference): each distance
is a dot product of *independent* cross-validation folds' mean differences,
whitened by an estimated (shrinkage-regularized) noise covariance, so
positive-bias-inducing self-products never appear. This unbiasedness is
exactly what makes it safe to compare raw model/brain RDM magnitudes
without a systematic inflation confound (verified in `test_analysis.py` by
a null-data test).

Works agnostically over real per-trial neural data (`NeuralDataset.rates`)
or model-side per-trial unit activations -- the function only needs
`[n_trials, n_units]` + per-trial condition labels.
"""
from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def _group_indices(labels: Sequence[Hashable]) -> dict:
    """Pure-Python grouping (label -> list of positions). Deliberately avoids
    numpy `==`/`unique` on object arrays of tuples: `arr == a_tuple` attempts
    to broadcast the tuple as if it were itself an array axis, producing a
    silent shape error rather than the elementwise Python `==` intended for
    opaque object-dtype elements. Condition labels are tuples or strings
    here (see `ConditionLabel.coarse_key()`/`.key()`), so this distinction
    is load-bearing, not a style choice."""
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
                    # Audit-fix review finding: this guard used to check only
                    # m1i/m2i, never m1j/m2j -- a NaN fold-mean for the SECOND
                    # condition in a pair (undefined because that condition
                    # had no trials in this fold) slipped through and
                    # propagated NaN into `val`, silently NaN-ing this RDM
                    # entry (and, via `compare_rdms`'s NaN-safe fallback,
                    # silently zeroing the WHOLE comparison for that
                    # session/run with no warning).
                    if np.any(np.isnan(m1i)) or np.any(np.isnan(m2i)) or np.any(np.isnan(m1j)) or np.any(np.isnan(m2j)):
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


def stratified_crossnobis_rdm(
    data: np.ndarray,
    condition_labels: Sequence[Hashable],
    stratum_labels: Sequence[Hashable],
    n_folds: int = 4,
    shrinkage: float = 0.1,
    seed: int = 0,
    min_trials_per_condition: int = 2,
    permute_seed: "int | None" = None,
) -> tuple:
    """Crossnobis RDM computed SEPARATELY within each stratum (e.g. load
    level) and assembled into one block-diagonal matrix, with cross-stratum
    entries left as NaN (never computed at all) rather than pooling every
    trial into one crossnobis run.

    Audit finding N1: a single crossnobis RDM built across ALL trials
    together necessarily includes condition PAIRS that differ in stratum
    (e.g. load=1 vs load=3) alongside pairs that differ only in the
    within-stratum condition of actual interest (e.g. held-item identity).
    When the stratum (load) structures activity far more strongly than
    identity does -- true of an undertrained recurrent network fed a load
    one-hot every tick -- those cross-stratum pairs dominate the RDM's
    off-diagonal variance, and the resulting correlation mostly measures
    "does this stratum differ" rather than "does identity differ at fixed
    stratum" (verified directly: corr(maintenance_raw_alignment,
    accuracy_load1) = -0.73 across 16 real cells, in the wrong direction,
    traced to exactly this). Stratifying and dropping cross-stratum pairs
    entirely removes that shortcut, rather than merely down-weighting it.

    `condition_labels` must NOT itself encode the stratum (pass the
    identity/category label alone; the stratum is `stratum_labels`,
    supplied separately) -- otherwise every condition would already be
    stratum-unique and this reduces to a no-op. Returned conds are
    `(stratum, condition)` tuples, tagged by stratum so two strata that
    happen to produce an identical bare condition label can never collide.

    `permute_seed`, when given, shuffles which trial gets which condition
    label WITHIN each stratum independently (trial data and stratum
    membership are untouched) before computing that stratum's crossnobis
    RDM -- the label-permutation null a caller can compare the true
    (`permute_seed=None`) statistic against (audit finding N2)."""
    data = np.asarray(data, dtype=float)
    cond_labels = list(condition_labels)
    strat_labels = list(stratum_labels)
    strata = sorted(set(strat_labels), key=str)

    rdms, conds_list = [], []
    for si, s in enumerate(strata):
        idx = [i for i, sl in enumerate(strat_labels) if sl == s]
        sub_data = data[idx]
        sub_labels = [cond_labels[i] for i in idx]
        if permute_seed is not None:
            perm_rng = np.random.RandomState(permute_seed + si)
            perm = perm_rng.permutation(len(sub_labels))
            sub_labels = [sub_labels[p] for p in perm]
        rdm, conds = _crossnobis_with_min_trials(
            sub_data, sub_labels, n_folds, shrinkage, seed, min_trials_per_condition
        )
        if rdm is None:
            continue
        rdms.append(rdm)
        conds_list.append([(s, c) for c in conds])

    if not rdms:
        return None, None

    total = sum(r.shape[0] for r in rdms)
    big = np.full((total, total), np.nan)
    all_conds: list = []
    offset = 0
    for rdm, conds in zip(rdms, conds_list):
        n = rdm.shape[0]
        big[offset : offset + n, offset : offset + n] = rdm
        all_conds.extend(conds)
        offset += n
    return big, all_conds


def _crossnobis_with_min_trials(
    data: np.ndarray,
    labels: list,
    n_folds: int,
    shrinkage: float,
    seed: int,
    min_trials_per_condition: int = 2,
) -> tuple:
    """Shared filter-then-crossnobis logic: drops conditions with fewer than
    `min_trials_per_condition` trials (a rare/singleton condition would
    otherwise force `n_folds_eff` down for the WHOLE call, per
    `analysis/rsa.py::_session_condition_rdm`'s original fix), then picks
    `n_folds_eff` from the rarest surviving condition's count. Returns
    (None, None) if fewer than 2 conditions or fewer than 2 folds survive.
    Factored out so `stratified_crossnobis_rdm` (per stratum) and the
    plain per-session path share one implementation."""
    counts: dict = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    keep = [i for i, l in enumerate(labels) if counts[l] >= min_trials_per_condition]
    data = data[keep]
    labels = [labels[i] for i in keep]
    if data.shape[0] == 0 or data.shape[1] == 0 or len(set(labels)) < 2:
        return None, None
    n_folds_eff = min(n_folds, min(labels.count(l) for l in set(labels)))
    if n_folds_eff < 2:
        return None, None
    return crossnobis_rdm(data, labels, n_folds=n_folds_eff, shrinkage=shrinkage, seed=seed)
