"""Model-to-brain representational similarity analysis and the noise
ceiling that every alignment score must be reported against, since no
alignment magnitude is interpretable in isolation.

The noise ceiling follows the standard split-reliability logic
(Nili/Diedrichsen/Kriegeskorte): each
session contributes an independent RDM (built from only that session's
units); upper bound = mean correlation of a session's RDM with the
grand-mean RDM (includes itself); lower bound = mean correlation of a
session's RDM with the mean of all *other* sessions' RDMs (leave-one-
session-out, LOSO) -- the honest lower estimate since no model can "know"
what one session's own noise looks like.

`_session_condition_rdm`/`noise_ceiling_from_dataset` default to the
original coarse `(load, probe_in_set, correct)` condition schema (used for
the sim-brain recovery gate and the pooled probe-epoch analysis), but accept
a `condition_fn` override -- used by `analysis/run_all.py` for the
identity-based maintenance-epoch schema (probe-epoch
in_set/lure/correct labels are properties of the response, not of the
maintenance-period stimulus, and must not be used to condition the delay-
period RDM).
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from brainalign_wm.analysis.rdm import _crossnobis_with_min_trials, crossnobis_rdm, stratified_crossnobis_rdm, vectorize_upper


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    """Spearman/Kendall correlation between two RDMs' upper triangles.

    NaN-pair-aware: a stratified/block-diagonal RDM
    (`rdm.stratified_crossnobis_rdm`) deliberately leaves cross-stratum
    entries as NaN rather than computing them at all -- pairs where
    EITHER side is NaN are dropped from the correlation instead of
    poisoning the whole result to a NaN-safe 0.0 (which would silently
    zero out every stratified comparison). Plain (non-stratified) RDMs
    never contain NaN, so this is a no-op for every pre-existing caller."""
    va, vb = vectorize_upper(rdm_a), vectorize_upper(rdm_b)
    mask = ~(np.isnan(va) | np.isnan(vb))
    va, vb = va[mask], vb[mask]
    if len(va) < 2:
        return 0.0
    if method == "spearman":
        r, _ = spearmanr(va, vb)
    elif method == "kendall":
        r, _ = kendalltau(va, vb)
    else:
        raise ValueError(f"unknown method {method!r}")
    return float(r) if r == r else 0.0  # NaN-safe (degenerate RDM -> 0)


def permutation_test_rdm(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    n_permutations: int = 1000,
    method: str = "spearman",
    seed: int = 0,
) -> tuple[float, float]:
    """Condition-label permutation test. Returns (observed_corr, p_value)."""
    rng = np.random.RandomState(seed)
    n = rdm_a.shape[0]
    observed = compare_rdms(rdm_a, rdm_b, method)
    null = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n)
        permuted = rdm_b[np.ix_(perm, perm)]
        null[i] = compare_rdms(rdm_a, permuted, method)
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_permutations + 1))
    return observed, p


def normalized_alignment(raw_corr: float, ceiling_upper: float) -> float:
    """Fraction of the noise ceiling explained; clipped to [0,1] (raw scores
    can slightly exceed a noisy ceiling estimate)."""
    if ceiling_upper <= 0:
        return 0.0
    return float(np.clip(raw_corr / ceiling_upper, 0.0, 1.0))


def _default_coarse_condition(row) -> tuple:
    return (row.load, row.probe_in_set, row.correct)


def _session_trial_patterns(ds, session: str, region, epoch: str, bin_ms: int):
    """Extracts (per-trial epoch-mean activity `[n_trials_sess, n_units_sess]`,
    that session's own trial rows) ONCE. Factored out of
    `_session_condition_rdm` so repeated re-derivations of a session's RDM
    under different condition-label assignments (`within_session_noise_
    ceiling`'s split-half resamples, a label-permutation null) don't
    re-invoke `ds.rates()` -- real adapters recompute spike histograms with
    no caching (see `run_all.py --max-sessions-per-dataset`), so refetching
    per resample/permutation would dominate wall-clock. Returns
    `(None, None)` for the same reasons `_session_condition_rdm` used to
    return `(None, None)` early (too few trials, no units in this session's
    region -- see the region-fallback note below, still load-bearing here)."""
    trials = ds.trials()
    session_trials = trials[trials.session == session]
    if len(session_trials) < 4:
        return None, None
    units = ds.units(region)
    # Exact session match (`uid.split("#")[0] == session`), NOT `startswith`:
    # a prefix match risks collision if one session id is a prefix of
    # another (e.g. "sub-1" vs "sub-10") -- doesn't currently happen in the
    # real Tier-A id scheme but is fragile. More importantly: if this
    # session genuinely has ZERO units in the requested region (real on
    # Tier A -- ~31% of sessions have no MFC units), do NOT fall back to
    # every OTHER session's units. `ds.rates()` zero-fills any (unit,
    # trial) pair where the unit's own session != the trial's session (by
    # design), so falling back would silently produce an all-zero,
    # spuriously "valid" RDM for exactly this region-family case.
    session_units = [u for u in units if u.split("#")[0] == str(session)]
    if not session_units:
        return None, None
    unit_idx = [i for i, u in enumerate(units) if u in set(session_units)]
    all_trial_ids = trials.trial_id.values
    trial_idx = [i for i, tid in enumerate(all_trial_ids) if tid in set(session_trials.trial_id)]
    rates = ds.rates(region, bin_ms, [epoch])  # [n_units(all), n_trials(all), n_bins]
    sub = rates[np.ix_(unit_idx, trial_idx)].mean(axis=2)  # [n_units_sess, n_trials_sess]
    data = sub.T  # [n_trials_sess, n_units_sess]
    return data, session_trials


def _session_condition_rdm(
    ds, session: str, region, epoch: str, bin_ms: int, n_folds: int = 2,
    condition_fn: Callable = _default_coarse_condition, seed: int = 0, min_trials_per_condition: int = 2,
    stratified: bool = False, permute_seed: "int | None" = None,
):
    """Crossnobis RDM built entirely from one session's own trials/units
    (never mixed with another session's data -- see `NeuralDataset.rates`'s
    caveat). `condition_fn(row)` labels each trial row (default: the coarse
    `(load, probe_in_set, correct)` schema); pass an identity-based
    `condition_fn` for epoch-appropriate maintenance/encode conditions.
    `seed` controls the internal fold assignment -- varied
    by `within_session_noise_ceiling` to get independent resampled RDM
    estimates of the same session for a reliability (ceiling) estimate.

    `stratified=True`: `condition_fn(row)` must then return
    `(stratum, condition)` -- e.g. `(load, held_item_tuple)` -- and the RDM
    is built SEPARATELY within each stratum, assembled block-diagonally via
    `rdm.stratified_crossnobis_rdm` (cross-stratum condition pairs are NaN,
    never computed, rather than pooled into one crossnobis run whose
    off-diagonal is dominated by the stratum difference; see that
    function's docstring). `permute_seed`, when given, shuffles condition
    labels WITHIN each stratum before computing its RDM -- a label-
    permutation null.

    Conditions with fewer than `min_trials_per_condition` trials in this
    session (or, if `stratified`, in that stratum) are dropped before
    computing the fold count: a fine-grained condition schema (e.g. the
    maintenance-epoch category multiset) will often have most conditions
    well-populated but a handful of rare combinations with only 1-2 trials
    -- `n_folds_eff` used to be derived from the SINGLE rarest condition's
    count across the WHOLE session, so one singleton condition silently
    zeroed out the entire session's RDM rather than just being excluded
    (found while validating this fix against real Tier-A data, where
    several category combinations recur only once per session)."""
    data, session_trials = _session_trial_patterns(ds, session, region, epoch, bin_ms)
    if data is None:
        return None, None
    labels = [condition_fn(row) for row in session_trials.itertuples()]

    if stratified:
        strata = [l[0] for l in labels]
        conds_only = [l[1] for l in labels]
        return stratified_crossnobis_rdm(
            data, conds_only, strata, n_folds=n_folds, seed=seed,
            min_trials_per_condition=min_trials_per_condition, permute_seed=permute_seed,
        )

    if permute_seed is not None:
        perm_rng = np.random.RandomState(permute_seed)
        perm = perm_rng.permutation(len(labels))
        labels = [labels[p] for p in perm]

    return _crossnobis_with_min_trials(data, labels, n_folds, shrinkage=0.1, seed=seed, min_trials_per_condition=min_trials_per_condition)


def _disjoint_half_indices(labels: list, seed: int) -> tuple[list, list]:
    """Splits trial positions into two DISJOINT index halves, stratified per
    condition label (each condition's own trials are shuffled and split
    roughly in half, so every well-populated condition contributes to both
    halves) -- the genuine split-half reliability `within_session_noise_
    ceiling`'s N3 fix needs. A condition with an odd trial count drops its
    leftover trial from BOTH halves rather than assigning it to one,
    avoiding a systematic size/composition imbalance between halves across
    resamples."""
    rng = np.random.RandomState(seed)
    groups: dict = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    half_a, half_b = [], []
    for idxs in groups.values():
        idxs = list(idxs)
        rng.shuffle(idxs)
        mid = len(idxs) // 2
        half_a.extend(idxs[:mid])
        half_b.extend(idxs[mid : 2 * mid])
    return half_a, half_b


def _half_rdm(data: np.ndarray, labels: list, n_folds: int, seed: int, stratified: bool, min_trials_per_condition: int):
    if stratified:
        strata = [l[0] for l in labels]
        conds_only = [l[1] for l in labels]
        return stratified_crossnobis_rdm(data, conds_only, strata, n_folds=n_folds, seed=seed, min_trials_per_condition=min_trials_per_condition)
    return _crossnobis_with_min_trials(data, labels, n_folds, shrinkage=0.1, seed=seed, min_trials_per_condition=min_trials_per_condition)


def within_session_noise_ceiling(
    ds, session: str, region, epoch: str, bin_ms: int,
    condition_fn: Callable = _default_coarse_condition, n_folds: int = 3, n_resamples: int = 8, seed: int = 0,
    stratified: bool = False, min_trials_per_condition: int = 2,
) -> tuple[float, float]:
    """(lower, upper) reliability of ONE session's own RDM, via repeated
    DISJOINT split-half resamples of that session's own trials -- used for
    the per-session, identity-based maintenance/encode-epoch alignment
    where held-item identity conditions are
    session-specific and essentially never shared across sessions/patients
    -- so a cross-session LOSO ceiling (`noise_ceiling_from_dataset`) would
    have an empty shared-condition intersection and always return 0.0.

    Audit fix N3: this used to re-derive the RDM `n_resamples` times from
    ALL of the session's trials, varying only the internal crossnobis fold
    seed (`_session_condition_rdm(..., seed=seed+r)`) -- since every
    resample reused the same full trial set, the resamples were
    near-identical and the "ceiling" saturated near 1 (observed
    ~0.88-0.91 for every real cell), measuring fold-reshuffle STABILITY of
    one estimate rather than split-half RELIABILITY of independent data.
    Fixed to match `pseudopopulation.pooled_noise_ceiling`'s split-half
    logic: each resample partitions the session's trials into two
    genuinely DISJOINT halves (`_disjoint_half_indices`, stratified per
    condition), builds an independent RDM from EACH half alone, and only
    the two halves' shared conditions are kept and aligned before either
    is added to the pool -- so no single trial ever contributes to both
    halves being compared. Same upper/lower aggregation logic as
    `noise_ceiling_from_dataset`/`pooled_noise_ceiling` (mean corr with the
    grand mean of halves; mean corr with the mean of all OTHER halves),
    applied to this session's own half-RDMs -- keeps raw alignment and
    ceiling on the SAME representation (A2c).

    `stratified`/`min_trials_per_condition` are forwarded identically to
    `_session_condition_rdm`'s N1 fix, so the ceiling is estimated on
    exactly the same (load-stratified, block-diagonal) representation the
    raw alignment uses -- a mismatched ceiling would silently misnormalize
    the raw score.

    `n_resamples`/`n_folds` default low (8/3): each half is itself a full
    crossnobis computation, so a resample costs roughly 2x a single
    `_session_condition_rdm` call. Found to dominate `run_all.py`'s
    wall-clock at higher settings across a full grid; increase locally if a
    tighter ceiling estimate is needed for a specific cell."""
    data, session_trials = _session_trial_patterns(ds, session, region, epoch, bin_ms)
    if data is None:
        return 0.0, 0.0
    labels = [condition_fn(row) for row in session_trials.itertuples()]

    half_rdms = []
    conds0 = None
    for r in range(n_resamples):
        half_a, half_b = _disjoint_half_indices(labels, seed=seed + r)
        if len(half_a) < 2 or len(half_b) < 2:
            continue
        rdm_a, conds_a = _half_rdm(data[half_a], [labels[i] for i in half_a], n_folds, seed + r, stratified, min_trials_per_condition)
        rdm_b, conds_b = _half_rdm(data[half_b], [labels[i] for i in half_b], n_folds, seed + r + 100_000, stratified, min_trials_per_condition)
        if rdm_a is None or rdm_b is None:
            continue
        shared = sorted(set(conds_a) & set(conds_b), key=str)
        if len(shared) < 2:
            continue
        if conds0 is None:
            conds0 = shared
        if shared != conds0:
            continue
        ia = [conds_a.index(c) for c in shared]
        ib = [conds_b.index(c) for c in shared]
        half_rdms.append(rdm_a[np.ix_(ia, ia)])
        half_rdms.append(rdm_b[np.ix_(ib, ib)])
    if len(half_rdms) < 2:
        return 0.0, 0.0
    # Cross-stratum (cross-load) entries are NaN in EVERY half-RDM by
    # construction (stratified mode) -- `nanmean` correctly propagates NaN
    # at those positions (there's genuinely nothing to average), but numpy
    # warns "Mean of empty slice" doing so; that warning is expected and
    # harmless here (not a sign of missing data), so it's suppressed
    # locally rather than spamming a full grid's worth of sessions.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stacked = np.stack(half_rdms)
        grand_mean = np.nanmean(stacked, axis=0)
        uppers, lowers = [], []
        for i in range(len(half_rdms)):
            uppers.append(compare_rdms(half_rdms[i], grand_mean))
            others_mean = np.nanmean([half_rdms[j] for j in range(len(half_rdms)) if j != i], axis=0)
            lowers.append(compare_rdms(half_rdms[i], others_mean))
    upper = float(np.clip(np.nanmean(uppers), 0.0, 1.0))
    lower = float(np.clip(np.nanmean(lowers), 0.0, upper))
    return lower, upper


def trial_level_split_half_ceiling(
    data: np.ndarray, labels: list, n_resamples: int = 8, seed: int = 0,
) -> tuple[float, float]:
    """(lower, upper) split-half ceiling for a raw per-trial Euclidean RDM
    (B2's task-model baseline correlates a per-trial
    task RDM against a per-trial Euclidean neural RDM, so its ceiling must
    be estimated on that same per-trial representation, not the
    condition-level crossnobis ceiling `within_session_noise_ceiling`
    produces). `_disjoint_half_indices` splits trials into two disjoint,
    per-label-stratified halves; individual trials aren't shared across
    halves, so each half's raw trials are pooled to a per-label mean
    pattern first (labels are the per-trial task condition, e.g. held
    item-set) and the two halves are correlated over the labels both
    halves have in common -- otherwise there's nothing to align a
    per-trial RDM against across disjoint trial sets."""
    from scipy.spatial.distance import pdist, squareform

    def _label_mean_rdm(idx: list) -> tuple:
        sub_labels = [labels[i] for i in idx]
        groups: dict = {}
        for i, l in zip(idx, sub_labels):
            groups.setdefault(l, []).append(i)
        conds = sorted(groups.keys(), key=str)
        patterns = np.array([data[groups[c]].mean(axis=0) for c in conds])
        return squareform(pdist(patterns, metric="euclidean")), conds

    half_rdms = []
    conds0 = None
    for r in range(n_resamples):
        half_a, half_b = _disjoint_half_indices(labels, seed=seed + r)
        if len(half_a) < 2 or len(half_b) < 2:
            continue
        rdm_a, conds_a = _label_mean_rdm(half_a)
        rdm_b, conds_b = _label_mean_rdm(half_b)
        shared = sorted(set(conds_a) & set(conds_b), key=str)
        if len(shared) < 2:
            continue
        if conds0 is None:
            conds0 = shared
        if shared != conds0:
            continue
        ia = [conds_a.index(c) for c in shared]
        ib = [conds_b.index(c) for c in shared]
        half_rdms.append(rdm_a[np.ix_(ia, ia)])
        half_rdms.append(rdm_b[np.ix_(ib, ib)])
    if len(half_rdms) < 2:
        return 0.0, 0.0
    stacked = np.stack(half_rdms)
    grand_mean = np.nanmean(stacked, axis=0)
    uppers, lowers = [], []
    for i in range(len(half_rdms)):
        uppers.append(compare_rdms(half_rdms[i], grand_mean))
        others_mean = np.nanmean([half_rdms[j] for j in range(len(half_rdms)) if j != i], axis=0)
        lowers.append(compare_rdms(half_rdms[i], others_mean))
    upper = float(np.clip(np.nanmean(uppers), 0.0, 1.0))
    lower = float(np.clip(np.nanmean(lowers), 0.0, upper))
    return lower, upper


def noise_ceiling_from_dataset(
    ds, region, epoch: str,
    condition_fn: Callable = _default_coarse_condition,
) -> tuple[float, float]:
    """(lower, upper) trial-split / leave-one-session-out reliability of the
    neural RDM, computed directly from a `NeuralDataset` -- a single shared
    implementation used by both `SimulatedBrain` and the real-data
    adapters. Audit fix M1: no longer accepts a dead `n_splits` parameter
    (the LOSO estimate below uses one RDM per session, not a resampling
    count, so it was never consulted).

    Sessions are aligned by condition identity, not by
    incidentally-matching RDM shape -- two sessions whose RDMs happen to
    have the same size but index DIFFERENT conditions must not be
    correlated as if aligned. Every session RDM used here is rebuilt over
    the same, explicitly shared, identically-ordered condition list before
    comparison, rather than relying on shape-matching alone."""
    bin_ms = getattr(ds, "bin_ms", 50)
    sessions = ds.sessions()
    session_rdm_conds: dict = {}
    for s in sessions:
        rdm, conds = _session_condition_rdm(ds, s, region, epoch, bin_ms, condition_fn=condition_fn)
        if rdm is None:
            continue
        session_rdm_conds[s] = (rdm, conds)
    if len(session_rdm_conds) < 2:
        return 0.0, 0.0

    # Shared condition set = conditions present in EVERY contributing
    # session (identity-based intersection, not shape-matching).
    shared = None
    for _s, (_rdm, conds) in session_rdm_conds.items():
        cset = set(conds)
        shared = cset if shared is None else (shared & cset)
    shared = sorted(shared, key=str) if shared else []
    if len(shared) < 2:
        return 0.0, 0.0

    rdms = []
    for s, (rdm, conds) in session_rdm_conds.items():
        idx = [conds.index(c) for c in shared]
        rdms.append(rdm[np.ix_(idx, idx)])
    if len(rdms) < 2:
        return 0.0, 0.0

    stacked = np.stack(rdms)
    grand_mean = stacked.mean(axis=0)
    uppers, lowers = [], []
    for i in range(len(rdms)):
        uppers.append(compare_rdms(rdms[i], grand_mean))
        others_mean = np.mean([rdms[j] for j in range(len(rdms)) if j != i], axis=0)
        lowers.append(compare_rdms(rdms[i], others_mean))
    upper = float(np.clip(np.nanmean(uppers), 0.0, 1.0))
    lower = float(np.clip(np.nanmean(lowers), 0.0, upper))
    return lower, upper
