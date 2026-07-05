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
identity-based maintenance-epoch schema (audit fix A2a/A2b: probe-epoch
in_set/lure/correct labels are properties of the response, not of the
maintenance-period stimulus, and must not be used to condition the delay-
period RDM).
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import kendalltau, spearmanr

from brainalign_wm.analysis.rdm import crossnobis_rdm, vectorize_upper


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray, method: str = "spearman") -> float:
    va, vb = vectorize_upper(rdm_a), vectorize_upper(rdm_b)
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


def _session_condition_rdm(
    ds, session: str, region, epoch: str, bin_ms: int, n_folds: int = 2,
    condition_fn: Callable = _default_coarse_condition, seed: int = 0, min_trials_per_condition: int = 2,
):
    """Crossnobis RDM built entirely from one session's own trials/units
    (never mixed with another session's data -- see `NeuralDataset.rates`'s
    caveat). `condition_fn(row)` labels each trial row (default: the coarse
    `(load, probe_in_set, correct)` schema); pass an identity-based
    `condition_fn` for epoch-appropriate maintenance/encode conditions
    (audit fix A2b). `seed` controls the internal fold assignment -- varied
    by `within_session_noise_ceiling` to get independent resampled RDM
    estimates of the same session for a reliability (ceiling) estimate.

    Conditions with fewer than `min_trials_per_condition` trials in this
    session are dropped before computing the fold count: a fine-grained
    condition schema (e.g. the maintenance-epoch category multiset) will
    often have most conditions well-populated but a handful of rare
    combinations with only 1-2 trials -- `n_folds_eff` used to be derived
    from the SINGLE rarest condition's count across the WHOLE session,
    so one singleton condition silently zeroed out the entire session's
    RDM rather than just being excluded (found while validating this fix
    against real Tier-A data, where several category combinations recur
    only once per session)."""
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
    # design), so falling back silently produced an all-zero, spuriously
    # "valid" RDM for exactly the region-family case A2d exists to prevent
    # -- found while validating this fix against real Tier-A MTL/MFC data.
    session_units = [u for u in units if u.split("#")[0] == str(session)]
    if not session_units:
        return None, None
    unit_idx = [i for i, u in enumerate(units) if u in set(session_units)]
    all_trial_ids = trials.trial_id.values
    trial_idx = [i for i, tid in enumerate(all_trial_ids) if tid in set(session_trials.trial_id)]
    rates = ds.rates(region, bin_ms, [epoch])  # [n_units(all), n_trials(all), n_bins]
    sub = rates[np.ix_(unit_idx, trial_idx)].mean(axis=2)  # [n_units_sess, n_trials_sess]
    data = sub.T  # [n_trials_sess, n_units_sess]
    labels = [condition_fn(row) for row in session_trials.itertuples()]

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
    rdm, conds = crossnobis_rdm(data, labels, n_folds=n_folds_eff, seed=seed)
    return rdm, conds


def within_session_noise_ceiling(
    ds, session: str, region, epoch: str, bin_ms: int,
    condition_fn: Callable = _default_coarse_condition, n_folds: int = 3, n_resamples: int = 8, seed: int = 0,
) -> tuple[float, float]:
    """(lower, upper) reliability of ONE session's own RDM, via repeated
    independent fold-resamplings of that session's own trials (rather than
    leave-one-SESSION-out): used for the per-session, identity-based
    maintenance/encode-epoch alignment (audit fix A2a/A2b), where held-item
    identity conditions are session-specific and essentially never shared
    across sessions/patients -- so a cross-session LOSO ceiling
    (`noise_ceiling_from_dataset`) would have an empty shared-condition
    intersection and always return 0.0. Same upper/lower aggregation logic
    as `noise_ceiling_from_dataset`, applied to resamples of one session
    instead of a set of sessions -- keeps raw alignment and ceiling on the
    SAME representation (A2c): both are per-session crossnobis RDMs over the
    identical condition set.

    `n_resamples`/`n_folds` default low (8/3): each resample is a full
    crossnobis computation (an O(n_cond^2 * n_folds^2) Python-level loop
    plus a covariance pseudo-inverse over all of a session's units) --
    found to dominate `run_all.py`'s wall-clock when this ran with the
    original defaults (30 resamples x 4 folds) across every session of
    every run. This trades ceiling-estimate smoothness for tractability
    across a full grid; increase locally if a tighter ceiling estimate is
    needed for a specific cell."""
    rdms = []
    conds0 = None
    for r in range(n_resamples):
        rdm, conds = _session_condition_rdm(ds, session, region, epoch, bin_ms, n_folds=n_folds, condition_fn=condition_fn, seed=seed + r)
        if rdm is None:
            continue
        if conds0 is None:
            conds0 = conds
        if conds == conds0:
            rdms.append(rdm)
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


def noise_ceiling_from_dataset(
    ds, region, epoch: str, n_splits: int = 100,
    condition_fn: Callable = _default_coarse_condition,
) -> tuple[float, float]:
    """(lower, upper) trial-split / leave-one-session-out reliability of the
    neural RDM, computed directly from a `NeuralDataset` -- a single shared
    implementation used by both `SimulatedBrain` and the real-data
    adapters. `n_splits` is accepted for interface compatibility (the LOSO
    estimate below uses one RDM per session, not a resampling count) --
    kept as a parameter so callers that pass it don't need updating.

    Audit fix A2e: sessions are aligned by condition IDENTITY, not by
    incidentally-matching RDM shape -- two sessions whose RDMs happen to
    have the same size but index DIFFERENT conditions must not be
    correlated as if aligned. Every session RDM used here is rebuilt over
    the same, explicitly shared, identically-ordered condition list before
    comparison (the previous shape-matching fallback, and its dead
    `common_conds`/`valid` variables, are removed -- audit fix B4)."""
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
