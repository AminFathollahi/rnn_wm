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
"""
from __future__ import annotations

from typing import Sequence

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


def _session_condition_rdm(ds, session: str, region, epoch: str, bin_ms: int, n_folds: int = 2):
    trials = ds.trials()
    session_trials = trials[trials.session == session]
    if len(session_trials) < 4:
        return None, None
    units = ds.units(region)
    session_units = [u for u in units if u.startswith(str(session))]
    if not session_units:
        session_units = units
    unit_idx = [i for i, u in enumerate(units) if u in set(session_units)]
    all_trial_ids = trials.trial_id.values
    trial_idx = [i for i, tid in enumerate(all_trial_ids) if tid in set(session_trials.trial_id)]
    rates = ds.rates(region, bin_ms, [epoch])  # [n_units(all), n_trials(all), n_bins]
    sub = rates[np.ix_(unit_idx, trial_idx)].mean(axis=2)  # [n_units_sess, n_trials_sess]
    data = sub.T  # [n_trials_sess, n_units_sess]
    labels = [
        (row.load, row.probe_in_set, row.correct)
        for row in session_trials.itertuples()
    ]
    if data.shape[1] == 0 or len(set(labels)) < 2:
        return None, None
    n_folds_eff = min(n_folds, min(labels.count(l) for l in set(labels)))
    if n_folds_eff < 2:
        return None, None
    rdm, conds = crossnobis_rdm(data, labels, n_folds=n_folds_eff)
    return rdm, conds


def noise_ceiling_from_dataset(ds, region, epoch: str, n_splits: int = 100) -> tuple[float, float]:
    """(lower, upper) trial-split / leave-one-session-out reliability of the
    neural RDM, computed directly from a `NeuralDataset` -- a single shared
    implementation used by both `SimulatedBrain` and the real-data
    adapters."""
    bin_ms = getattr(ds, "bin_ms", 50)
    sessions = ds.sessions()
    session_rdms = {}
    common_conds = None
    for s in sessions:
        rdm, conds = _session_condition_rdm(ds, s, region, epoch, bin_ms)
        if rdm is None:
            continue
        session_rdms[s] = (rdm, conds)
        common_conds = conds if common_conds is None else common_conds
    valid = {s: r for s, (r, c) in session_rdms.items() if r.shape == session_rdms[list(session_rdms)[0]][0].shape}
    rdms = [v[0] for v in session_rdms.values()]
    if len(rdms) < 2:
        return 0.0, 0.0
    # align shapes: keep only sessions whose RDM matches the modal shape
    shapes = [r.shape for r in rdms]
    modal_shape = max(set(shapes), key=shapes.count)
    rdms = [r for r in rdms if r.shape == modal_shape]
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
