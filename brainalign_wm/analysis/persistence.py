"""Single-neuron persistence, selectivity, and load-tuning measures.
Operates on `[n_units, n_conditions, n_timebins]` condition-mean rate
tensors plus a matching list of condition metadata (dicts or namedtuples
with at least `.load`, `.item_id`/`.category`), so the same code runs on
model units and on real or simulated neurons alike.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sstats


@dataclass
class PersistenceResult:
    persistent_index: np.ndarray  # [n_units]
    selectivity_f: np.ndarray  # [n_units] one-way ANOVA F-stat across conditions
    selectivity_p: np.ndarray  # [n_units]
    load_beta: np.ndarray  # [n_units] slope of rate on load
    load_r2: np.ndarray  # [n_units]


def persistent_activity_index(
    maintain_rates: np.ndarray, baseline_rates: np.ndarray, condition_labels: np.ndarray
) -> np.ndarray:
    """[n_units, n_conditions] maintenance rate vs. [n_units, n_conditions]
    (or [n_units]) baseline -> per-unit index: (preferred-condition maintain
    rate - baseline) / (|baseline| + 1), for the condition each unit fires
    most for (memoranda-selectivity requires it depend on which item/
    category is held, not just be elevated overall).

    Uses `abs(base) + 1.0`, not `base + 1.0`: this function runs on both
    real firing rates (always >= 0, where `abs(base) == base` -- no
    behaviour change) and model GRU hidden-unit activations, which are
    tanh/sigmoid-bounded and can be negative. `base == -1.0` (a fully
    saturated-negative unit during the baseline epoch, not rare for a
    trained GRU) would otherwise divide by exactly zero and return `inf`
    rather than erroring. `abs(base) + 1.0` keeps the near-zero-baseline
    regularization while staying strictly positive for any real `base`.

    That `+ 1.0` is an absolute-rate regularizer, so the index depends on
    the units the activity is measured in and is not comparable between
    bounded model activations and firing rates in spikes/s. It is retained
    for continuity; use `standardized_persistence_index` for any
    model-versus-brain comparison."""
    baseline = baseline_rates if baseline_rates.ndim > 1 else baseline_rates[:, None]
    preferred = maintain_rates.max(axis=1)
    base = baseline.mean(axis=1)
    return (preferred - base) / (np.abs(base) + 1.0)


def standardized_persistence_index(
    maintain_by_trial: np.ndarray,
    baseline_rates: np.ndarray,
    condition_labels: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Scale-free companion to `persistent_activity_index`.

    `maintain_by_trial` is [n_units, n_trials] maintenance-epoch activity,
    `baseline_rates` is [n_units] (or [n_units, n_conditions]) baseline
    activity, `condition_labels` is [n_trials]. Returns a per-unit index:
    the unit's maintenance elevation over baseline at its preferred
    condition, divided by that unit's own across-trial SD of maintenance
    activity.

    Dividing by the unit's own variability rather than by `|baseline| + 1`
    makes the index invariant to a multiplicative rescaling of the
    activity, so bounded model activations and firing rates in spikes/s
    land on the same scale and a model-vs-brain difference cannot be a
    units difference.

    The preferred condition is selected on one random half of each
    condition's trials and the index is evaluated on the disjoint other
    half (both directions, averaged), so the maximum over conditions is not
    evaluated on the data that chose it -- an uncross-validated maximum is
    biased upward by exactly the noise it selected on. Units with zero
    across-trial variance yield NaN."""
    maintain_by_trial = np.asarray(maintain_by_trial, dtype=float)
    labels = np.asarray(condition_labels)
    baseline = np.asarray(baseline_rates, dtype=float)
    base = baseline.mean(axis=1) if baseline.ndim > 1 else baseline
    n_units, n_trials = maintain_by_trial.shape
    conditions = np.unique(labels)

    rng = np.random.RandomState(seed)
    half = np.zeros(n_trials, dtype=int)
    for condition in conditions:
        idx = rng.permutation(np.flatnonzero(labels == condition))
        half[idx[len(idx) // 2:]] = 1

    half_means = np.full((2, n_units, len(conditions)), np.nan)
    for h in (0, 1):
        for j, condition in enumerate(conditions):
            mask = (labels == condition) & (half == h)
            if mask.any():
                half_means[h, :, j] = maintain_by_trial[:, mask].mean(axis=1)

    unit_rows = np.arange(n_units)
    evaluated = np.full((2, n_units), np.nan)
    for selecting, evaluating in ((0, 1), (1, 0)):
        usable = ~np.all(np.isnan(half_means[selecting]), axis=1)
        if not usable.any():
            continue
        preferred = np.nanargmax(np.where(usable[:, None], half_means[selecting], -np.inf), axis=1)
        evaluated[evaluating, usable] = half_means[evaluating][unit_rows, preferred][usable]

    preferred_activity = np.nanmean(evaluated, axis=0)
    sd = maintain_by_trial.std(axis=1, ddof=1) if n_trials > 1 else np.zeros(n_units)
    index = np.full(n_units, np.nan)
    valid = sd > 0
    index[valid] = (preferred_activity[valid] - base[valid]) / sd[valid]
    return index


def selectivity_anova(rates_by_condition: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """rates_by_condition: list of [n_units, n_trials_in_cond] arrays (one
    per condition, unequal trial counts OK). Returns (F, p) per unit via a
    one-way ANOVA across conditions."""
    n_units = rates_by_condition[0].shape[0]
    F = np.zeros(n_units)
    P = np.ones(n_units)
    for u in range(n_units):
        groups = [c[u, :] for c in rates_by_condition if c.shape[1] > 0]
        if len(groups) < 2:
            continue
        f, p = sstats.f_oneway(*groups)
        F[u] = f if f == f else 0.0
        P[u] = p if p == p else 1.0
    return F, P


def load_tuning(maintain_rate_by_trial: np.ndarray, loads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """maintain_rate_by_trial: [n_units, n_trials]; loads: [n_trials].
    Returns (beta, r2) per unit from a simple linear regression on load."""
    n_units = maintain_rate_by_trial.shape[0]
    beta = np.zeros(n_units)
    r2 = np.zeros(n_units)
    x = loads.astype(float)
    x_c = x - x.mean()
    denom = (x_c**2).sum()
    for u in range(n_units):
        y = maintain_rate_by_trial[u]
        y_c = y - y.mean()
        b = (x_c @ y_c) / denom if denom > 0 else 0.0
        pred = b * x_c
        ss_res = ((y_c - pred) ** 2).sum()
        ss_tot = (y_c**2).sum()
        beta[u] = b
        r2[u] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2
