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

    Uses `abs(base) + 1.0`, not `base + 1.0`: this function is called on
    both real neural firing rates (always >= 0, where `abs(base) == base`
    -- no behavior change) and model GRU hidden-unit activations
    (`dynamics_and_persistence.py::persistence_index_for_session`), which
    are tanh/sigmoid-bounded and CAN be negative. `base == -1.0` (a fully
    saturated-negative unit during the baseline epoch, a real, non-rare
    occurrence for a trained GRU) previously divided by exactly zero,
    producing a meaningless `inf` "index" for that unit rather than an
    error -- found via a recurring `RuntimeWarning: divide by zero` while
    auditing a real post-grid analysis run. `abs(base) + 1.0` keeps the
    same near-zero-baseline regularization intent while staying strictly
    positive for any real-valued `base`."""
    baseline = baseline_rates if baseline_rates.ndim > 1 else baseline_rates[:, None]
    preferred = maintain_rates.max(axis=1)
    base = baseline.mean(axis=1)
    return (preferred - base) / (np.abs(base) + 1.0)


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
