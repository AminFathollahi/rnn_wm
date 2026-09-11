"""Task-structure reference RDMs and RDM-level variance partitioning.

Any model that performs the Sternberg task necessarily reproduces the
task's own condition variables -- memory load, whether the probe was in
the held set, and whether the response was correct. A model<->neural RDM
correlation built on those same condition labels is therefore not by
itself evidence of shared representation: the task structure alone
predicts part of it.

This module builds the RDMs implied by the condition labels alone (one
per condition field, plus their normalized Hamming combination) and
computes how much of the model<->neural correlation survives removing
them. The semipartial estimator ranks each RDM vector to quantiles before
regressing, so it is the rank-based counterpart of the Spearman
correlation `rsa.compare_rdms` reports, and its fitted coefficients
transfer between vectors of different length -- which is what makes the
out-of-sample (fit on one set of sessions, evaluate on another) variant
well defined.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from brainalign_wm.analysis.rdm import vectorize_upper

# Field order of the probe-epoch condition label built by
# `run_all._coarse_condition`: (load, probe-in-set, correct).
PROBE_FIELD_NAMES = ("load", "response", "correctness")


def label_field_rdm(conditions: list[tuple], field: int) -> np.ndarray:
    """0 where two conditions share the value of one label field, 1 otherwise."""
    values = np.array([c[field] for c in conditions], dtype=object)
    return (values[:, None] != values[None, :]).astype(float)


def normalized_hamming_rdm(conditions: list[tuple]) -> np.ndarray:
    """Mean per-field mismatch over every field of the condition label."""
    n_fields = len(conditions[0])
    return np.mean([label_field_rdm(conditions, f) for f in range(n_fields)], axis=0)


def probe_task_structure_rdms(conditions: list[tuple]) -> dict[str, np.ndarray]:
    """The four label-only reference RDMs for `(load, in_set, correct)`
    conditions: one per field, plus their normalized Hamming combination."""
    rdms = {f"{name}_only": label_field_rdm(conditions, i) for i, name in enumerate(PROBE_FIELD_NAMES)}
    rdms["combined_task_structure"] = normalized_hamming_rdm(conditions)
    return rdms


def _quantile_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks rescaled to (0, 1), so vectors of different length are on a
    common scale and a regression fitted on one transfers to another."""
    return rankdata(v) / (len(v) + 1.0)


def _shared_finite(*vectors: np.ndarray) -> list[np.ndarray]:
    """Restrict every vector to the entries finite in all of them.
    Stratified RDMs leave cross-stratum entries NaN by construction."""
    mask = np.ones(len(vectors[0]), dtype=bool)
    for v in vectors:
        mask &= np.isfinite(v)
    return [v[mask] for v in vectors]


def fit_removal(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Least-squares (intercept, slope) of quantile-ranked `y` on quantile-
    ranked `x`. Applying these to another vector's ranks removes the same
    linear dependence on `x` without refitting on the evaluation data."""
    yr, xr = _quantile_ranks(y), _quantile_ranks(x)
    slope, intercept = np.polyfit(xr, yr, 1)
    return float(intercept), float(slope)


def apply_removal(y: np.ndarray, x: np.ndarray, coefficients: tuple[float, float]) -> np.ndarray:
    intercept, slope = coefficients
    return _quantile_ranks(y) - (intercept + slope * _quantile_ranks(x))


def semipartial_correlation(neural: np.ndarray, model: np.ndarray, task: np.ndarray) -> float:
    """Correlation between the neural and model RDM vectors after removing
    the task-structure RDM from each. Fitted and evaluated on the same
    entries (in-sample)."""
    return out_of_sample_semipartial((neural, model, task), (neural, model, task))


def out_of_sample_semipartial(
    fit: tuple[np.ndarray, np.ndarray, np.ndarray],
    evaluate: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> float:
    """Semipartial correlation whose task-removal coefficients come from
    `fit` (neural, model, task) and whose correlation is measured on
    `evaluate`. Passing the same triple twice gives the in-sample value."""
    fit_neural, fit_model, fit_task = _shared_finite(*fit)
    ev_neural, ev_model, ev_task = _shared_finite(*evaluate)
    if len(fit_neural) < 3 or len(ev_neural) < 3:
        return float("nan")
    neural_resid = apply_removal(ev_neural, ev_task, fit_removal(fit_neural, fit_task))
    model_resid = apply_removal(ev_model, ev_task, fit_removal(fit_model, fit_task))
    if neural_resid.std() == 0 or model_resid.std() == 0:
        return float("nan")
    return float(np.corrcoef(neural_resid, model_resid)[0, 1])


def rdm_vectors(*rdms: np.ndarray) -> list[np.ndarray]:
    return [vectorize_upper(r) for r in rdms]
