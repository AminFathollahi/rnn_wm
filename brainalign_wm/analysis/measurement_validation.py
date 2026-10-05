"""What the two RDM-alignment measures actually measure.

Four things live here, all read-only with respect to the reported
alignment DVs and their acceptance gate:

  * **Task-structure baselines.** The label-only reference RDMs from
    `analysis/task_structure.py` are pushed through the identical
    comparison the trained model goes through -- same neural RDM, same
    shared-condition set, same `rsa.compare_rdms` -- so a task-structure
    correlation and a model correlation are directly comparable numbers.
  * **The model's additional contribution.** A semipartial correlation on
    the RDM vectors, with the combined task-structure RDM removed from
    both sides, estimated out-of-sample by splitting SESSIONS (never
    trials) in half.
  * **A sensitivity curve for the per-session stratified estimator.** A
    synthetic model side is built as a convex mixture of the real neural
    condition geometry and an independent isotropic geometry, then pushed
    through the real per-session path. Recovered against true alignment
    at the real trial counts and condition sparsity gives the estimator's
    detection floor.
  * **Epoch crossed with method.** The maintenance DV is per-session,
    load-stratified, on real held-item identity; the probe DV is pooled
    across sessions, unstratified, on coarse (load, in-set, correct)
    conditions. Epoch and method are therefore confounded in the reported
    numbers, and only applying each method to each epoch separates them.

Interval estimates resample PATIENTS, not sessions or trials: a patient
contributes several sessions and many units, so a session- or unit-level
resample would treat correlated observations as independent.
"""
from __future__ import annotations

import zlib
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from brainalign_wm.analysis.rdm import stratified_crossnobis_rdm, vectorize_upper
from brainalign_wm.analysis.rsa import compare_rdms
from brainalign_wm.analysis.task_structure import (
    label_field_rdm,
    normalized_hamming_rdm,
    out_of_sample_semipartial,
    probe_task_structure_rdms,
    rdm_vectors,
    semipartial_correlation,
)

MAINTENANCE_FIELD_NAMES = ("load", "held_set")


# ----------------------------------------------------------- session views --

class SessionSubsetView:
    """A neural dataset restricted to a subset of its sessions.

    Splitting sessions is the only way to hold out data for the pooled
    probe estimator: trials within a session share units and drift, so a
    trial-level split leaks. Units from excluded sessions are kept in the
    unit list but contribute nothing, because
    `pseudopopulation.build_condition_fold_means` only ever fills a unit
    from its own session's trials."""

    def __init__(self, dataset, sessions):
        self._dataset = dataset
        self._sessions = set(sessions)
        all_trials = dataset.trials()
        self._mask = all_trials.session.isin(self._sessions).values
        self._trials = all_trials[self._mask].reset_index(drop=True)
        self.bin_ms = dataset.bin_ms

    def units(self, region=None):
        return self._dataset.units(region)

    def trials(self):
        return self._trials

    def rates(self, region, bin_ms, epochs):
        return self._dataset.rates(region, bin_ms, epochs)[:, self._mask, :]

    def sessions(self):
        return sorted(self._sessions)

    def patient_of(self, session):
        return self._dataset.patient_of(session)


def split_sessions(sessions: list[str]) -> tuple[list[str], list[str]]:
    """Deterministic halves, interleaved so both halves span the session
    ordering (which is dataset-ordered, so a contiguous split would put
    the two datasets in different halves)."""
    ordered = sorted(sessions)
    return ordered[0::2], ordered[1::2]


# ----------------------------------------------------------- interval ------

def patient_cluster_bootstrap_ci(
    values: np.ndarray, patients: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval for the mean of `values`, resampling whole
    patients with replacement."""
    values = np.asarray(values, dtype=float)
    patients = np.asarray(patients)
    finite = np.isfinite(values)
    values, patients = values[finite], patients[finite]
    if len(values) == 0:
        return float("nan"), float("nan")
    unique = np.unique(patients)
    by_patient = {p: values[patients == p] for p in unique}
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        means[b] = np.concatenate([by_patient[p] for p in drawn]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pooled_rdm_patient_bootstrap_ci(
    fold_means: np.ndarray, conditions: list, unit_patients: np.ndarray, model_rdm: np.ndarray,
    condition_index: list[int], n_boot: int = 200, seed: int = 0,
) -> tuple[float, float]:
    """Percentile interval for a pooled-pseudopopulation alignment,
    resampling whole patients on the UNIT axis of the condition fold means
    and rebuilding the neural RDM from each resample. The model RDM is
    fixed; the interval describes the neural sampling variability the
    pooled estimator is exposed to."""
    from brainalign_wm.analysis.pseudopopulation import cv_euclidean_rdm

    unique = np.unique(unit_patients)
    index_by_patient = {p: np.where(unit_patients == p)[0] for p in unique}
    rng = np.random.RandomState(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        cols = np.concatenate([index_by_patient[p] for p in drawn])
        rdm = cv_euclidean_rdm(fold_means[:, :, cols], conditions)
        values[b] = compare_rdms(model_rdm, rdm[np.ix_(condition_index, condition_index)])
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ------------------------------------------------- probe task structure ----

def probe_task_structure_rows(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """Label-only baselines and the model's semipartial contribution at
    the probe epoch, on the pooled coarse-condition estimator's own
    neural RDM and shared-condition set."""
    from brainalign_wm.analysis.pseudopopulation import build_condition_fold_means, _unit_session
    from brainalign_wm.analysis.run_all import _coarse_condition, _probe_alignment_for_run

    observed = _probe_alignment_for_run(run_id, model_df, dandi_data, region, return_rdms=True)
    base = {"run_id": run_id, "epoch": "probe", "region": region or "pooled", "method": "pooled_coarse"}
    if observed.get("status") != "ok":
        return [dict(base, predictor="trained_model", status=observed.get("status"))]

    model_rdm, neural_rdm, conditions = observed["model_rdm"], observed["neural_rdm"], observed["conditions"]
    model_vec, neural_vec = rdm_vectors(model_rdm, neural_rdm)

    fold_means, all_conditions = build_condition_fold_means(
        dandi_data, region, dandi_data.bin_ms, "probe", _coarse_condition, n_folds=4, seed=0
    )
    condition_index = [all_conditions.index(c) for c in conditions]
    unit_patients = np.array([dandi_data.patient_of(_unit_session(u)) for u in dandi_data.units(region)])
    model_ci = pooled_rdm_patient_bootstrap_ci(
        fold_means, all_conditions, unit_patients, model_rdm, condition_index
    )

    rows = [dict(
        base, predictor="trained_model", status="ok", raw_alignment=observed["raw_alignment"],
        ci_lower=model_ci[0], ci_upper=model_ci[1], n_conditions=len(conditions),
        noise_ceiling_upper=observed["noise_ceiling_upper"],
    )]

    task_rdms = probe_task_structure_rdms(conditions)
    for name, task_rdm in task_rdms.items():
        ci = pooled_rdm_patient_bootstrap_ci(fold_means, all_conditions, unit_patients, task_rdm, condition_index)
        rows.append(dict(
            base, predictor=name, status="ok", raw_alignment=compare_rdms(task_rdm, neural_rdm),
            ci_lower=ci[0], ci_upper=ci[1], n_conditions=len(conditions),
        ))

    combined_vec = vectorize_upper(task_rdms["combined_task_structure"])
    rows.append(dict(
        base, predictor="model_semipartial_in_sample", status="ok",
        raw_alignment=semipartial_correlation(neural_vec, model_vec, combined_vec),
        n_conditions=len(conditions),
    ))
    rows.append(_probe_out_of_sample_row(run_id, model_df, dandi_data, region, conditions, base))
    return rows


def _probe_out_of_sample_row(run_id, model_df, dandi_data, region, conditions, base) -> dict:
    """Semipartial contribution with the task regression fitted on one half
    of the sessions and evaluated on the other, averaged over both
    directions. Both halves are scored on the same condition set so their
    RDM vectors are comparable."""
    from brainalign_wm.analysis.run_all import _probe_alignment_for_run

    if "session" not in model_df.columns:
        return dict(base, predictor="model_semipartial_out_of_sample", status="no_session_column")
    half_a, half_b = split_sessions(sorted(model_df.session.unique()))
    triples = []
    for half in (half_a, half_b):
        result = _probe_alignment_for_run(
            run_id, model_df[model_df.session.isin(half)], SessionSubsetView(dandi_data, half), region,
            conditions_to_use=conditions, return_rdms=True, compute_ceiling=False,
        )
        if result.get("status") != "ok":
            return dict(base, predictor="model_semipartial_out_of_sample", status="half_" + str(result.get("status")))
        task_rdm = normalized_hamming_rdm(result["conditions"])
        neural_vec, model_vec, task_vec = rdm_vectors(result["neural_rdm"], result["model_rdm"], task_rdm)
        triples.append((neural_vec, model_vec, task_vec))
    values = [out_of_sample_semipartial(triples[0], triples[1]), out_of_sample_semipartial(triples[1], triples[0])]
    return dict(
        base, predictor="model_semipartial_out_of_sample", status="ok",
        raw_alignment=float(np.mean(values)), n_conditions=len(conditions),
        n_sessions_fit=len(half_a), n_sessions_held_out=len(half_b),
    )


# ------------------------------------------- maintenance task structure ----

def _paired_trial_labels(model_df: pd.DataFrame, dandi_data, session_id: str, condition_fn) -> list:
    """Condition labels for exactly the trials `run_all._paired_trial_
    patterns` retains: the session's trials, in table order, that the
    model replay produced maintenance activity for."""
    session_trials = dandi_data.trials()
    session_trials = session_trials[session_trials.session == session_id].reset_index(drop=True)
    maintain = model_df[(model_df.session == session_id) & (model_df.epoch == "maintain")]
    present = {int(t) for t in maintain.trial_id.unique()}
    return [condition_fn(row) for pos, row in enumerate(session_trials.itertuples()) if pos in present]


def maintenance_task_structure_rows(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """Label-only baselines and the model's semipartial contribution at
    the maintenance epoch, on the per-trial representation the held-item-
    set task RDM is defined on (a condition-level task RDM is constant
    within a load stratum, so it has no variance to correlate)."""
    from brainalign_wm.analysis.run_all import (
        _maintenance_condition_fn_stratified, _paired_trial_patterns, _task_model_rdm_per_trial,
    )

    base = {"run_id": run_id, "epoch": "maintain", "region": region or "pooled", "method": "per_session_trial_level"}
    all_trials = dandi_data.trials()
    per_session: list[dict] = []
    for session_id in sorted(set(model_df.session.unique())):
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn_stratified(session_trials)
        model_patterns, neural_patterns = _paired_trial_patterns(model_df, dandi_data, session_id, region)
        if model_patterns.shape[0] < 8:
            continue
        labels = _paired_trial_labels(model_df, dandi_data, session_id, condition_fn)
        if len(labels) != model_patterns.shape[0] or len(set(labels)) < 2:
            continue
        model_rdm = squareform(pdist(model_patterns, metric="euclidean"))
        neural_rdm = squareform(pdist(neural_patterns, metric="euclidean"))
        task_rdms = {
            f"{name}_only": label_field_rdm(labels, i) for i, name in enumerate(MAINTENANCE_FIELD_NAMES)
        }
        task_rdms["combined_task_structure"] = _task_model_rdm_per_trial(labels)
        neural_vec, model_vec = rdm_vectors(neural_rdm, model_rdm)
        combined_vec = vectorize_upper(task_rdms["combined_task_structure"])
        entry = {
            "session": session_id, "patient": dandi_data.patient_of(session_id), "n_trials": len(labels),
            "trained_model": compare_rdms(model_rdm, neural_rdm),
            "model_semipartial_in_sample": semipartial_correlation(neural_vec, model_vec, combined_vec),
            "_vectors": (neural_vec, model_vec, combined_vec),
        }
        for name, task_rdm in task_rdms.items():
            entry[name] = compare_rdms(task_rdm, neural_rdm)
        per_session.append(entry)

    if not per_session:
        return [dict(base, predictor="trained_model", status="no_usable_sessions")]

    patients = np.array([e["patient"] for e in per_session])
    predictors = ["trained_model", "model_semipartial_in_sample", "combined_task_structure"] + [
        f"{name}_only" for name in MAINTENANCE_FIELD_NAMES
    ]
    rows = []
    for predictor in predictors:
        values = np.array([e[predictor] for e in per_session], dtype=float)
        low, high = patient_cluster_bootstrap_ci(values, patients)
        rows.append(dict(
            base, predictor=predictor, status="ok", raw_alignment=float(np.nanmean(values)),
            ci_lower=low, ci_upper=high, n_sessions=len(values), n_patients=len(set(patients)),
        ))

    half_a, half_b = split_sessions([e["session"] for e in per_session])

    def stacked(sessions):
        chosen = [e["_vectors"] for e in per_session if e["session"] in set(sessions)]
        if not chosen:
            return None
        return tuple(np.concatenate([c[i] for c in chosen]) for i in range(3))

    fit, held_out = stacked(half_a), stacked(half_b)
    if fit is not None and held_out is not None:
        values = [out_of_sample_semipartial(fit, held_out), out_of_sample_semipartial(held_out, fit)]
        rows.append(dict(
            base, predictor="model_semipartial_out_of_sample", status="ok",
            raw_alignment=float(np.mean(values)), n_sessions=len(per_session),
            n_sessions_fit=len(half_a), n_sessions_held_out=len(half_b),
        ))
    return rows


# ------------------------------------------------ per-session stratified ---

def stratified_session_alignment(
    model_df: pd.DataFrame, dandi_data, region, epoch: str = "maintain",
    synthetic_model: Callable | None = None, sessions: list[str] | None = None,
) -> list[dict]:
    """Per-session, load-stratified crossnobis alignment -- the estimator
    the maintenance DV uses -- with the epoch left open and the model side
    optionally replaced by a synthetic pattern generator.

    At `epoch="maintain"` with no generator this reproduces
    `run_all._maintenance_alignment_for_run`'s `subpop="all"` rows value
    for value: same condition function, same fold counts, same
    shared-condition floor, same minimum valid-pair guard, same
    `compare_rdms`. `synthetic_model(neural_data, neural_labels, session_id)`
    returns `(patterns, labels)` for the model side, which is how a known
    ground-truth geometry is pushed through the real path."""
    from brainalign_wm.analysis import rsa as rsa_module
    from brainalign_wm.analysis.run_all import (
        MIN_SHARED_CONDITIONS_MAINTENANCE, MIN_VALID_PAIRS_MAINTENANCE, _filter_min_trials,
        _maintenance_condition_fn_stratified, _model_epoch_patterns, _n_valid_pairs,
        _shared_conditions_or_none,
    )

    all_trials = dandi_data.trials()
    if sessions is None:
        sessions = sorted(set(model_df.session.unique())) if model_df is not None else sorted(dandi_data.sessions())
    rows = []
    for session_id in sessions:
        session_trials = all_trials[all_trials.session == session_id]
        if len(session_trials) == 0:
            continue
        condition_fn = _maintenance_condition_fn_stratified(session_trials)

        neural_data, neural_session_trials = rsa_module._session_trial_patterns(
            dandi_data, session_id, region, epoch, dandi_data.bin_ms
        )
        if neural_data is None:
            continue
        neural_labels = [condition_fn(row) for row in neural_session_trials.itertuples()]
        neural_rdm, neural_conds = stratified_crossnobis_rdm(
            neural_data, [l[1] for l in neural_labels], [l[0] for l in neural_labels], n_folds=2, seed=0
        )
        if neural_rdm is None:
            continue

        if synthetic_model is None:
            patterns, labels = _model_epoch_patterns(model_df[model_df.session == session_id], epoch, condition_fn)
            patterns, labels = _filter_min_trials(patterns, labels, min_count=2)
        else:
            patterns, labels = synthetic_model(neural_data, neural_labels, session_id)
        if len(patterns) == 0 or len(set(labels)) < 2:
            continue
        model_rdm, model_conds = stratified_crossnobis_rdm(
            patterns, [l[1] for l in labels], [l[0] for l in labels], n_folds=4, seed=0
        )
        if model_rdm is None:
            continue

        shared = _shared_conditions_or_none(model_conds, neural_conds, floor=MIN_SHARED_CONDITIONS_MAINTENANCE)
        if shared is None:
            continue
        m_idx = [model_conds.index(c) for c in shared]
        n_idx = [neural_conds.index(c) for c in shared]
        model_sub = model_rdm[np.ix_(m_idx, m_idx)]
        neural_sub = neural_rdm[np.ix_(n_idx, n_idx)]
        if _n_valid_pairs(model_sub, neural_sub) < MIN_VALID_PAIRS_MAINTENANCE:
            continue
        rows.append({
            "session": session_id, "patient": dandi_data.patient_of(session_id),
            "raw_alignment": compare_rdms(model_sub, neural_sub), "n_shared_conditions": len(shared),
        })
    return rows


# ------------------------------------------------------ sensitivity curve --

def _stratified_euclidean_rdm(patterns: np.ndarray, conditions: list) -> np.ndarray:
    """Noise-free counterpart of `rdm.stratified_crossnobis_rdm`: Euclidean
    distances between condition-level patterns, with cross-stratum entries
    left NaN so it is comparable pair for pair."""
    rdm = squareform(pdist(patterns, metric="euclidean"))
    strata = np.array([c[0] for c in conditions])
    rdm[strata[:, None] != strata[None, :]] = np.nan
    np.fill_diagonal(rdm, 0.0)
    return rdm


def geometry_mixture_generator(weight: float, seed: int = 0) -> Callable:
    """A synthetic model side whose condition geometry is `weight` of the
    real neural condition geometry and `1 - weight` of an independent
    isotropic geometry matched to it in per-unit mean and across-condition
    scale, plus per-trial noise at the real within-condition residual
    scale. At weight 1 the synthetic trials are a statistically matched
    resample of the neural data; at weight 0 they carry no shared
    geometry at all.

    Also reports, per session, the noise-free correlation between the
    mixed and the real condition geometry -- the true alignment the
    estimator is being asked to recover."""
    state = {"true_by_session": {}}

    def generate(neural_data: np.ndarray, neural_labels: list, session_id: str):
        # `hash()` on a str is salted per process, so deriving the seed from it
        # would give a different mixture every interpreter run at the same
        # nominal `seed`. crc32 over the encoded key is stable across processes.
        key = f"{session_id}|{weight:.6f}|{seed}".encode()
        rng = np.random.RandomState(zlib.crc32(key) % (2 ** 31))
        conditions = sorted(set(neural_labels), key=str)
        index = {c: i for i, c in enumerate(conditions)}
        means = np.stack([neural_data[[i for i, l in enumerate(neural_labels) if l == c]].mean(axis=0)
                          for c in conditions])
        random_geometry = rng.randn(*means.shape) * means.std(axis=0, keepdims=True) + means.mean(axis=0, keepdims=True)
        mixed = weight * means + (1.0 - weight) * random_geometry
        residual_scale = (neural_data - means[[index[l] for l in neural_labels]]).std(axis=0, keepdims=True)
        patterns = mixed[[index[l] for l in neural_labels]] + rng.randn(*neural_data.shape) * residual_scale
        state["true_by_session"][session_id] = compare_rdms(
            _stratified_euclidean_rdm(mixed, conditions), _stratified_euclidean_rdm(means, conditions)
        )
        return patterns, list(neural_labels)

    generate.true_by_session = state["true_by_session"]
    return generate


def sensitivity_curve(
    dandi_data, region, weights: list[float], sessions: list[str] | None = None, seed: int = 0, n_boot: int = 2000,
) -> list[dict]:
    """Recovered against true per-session stratified alignment as the
    synthetic model side's share of the real neural condition geometry is
    swept. The low end is the estimator's detection floor at the real
    per-session trial counts and condition sparsity."""
    rows = []
    for weight in weights:
        generator = geometry_mixture_generator(weight, seed=seed)
        per_session = stratified_session_alignment(
            None, dandi_data, region, epoch="maintain", synthetic_model=generator, sessions=sessions
        )
        if not per_session:
            rows.append({"region": region or "pooled", "mixing_weight": weight, "status": "no_usable_sessions"})
            continue
        recovered = np.array([r["raw_alignment"] for r in per_session], dtype=float)
        patients = np.array([r["patient"] for r in per_session])
        true_values = np.array([generator.true_by_session[r["session"]] for r in per_session], dtype=float)
        low, high = patient_cluster_bootstrap_ci(recovered, patients, n_boot=n_boot, seed=seed)
        rows.append({
            "region": region or "pooled", "mixing_weight": weight, "status": "ok",
            "true_rdm_correlation": float(np.nanmean(true_values)),
            "recovered_alignment": float(np.nanmean(recovered)),
            "ci_lower": low, "ci_upper": high,
            "n_sessions": len(per_session), "n_patients": len(set(patients)),
        })
    return rows


# ------------------------------------------------------ epoch x method -----

def epoch_method_rows(run_id: str, model_df: pd.DataFrame, dandi_data, region) -> list[dict]:
    """The four cells of epoch (maintain, probe) crossed with method
    (per-session load-stratified on real identity; pooled across sessions,
    unstratified, on coarse conditions). The diagonal reproduces the two
    reported DVs; the off-diagonal says how much of the gap between them
    is the method rather than the epoch."""
    from brainalign_wm.analysis.run_all import _probe_alignment_for_run

    base = {"run_id": run_id, "region": region or "pooled"}
    rows = []
    for epoch in ("maintain", "probe"):
        pooled = _probe_alignment_for_run(
            run_id, model_df, dandi_data, region, epoch=epoch, compute_ceiling=False
        )
        rows.append(dict(
            base, epoch=epoch, method="pooled_coarse", status=pooled.get("status"),
            raw_alignment=pooled.get("raw_alignment"), n_conditions=pooled.get("n_shared_conditions"),
        ))
        per_session = stratified_session_alignment(model_df, dandi_data, region, epoch=epoch)
        if not per_session:
            rows.append(dict(base, epoch=epoch, method="per_session_stratified", status="no_usable_sessions"))
            continue
        values = np.array([r["raw_alignment"] for r in per_session], dtype=float)
        patients = np.array([r["patient"] for r in per_session])
        low, high = patient_cluster_bootstrap_ci(values, patients)
        rows.append(dict(
            base, epoch=epoch, method="per_session_stratified", status="ok",
            raw_alignment=float(np.nanmean(values)), ci_lower=low, ci_upper=high,
            n_sessions=len(values), n_patients=len(set(patients)),
        ))
    return rows
