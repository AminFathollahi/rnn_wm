"""H5 (dynamic vs. stable delay coding) and H6 (persistent activity)
analyses, wired against real Tier-A data and the model's activity logs
(audit fix C3/C4 -- these were implemented in `analysis/{cross_temporal,
persistence}.py` but never invoked outside the sim-brain recovery gate).

Both hinge on a per-session, per-tick/bin time series, decoded or indexed
against a variable with enough repeated trials to be tractable. Item/
category identity (as used for the maintenance-epoch RSA condition, see
`run_all.py::_maintenance_condition_fn`) has enough repeats on only a
subset of sessions (dataset 000469; see that module's docstring); LOAD
recurs on every session (dozens of trials per load level), and is still a
genuine, epoch-appropriate (not post-hoc) delay-period variable, so it is
used here as the shared decodable/indexed factor for both model and
neural data, giving full session coverage for H5/H6 regardless of which
maintenance-condition schema a given session supports.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def model_epoch_timeseries(df: pd.DataFrame, epoch: str) -> tuple[np.ndarray, np.ndarray]:
    """df: one run's activity log -- MUST already be filtered to a single
    session (`trial_id` is assigned per-session by `generate_activity_logs.
    replay_session` and is not globally unique across sessions; grouping by
    it alone on a multi-session log would merge unrelated trials -- see
    `run_all.py::_model_epoch_patterns`'s docstring for the bug this caused
    elsewhere). Returns (X [n_trials, n_ticks, n_units], loads [n_trials])
    -- only trials with the modal tick count for this epoch are kept
    (curriculum warmup trials can have a shorter maintain window; the bulk
    of trials at a fixed load share one length)."""
    sub = df[df.epoch == epoch]
    if len(sub) == 0:
        return np.zeros((0, 0, 0)), np.zeros((0,))
    if "session" in sub.columns and sub["session"].nunique() > 1:
        raise ValueError("model_epoch_timeseries requires a single-session-filtered df; got multiple sessions")
    is_flat = sub["h_flat"].iloc[0] is not None
    per_trial = []
    for trial_id, g in sub.groupby("trial_id"):
        g = g.sort_values("t")
        if is_flat:
            mat = np.stack(g["h_flat"].to_numpy())
        else:
            mat = np.concatenate([np.stack(g["h_worker"].to_numpy()), np.stack(g["h_manager"].to_numpy())], axis=1)
        per_trial.append((int(g["load"].iloc[0]), mat))
    lengths = [m.shape[0] for _, m in per_trial]
    modal_len = max(set(lengths), key=lengths.count)
    kept = [(load, m) for load, m in per_trial if m.shape[0] == modal_len]
    if not kept:
        return np.zeros((0, 0, 0)), np.zeros((0,))
    X = np.stack([m for _, m in kept])  # [n_trials, n_ticks, n_units]
    loads = np.array([load for load, _ in kept])
    return X, loads


def neural_session_epoch_timeseries(ds, session_id: str, region, epoch: str, bin_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """(X [n_trials, n_bins, n_units], loads [n_trials]) for ONE session's
    own trials/units only (never mixed with another session -- see
    `NeuralDataset.rates`'s caveat)."""
    trials = ds.trials()
    session_trials = trials[trials.session == session_id]
    if len(session_trials) < 4:
        return np.zeros((0, 0, 0)), np.zeros((0,))
    units = ds.units(region)
    # Exact session match + no fallback-to-all-units when this session has
    # zero units in `region` -- same bug class as `rsa.py::
    # _session_condition_rdm` (fallback silently zero-fills via `rates()`'s
    # cross-session zero-fill, producing a spuriously "valid" result).
    session_units = [u for u in units if u.split("#")[0] == str(session_id)]
    if not session_units:
        return np.zeros((0, 0, 0)), np.zeros((0,))
    unit_idx = [i for i, u in enumerate(units) if u in set(session_units)]
    all_trial_ids = trials.trial_id.values
    trial_idx = [i for i, tid in enumerate(all_trial_ids) if tid in set(session_trials.trial_id)]
    rates = ds.rates(region, bin_ms, [epoch])  # [n_units(all), n_trials(all), n_bins]
    sub = rates[np.ix_(unit_idx, trial_idx)]  # [n_units_sess, n_trials_sess, n_bins]
    X = sub.transpose(1, 2, 0)  # [n_trials, n_bins, n_units]
    loads = session_trials.load.values
    return X, loads


def stability_index_for_session(model_df: pd.DataFrame, ds, session_id: str, region, bin_ms: int, n_folds: int = 3, seed: int = 0) -> Optional[dict]:
    """H5: cross-temporal decoding of LOAD across maintenance timebins, for
    both the model (this session's replayed trials) and the neural data
    (this session's own trials/units). Returns
    {"model_stability", "neural_stability"} or None if either side has too
    few trials/classes."""
    from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding, stability_index

    sess_df = model_df[model_df["session"] == session_id] if "session" in model_df.columns else model_df.iloc[0:0]
    X_model, y_model = model_epoch_timeseries(sess_df, "maintain")
    X_neural, y_neural = neural_session_epoch_timeseries(ds, session_id, region, "maintain", bin_ms)

    def _drop_rare_classes(X: np.ndarray, y: np.ndarray):
        # StratifiedKFold requires EVERY class present to have >= n_folds
        # members -- a single rare class (e.g. a curriculum-driven load
        # imbalance at smoke tier) must be dropped entirely, not just
        # outnumbered by well-represented ones.
        if X.shape[0] == 0:
            return X, y
        counts = pd.Series(y).value_counts()
        keep_classes = set(counts[counts >= n_folds].index)
        mask = np.array([v in keep_classes for v in y])
        return X[mask], y[mask]

    X_model, y_model = _drop_rare_classes(X_model, y_model)
    X_neural, y_neural = _drop_rare_classes(X_neural, y_neural)

    def _decodable(X: np.ndarray, y: np.ndarray) -> bool:
        return X.shape[0] >= 6 and len(set(y.tolist())) >= 2

    if not _decodable(X_model, y_model) or not _decodable(X_neural, y_neural):
        return None
    mat_model = cross_temporal_decoding(X_model, y_model, n_folds=n_folds, seed=seed)
    mat_neural = cross_temporal_decoding(X_neural, y_neural, n_folds=n_folds, seed=seed)
    return {"model_stability": stability_index(mat_model), "neural_stability": stability_index(mat_neural)}


def persistence_index_for_session(model_df: pd.DataFrame, ds, session_id: str, region, bin_ms: int) -> Optional[dict]:
    """H6: per-unit persistent-activity index (maintenance rate vs. a
    fixation baseline, at each unit's own preferred load), for model units
    and neural units of this session. Returns
    {"model_index": array, "neural_index": array} or None if data is too
    sparse."""
    from brainalign_wm.analysis.persistence import persistent_activity_index

    sess_df = model_df[model_df["session"] == session_id] if "session" in model_df.columns else model_df.iloc[0:0]
    X_maintain, loads_m = model_epoch_timeseries(sess_df, "maintain")
    X_fix, loads_f = model_epoch_timeseries(sess_df, "fixation")
    if X_maintain.shape[0] < 6 or X_fix.shape[0] < 6:
        model_index = None
    else:
        n_units = X_maintain.shape[2]
        loads_sorted = sorted(set(loads_m.tolist()))
        maintain_by_cond = np.stack([X_maintain[loads_m == l].mean(axis=(0, 1)) for l in loads_sorted], axis=1)  # [n_units, n_cond]
        baseline = X_fix.mean(axis=(0, 1))  # [n_units]
        model_index = persistent_activity_index(maintain_by_cond, baseline[:, None], np.array(loads_sorted))

    X_neural_m, loads_nm = neural_session_epoch_timeseries(ds, session_id, region, "maintain", bin_ms)
    X_neural_f, loads_nf = neural_session_epoch_timeseries(ds, session_id, region, "fixation", bin_ms)
    if X_neural_m.shape[0] < 6 or X_neural_f.shape[0] < 6:
        neural_index = None
    else:
        loads_sorted_n = sorted(set(loads_nm.tolist()))
        maintain_by_cond_n = np.stack(
            [X_neural_m[loads_nm == l].mean(axis=(0, 1)) for l in loads_sorted_n], axis=1
        )  # [n_units, n_cond]
        baseline_n = X_neural_f.mean(axis=(0, 1))
        neural_index = persistent_activity_index(maintain_by_cond_n, baseline_n[:, None], np.array(loads_sorted_n))

    if model_index is None or neural_index is None:
        return None
    return {"model_index": model_index, "neural_index": neural_index}
