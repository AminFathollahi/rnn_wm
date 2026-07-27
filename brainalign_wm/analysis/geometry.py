"""Representational-geometry analysis for the digital-twin comparison
(Phase 8, comments.txt §5). Mirrors the companion project's
(`../../wm_dynamics/src/geometry.py`) conventions for participation ratio,
principal angles, and decoding-axis rotation, so numbers from this repo are
directly comparable to the companion's human/macaque analyses -- read there
before extending this file rather than reinventing a method it already has.

CONSTRAINT (item 8.3, C3): every function here takes SINGLE-TRIAL ensembles
-- `Z: [n_trials, n_timebins, n_units]` with one row per real trial, never a
trial-averaged mean tucked into a fake trial axis. Trial-averaging manufactures
a spurious contraction: averaging away trial-to-trial phase/state variability
shrinks the apparent spread of the population before any covariance/PCA
statistic ever sees it, so a participation ratio or rotation angle measured on
a trial average is a downward-biased estimate of the true single-trial
geometry. `test_geometry.py::test_trial_averaging_biases_dynamics_estimate`
demonstrates this on synthetic data with a planted per-trial flow.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------- participation ratio (item 8.2, mirrors companion) ----------------

def participation_ratio(eigenvalues: np.ndarray) -> float:
    """PR = (sum lambda_i)^2 / sum(lambda_i^2); mirrors
    `wm_dynamics/src/geometry.py::participation_ratio` exactly (same
    formula, Abbott et al. 2011) so numbers are directly comparable."""
    lam = np.asarray(eigenvalues, dtype=float)
    lam = lam[lam > 0]
    if lam.size == 0:
        return 0.0
    return float((lam.sum() ** 2) / (lam**2).sum())


def pca_participation_ratio(X: np.ndarray) -> float:
    """PR of X's own covariance spectrum. X: [n_samples, n_units]."""
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    return participation_ratio(s**2)


def participation_ratio_by_load(
    Z: np.ndarray, loads: np.ndarray, delay_bins: slice
) -> dict[int, float]:
    """Item 8.2: PR of the delay-period state, per load. `Z`:
    [n_trials, n_timebins, n_units] (single-trial, per C3). For each load,
    pools every SINGLE-TRIAL state vector within `delay_bins` (trials x
    delay-timebins, not trial-averaged) and reports its PR -- the test is
    whether PR stays flat across loads (human data) or expands (slot-model
    prediction)."""
    loads = np.asarray(loads)
    out: dict[int, float] = {}
    for load in sorted(np.unique(loads).tolist()):
        X = Z[loads == load][:, delay_bins, :]
        X = X.reshape(-1, X.shape[-1])  # pool (trials x delay-timebins), single-trial rows
        out[int(load)] = pca_participation_ratio(X)
    return out


def mean_speed(Z: np.ndarray) -> float:
    """Mean per-tick displacement magnitude: mean over (trials x
    consecutive-timepoint pairs) of `||Z[t+1] - Z[t]||`. `Z`:
    `[n_trials, n_timebins, n_units]`, or `[n_timebins, n_units]` for a
    single (e.g. trial-averaged) trajectory. Item 8.3's minimal dynamics
    probe: computed on a SINGLE-TRIAL ensemble it recovers a flow's true
    local speed; computed on a trial-averaged trajectory built from
    time-jittered copies of the same flow, jitter-smoothing damps the
    averaged trajectory's amplitude and hence its apparent speed -- the
    "spurious contraction" this module's docstring warns about. See
    `test_geometry.py::test_trial_averaging_biases_dynamics_estimate`."""
    if Z.ndim == 2:
        Z = Z[None]
    diffs = Z[:, 1:, :] - Z[:, :-1, :]
    return float(np.linalg.norm(diffs, axis=-1).mean())


# ---------------- decoding-axis rotation (item 8.1) ----------------

def _fit_decoding_axis(Z: np.ndarray, labels: np.ndarray, t: int) -> np.ndarray:
    """Unit-normalized logistic-regression decoding axis at timepoint `t`.
    Mirrors `wm_dynamics/src/geometry.py::_fit_axis_weights`'s per-timepoint
    fit (binary here; content/context labels in this repo's usage are
    binary-encodable per test/decoder call). Returns `[n_units]`."""
    X = StandardScaler().fit_transform(Z[:, t, :])
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
    clf.fit(X, labels)
    w = clf.coef_[0]
    norm = np.linalg.norm(w)
    return w / norm if norm > 1e-12 else w


def axis_rotation_angle(Z: np.ndarray, labels: np.ndarray, t_start: int, t_end: int) -> float:
    """Angle (degrees, in [0, 90]) between the decoding axis for `labels`
    at `t_start` vs. `t_end`. `axis_angular_velocity` in the companion
    project reports this as a rate (rad/s) swept over many steps; this
    function reports the single-pair angle directly in degrees, which is
    what item 8.1's "rotated 90 degrees by t+10" framing asks for."""
    w0 = _fit_decoding_axis(Z, labels, t_start)
    w1 = _fit_decoding_axis(Z, labels, t_end)
    cos = np.clip(np.abs(float(w0 @ w1)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def content_context_rotation(
    Z: np.ndarray, content_labels: np.ndarray, context_labels: np.ndarray, t_start: int, t_end: int
) -> dict:
    """Item 8.1's headline test: does the CONTENT axis rotate more than the
    CONTEXT axis over the same delay window? Returns both angles and their
    paired difference (content - context); positive means content rotates
    more, matching the human-data direction (paired difference 0.102,
    p=0.008 there -- this function reports the analogous quantity on
    whatever `Z`/labels it's given, not that literal number)."""
    content_deg = axis_rotation_angle(Z, content_labels, t_start, t_end)
    context_deg = axis_rotation_angle(Z, context_labels, t_start, t_end)
    return {
        "content_rotation_deg": content_deg,
        "context_rotation_deg": context_deg,
        "paired_difference_deg": content_deg - context_deg,
    }


# ---------------- CTG matrix (item 8.1, mirrors companion's LinearSVC+AUC) ----------------

def cross_temporal_generalization_auc(
    Z: np.ndarray, labels: np.ndarray, n_splits: int = 5, seed: int = 0
) -> np.ndarray:
    """CTG matrix via LinearSVC decision function + ROC-AUC, mirroring
    `wm_dynamics/src/geometry.py::cross_temporal_generalization` exactly
    (LinearSVC, not LogisticRegression -- a DIFFERENT convention from this
    repo's own `analysis/cross_temporal.py::cross_temporal_decoding`,
    deliberately: item 8.1 needs numbers comparable to the companion
    project's own CTG results, item 8.6 is the one that reuses this repo's
    own `cross_temporal_decoding`). `Z`: [n_trials, n_timebins, n_units];
    `labels`: [n_trials] binary. Returns `[n_timebins, n_timebins]` AUC."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import LinearSVC

    N, T, _k = Z.shape
    labels = np.asarray(labels)
    auc_mat = np.full((T, T), np.nan)
    if len(np.unique(labels)) < 2:
        return auc_mat

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(N), labels))

    for t1 in range(T):
        fold_mats = np.zeros((len(folds), T))
        for fi, (tr_idx, te_idx) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(Z[tr_idx, t1, :])
            clf = LinearSVC(C=1.0, max_iter=2000)
            clf.fit(X_tr, labels[tr_idx])
            for t2 in range(T):
                X_te = scaler.transform(Z[te_idx, t2, :])
                scores = clf.decision_function(X_te)
                y_te = labels[te_idx]
                if len(np.unique(y_te)) < 2:
                    continue
                try:
                    fold_mats[fi, t2] = roc_auc_score(y_te, scores)
                except ValueError:
                    fold_mats[fi, t2] = np.nan
        auc_mat[t1] = np.nanmean(fold_mats, axis=0)
    return auc_mat


# ---------------- LIBBY21 stable/switching units (item 8.1) ----------------

def libby_stable_switching_units(
    Z: np.ndarray, content_labels: np.ndarray, encode_bins: slice, maintain_bins: slice
) -> dict:
    """[LIBBY21] per-unit decomposition: for each unit, the sign of its
    content-selectivity (mean activity difference between the two content
    classes) during `encode_bins` vs. `maintain_bins`. A unit is "stable"
    if the sign is preserved, "switching" if it flips. `content_labels`
    must be binary (exactly 2 classes) -- selectivity SIGN is only
    well-defined for a two-class contrast.

    `Z`: [n_trials, n_timebins, n_units], single-trial (C3). Returns
    `{"stable_frac": float, "switching_frac": float, "is_stable": [n_units] bool}`.
    """
    classes = np.unique(content_labels)
    if len(classes) != 2:
        raise ValueError(f"libby_stable_switching_units needs exactly 2 content classes, got {len(classes)}")
    a_mask = content_labels == classes[0]
    b_mask = content_labels == classes[1]

    def _selectivity_sign(bins: slice) -> np.ndarray:
        # mean over (trials-in-class x timebins-in-window), per unit
        mean_a = Z[a_mask][:, bins, :].reshape(-1, Z.shape[-1]).mean(axis=0)
        mean_b = Z[b_mask][:, bins, :].reshape(-1, Z.shape[-1]).mean(axis=0)
        return np.sign(mean_a - mean_b)

    sign_encode = _selectivity_sign(encode_bins)
    sign_maintain = _selectivity_sign(maintain_bins)
    is_stable = sign_encode == sign_maintain
    n = is_stable.size
    return {
        "stable_frac": float(is_stable.sum() / n) if n else 0.0,
        "switching_frac": float((~is_stable).sum() / n) if n else 0.0,
        "is_stable": is_stable,
    }
