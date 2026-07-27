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

try:
    import scipy.linalg as _sla
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


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


# ---------------- DMD ensemble fit (item 8.4a, mirrors companion's ensemble_dmd) ----------------

def _dmd_from_pairs(X1: np.ndarray, X2: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact-DMD operator fit from explicit snapshot pairs (not necessarily
    contiguous -- pairs may be pooled across trials). Mirrors
    `wm_dynamics/src/dynamics.py::_dmd_from_pairs` exactly. `X1, X2`:
    `[d, M]`. Returns `(A: [d,d] real, eigenvalues: [r] complex)`."""
    U, s, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(r, len(s))
    U, s, Vt = U[:, :r], s[:r], Vt[:r]
    S_inv = np.diag(1.0 / s)
    Atilde = U.T @ X2 @ Vt.T @ S_inv
    lam, W = np.linalg.eig(Atilde)
    Phi = X2 @ Vt.T @ S_inv @ W
    A = np.real(Phi @ np.diag(lam) @ np.linalg.pinv(Phi))
    return A, lam


def dmd_ensemble_fit(
    Z_trials: np.ndarray, r: int, dt: float = 1.0, n_splits: int = 5, n_null: int = 50,
    rng: np.random.Generator | None = None,
) -> dict:
    """DMD fit on POOLED single-trial transition pairs (item 8.4a, C3):
    mirrors `wm_dynamics/src/dynamics.py::ensemble_dmd` exactly -- fitting
    DMD to a trial-AVERAGED mean trajectory hits near-perfect R^2 by
    construction and is confounded by the trial-averaging contraction this
    module's docstring warns about, so this stacks every trial's
    `(x_t -> x_{t+1})` pairs into one snapshot-pair regression, fits ONE
    operator `A` on the ensemble, and validates it out-of-sample (trial-wise
    CV) plus a circular-shift null (destroys true pairing, preserves each
    trial's marginal statistics -- an operator fit to the null should show
    near-chance one-step R^2).

    `Z_trials`: `[n_trials, n_timebins, n_units]` (single-trial, per C3).
    Returns `{"A", "eigenvalues", "div_scalar", "r2_insample", "r2_cv",
    "r2_cv_std", "r2_null", "r2_null_std", "n_trials"}`."""
    if rng is None:
        rng = np.random.default_rng(0)
    N, T, d = Z_trials.shape
    r_use = min(r, d, N * (T - 1) - 1)

    def _pairs(Z_sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X1 = Z_sub[:, :-1, :].reshape(-1, d).T
        X2 = Z_sub[:, 1:, :].reshape(-1, d).T
        return X1, X2

    def _r2(A: np.ndarray, X1: np.ndarray, X2: np.ndarray) -> float:
        pred = A @ X1
        ss_res = np.sum((X2 - pred) ** 2)
        ss_tot = np.sum((X2 - X2.mean(axis=1, keepdims=True)) ** 2)
        return float(1.0 - ss_res / (ss_tot + 1e-10))

    X1_all, X2_all = _pairs(Z_trials)
    A_all, lam_all = _dmd_from_pairs(X1_all, X2_all, r_use)
    div_scalar = float(np.sum(np.log(np.abs(lam_all) + 1e-300))) / dt
    r2_insample = _r2(A_all, X1_all, X2_all)

    trial_idx = rng.permutation(N)
    folds = np.array_split(trial_idx, min(n_splits, N))
    r2_cv_list = []
    for k in range(len(folds)):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        if len(tr) < 2 or len(te) < 1:
            continue
        X1_tr, X2_tr = _pairs(Z_trials[tr])
        X1_te, X2_te = _pairs(Z_trials[te])
        A_tr, _ = _dmd_from_pairs(X1_tr, X2_tr, r_use)
        r2_cv_list.append(_r2(A_tr, X1_te, X2_te))
    r2_cv = float(np.mean(r2_cv_list)) if r2_cv_list else float("nan")
    r2_cv_std = float(np.std(r2_cv_list)) if r2_cv_list else float("nan")

    r2_null_list = []
    for _ in range(n_null):
        Z_shift = np.empty_like(Z_trials)
        for i in range(N):
            shift = int(rng.integers(1, max(T - 1, 2)))
            Z_shift[i] = np.roll(Z_trials[i], shift, axis=0)
        X1_s, X2_s = _pairs(Z_shift)
        A_s, _ = _dmd_from_pairs(X1_s, X2_s, r_use)
        r2_null_list.append(_r2(A_s, X1_s, X2_s))
    r2_null = float(np.mean(r2_null_list))
    r2_null_std = float(np.std(r2_null_list))

    return {
        "A": A_all, "eigenvalues": lam_all, "div_scalar": div_scalar,
        "r2_insample": r2_insample, "r2_cv": r2_cv, "r2_cv_std": r2_cv_std,
        "r2_null": r2_null, "r2_null_std": r2_null_std, "n_trials": int(N),
    }


# ---------------- dominant direction + LQR gain (item 8.4b/8.4d, mirrors companion) ----------------

def canonicalize_eigenvector_phase(v: np.ndarray) -> np.ndarray:
    """Deterministic real direction from a (possibly complex) eigenvector --
    mirrors `wm_dynamics/src/control.py::canonicalize_eigenvector_phase`
    exactly. `numpy.linalg.eig` fixes eigenvectors only up to an arbitrary
    phase; this rotates by the phase that makes the largest-magnitude entry
    real and positive, then takes the real part, so repeated fits of the
    SAME physical mode give the SAME sign/direction."""
    idx = int(np.argmax(np.abs(v)))
    phase = np.angle(v[idx])
    v_rot = v * np.exp(-1j * phase)
    real_part = v_rot.real
    return real_part / (np.linalg.norm(real_part) + 1e-12)


def unstable_eigenvector(A: np.ndarray) -> tuple[np.ndarray, float]:
    """Item 8.4b: the slowest-decaying (or fastest-growing) direction --
    eigenvector of `A`'s eigenvalue with the largest real part. Mirrors
    `wm_dynamics/src/control.py::unstable_eigenvector` exactly. Returns
    `(v_star: [n] real unit-norm phase-canonicalized, max_re_eig: float)`."""
    eigs, vecs = np.linalg.eig(A)
    idx = int(np.argmax(eigs.real))
    v_star = canonicalize_eigenvector_phase(vecs[:, idx])
    return v_star, float(eigs[idx].real)


def dare_solve(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discrete Algebraic Riccati Equation solve. Mirrors
    `wm_dynamics/src/control.py::dare_solve` exactly (scipy if available,
    value-iteration fallback otherwise). Returns `(P, K)`, `K` the LQR gain
    (`u = -K x`)."""
    if _HAS_SCIPY:
        P = _sla.solve_discrete_are(A, B, Q, R)
    else:
        P = Q.copy().astype(float)
        for _ in range(10000):
            P_new = (
                A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
            )
            if np.max(np.abs(P_new - P)) < 1e-10:
                P = P_new
                break
            P = P_new
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K


def orthogonalization_index(X: np.ndarray, labels: np.ndarray) -> float:
    """[BASHIVAN24] eq. 1, item 8.7: `O = E[triu(1 - |cos(w_i, w_j)|)]`
    over decision-hyperplane normals `w_i`, one per class (one-vs-rest
    logistic regression -- `_fit_decoding_axis`'s binary case generalized
    to >2 classes). `X`: `[n_trials, n_features]` (a single timepoint's
    state, or any other feature space -- RNN hidden state and cached
    ResNet features are both valid inputs here per item 8.7's "apply to
    BOTH" ask, this function is agnostic to which). O near 0: class axes
    nearly parallel (entangled); O near 1: nearly orthogonal (each class
    has its own dedicated direction)."""
    classes = np.unique(labels)
    if len(classes) < 2:
        return 0.0
    Xs = StandardScaler().fit_transform(X)
    axes = []
    for c in classes:
        y = (labels == c).astype(int)
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(Xs, y)
        w = clf.coef_[0]
        norm = np.linalg.norm(w)
        axes.append(w / norm if norm > 1e-12 else w)
    axes = np.stack(axes)
    n = len(axes)
    cos_terms = [1.0 - abs(float(axes[i] @ axes[j])) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(cos_terms)) if cos_terms else 0.0


def task_irrelevant_decoding(X: np.ndarray, labels: np.ndarray, n_folds: int = 4, seed: int = 0) -> float:
    """Item 8.8: cross-validated decode accuracy of a TASK-IRRELEVANT label
    (serial position -- primary, or category -- secondary) from delay-
    period state `X: [n_trials, n_features]`. A load-3 Sternberg trial's
    set-membership judgment does not need item order, so above-chance
    position decoding means the network retains a full structured
    representation, not a task-sufficient one ([BASHIVAN24]'s headline
    result). Plain stratified-CV logistic regression -- reuses the same
    `StandardScaler`+`LogisticRegression` recipe as the rest of this
    module, no companion equivalent needed (this is a decode-accuracy
    question, not a geometry-fitting one)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=1000)
    n_folds = min(n_folds, int(np.min(np.bincount(np.unique(labels, return_inverse=True)[1]))))
    if n_folds < 2:
        return float("nan")
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, Xs, labels, cv=cv)
    return float(scores.mean())


def lqr_gain(A: np.ndarray, B: np.ndarray, q_state: float = 1.0, r_control: float = 1.0) -> dict:
    """Design an LQR controller with identity-scaled cost matrices
    (`Q = q_state*I`, `R = r_control*I`). Mirrors
    `wm_dynamics/src/control.py::lqr_design` exactly. Returns `{"P", "K",
    "closed_loop_A", "is_stable", "closed_loop_eigenvalues", "q_state",
    "r_control"}`."""
    n, m = A.shape[0], B.shape[1]
    Q_mat = q_state * np.eye(n)
    R_mat = r_control * np.eye(m)
    P, K = dare_solve(A, B, Q_mat, R_mat)
    A_cl = A - B @ K
    eigs = np.linalg.eigvals(A_cl)
    return {
        "P": P, "K": K, "closed_loop_A": A_cl,
        "is_stable": bool(np.all(np.abs(eigs) < 1.0)),
        "closed_loop_eigenvalues": eigs, "q_state": q_state, "r_control": r_control,
    }
