"""Phase 8b (comments.txt §5, items 8.4-8.6): synthetic-data tests with a
KNOWN planted answer for every `analysis/twin.py` function (plus the DMD/
LQR additions to `analysis/geometry.py`) -- item 8's acceptance criterion:
"a geometry function that cannot recover a planted ground truth is
worthless." Also the FIRST correctness test anywhere in this repo for the
pre-existing `analysis/cross_temporal.py::cross_temporal_decoding`/
`stability_index` (item 8.6 reuses them; they had zero test coverage)."""
import numpy as np
import pytest

from brainalign_wm.analysis.geometry import dmd_ensemble_fit, unstable_eigenvector
from brainalign_wm.analysis.twin import (
    correct_vs_incorrect_geometry,
    cross_temporal_by_position,
    on_demand_lqr_control,
    perturb_and_measure_decodability,
)


# ---------------- 8.4a/8.4b: DMD ensemble fit + dominant direction ----------------

def _planted_linear_system(rng, n=3, lam=(0.95, 0.5, 0.2)):
    """Symmetric A (real, orthonormal eigenbasis) with a KNOWN dominant
    eigenvector -- eigenvalue 0.95 is the largest real part, so
    `unstable_eigenvector` should recover its eigenvector V[:, 0]."""
    M = rng.randn(n, n)
    V, _ = np.linalg.qr(M)
    A = V @ np.diag(lam) @ V.T
    return A, V[:, 0]


def test_dmd_ensemble_fit_recovers_dominant_eigenvector_and_nulls_correctly():
    """Generate a genuine multi-trial ensemble (per-trial random ICs, small
    process noise -- single-trial, not one deterministic curve) from a
    KNOWN linear system, fit via `dmd_ensemble_fit`, and check:
    (1) the recovered dominant eigenvector aligns with the planted one,
    (2) r2_cv (out-of-sample, genuine trials) is high and close to
        r2_insample -- the linear fit genuinely generalizes,
    (3) r2_null (circular-shift control) is CLEARLY, substantially lower
        than r2_cv -- not necessarily near chance (see note below)."""
    rng = np.random.RandomState(42)
    n = 3
    A, v_star_planted = _planted_linear_system(rng, n=n)

    N, T = 150, 20
    process_noise = 0.02
    Z = np.zeros((N, T, n))
    for i in range(N):
        x = rng.randn(n) * 1.0
        for t in range(T):
            Z[i, t] = x
            x = A @ x + rng.randn(n) * process_noise

    out = dmd_ensemble_fit(Z, r=3, rng=np.random.default_rng(0))
    print("r2_insample", out["r2_insample"], "r2_cv", out["r2_cv"], "r2_null", out["r2_null"])

    v_hat, max_re_eig = unstable_eigenvector(out["A"])
    alignment = abs(float(v_hat @ v_star_planted))
    print("eigenvector alignment", alignment, "max_re_eig", max_re_eig, "planted", 0.95)

    assert alignment > 0.99, alignment
    assert max_re_eig == pytest.approx(0.95, abs=0.02)
    assert out["r2_cv"] > 0.95
    assert out["r2_cv"] - out["r2_null"] > 0.15, (
        "r2_null should be clearly, substantially lower than r2_cv -- NOTE: for a "
        "purely AUTONOMOUS, time-invariant synthetic system like this one, the "
        "companion's circular-per-trial-shift null is a comparatively weak "
        "discriminator (it only corrupts the single wrap-around pair per trial, "
        "~1/(T-1) of the pairs -- the other ~95% remain genuine (x_t, x_{t+1}) "
        "pairs, just relabeled), so r2_null does NOT collapse to chance here "
        "(observed ~0.79 vs r2_cv~0.997). It still reliably discriminates (a "
        "consistent, highly reproducible ~0.2 gap, std ~5e-4) -- the null's full "
        "power is against NON-stationary/task-locked real data (its actual "
        "intended use, item 8.10/Phase 11), not an idealized autonomous LTI toy."
    )


# ---------------- 8.4c: causal perturbation vs. decodability ----------------

def test_perturbation_along_dominant_direction_hurts_decodability_most():
    """Content is encoded along a KNOWN direction (v_star); a KNOWN
    orthogonal direction (v_context) carries no content information.
    Perturbing along v_star should hurt a content decoder far more than
    perturbing along v_context or a random direction -- item 8.4c's
    prediction, structurally."""
    rng = np.random.RandomState(7)
    n_trials, k = 300, 4
    amp, noise_sigma = 2.0, 0.3
    content_labels = rng.randint(0, 2, size=n_trials)
    content_sign = np.where(content_labels == 1, 1.0, -1.0)

    v_star = np.array([1.0, 0.0, 0.0, 0.0])
    v_context = np.array([0.0, 1.0, 0.0, 0.0])

    x0_batch = np.zeros((n_trials, k))
    x0_batch[:, 0] = content_sign * amp
    x0_batch += rng.randn(n_trials, k) * noise_sigma

    def step_fn(x):
        # slow WM-delay-like persistence: small decay + noise
        return 0.98 * x + np.random.randn(k) * 0.05

    out = perturb_and_measure_decodability(
        step_fn, x0_batch, content_labels, v_star, v_context,
        t_perturb=5, t_final=10, perturb_scale=3.0, n_random_dirs=10,
        rng=np.random.default_rng(1),
    )
    print("perturbation result:", out)

    assert out["vstar_drop"] > out["random_drop_mean"] > out["context_drop"] - 1e-6
    assert out["vstar_drop"] > 0.3
    assert out["context_drop"] < 0.05


# ---------------- 8.4d: on-demand LQR controller ----------------

def test_on_demand_lqr_controller_reduces_drift_with_partial_duty_cycle():
    """An unstable-but-controllable system (A = 1.05*I, B = I): without
    control, state drifts away under noise; the on-demand controller
    (fires only when decoded confidence drops below threshold) should
    show lower drift AND higher decodability than the no-control baseline,
    with a duty cycle strictly between 0 and 1 (proof it's genuinely
    gated, not always-on or inert)."""
    n = 2
    A = 1.05 * np.eye(n)
    B = np.eye(n)
    x0 = np.array([0.1, 0.1])

    def confidence(x):
        return float(np.exp(-np.linalg.norm(x)))

    out = on_demand_lqr_control(
        A, B, x0, T=60, decode_confidence_fn=confidence, threshold=0.85,
        process_noise_sigma=0.05, rng=np.random.default_rng(0),
    )
    print("on-demand LQR result:", out)

    assert out["drift_reduction"] > 0
    assert out["decodability_lift"] > 0
    assert 0.0 < out["duty_cycle"] < 1.0


def test_on_demand_lqr_controller_never_fires_when_threshold_unreachable():
    """A threshold of 0 can never be crossed by a confidence in (0, 1] --
    the controller should never fire (duty_cycle == 0), proving the gate
    genuinely depends on the trigger condition rather than firing
    unconditionally."""
    n = 2
    A = 1.05 * np.eye(n)
    B = np.eye(n)
    x0 = np.array([0.1, 0.1])

    def confidence(x):
        return float(np.exp(-np.linalg.norm(x)))

    out = on_demand_lqr_control(
        A, B, x0, T=30, decode_confidence_fn=confidence, threshold=0.0,
        process_noise_sigma=0.05, rng=np.random.default_rng(0),
    )
    assert out["duty_cycle"] == 0.0


# ---------------- 8.5: correct vs. incorrect trial split ----------------

def test_correct_vs_incorrect_geometry_recovers_planted_error_signature():
    """Plant MORE drift and MORE content-axis rotation on 'incorrect'
    trials than 'correct' ones; assert the split-comparison function
    recovers both directions (item 8.5: "does the manifold differ on
    error trials -- more drift ... larger rotation")."""
    rng = np.random.RandomState(11)
    n_trials, T, k = 300, 11, 4
    correct = rng.rand(n_trials) < 0.5
    content_labels = rng.randint(0, 2, size=n_trials)
    context_labels = rng.randint(0, 2, size=n_trials)
    content_sign = np.where(content_labels == 1, 1.0, -1.0)
    context_sign = np.where(context_labels == 1, 1.0, -1.0)

    Z = np.zeros((n_trials, T, k))
    for i in range(n_trials):
        noise_sigma = 0.2 if correct[i] else 0.6
        rot_max = 5.0 if correct[i] else 60.0
        for t in range(T):
            theta = np.deg2rad(rot_max * t / (T - 1))
            w_content = np.array([np.cos(theta), np.sin(theta), 0, 0])
            w_context = np.array([0, 0, 1.0, 0])
            Z[i, t] = (
                2.0 * content_sign[i] * w_content + 2.0 * context_sign[i] * w_context
                + rng.randn(k) * noise_sigma
            )

    out = correct_vs_incorrect_geometry(Z, correct, content_labels, context_labels, t_start=0, t_end=T - 1)
    print("correct:", out["correct"])
    print("incorrect:", out["incorrect"])
    print("diff (incorrect - correct):", out["incorrect_minus_correct"])

    diff = out["incorrect_minus_correct"]
    assert diff["drift_speed"] > 0
    assert diff["content_rotation_deg"] > 20.0


# ---------------- 8.6: cross-temporal generalization of memorandum identity ----------------

def test_cross_temporal_by_position_distinguishes_stable_from_dynamic_code():
    """FIRST-EVER correctness test of `cross_temporal.cross_temporal_decoding`/
    `stability_index` (pre-existing, zero prior test coverage). Position 1's
    code uses a FIXED decoding axis the whole trial (should generalize well
    across time -> high stability_index); position 2's axis alternates
    between two orthogonal directions every tick (narrow-diagonal, dynamic
    code -> lower stability_index). If the pre-existing code does NOT show
    this distinction, that is a real bug to report, not to work around."""
    rng = np.random.RandomState(21)
    n_trials, T, k = 200, 8, 4

    labels_stable = rng.randint(0, 2, size=n_trials)
    sign_stable = np.where(labels_stable == 1, 1.0, -1.0)
    w_stable = np.array([1.0, 0, 0, 0])
    Z_stable = rng.randn(n_trials, T, k) * 0.3
    for t in range(T):
        Z_stable[:, t, :] += 2.0 * np.outer(sign_stable, w_stable)

    labels_dynamic = rng.randint(0, 2, size=n_trials)
    sign_dynamic = np.where(labels_dynamic == 1, 1.0, -1.0)
    Z_dynamic = rng.randn(n_trials, T, k) * 0.3
    for t in range(T):
        w = np.array([0, 1.0, 0, 0]) if t % 2 == 0 else np.array([0, 0, 1.0, 0])
        Z_dynamic[:, t, :] += 2.0 * np.outer(sign_dynamic, w)

    out_stable = cross_temporal_by_position(Z_stable, {1: labels_stable}, n_folds=4, seed=0)
    out_dynamic = cross_temporal_by_position(Z_dynamic, {2: labels_dynamic}, n_folds=4, seed=0)
    si_stable = out_stable[1]["stability_index"]
    si_dynamic = out_dynamic[2]["stability_index"]
    print("stable stability_index:", si_stable, "dynamic stability_index:", si_dynamic)

    assert si_stable > si_dynamic + 0.15
    assert si_stable > 0.9  # a truly time-invariant axis should generalize almost perfectly
