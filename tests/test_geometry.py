"""Phase 8a (comments.txt §5, items 8.1-8.3): synthetic-data tests with a
KNOWN planted answer for every `analysis/geometry.py` function -- a
geometry function that cannot recover a planted ground truth is worthless
(the phase's own acceptance criterion)."""
import numpy as np
import pytest

from brainalign_wm.analysis.geometry import (
    axis_rotation_angle,
    content_context_rotation,
    cross_temporal_generalization_auc,
    libby_stable_switching_units,
    mean_speed,
    orthogonalization_index,
    participation_ratio,
    participation_ratio_by_load,
    pca_participation_ratio,
    task_irrelevant_decoding,
)


# ---------------- 8.1: content vs. context axis rotation (headline test) ----------------

def test_content_axis_rotates_more_than_context_axis():
    """Plant a content-decoding axis that rotates 90 degrees from t_start
    to t_end and a context-decoding axis that does not rotate at all over
    the same window; assert the function recovers ~90 for content, ~0 for
    context, and a large positive paired difference -- mirroring the human
    data's directional finding (content rotates more) without trying to
    hit its literal magnitude."""
    rng = np.random.RandomState(0)
    n_trials = 400
    T = 11  # t=0..10, content axis sweeps 0 -> 90 degrees linearly
    k = 4
    amp = 3.0
    noise_sigma = 0.3

    content_labels = rng.randint(0, 2, size=n_trials)
    context_labels = rng.randint(0, 2, size=n_trials)
    content_sign = np.where(content_labels == 1, 1.0, -1.0)
    context_sign = np.where(context_labels == 1, 1.0, -1.0)

    Z = rng.randn(n_trials, T, k) * noise_sigma
    for t in range(T):
        theta = np.deg2rad(90.0 * t / (T - 1))
        w_content = np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])
        w_context = np.array([0.0, 0.0, 1.0, 0.0])  # fixed direction, every t
        Z[:, t, :] += amp * np.outer(content_sign, w_content)
        Z[:, t, :] += amp * np.outer(context_sign, w_context)

    result = content_context_rotation(Z, content_labels, context_labels, t_start=0, t_end=T - 1)
    print("content_context_rotation:", result)

    assert result["content_rotation_deg"] > 60.0, result
    assert result["context_rotation_deg"] < 25.0, result
    assert result["paired_difference_deg"] > 35.0, result


def test_axis_rotation_angle_near_zero_for_static_axis():
    rng = np.random.RandomState(1)
    n_trials, T, k = 300, 5, 3
    labels = rng.randint(0, 2, size=n_trials)
    sign = np.where(labels == 1, 1.0, -1.0)
    w = np.array([1.0, 0.0, 0.0])
    Z = rng.randn(n_trials, T, k) * 0.2
    for t in range(T):
        Z[:, t, :] += 2.0 * np.outer(sign, w)
    angle = axis_rotation_angle(Z, labels, t_start=0, t_end=T - 1)
    assert angle < 20.0, angle


# ---------------- 8.1: CTG matrix (mirrors companion's LinearSVC+AUC) ----------------

def test_ctg_matrix_diagonal_high_for_stable_code():
    """A code whose decoding axis never moves should generalize well
    across every (train_t, test_t) pair -- both diagonal AND off-diagonal
    AUC should be high."""
    rng = np.random.RandomState(2)
    n_trials, T, k = 200, 6, 3
    labels = rng.randint(0, 2, size=n_trials)
    sign = np.where(labels == 1, 1.0, -1.0)
    w = np.array([1.0, 0.0, 0.0])
    Z = rng.randn(n_trials, T, k) * 0.3
    for t in range(T):
        Z[:, t, :] += 2.5 * np.outer(sign, w)
    auc = cross_temporal_generalization_auc(Z, labels, n_splits=4, seed=0)
    assert np.nanmean(np.diag(auc)) > 0.9
    off_mask = ~np.eye(T, dtype=bool)
    assert np.nanmean(auc[off_mask]) > 0.85


# ---------------- 8.1: LIBBY21 stable/switching per-unit decomposition ----------------

def test_libby_stable_switching_units_recovers_planted_fractions():
    rng = np.random.RandomState(3)
    n_trials, T, n_units = 300, 10, 10
    encode_bins = slice(0, 3)
    maintain_bins = slice(6, 9)
    n_stable = 7  # units 0..6 keep sign; units 7..9 flip
    encode_sign = np.ones(n_units)
    maintain_sign = np.ones(n_units)
    maintain_sign[n_stable:] = -1.0
    planted_stable = encode_sign == maintain_sign

    content_labels = rng.randint(0, 2, size=n_trials)
    class_val = np.where(content_labels == 0, 1.0, -1.0)  # classes[0] gets +1

    Z = rng.randn(n_trials, T, n_units) * 0.1
    for t in range(T):
        if encode_bins.start <= t < encode_bins.stop:
            Z[:, t, :] += np.outer(class_val, encode_sign)
        elif maintain_bins.start <= t < maintain_bins.stop:
            Z[:, t, :] += np.outer(class_val, maintain_sign)

    out = libby_stable_switching_units(Z, content_labels, encode_bins, maintain_bins)
    print("libby_stable_switching_units:", out["stable_frac"], out["switching_frac"])

    assert np.array_equal(out["is_stable"], planted_stable)
    assert out["stable_frac"] == pytest.approx(0.7)
    assert out["switching_frac"] == pytest.approx(0.3)


def test_libby_stable_switching_units_rejects_non_binary_labels():
    Z = np.zeros((10, 5, 4))
    labels = np.array([0, 1, 2] * 3 + [0])
    with pytest.raises(ValueError):
        libby_stable_switching_units(Z, labels, slice(0, 2), slice(3, 5))


# ---------------- 8.2: participation ratio vs. load ----------------

def _make_load_data(rng, n_active_by_load: dict[int, int], n_units=9, n_trials_per_load=200, T=6):
    delay_bins = slice(2, 5)
    loads, rows = [], []
    for load, n_active in n_active_by_load.items():
        for _ in range(n_trials_per_load):
            loads.append(load)
            trial = np.zeros((T, n_units))
            trial[:, :n_active] = rng.randn(T, n_active) * 1.0
            trial[:, n_active:] = rng.randn(T, n_units - n_active) * 0.01
            rows.append(trial)
    return np.stack(rows), np.array(loads), delay_bins


def test_participation_ratio_stays_flat_when_rank_is_load_invariant():
    rng = np.random.RandomState(4)
    Z, loads, delay_bins = _make_load_data(rng, {1: 3, 2: 3, 3: 3})
    pr = participation_ratio_by_load(Z, loads, delay_bins)
    print("PR (flat, planted rank=3 at every load):", pr)
    values = list(pr.values())
    assert max(values) - min(values) < 1.0, pr
    for v in values:
        assert abs(v - 3.0) < 1.2, pr


def test_participation_ratio_expands_when_rank_grows_with_load():
    rng = np.random.RandomState(5)
    Z, loads, delay_bins = _make_load_data(rng, {1: 2, 2: 4, 3: 6})
    pr = participation_ratio_by_load(Z, loads, delay_bins)
    print("PR (expanding, planted rank=2/4/6):", pr)
    assert pr[1] < pr[2] < pr[3], pr
    assert pr[3] - pr[1] > 2.0, pr


def test_participation_ratio_formula_matches_known_values():
    # equal eigenvalues -> PR == D; one dominant eigenvalue -> PR ~= 1
    assert participation_ratio(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(4.0)
    assert participation_ratio(np.array([100.0, 1e-6, 1e-6, 1e-6])) == pytest.approx(1.0, abs=0.01)


# ---------------- 8.3: single-trial vs. trial-averaged dynamics estimate ----------------

def test_trial_averaging_biases_dynamics_estimate():
    """Plant a circular flow (known, constant angular speed) with a small
    per-trial TIME-JITTER (each trial is the same trajectory shifted by a
    random integer number of ticks). The single-trial ensemble's own
    within-trial finite differences recover the true speed regardless of
    jitter (a per-trial constant time-shift does not change a trial's own
    local dynamics). Averaging trials together BEFORE measuring speed acts
    like a low-pass filter over the jitter distribution: it damps the
    averaged trajectory's amplitude and hence its apparent speed -- the
    "spurious contraction" comments.txt warns about, and the reason every
    dynamics fit in this module must use single-trial ensembles (C3)."""
    rng = np.random.RandomState(6)
    n_trials = 400
    T = 40
    n_units = 3
    radius = 5.0
    omega = 2 * np.pi / T
    max_jitter = 10  # up to 1/4 period, deliberately large to give a robust, unambiguous effect
    noise_sigma = 0.02

    Z = np.zeros((n_trials, T, n_units))
    for i in range(n_trials):
        jitter = rng.randint(-max_jitter, max_jitter + 1)
        for t in range(T):
            phase = omega * (t + jitter)
            Z[i, t, 0] = radius * np.cos(phase)
            Z[i, t, 1] = radius * np.sin(phase)
            Z[i, t, :] += rng.randn(n_units) * noise_sigma

    single_trial_speed = mean_speed(Z)
    trial_averaged_speed = mean_speed(Z.mean(axis=0))
    print(f"single_trial_speed={single_trial_speed:.4f} trial_averaged_speed={trial_averaged_speed:.4f}")

    true_speed = radius * omega  # local speed of the underlying circular flow
    assert abs(single_trial_speed - true_speed) < 0.15 * true_speed
    assert trial_averaged_speed < 0.85 * single_trial_speed


def test_mean_speed_zero_for_static_state():
    Z = np.ones((10, 5, 3))
    assert mean_speed(Z) == pytest.approx(0.0)


# ---------------- 8.7: orthogonalization index [BASHIVAN24] ----------------

def test_orthogonalization_index_near_one_for_many_orthogonal_classes():
    """N classes, one-hot along N dedicated axes. NOTE: for one-vs-rest
    normals on a K-class one-hot simplex, the pairwise cosine has a known
    closed form cos = -1/(K-1) (each OvR hyperplane must push away from
    the OTHER K-1 classes' shared centroid, not just its own axis) -- NOT
    0 even though the class centers themselves are exactly orthogonal, so
    a small K (e.g. 3, cos=-0.5, O=0.5) does NOT give O near 1. Using
    K=10 (cos=-1/9, O~0.89) makes the -> 1 limit as K grows unambiguous."""
    rng = np.random.RandomState(0)
    n_per_class = 150
    n_classes = 10
    X, labels = [], []
    for c in range(n_classes):
        centers = np.zeros((n_per_class, n_classes))
        centers[:, c] = 3.0
        X.append(centers + rng.randn(n_per_class, n_classes) * 0.3)
        labels.append(np.full(n_per_class, c))
    X = np.concatenate(X)
    labels = np.concatenate(labels)
    O = orthogonalization_index(X, labels)
    print("orthogonalization_index (10-class one-hot):", O)
    assert O > 0.8, O


def test_orthogonalization_index_near_zero_for_antipodal_classes():
    """2 classes offset in opposite directions along the SAME 1-D axis --
    with exactly 2 classes, the two one-vs-rest normals are exact mirror
    images of each other (`w_1 = -w_0`) by construction, so
    `|cos(w_0, w_1)| == 1` exactly and O == 0 exactly: the classes are
    fully entangled on one shared axis, no dedicated direction each."""
    rng = np.random.RandomState(1)
    n = 200
    w = np.array([1.0, 0.0, 0.0, 0.0])
    X = np.concatenate([
        -2.0 * w[None, :] + rng.randn(n, 4) * 0.2,
        2.0 * w[None, :] + rng.randn(n, 4) * 0.2,
    ])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    O = orthogonalization_index(X, labels)
    print("orthogonalization_index (2-class same-axis):", O)
    assert O < 1e-6, O


# ---------------- 8.8: task-irrelevant decoding [BASHIVAN24] ----------------

def test_task_irrelevant_decoding_recovers_perfectly_decodable_label():
    rng = np.random.RandomState(2)
    n = 300
    labels = rng.randint(0, 2, size=n)
    sign = np.where(labels == 1, 1.0, -1.0)
    X = np.zeros((n, 5))
    X[:, 0] = sign * 3.0
    X += rng.randn(n, 5) * 0.2
    acc = task_irrelevant_decoding(X, labels, n_folds=4, seed=0)
    print("task_irrelevant_decoding (decodable):", acc)
    assert acc > 0.95, acc


def test_task_irrelevant_decoding_near_chance_for_independent_label():
    rng = np.random.RandomState(3)
    n = 300
    X = rng.randn(n, 5)
    labels = rng.randint(0, 2, size=n)  # independent of X
    acc = task_irrelevant_decoding(X, labels, n_folds=4, seed=0)
    print("task_irrelevant_decoding (independent):", acc)
    assert 0.35 < acc < 0.65, acc
