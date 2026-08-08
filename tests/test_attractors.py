import numpy as np
import torch

from brainalign_wm.analysis.attractors import find_fixed_points, summarize_fixed_points


def _tanh_map(a: float):
    """h' = tanh(a*h): analytic bistable 1-D map. h=0 is a fixed point
    with derivative a (unstable for a>1); h=+-h* (tanh(a*h*)=h*) are
    stable for a=2 (derivative there is a*(1-h*^2) ~= 0.166)."""
    def step(h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(a * h)
    return step


def test_finds_known_stable_and_unstable_fixed_points():
    step = _tanh_map(2.0)
    h0 = np.array([[-2.0], [-0.01], [0.0], [0.01], [2.0]], dtype=np.float32)
    fps = find_fixed_points(step, h0, n_iters=2000, lr=0.05)

    locations = sorted(fp.h[0] for fp in fps)
    assert len(locations) == 3
    neg, zero, pos = locations
    assert abs(neg + 0.9575) < 1e-2
    assert abs(zero) < 1e-3
    assert abs(pos - 0.9575) < 1e-2

    by_loc = sorted(fps, key=lambda fp: fp.h[0])
    assert by_loc[0].is_stable  # h ~= -0.9575
    assert not by_loc[1].is_stable  # h == 0
    assert by_loc[2].is_stable  # h ~= +0.9575

    for fp in fps:
        assert fp.speed < 1e-6
        assert fp.jacobian.shape == (1, 1)


def test_unconverged_starts_are_dropped_not_reported():
    """A start that never reaches speed_tol must not appear as a spurious
    fixed point at whatever speed it stalled at."""
    step = _tanh_map(2.0)
    h0 = np.array([[0.5]], dtype=np.float32)
    fps = find_fixed_points(step, h0, n_iters=1, lr=1e-4, speed_tol=1e-12)
    assert fps == []


def test_summarize_fixed_points_pools_and_splits_subpop_blocks():
    step = _tanh_map(2.0)
    h0 = np.array([[-2.0, -2.0], [2.0, 2.0]], dtype=np.float32)

    def joint_step(h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(2.0 * h)  # two decoupled copies of the same 1-D map

    fps = find_fixed_points(joint_step, h0, n_iters=2000, lr=0.05)
    summary = summarize_fixed_points(fps, subpop_slices={"a": slice(0, 1), "b": slice(1, 2)})
    assert summary["n_fixed_points"] == 2
    assert summary["n_stable_fixed_points"] == 2
    assert summary["a_max_eig_modulus"] is not None
    assert summary["b_max_eig_modulus"] == summary["a_max_eig_modulus"]  # decoupled identical blocks


def test_marginal_classification_for_near_unity_eigenvalue():
    """A slow-point manifold (line-attractor-like dynamics) has eigenvalue
    moduli just barely over 1 -- neither `is_stable` nor a strongly-
    repelling saddle. `is_marginal` must catch that case."""
    def step(h: torch.Tensor) -> torch.Tensor:
        return 1.0001 * h

    h0 = np.array([[1.0]], dtype=np.float32)
    fps = find_fixed_points(step, h0, n_iters=3000, lr=0.05, speed_tol=1e-4)
    assert len(fps) == 1
    assert fps[0].is_marginal
    assert not fps[0].is_stable


def test_summarize_empty_fixed_points_returns_nones_not_crash():
    summary = summarize_fixed_points([], subpop_slices={"worker": slice(0, 2)})
    assert summary["n_fixed_points"] == 0
    assert summary["mean_speed"] is None
    assert summary["worker_max_eig_modulus"] is None
