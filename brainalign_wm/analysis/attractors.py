"""Fixed-point (attractor) analysis of a trained recurrent cell's own
autonomous dynamics (Sussillo & Barak 2013, "Opening the Black Box"):
given a one-step transition function `h_{t+1} = step_fn(h_t)` -- every
other per-tick input (the visual drive, gate biases, Hebbian traces) held
fixed by the caller at a representative value, see
`scripts/analyze_attractors.py` for how a maintenance-epoch,
no-new-stimulus `step_fn` is built per cell/subpopulation -- find every
h* with `step_fn(h*) ~= h*` ("slow points" at zero tolerance are exact
fixed points; this finder reports only points that actually reach the
speed tolerance, never a "closest we got" point at whatever speed
gradient descent stalled at), then classify each by the eigenvalues of
d(step_fn)/dh at h*: a discrete-time fixed point is locally stable
(an attractor) iff every eigenvalue's modulus is < 1, and oscillatory
(a spiral rather than a node) iff its dominant eigenvalue has a nonzero
imaginary part.

Torch (not numpy, unlike every other module under `analysis/`) is
required here: h* is found by gradient descent through the model's own
`forward` and the stability classification needs the Jacobian of that
same differentiable function, not a static array -- there is no
weights-only shortcut the way `network_properties.py` has one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch


@dataclass
class FixedPoint:
    h: np.ndarray             # [H] fixed-point location
    speed: float               # 0.5*||step_fn(h)-h||^2 at convergence
    jacobian: np.ndarray        # [H, H] d(step_fn)/dh at h, real-valued
    eigenvalues: np.ndarray     # [H] complex eigenvalues of `jacobian`
    max_eig_modulus: float
    is_stable: bool             # max_eig_modulus < stability_tol
    is_marginal: bool           # |max_eig_modulus - 1| < marginal_tol: a slow point on a
                                 # quasi-continuous/line-attractor manifold (Sussillo & Barak
                                 # 2013) rather than a strict node -- neither `is_stable` nor
                                 # a strongly-repelling saddle, and the functionally relevant
                                 # case for WM maintenance over realistic trial lengths.
    is_oscillatory: bool        # dominant eigenvalue has a nonzero imaginary part


def _jacobian(step_fn: Callable[[torch.Tensor], torch.Tensor], h_star: torch.Tensor) -> np.ndarray:
    h_star = h_star.detach().clone()

    def f(h: torch.Tensor) -> torch.Tensor:
        return step_fn(h.unsqueeze(0)).squeeze(0)

    jac = torch.autograd.functional.jacobian(f, h_star)
    return jac.detach().cpu().numpy().astype(np.float64)


def find_fixed_points(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    h0: np.ndarray,
    *,
    n_iters: int = 5000,
    lr: float = 1e-2,
    speed_tol: float = 1e-6,
    dedup_tol: Optional[float] = None,
    stability_tol: float = 1.0,
    marginal_tol: float = 1e-3,
) -> list[FixedPoint]:
    """Sussillo & Barak (2013): from every row of `h0` ([n_starts, H],
    typically real hidden states sampled from a replayed trial's
    maintenance epoch -- C3, single-trial, never a trial average --
    optionally jittered with noise by the caller for broader coverage),
    run Adam on the scalar speed `q(h) = 0.5*||step_fn(h)-h||^2` for
    `n_iters` steps (all starts optimized together as one batch, since
    `step_fn` is itself batched and the per-row speed terms have
    block-diagonal gradients -- no cross-talk between starts), keep only
    rows whose final speed is below `speed_tol`, deduplicate points within
    `dedup_tol` of each other (keeping the lower-speed copy; `dedup_tol`
    defaults to 5% of `h0`'s median row norm when not given), and classify
    each survivor via the Jacobian of `step_fn` at that point.
    """
    h0_arr = np.asarray(h0, dtype=np.float32)
    if h0_arr.ndim != 2:
        raise ValueError(f"h0 must be [n_starts, H], got shape {h0_arr.shape}")
    if dedup_tol is None:
        dedup_tol = 0.05 * float(np.median(np.linalg.norm(h0_arr, axis=1)) + 1e-8)

    h = torch.as_tensor(h0_arr).clone().requires_grad_(True)
    opt = torch.optim.Adam([h], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        speed_per_row = 0.5 * ((step_fn(h) - h) ** 2).sum(dim=-1)
        speed_per_row.sum().backward()
        opt.step()
    with torch.no_grad():
        final_speed = (0.5 * ((step_fn(h) - h) ** 2).sum(dim=-1)).cpu().numpy()

    converged = [(h[i].detach(), float(final_speed[i])) for i in range(h0_arr.shape[0]) if final_speed[i] < speed_tol]
    kept: list[tuple[torch.Tensor, float]] = []
    for h_i, speed_i in sorted(converged, key=lambda t: t[1]):
        if all(torch.norm(h_i - h_k).item() > dedup_tol for h_k, _ in kept):
            kept.append((h_i, speed_i))

    results = []
    for h_star, speed in kept:
        jac = _jacobian(step_fn, h_star)
        eigvals = np.linalg.eigvals(jac)
        max_mod = float(np.max(np.abs(eigvals)))
        dominant = eigvals[int(np.argmax(np.abs(eigvals)))]
        results.append(FixedPoint(
            h=h_star.cpu().numpy(), speed=speed, jacobian=jac, eigenvalues=eigvals,
            max_eig_modulus=max_mod, is_stable=max_mod < stability_tol,
            is_marginal=abs(max_mod - 1.0) < marginal_tol,
            is_oscillatory=bool(abs(dominant.imag) > 1e-3 * max(abs(dominant), 1e-8)),
        ))
    return results


def summarize_fixed_points(
    fixed_points: list[FixedPoint], subpop_slices: Optional[dict[str, slice]] = None,
) -> dict:
    """Per-run scalar DVs. `subpop_slices` (e.g. `{"worker": slice(0,196),
    "manager": slice(196,324)}` for an S=1 run's joint state vector) does
    NOT re-run the finder on an isolated subsystem -- it reads off the
    SAME joint fixed points' block-diagonal sub-Jacobian
    (`jacobian[sl, sl]`), i.e. "how attracting is this joint fixed point
    along the worker/manager axes alone", which is the well-defined
    question given the worker and manager are coupled every tick
    (`HRLCore.forward`) and have no independent attractor of their own to
    isolate."""
    out = {
        "n_fixed_points": len(fixed_points),
        "n_stable_fixed_points": sum(1 for fp in fixed_points if fp.is_stable),
        "n_marginal_fixed_points": sum(1 for fp in fixed_points if fp.is_marginal),
        "n_oscillatory_fixed_points": sum(1 for fp in fixed_points if fp.is_oscillatory),
        "mean_speed": float(np.mean([fp.speed for fp in fixed_points])) if fixed_points else None,
        "max_eig_modulus": float(np.max([fp.max_eig_modulus for fp in fixed_points])) if fixed_points else None,
        "mean_eig_modulus": float(np.mean([fp.max_eig_modulus for fp in fixed_points])) if fixed_points else None,
    }
    if subpop_slices:
        for name, sl in subpop_slices.items():
            mods = [float(np.max(np.abs(np.linalg.eigvals(fp.jacobian[sl, sl])))) for fp in fixed_points]
            out[f"{name}_max_eig_modulus"] = float(np.max(mods)) if mods else None
            out[f"{name}_n_stable_fixed_points"] = int(sum(1 for m in mods if m < 1.0))
    return out
