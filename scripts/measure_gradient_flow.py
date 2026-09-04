#!/usr/bin/env python3
"""Measures the gradient that actually reaches each tick's hidden state in
a full Sternberg trial, at model initialization (no training), instead of
inferring "vanishing gradient" from a flat training loss curve.

For a handful of recurrent cores, this builds the model fresh, runs one
batch of real trials through the front end and core, backpropagates a
single-tick loss anchored at the probe epoch (the only loss that must flow
backward through the entire maintenance delay to reach the encode epoch),
and records ||dL/dh_t|| at every tick via `retain_grad()`. It reports the
attenuation between the start of the maintain epoch and the first encode
tick -- the delay the model has to bridge to solve the task -- alongside
the measured spectral radius of each core's effective recurrent matrix.

Usage:
  python scripts/measure_gradient_flow.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from brainalign_wm.training.train import (
    ROOT, _build_model, _image_features_all_ticks, _init_state, _load_full_config, _step_core, _target_action,
)

LOADS = (1, 3)
BATCH_SIZE = 32
RESCALED_RADIUS = 1.0


def _spectral_radius(weight: torch.Tensor, mask: Optional[torch.Tensor] = None):
    """Spectral radius of the effective (masked) recurrent weight. A GRU's
    `weight_hh` stacks one square block per gate ([n_gates*H, H]); each
    block is its own recurrent matrix, so this returns a list of radii for
    a stacked weight and a single float for a plain [H, H] one."""
    effective = weight.detach() if mask is None else weight.detach() * mask
    H = effective.shape[-1]
    n_blocks = effective.shape[0] // H
    if n_blocks == 1:
        return torch.linalg.eigvals(effective).abs().max().item()
    return [torch.linalg.eigvals(effective[i * H:(i + 1) * H]).abs().max().item() for i in range(n_blocks)]


def _rescale_to_spectral_radius(weight: torch.Tensor, target: float, mask: Optional[torch.Tensor] = None) -> None:
    """In-place rescale of a square recurrent weight so its (optionally
    masked) effective spectral radius equals `target`."""
    with torch.no_grad():
        radius = _spectral_radius(weight, mask)
        if radius > 1e-12:
            weight.mul_(target / radius)


def _sample_trials(task_gen, load: int, batch_size: int, seed: int):
    """`batch_size` independent trials at a fixed load, under the
    curriculum's final ("target") distribution -- the settled task
    configuration a fully-trained run is actually evaluated against, not an
    early-curriculum shortcut."""
    params = task_gen.curriculum.params_for(10**9, 10**9)
    assert params["phase"] == "target"
    batch = []
    for b in range(batch_size):
        rng = np.random.RandomState(np.random.SeedSequence([seed, b]).generate_state(4))
        batch.append(
            task_gen.sternberg.generate_trial(
                rng=rng, loads=[load], lure_fraction=params["lure_fraction"],
                maintain_steps=params["maintain_steps"], trial_id=b, load_weights=None, split="test",
            )
        )
    return batch


def _run_and_backprop(front_end, core, heads, S: int, trial_steps_batch, image_bank, feature_dim, device):
    """One forward pass through a full trial, backpropagating a single
    cross-entropy loss at the first probe tick. Returns, per tick, the
    tracked hidden-state tensor(s) (with `.retain_grad()` already called)
    so their `.grad` is populated after this returns, plus the trial's
    per-tick epoch labels."""
    B = len(trial_steps_batch)
    T = len(trial_steps_batch[0])
    epochs = [trial_steps_batch[0][i].epoch for i in range(T)]
    probe_i = epochs.index("probe")

    all_v = _image_features_all_ticks(image_bank, trial_steps_batch, feature_dim, device)
    all_c = torch.tensor(
        [[trial_steps_batch[b][i].c_t for b in range(B)] for i in range(T)], dtype=torch.float32, device=device,
    )

    state = _init_state(core, S, P=0, batch=B, device=device)
    tracked = []
    logits_at_probe = None
    for i in range(T):
        z_t = front_end(all_v[i], all_c[i])
        h_star, new_state, _ = _step_core(core, S, M=0, P=0, z_t=z_t, state=state, t=i, gate_bias=None)
        if S == 0:
            new_state["h"].retain_grad()
            tracked.append({"flat": new_state["h"]})
        else:
            new_state["h_worker"].retain_grad()
            new_state["h_manager"].retain_grad()
            tracked.append({"h_worker": new_state["h_worker"], "h_manager": new_state["h_manager"]})
        if i == probe_i:
            _, _, logits_at_probe = heads(h_star)
        state = new_state

    target = torch.tensor(
        [_target_action("probe", bool(trial_steps_batch[b][probe_i].in_set)) for b in range(B)],
        dtype=torch.long, device=device,
    )
    loss = F.cross_entropy(logits_at_probe, target)
    loss.backward()
    return tracked, epochs, probe_i, loss.item()


def _grad_norm(tracked_tick: dict, path: str) -> Optional[float]:
    tensor = tracked_tick.get(path)
    if tensor is None or tensor.grad is None:
        return None
    return tensor.grad.norm().item()


def _measure(label: str, front_end, core, heads, S: int, task_gen, image_bank, feature_dim, device, paths):
    """`paths`: list of (path_label, weight, mask) for the spectral-radius
    report -- one entry for a flat core, two (worker, manager) for a
    hierarchical one."""
    rows = []
    radii = {name: _spectral_radius(w, m) for name, w, m in paths}
    for load in LOADS:
        for p in core.parameters():
            if p.grad is not None:
                p.grad = None
        trial_steps_batch = _sample_trials(task_gen, load, BATCH_SIZE, seed=load)
        tracked, epochs, probe_i, loss_value = _run_and_backprop(
            front_end, core, heads, S, trial_steps_batch, image_bank, feature_dim, device,
        )
        first_encode_i = epochs.index("encode")
        maintain_start_i = epochs.index("maintain")
        for path_name, _, _ in paths:
            grad_probe = _grad_norm(tracked[probe_i], path_name)
            grad_maintain = _grad_norm(tracked[maintain_start_i], path_name)
            grad_encode = _grad_norm(tracked[first_encode_i], path_name)
            ratio = (grad_maintain / grad_encode) if grad_encode and grad_encode > 1e-300 else float("inf")
            rows.append({
                "core": label, "path": path_name, "load": load, "n_ticks": len(epochs),
                "spectral_radius": radii[path_name], "probe_loss": loss_value,
                "grad_at_probe": grad_probe, "grad_at_maintain_start": grad_maintain,
                "grad_at_first_encode": grad_encode, "maintain_to_encode_attenuation": ratio,
            })
    return rows


def _self_check() -> None:
    """Two-tick linear toy recurrence with a known analytic gradient ratio:
    h0=0, h1 = w*h0 + x1, h2 = w*h1 + x2, loss = 0.5*h2**2. Analytically
    dL/dh2 = h2 and dL/dh1 = w*h2, so ||dL/dh1|| / ||dL/dh2|| == |w|
    exactly. If retain_grad()-based measurement is wrong, this catches it
    before the real trial numbers are trusted."""
    w = torch.tensor(0.7)
    x1, x2 = torch.tensor(1.3, requires_grad=True), torch.tensor(-0.4, requires_grad=True)
    h0 = torch.zeros(1)
    h1 = w * h0 + x1
    h1.retain_grad()
    h2 = w * h1 + x2
    h2.retain_grad()
    loss = 0.5 * h2 ** 2
    loss.backward()
    measured_ratio = (h1.grad.abs() / h2.grad.abs()).item()
    assert abs(measured_ratio - w.item()) < 1e-6, f"self-check failed: measured ratio {measured_ratio} != w {w.item()}"
    assert abs(h2.grad.item() - h2.item()) < 1e-6, "self-check failed: dL/dh2 != h2"


def main() -> int:
    _self_check()
    print("[measure_gradient_flow] self-check passed (two-tick toy recurrence, analytic gradient ratio recovered)")

    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    full_cfg = _load_full_config()
    device = torch.device("cpu")
    m = full_cfg["model"]
    feature_dim = m["feature_dim"]

    image_bank = ImageTokenBank(
        stimuli_root=Path(full_cfg["paths"]["stimuli"]), categories=full_cfg["task"]["categories"],
        feature_cache_path=Path(full_cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(full_cfg, image_bank, seed=0)

    torch.manual_seed(0)
    all_rows = []

    vanilla_cfg = {**full_cfg, "model": {**m, "substrate": "vanilla"}}
    gru_cfg = {**full_cfg, "model": {**m, "substrate": "gru"}}

    front_end, core, heads = _build_model(vanilla_cfg, S=0, M=0, P=0, device=device)
    all_rows += _measure(
        "flat_vanilla_init_default", front_end, core, heads, 0, task_gen, image_bank, feature_dim, device,
        paths=[("flat", core.weight_hh, core.mask)],
    )

    _rescale_to_spectral_radius(core.weight_hh, RESCALED_RADIUS, core.mask)
    all_rows += _measure(
        f"flat_vanilla_radius_{RESCALED_RADIUS:.2f}", front_end, core, heads, 0, task_gen, image_bank, feature_dim,
        device, paths=[("flat", core.weight_hh, core.mask)],
    )

    front_end, core, heads = _build_model(vanilla_cfg, S=1, M=0, P=0, device=device)
    all_rows += _measure(
        "hierarchical_vanilla_init_default", front_end, core, heads, 1, task_gen, image_bank, feature_dim, device,
        paths=[("h_worker", core.worker.weight_hh, core.worker.mask), ("h_manager", core.manager.weight_hh, core.manager.mask)],
    )

    front_end, core, heads = _build_model(gru_cfg, S=0, M=0, P=0, device=device)
    all_rows += _measure(
        "flat_gru", front_end, core, heads, 0, task_gen, image_bank, feature_dim, device,
        paths=[("flat", core.cell.weight_hh, core.cell.mask)],
    )

    header = (
        f"{'core':<28} {'path':<10} {'load':>4} {'ticks':>5} {'radius':>8} "
        f"{'grad@encode1':>13} {'grad@maintain0':>15} {'grad@probe':>11} {'attenuation':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in all_rows:
        radius = r["spectral_radius"]
        radius_str = (
            "/".join(f"{v:.3f}" for v in radius) if isinstance(radius, (list, tuple)) else f"{radius:.3f}"
        )
        print(
            f"{r['core']:<28} {r['path']:<10} {r['load']:>4} {r['n_ticks']:>5} {radius_str:>8} "
            f"{r['grad_at_first_encode']:>13.3e} {r['grad_at_maintain_start']:>15.3e} "
            f"{r['grad_at_probe']:>11.3e} {r['maintain_to_encode_attenuation']:>13.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
