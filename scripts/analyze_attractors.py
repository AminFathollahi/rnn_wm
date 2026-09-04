#!/usr/bin/env python3
"""Attractor structure (fixed points of the recurrent core's own autonomous
maintenance-epoch dynamics, `analysis/attractors.py`) for every completed
checkpoint in `results/manifest.jsonl` that already has a replayed activity
log at `results/activity_logs/{run_id}.parquet` -- generating a fresh log
for a run that doesn't have one yet is out of scope here, exactly the same
scoping `analyze_network_properties.py` uses for its mixed-selectivity
piece (that's the neural-alignment pipeline's job, `analysis/run_all.py`).

The one-step transition function held fixed everything except the hidden
state at its maintenance-epoch value: the visual input during `maintain` is
provably identical on every trial/session (`v_t=0`, `c_t=context_vector
("maintain")`, so `z_t = front_end(v_t, c_t)` is one fixed vector -- see
`brainalign_wm/tasks/sternberg.py::context_vector`'s docstring: load must
be carried in recurrent state, not read off a live per-tick broadcast, so
there is no information the model receives during maintenance beyond that
fixed z_t), the reflective gate bias/raw surprise signal at its resting
(R_t=0, i.e. "no surprise") value, the plastic worker's Hebbian trace at
its trial-initial (zero) value, and -- for S=1 -- the tick phase at a
manager-update tick (t=0; the manager's clock divides any run length, so
t=0 always satisfies `t % manager_period == 0`; off-tick, `h_manager` is
an exact no-op per `HRLCore.forward`'s own docstring, so this is the
system's real discrete-time map sampled at its own natural clock, not an
arbitrary choice of `t`). Starting points for the search are the model's
own REAL maintenance-epoch hidden states from the replayed log (C3:
single-trial, never averaged), plus Gaussian-jittered copies for broader
basin coverage -- never synthetic/random-only starts.

Needs zero brain data, like `analyze_network_properties.py` -- fixed
points are a pure property of the trained model's own dynamics.

Usage:
  python scripts/analyze_attractors.py
  python scripts/analyze_attractors.py --campaign-only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from brainalign_wm.config import get_path, load_config
from brainalign_wm.analysis.attractors import find_fixed_points, summarize_fixed_points
from brainalign_wm.tasks.sternberg import context_vector
from brainalign_wm.training.generate_activity_logs import _load_checkpoint, _parse_run_id, _run_id_extras
from brainalign_wm.training.train import ROOT, _build_model, _load_full_config

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"
ACTIVITY_LOGS = get_path("activity_logs")
OUT_PATH = RESULTS / "attractor_properties.jsonl"

MAX_REAL_STARTS = 64       # real maintenance-epoch hidden states sampled per run (bounds optimizer cost)
N_JITTER_PER_START = 2     # Gaussian-jittered copies per real start, for basin coverage beyond exactly-observed states
JITTER_REL_SIGMA = 0.1     # jitter std, as a fraction of that run's real-state std
SAMPLE_SEED = 0


def _resolved_cfg_for_run(run_id: str, full_cfg: dict) -> dict:
    """Older one-off pilot runs (pre-dating `run_grid.py`'s campaign
    system, e.g. `HIERGRU_SUP_s0`, trained with `manager_units=24` vs.
    today's config default) can have hyperparameters that no longer match
    `configs/config.yaml` -- their own `results/resolved_config_{run_id
    lowercased}.yaml`, written at train time, is ground truth for those
    (CLAUDE.md: "check its manifest row, resolved config, and metrics").
    Every `M#####_{SUP,RL}_s*` campaign run shares one config across all
    120 cells and has no such per-run file, so this falls back to the
    shared `full_cfg` for those (matching `analyze_network_properties.py`'s
    existing assumption, which IS correct for the campaign)."""
    path = RESULTS / f"resolved_config_{run_id.lower()}.yaml"
    if path.exists():
        import yaml

        with path.open() as f:
            return load_config(path)
    return full_cfg


_CAMPAIGN_MODEL_ID = re.compile(r"M[01]{5}")


def _is_campaign_core_record(rec: dict) -> bool:
    """Return whether a manifest row belongs to the five-arm Core campaign."""
    model_id = str(rec.get("model_id", ""))
    supervision = rec.get("supervision")
    return (
        _CAMPAIGN_MODEL_ID.fullmatch(model_id) is not None
        and supervision in {"SUP", "RL"}
        and str(rec.get("run_id", "")).startswith(f"{model_id}_{supervision}_s")
        and all(key in rec for key in ("S", "M", "P", "T", "D", "seed"))
    )


def _completed_run_ids(campaign_only: bool = False) -> list[str]:
    if not MANIFEST.exists():
        return []
    seen = {}
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[analyze_attractors] skipping malformed manifest line: {line[:200]!r}", flush=True)
                continue
            if (rec.get("status") == "completed"
                    and not rec.get("archived", False)
                    and (not campaign_only or _is_campaign_core_record(rec))):
                seen[rec["run_id"]] = rec
    return list(seen.keys())


def _maintenance_input(front_end, feature_dim: int, device) -> torch.Tensor:
    v0 = torch.zeros(1, feature_dim, device=device)
    c0 = torch.tensor([context_vector("maintain", encoded_count=0)], dtype=torch.float32, device=device)
    with torch.no_grad():
        return front_end(v0, c0).detach()


def _build_step_fn(front_end, core, cfg: dict, S: int, M: int, P: int, pbwm_gate: bool, device):
    """Returns (step_fn, subpop_slices). `step_fn` maps `[B, H]` -> `[B, H]`
    (`H` = flat_units for S=0, worker_units+manager_units for S=1).
    `subpop_slices` is None for S=0 (nothing to split) or
    `{"worker": slice, "manager": slice}` for S=1."""
    z_star = _maintenance_input(front_end, cfg["model"]["feature_dim"], device)

    def _expand(t: torch.Tensor, B: int) -> torch.Tensor:
        return t.expand(B, *t.shape[1:])

    if S == 0:
        gate_width = core.hidden_dim

        def step(h: torch.Tensor) -> torch.Tensor:
            B = h.shape[0]
            z_t = _expand(z_star, B)
            gate_bias = torch.zeros(B, gate_width, device=device) if M else None
            if P:
                hebb0 = torch.zeros(B, 3 * core.hidden_dim, core.hidden_dim, device=device)
                h_t, _, _ = core(z_t, h, hebb0, extra_update_bias=gate_bias)
            else:
                h_t, _ = core(z_t, h, extra_update_bias=gate_bias)
            return h_t

        return step, None

    Hw, Hm = core.worker_units, core.manager_units
    gate_width = Hm

    def step(h: torch.Tensor) -> torch.Tensor:
        B = h.shape[0]
        hw, hm = h[:, :Hw], h[:, Hw:]
        z_t = _expand(z_star, B)
        state = {"h_worker": hw, "h_manager": hm, "g": core.g_proj(hm)}
        gate_bias, R_t = None, None
        if pbwm_gate:
            R_t = torch.zeros(B, 1, device=device)
            state["c_manager"] = core.manager.init_cell(B, device)
        elif M:
            gate_bias = torch.zeros(B, gate_width, device=device)
        if P:
            state["hebb_worker"] = torch.zeros(B, 3 * Hw, Hw, device=device)
        new_state, _ = core(z_t, state, t=0, gate_bias=gate_bias, R_t=R_t)
        return torch.cat([new_state["h_worker"], new_state["h_manager"]], dim=-1)

    return step, {"worker": slice(0, Hw), "manager": slice(Hw, Hw + Hm)}


def _sample_starts(run_id: str, S: int) -> "np.ndarray | None":
    log_path = ACTIVITY_LOGS / f"{run_id}.parquet"
    if not log_path.exists():
        return None
    df = pd.read_parquet(log_path)
    df = df[df["epoch"] == "maintain"]
    if len(df) == 0:
        return None
    if S == 0:
        rows = df["h_flat"].dropna().tolist()
    else:
        hw = df["h_worker"].dropna()
        hm = df["h_manager"].dropna()
        joint = pd.concat([hw, hm], axis=1).dropna()
        # `w`/`m` come back from parquet as numpy arrays, not Python lists --
        # `w + m` would silently attempt elementwise addition (and crash on
        # any worker/manager width mismatch) instead of concatenating.
        rows = [np.concatenate([w, m]) for w, m in zip(joint["h_worker"], joint["h_manager"])]
    if not rows:
        return None
    real = np.asarray(rows, dtype=np.float32)
    rng = np.random.RandomState(SAMPLE_SEED)
    if real.shape[0] > MAX_REAL_STARTS:
        idx = rng.choice(real.shape[0], size=MAX_REAL_STARTS, replace=False)
        real = real[idx]
    sigma = JITTER_REL_SIGMA * (real.std(axis=0, keepdims=True) + 1e-6)
    jittered = [real + rng.normal(scale=sigma, size=real.shape).astype(np.float32) for _ in range(N_JITTER_PER_START)]
    return np.concatenate([real, *jittered], axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--campaign-only", action="store_true",
        help="analyze only active completed five-arm Core campaign runs; excludes pilots, "
             "capacity sweeps, local-learning extensions, and other checkpoint variants",
    )
    args = ap.parse_args(argv)

    full_cfg = _load_full_config()
    device = torch.device("cpu")
    run_ids = _completed_run_ids(campaign_only=args.campaign_only)
    print(f"[analyze_attractors] {len(run_ids)} completed runs in manifest", flush=True)

    already_done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    already_done.add(json.loads(line)["run_id"])
                except json.JSONDecodeError:
                    print(f"[analyze_attractors] skipping malformed output line: {line[:200]!r}", flush=True)

    n_written = 0
    with OUT_PATH.open("a") as out_f:
        for run_id in run_ids:
            if run_id in already_done:
                continue
            ckpt_path = RESULTS / "checkpoints" / run_id / "ckpt.pt"
            if not ckpt_path.exists():
                print(f"[analyze_attractors]   skipping {run_id} (no checkpoint on disk)", flush=True)
                continue
            model_id, S, M, P, T, D, seed = _parse_run_id(run_id)
            cfg = _resolved_cfg_for_run(run_id, full_cfg)
            if cfg["model"].get("substrate", "gru") != "gru":
                print(f"[analyze_attractors]   skipping {run_id} (non-GRU substrate, out of scope)", flush=True)
                continue
            starts = _sample_starts(run_id, S)
            if starts is None:
                print(f"[analyze_attractors]   skipping {run_id} (no activity log / no maintain-epoch rows yet)", flush=True)
                continue

            t0 = time.time()
            pbwm_gate, identity_catch_fraction, flat_units_mult = _run_id_extras(model_id)
            if identity_catch_fraction:
                cfg = {**cfg, "task": {**cfg["task"], "identity_catch_fraction": identity_catch_fraction}}
            if flat_units_mult != 1:
                cfg = {**cfg, "model": {**cfg["model"], "flat_units": cfg["model"]["flat_units"] * flat_units_mult}}
            front_end, core, heads = _build_model(cfg, S, M, P, device, pbwm_gate=pbwm_gate)
            _load_checkpoint(front_end, core, heads, run_id, device)
            front_end.eval(); core.eval()

            step_fn, subpop_slices = _build_step_fn(front_end, core, cfg, S, M, P, pbwm_gate, device)
            fixed_points = find_fixed_points(step_fn, starts)
            summary = summarize_fixed_points(fixed_points, subpop_slices=subpop_slices)

            rec = {"run_id": run_id, "model_id": model_id, "S": S, "M": M, "P": P, "T": T, "D": D, "seed": seed,
                   "n_starts": int(starts.shape[0]), **summary}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_written += 1
            print(f"[analyze_attractors] >>> {run_id}: {summary['n_fixed_points']} fixed points "
                  f"({summary['n_stable_fixed_points']} stable) in {time.time()-t0:.1f}s", flush=True)

    print(f"[analyze_attractors] wrote {n_written} new records to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
