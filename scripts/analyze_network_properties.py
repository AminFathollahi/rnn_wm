#!/usr/bin/env python3
"""Internal brain-like organization metrics (§4.2, reviewer suggestion
applied 2026-07-08) for every completed checkpoint in
`results/manifest.jsonl`: weight entropy, Louvain modularity Q, and
Watts-Strogatz small-worldness on each recurrent component's own
effective connectivity graph (`analysis/network_properties.py`), plus
per-session mixed-selectivity (load x held-item interaction) wherever a
replayed activity log already exists at
`results/activity_logs/{run_id}.parquet` -- generating a fresh log for
runs that don't have one is out of scope here (that's the neural-
alignment pipeline's job, `analysis/run_all.py`); this script only uses
what's already on disk for the mixed-selectivity piece.

Needs zero brain data (weight_entropy/modularity_q/small_worldness are
pure functions of trained weights; mixed_selectivity_index only needs the
model's own replayed activity, not the neural recordings it was replayed
against) -- independent evidence for whether S/M/P produce more brain-
like *internal* organization, orthogonal to every neural-alignment DV.

Usage:
  python scripts/analyze_network_properties.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from brainalign_wm.analysis.network_properties import (
    gru_effective_connectivity, mixed_selectivity_index, modularity_q, small_worldness, weight_entropy,
)
from brainalign_wm.training.generate_activity_logs import _load_checkpoint, _parse_run_id, _run_id_extras
from brainalign_wm.training.train import ROOT, _build_model, _load_full_config

MANIFEST = ROOT / "results" / "manifest.jsonl"
ACTIVITY_LOGS = ROOT / "results" / "activity_logs"
OUT_PATH = ROOT / "results" / "network_properties.jsonl"


def _completed_run_ids() -> list[str]:
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
                print(f"[analyze_network_properties] skipping malformed manifest line: {line[:200]!r}", flush=True)
                continue
            if rec.get("status") == "completed" and not rec.get("archived", False):
                seen[rec["run_id"]] = rec  # last write wins (resumed/rerun rows)
    return list(seen.keys())


def _core_components(core, S: int) -> dict[str, np.ndarray]:
    """{"flat": weight_hh} for S=0 (GRU substrate: `_GatedFlatCore.cell`
    exposes `.weight_hh` directly), or {"worker": ..., "manager": ...} for
    S=1 -- see `models/hrl.py`:
    `HRLCore.worker`/`.manager` are themselves `MaskedGRUCell`/
    `PlasticGRUCell`/`PBWMManagerCell` instances with `.weight_hh` directly
    (no further `.cell` nesting), unlike the S=0 wrapper."""
    if S == 0:
        return {"flat": core.cell.weight_hh.detach().numpy()}
    return {
        "worker": core.worker.weight_hh.detach().numpy(),
        "manager": core.manager.weight_hh.detach().numpy(),
    }


def _mixed_selectivity_for_run(run_id: str) -> "dict | None":
    log_path = ACTIVITY_LOGS / f"{run_id}.parquet"
    if not log_path.exists():
        return None
    df = pd.read_parquet(log_path)
    per_session = []
    for session_id, g in df.groupby("session"):
        result = mixed_selectivity_index(g, epoch="maintain")
        if result is not None:
            per_session.append(result)
    if not per_session:
        return None
    pooled_mean = float(np.mean([r["population_mean"] for r in per_session]))
    return {"population_mean": pooled_mean, "n_sessions": len(per_session)}


def main() -> int:
    import time

    full_cfg = _load_full_config()
    device = torch.device("cpu")
    run_ids = _completed_run_ids()
    print(f"[analyze_network_properties] {len(run_ids)} completed runs in manifest", flush=True)

    # Incremental append (not a batch write at the end): small_worldness's
    # random-reference rewiring is the dominant cost per run and can take
    # minutes on a 256-512-unit connectivity graph -- an earlier version of
    # this script buffered every record in memory and wrote them all only
    # after the full loop, so a `timeout`-killed or interrupted run lost
    # ALL progress, not just the tail. Writing (and flushing) one line per
    # run as it finishes means a partial run leaves a partial, still-usable
    # results file, and a rerun can skip already-computed run_ids.
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
                    print(f"[analyze_network_properties] skipping malformed output line: {line[:200]!r}", flush=True)

    # Audit fix M3: `with` block instead of a bare open()/close() pair -- an
    # exception mid-loop used to leak the fd (close() was never reached).
    n_written = 0
    with OUT_PATH.open("a") as out_f:
        for run_id in run_ids:
            if run_id in already_done:
                continue
            ckpt_path = ROOT / "results" / "checkpoints" / run_id / "ckpt.pt"
            if not ckpt_path.exists():
                print(f"[analyze_network_properties]   skipping {run_id} (no checkpoint on disk)", flush=True)
                continue
            t0 = time.time()
            model_id, S, M, P, T, D, seed = _parse_run_id(run_id)
            pbwm_gate, identity_catch_fraction, flat_units_mult = _run_id_extras(model_id)
            cfg = full_cfg
            if identity_catch_fraction:  # Heads(identity_aux=...) changes param shapes -- see generate_activity_logs.py
                cfg = {**cfg, "task": {**cfg["task"], "identity_catch_fraction": identity_catch_fraction}}
            if flat_units_mult != 1:
                cfg = {**cfg, "model": {**cfg["model"], "flat_units": cfg["model"]["flat_units"] * flat_units_mult}}
            front_end, core, heads = _build_model(cfg, S, M, P, device, pbwm_gate=pbwm_gate)
            _load_checkpoint(front_end, core, heads, run_id, device)

            rec = {"run_id": run_id, "model_id": model_id, "S": S, "M": M, "P": P, "T": T, "D": D, "seed": seed}
            for name, weight_hh in _core_components(core, S).items():
                C = gru_effective_connectivity(weight_hh)
                rec[f"{name}_weight_entropy"] = weight_entropy(weight_hh)
                rec[f"{name}_modularity_q"] = modularity_q(C, seed=0)
                rec[f"{name}_small_worldness"] = small_worldness(C, seed=0)
            mixed_sel = _mixed_selectivity_for_run(run_id)
            if mixed_sel is not None:
                rec["mixed_selectivity_mean"] = mixed_sel["population_mean"]
                rec["mixed_selectivity_n_sessions"] = mixed_sel["n_sessions"]
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_written += 1
            print(f"[analyze_network_properties] >>> {run_id} done in {time.time()-t0:.1f}s", flush=True)

    print(f"[analyze_network_properties] wrote {n_written} new records to {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
