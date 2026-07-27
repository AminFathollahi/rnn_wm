#!/usr/bin/env python3
"""Item 8.10 (comments.txt §5): driver over all completed runs in
`results/manifest.jsonl` -> `results/geometry_results.csv`.

For each `status == "completed"` manifest entry with a matching
`results/activity_logs/{run_id}.parquet` (written by
`brainalign_wm/training/generate_activity_logs.py`), builds a single-trial
ensemble (C3 -- no trial-averaging) from the delay/maintain-epoch `h_flat`
ticks and computes `mean_speed`/`pca_participation_ratio`
(`analysis/geometry.py`), plus -- if a matching checkpoint exists --
the four [SHAKIBA26] topology metrics (`analysis/network_properties.py`)
from the trained `weight_hh`. One row per run.

SCAFFOLDING, NOT THE FINAL ANALYSIS: no real training runs exist yet (no
`results/manifest.jsonl` at all), so the currently-exercised path is the
graceful empty case below -- a header-only CSV plus an explanatory message,
exit 0. Does not attempt every 8.1-8.9 metric (content/context rotation,
orthogonalization index, task-irrelevant decoding all need per-trial
content/context/position labels this driver doesn't try to reconstruct from
the log schema alone) -- that full-coverage pass is Phase 11's job, once
real trained checkpoints exist to design it against.

Usage:
  python scripts/run_geometry.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import MANIFEST, RESULTS, load_all_records  # noqa: E402

from brainalign_wm.analysis.geometry import mean_speed, pca_participation_ratio  # noqa: E402
from brainalign_wm.analysis.network_properties import (  # noqa: E402
    degree_assortativity,
    gru_effective_connectivity,
    modularity_q,
    small_worldness,
    weight_entropy,
)

ACTIVITY_LOGS = RESULTS / "activity_logs"
OUT_CSV = RESULTS / "geometry_results.csv"

COLUMNS = [
    "run_id", "model_id", "seed", "status", "n_trials",
    "mean_speed", "pca_participation_ratio",
    "weight_entropy_hist", "weight_entropy_kde",
    "modularity_q_louvain", "modularity_q_cnm",
    "small_worldness", "degree_assortativity", "error",
]


def _build_single_trial_ensemble(df: pd.DataFrame) -> np.ndarray:
    """[n_trials, n_timebins, n_units] from `h_flat`, restricted to
    epoch=='maintain' (the delay period -- matches
    `network_properties.py::_epoch_trial_means`'s own epoch-filtering
    convention), each trial's ticks sorted by `t` (C3: one row per REAL
    trial, never averaged). Trials whose tick count doesn't match the
    session's modal tick count are dropped (a single array needs one
    fixed T; a handful of truncated/aborted trials shouldn't blank the
    whole ensemble)."""
    sub = df[df["epoch"] == "maintain"]
    if len(sub) == 0:
        return np.zeros((0, 0, 0))
    per_trial = []
    for _, g in sub.groupby("trial_id"):
        g = g.sort_values("t")
        per_trial.append(np.stack(g["h_flat"].to_numpy()))
    lengths = [len(x) for x in per_trial]
    if not lengths:
        return np.zeros((0, 0, 0))
    modal_T = max(set(lengths), key=lengths.count)
    per_trial = [x for x in per_trial if len(x) == modal_T]
    return np.stack(per_trial) if per_trial else np.zeros((0, 0, 0))


def _topology_metrics(run_id: str, model_id: str) -> dict:
    """Best-effort: load the trained checkpoint and read `core.cell.weight_hh`
    to compute the four [SHAKIBA26] topology metrics. Returns {} (not an
    error) if there's no checkpoint, or this cell's substrate has no single
    `weight_hh` matrix to read (S=1 worker/manager split, vanilla substrate)
    -- item 8.10's job is a driver over whatever is available today, not to
    force every architecture to expose this."""
    ckpt_path = RESULTS / "checkpoints" / run_id / "ckpt.pt"
    if not ckpt_path.exists():
        return {}
    from brainalign_wm.training.generate_activity_logs import _load_checkpoint, _parse_model_id, _run_id_extras
    from brainalign_wm.training.train import _build_model, _load_full_config

    S, M, P, _T, _D = _parse_model_id(model_id)
    pbwm_gate, _idcatch, _mult = _run_id_extras(model_id)
    full_cfg = _load_full_config()
    device = torch.device("cpu")
    front_end, core, heads = _build_model(full_cfg, S, M, P, device, pbwm_gate=pbwm_gate)
    _load_checkpoint(front_end, core, heads, run_id, device)
    if not hasattr(core, "cell") or not hasattr(core.cell, "weight_hh"):
        return {}
    weight_hh = core.cell.weight_hh.detach().cpu().numpy()
    C = gru_effective_connectivity(weight_hh)
    return {
        "weight_entropy_hist": weight_entropy(weight_hh, method="histogram"),
        "weight_entropy_kde": weight_entropy(weight_hh, method="kde"),
        "modularity_q_louvain": modularity_q(C, method="louvain"),
        "modularity_q_cnm": modularity_q(C, method="cnm"),
        "small_worldness": small_worldness(C),
        "degree_assortativity": degree_assortativity(C),
    }


def _write_csv(rows: list[dict]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in COLUMNS})


def main() -> int:
    RESULTS.mkdir(exist_ok=True)
    records = [r for r in load_all_records(MANIFEST) if r.get("status") == "completed"]
    rows = []
    for rec in records:
        run_id = rec["run_id"]
        log_path = ACTIVITY_LOGS / f"{run_id}.parquet"
        if not log_path.exists():
            continue
        row = {"run_id": run_id, "model_id": rec.get("model_id", "?"), "seed": rec.get("seed", "?")}
        try:
            df = pd.read_parquet(log_path)
            Z = _build_single_trial_ensemble(df)
            row["n_trials"] = int(Z.shape[0])
            if Z.shape[0] >= 2:
                row["mean_speed"] = mean_speed(Z)
                row["pca_participation_ratio"] = pca_participation_ratio(Z.reshape(-1, Z.shape[-1]))
            row.update(_topology_metrics(run_id, row["model_id"]))
            row["status"] = "ok"
        except Exception as e:  # noqa: BLE001 -- one bad run shouldn't blank the whole CSV (run_grid.py's own isolation philosophy)
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)

    if not rows:
        _write_csv([])
        print(
            f"[run_geometry] no completed runs with a matching activity log found under "
            f"{ACTIVITY_LOGS} (manifest: {MANIFEST}) -- wrote header-only {OUT_CSV}. "
            f"Run brainalign_wm/training/generate_activity_logs.py after training a model "
            f"to populate this.",
            flush=True,
        )
        return 0

    _write_csv(rows)
    print(f"[run_geometry] wrote {len(rows)} row(s) to {OUT_CSV}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
