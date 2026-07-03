"""Single-run training entrypoint called by ../../run_grid.py.

TODO(sonnet5): implement per protocol §5 (models), §6 (mechanisms), §7 (task),
§0.2/§14 (gates & logging). Until then this raises NotImplementedError; run_grid
catches it per-run (isolation) or you can drive the orchestration with `--scaffold`.

Contract:
    train_one(run: dict, cfg: dict) -> dict
      run = {"run_id","model_id","S","M","L","seed"}
      returns {"status": "completed"|"failed",
               "gates": {"load1>0.95": bool, "load3>0.80": bool},
               "accuracy": {"load1": float, "load2": float, "load3": float},
               "rung": int}     # local-learning fallback rung reached (§6.2)

Requirements when implemented:
  * seed every RNG from run["seed"] (determinism, §11.4);
  * checkpoint every N steps to results/checkpoints/<run_id>/ and RESUME if present;
  * emit the §9.1 logging schema to Parquet;
  * for L=1, walk the fallback ladder and record the rung;
  * NEVER substitute BPTT for an L=1 cell, and never fabricate a gate pass.
"""
from __future__ import annotations


def train_one(run: dict, cfg: dict) -> dict:
    raise NotImplementedError(
        "train_one is a stub — implement it (protocol §5–§7) or run run_grid.py --scaffold"
    )
