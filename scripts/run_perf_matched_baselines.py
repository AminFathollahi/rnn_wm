#!/usr/bin/env python3
"""Performance-matched control baselines off M00000, the baseline cell
(§4.1, reviewer suggestion applied 2026-07-08; re-anchored to the 5-arm
ablation battery's M00000 in the v6.0 pivot): flat GRUs given a generic
capacity/regularization boost instead of a specific biological
constraint, so that alignment differences between the Core ablation cells
and these controls can be attributed to the *biological* mechanism rather
than to accuracy or regularization strength alone.

  - flat_gru_2x: `model.flat_units` doubled (matches a hierarchical
    model's higher parameter count without adding hierarchy).
  - flat_gru_l1: `train.l1_weight_penalty` on the core's own weight
    matrices (matches Hebbian plasticity's implicit sparsification
    without the spatial/associative structure Hebbian updates add).
  - flat_gru_dropout: `model.core_dropout_p` on the recurrent state
    (matches the noise/regularization *effect* of Hebbian plasticity or
    recurrent noise without the specific mechanism).

All three are trained from M00000's own architecture (S=0,M=0,P=0,T=0,D=0),
BPTT throughout -- only the one extra knob differs from the Core M00000 cell.

Run_id convention (consumed by `training/generate_activity_logs.py::
_run_id_extras`, `analysis/run_all.py`'s existing `model_id != "M{S}{M}{P}{T}{D}"`
variant guard, which already covers any non-Core-exact model_id with no
change needed): `M00000_2x_s{seed}`, `M00000_l1_s{seed}`, `M00000_dropout_s{seed}`.

Usage:
  python scripts/run_perf_matched_baselines.py --seeds 3 --budget 8h
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import (  # noqa: E402
    MANIFEST, REPORT, RESULTS, ROOT,
    build_resolved_config, config_hash, git_commit, load_completed, parse_budget, resolve_train_fn,
    resolved_config_path, write_report,
)

# `flat_units` doubling happens via an absolute override (2 * Core
# default, read from config.yaml at launch time below) rather than a
# multiplier baked into the arm dict, so the arm survives a future
# `model.flat_units` config change without silently drifting.
# M7 fix: `l1_weight_penalty` (not the old ad hoc `l1_weight`) so the
# per-run override key matches `train.py`'s config-key convention.
ARMS = [
    {"suffix": "2x"},
    {"suffix": "l1", "l1_weight_penalty": 0.001},
    {"suffix": "dropout", "core_dropout_p": 0.1},
]

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_perf_matched_baselines] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int], base_flat_units: int) -> list[dict]:
    """Seed-major ordering (breadth before depth), same convention as
    `run_grid.enumerate_runs`."""
    runs = []
    for seed in seeds:
        for arm in ARMS:
            extra = {k: v for k, v in arm.items() if k != "suffix"}
            if arm["suffix"] == "2x":
                extra["flat_units"] = base_flat_units * 2
            model_id = f"M00000_{arm['suffix']}"
            runs.append({
                "model_id": model_id, "S": 0, "M": 0, "P": 0, "T": 0, "D": 0,
                "seed": seed, "run_id": f"{model_id}_s{seed}", **extra,
            })
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="8h", help="wall-clock budget, e.g. 8h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--tier", type=str, default="dev", choices=["smoke", "dev", "full"],
                     help="compute tier (default: dev, a sanity-check tier -- the full study uses --tier full)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    completed = load_completed(MANIFEST)
    commit = git_commit()

    import yaml

    full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    base_flat_units = int(full_cfg.get("model", {}).get("flat_units", 256))
    runs = enumerate_runs(seeds, base_flat_units)
    cfg = {"steps": full_cfg.get("tiers", {}).get(args.tier, {}).get("steps", 20000)}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get(args.tier, {}), args.tier)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("perf").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_perf_matched_baselines] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_perf_matched_baselines] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_perf_matched_baselines] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_perf_matched_baselines] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_perf_matched_baselines] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
