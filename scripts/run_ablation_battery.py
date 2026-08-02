#!/usr/bin/env python3
"""Bio-plausible ablation battery off M11111, the full reference cell
(master protocol §4.4, comments.txt item 5 / 2026-07-08 v6.0 re-anchor):
three single-factor perturbations -- +energy cost, +noise, +pbwm_gate --
each trained for >=5 seeds and later compared against the Core
M11111_s{seed} cells already in `results/manifest.jsonl`.

A separate CLI from `run_grid.py` because each arm needs a per-run
override (`train_one`'s `run` dict: `energy_cost_weight` /
`recurrent_noise_sigma` / `pbwm_gate`), not a global config.yaml edit --
`run_grid.py`'s CLI has no notion of per-run overrides. Reuses
`run_grid.py`'s manifest/report/signal-handling machinery directly so both
scripts stay resumable and produce a consistent `results/manifest.jsonl`
and `results/RUN_REPORT.md`.

Run_id convention (consumed by `training/generate_activity_logs.py::
_run_id_extras` and `analysis/run_all.py::_is_ablation_or_catch_variant`):
`M11111_energy_s{seed}`, `M11111_noise_s{seed}`, `M11111_pbwm_s{seed}`.

Usage:
  python scripts/run_ablation_battery.py --seeds 5 --budget 10h
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

# input_noise_sigma (config.yaml model.input_noise_sigma) is the Core's
# existing input-noise scale; §4.4 explicitly directs "+noise" to reuse
# that same magnitude extended to the recurrent state.
ARMS = [
    {"suffix": "energy", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1, "energy_cost_weight": 0.01},
    {"suffix": "noise", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1, "recurrent_noise_sigma": 0.05},
    {"suffix": "pbwm", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1, "pbwm_gate": True},
]

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_ablation_battery] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    """Seed-major ordering (breadth before depth), same convention as
    `run_grid.enumerate_runs`."""
    runs = []
    for seed in seeds:
        for arm in ARMS:
            extra = {k: v for k, v in arm.items() if k not in ("suffix", "S", "M", "P", "T", "D")}
            model_id = f"M{arm['S']}{arm['M']}{arm['P']}{arm['T']}{arm['D']}_{arm['suffix']}"
            runs.append({
                "model_id": model_id, "S": arm["S"], "M": arm["M"], "P": arm["P"], "T": arm["T"], "D": arm["D"],
                "seed": seed, "run_id": f"{model_id}_s{seed}", **extra,
            })
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="10h", help="wall-clock budget, e.g. 10h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--tier", type=str, default="dev", choices=["smoke", "dev", "full"],
                     help="compute tier (default: dev, a sanity-check tier -- the full study uses --tier full)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    runs = enumerate_runs(seeds)
    completed = load_completed(MANIFEST)
    commit = git_commit()

    import yaml

    full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = {"steps": full_cfg.get("tiers", {}).get(args.tier, {}).get("steps", 20000)}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get(args.tier, {}), args.tier)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("ablation").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_ablation_battery] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_ablation_battery] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_ablation_battery] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_ablation_battery] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_ablation_battery] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
