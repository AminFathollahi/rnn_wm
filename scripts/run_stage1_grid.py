#!/usr/bin/env python3
"""Stage 1 (comments.txt §4/§11.3): map the training regime on the cheap
vanilla-tanh-RNN substrate before spending compute on the GRU
bio-plausibility battery (Stage 2). Factorial:

    supervision  {SUP, RL}
  x structure    {S=0 flat, S=1 hierarchical}
  x diet         {wm_only, multitask}
  = 8 cells x 3 seeds = 24 runs.

Every cell is `model.substrate=vanilla`, M=0, P=0 (M/P are Stage 2 arms,
not Stage 1's factorial -- see `_build_model`'s vanilla paths). Distinct
run_id scheme (`ST1_*`) from both Stage 2's `M{s}{m}{p}{t}{d}_s{seed}`
(run_grid.py's CELLS) and Phase 11.1's `M{s}0000_pilot_s{seed}`, so
`load_completed` can never conflate a Stage 1 run with either.

Resumable exactly like `run_grid.py`: completed runs recorded in
`results/manifest.jsonl` are skipped on a re-invocation, and `train_one`
itself resumes each individual run from its own last checkpoint.

`--max-steps` has no default -- comments.txt 11.2 requires it be set from
the PILOT's real numbers (1.5x the slower of S=0/S=1's steps_to_criterion,
rounded up to the nearest 10k), a deliberate value chosen once per pilot
result, not a silently-reused constant.

Usage:
  python scripts/run_stage1_grid.py --max-steps 100000 --budget 48h
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import (  # noqa: E402
    MANIFEST, REPORT, RESULTS, ROOT,
    build_resolved_config, config_hash, git_commit, load_completed, parse_budget,
    resolve_train_fn, resolved_config_path, write_report,
)

STAGE1_CELLS = [
    {
        "model_id": f"ST1_S{s}_{sup}_{diet_tag}", "S": s, "M": 0, "P": 0,
        "supervision": sup, "diet": diet, "substrate": "vanilla",
    }
    for s in (0, 1)
    for sup in ("SUP", "RL")
    for diet, diet_tag in (("wm_only", "wm"), ("multitask", "mt"))
]

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_stage1] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_stage1_runs(seeds: list[int]) -> list[dict]:
    """Seed-major (breadth before depth), same convention as
    `run_grid.enumerate_runs`."""
    return [{**cell, "seed": seed, "run_id": f"{cell['model_id']}_s{seed}"} for seed in seeds for cell in STAGE1_CELLS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, required=True,
                    help="comments.txt 11.2: 1.5x the slower pilot's steps_to_criterion, rounded up to 10k")
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1); study default 3 (§4)")
    ap.add_argument("--budget", type=str, default="72h", help="wall-clock budget, e.g. 48h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--scaffold", action="store_true", help="force the synthetic stub (no deps)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    runs = enumerate_stage1_runs(seeds)
    completed = load_completed(MANIFEST)
    commit = git_commit()

    import yaml

    full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    cfg = {"steps": args.max_steps}

    resolved_cfg = build_resolved_config(full_cfg, {"steps": args.max_steps}, "stage1")
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("stage1").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(args.scaffold)

    print(f"[run_stage1] max_steps={args.max_steps} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={len(completed)} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_stage1] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_stage1] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": "stage1", "started": started, "phase": "11.3_stage1"}
        print(f"[run_stage1] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same convention as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point, same as run_grid.py
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_stage1] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
