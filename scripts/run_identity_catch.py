#!/usr/bin/env python3
"""Identity-report catch-trial sub-experiment (master protocol §9.4a,
comments.txt item 6): {M00000, M11111} x >=4 seeds at
`task.identity_catch_fraction=0.12` (re-anchored to the 5-arm ablation
battery's baseline/full-reference cells in the v6.0 pivot), dissociating
"the match/non-match objective didn't need identity" from "the network
can't preserve it" -- compare each cell's maintenance-epoch alignment
WITH vs. WITHOUT the identity catch trials (§9.4a's interpretation
table), at matched match/non-match accuracy.

Separate CLI from `run_grid.py` for the same reason as
`run_ablation_battery.py`: `identity_catch_fraction` is a per-run
override (`train_one`'s `run` dict), not a global config.yaml edit.

Run_id convention (consumed by `training/generate_activity_logs.py::
_run_id_extras` and `analysis/run_all.py::_is_ablation_or_catch_variant`):
`M00000_idcatch_s{seed}`, `M11111_idcatch_s{seed}`.

Usage:
  python scripts/run_identity_catch.py --seeds 4 --budget 10h
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

CELLS = [
    {"suffix": "idcatch", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0},
    {"suffix": "idcatch", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1},
]
IDENTITY_CATCH_FRACTION = 0.12

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_identity_catch] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    """Seed-major ordering (breadth before depth), same convention as
    `run_grid.enumerate_runs`."""
    runs = []
    for seed in seeds:
        for cell in CELLS:
            model_id = f"M{cell['S']}{cell['M']}{cell['P']}{cell['T']}{cell['D']}_{cell['suffix']}"
            runs.append({
                "model_id": model_id, "S": cell["S"], "M": cell["M"], "P": cell["P"], "T": cell["T"], "D": cell["D"],
                "seed": seed, "run_id": f"{model_id}_s{seed}",
                "identity_catch_fraction": IDENTITY_CATCH_FRACTION,
            })
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4, help="number of seeds (0..N-1); breadth-first")
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
    resolved_config_path("identity").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_identity_catch] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_identity_catch] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_identity_catch] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_identity_catch] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_identity_catch] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
