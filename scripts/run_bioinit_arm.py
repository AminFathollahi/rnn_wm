#!/usr/bin/env python3
"""M00001_bioinit (item 8.9b, comments.txt §5): arm D (S=0,M=0,P=0,T=0,D=1,
the existing +D ablation-battery cell) with the bio-statistics weight init
(`brainalign_wm.models.gru_cell.bioinit_weight_hh`) instead of the uniform
default -- tests whether initialization rescues the Dale sign-penalty cost
[SHAKIBA26] Table 3 predicts (arm D costs substantial performance under the
current uniform-random init; a bio-derived init should recover most of it).

Not enumerated in `run_grid.py`'s auto-generated `CELLS` (a single
supplementary variant, not part of the 5-bit ablation battery), same
pattern as `run_perf_matched_baselines.py`'s M00000_2x/l1/dropout arms --
`bioinit: True` is a per-run override `train_one` reads (mirrors
`pbwm_gate`'s convention exactly, see `training/train.py::train_one`).

Usage:
  python scripts/run_bioinit_arm.py --seeds 3 --budget 8h
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

MODEL_ID = "M00001_bioinit"
_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_bioinit_arm] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    return [
        {
            "model_id": MODEL_ID, "S": 0, "M": 0, "P": 0, "T": 0, "D": 1,
            "bioinit": True, "seed": seed, "run_id": f"{MODEL_ID}_s{seed}",
        }
        for seed in seeds
    ]


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
    runs = enumerate_runs(seeds)
    cfg = {"steps": full_cfg.get("tiers", {}).get(args.tier, {}).get("steps", 20000)}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get(args.tier, {}), args.tier)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("bioinit").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_bioinit_arm] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_bioinit_arm] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_bioinit_arm] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_bioinit_arm] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_bioinit_arm] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
