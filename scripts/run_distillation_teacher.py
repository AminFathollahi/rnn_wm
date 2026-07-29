#!/usr/bin/env python3
"""Item 9.2 (comments.txt §5): trains M00000 (the plain S=0,M=0,P=0,T=0,D=0
cell, canonical `flat_units=128` from `configs/config.yaml`) at `--tier full`
to serve as the distillation TEACHER -- "Teacher = a trained full-size cell."

None of Phase 9a's `M00000_H*` checkpoints qualify: they were all trained at
dev tier (20000 steps, 1 seed) and NONE met the §3 accuracy criterion
(`criterion_met_frac == 0.0` for every H, see PHASE_LOG.md's Phase 9a entry)
-- distilling "behaviour at the criterion" (9.2a) from a teacher that never
reached the criterion itself would be meaningless. `run_grid.py` itself
would work but enumerates all 8 ablation-battery cells breadth-first per
seed (~8x the wall-clock for one teacher) -- same reasoning as
`run_bioinit_arm.py`/`run_capacity_curve.py` for a single supplementary
cell outside the battery.

Run_id: `M00000_teacher_s{seed}` (distinct from both `run_grid.py`'s plain
`M00000_s{seed}` -- reserved for the real Core-battery run, Phase 11 -- and
`run_capacity_curve.py`'s `M00000_H128_s{seed}` -- dev tier only).

Usage:
  python scripts/run_distillation_teacher.py --seeds 1 --tier full --budget 8h
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

MODEL_ID = "M00000_teacher"
_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_distillation_teacher] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    return [
        {
            "model_id": MODEL_ID, "S": 0, "M": 0, "P": 0, "T": 0, "D": 0,
            "seed": seed, "run_id": f"{MODEL_ID}_s{seed}",
        }
        for seed in seeds
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="8h", help="wall-clock budget, e.g. 8h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--tier", type=str, default="full", choices=["smoke", "dev", "full"],
                     help="compute tier (default: full -- this teacher must actually reach the §3 criterion)")
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
    cfg = {"steps": full_cfg.get("tiers", {}).get(args.tier, {}).get("steps", 150000)}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get(args.tier, {}), args.tier)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("distillation_teacher").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_distillation_teacher] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_distillation_teacher] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_distillation_teacher] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_distillation_teacher] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_distillation_teacher] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[run_distillation_teacher] <<< {run['run_id']} status={rec.get('status')} "
              f"wall_s={rec['wall_clock_s']} criterion_met={rec.get('criterion_met')}", flush=True)

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
