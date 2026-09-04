#!/usr/bin/env python3
"""Item 9.1 (comments.txt §5): capacity curve. Trains M00000's own
architecture (S=0,M=0,P=0,T=0,D=0) from scratch at H (`model.flat_units`)
in {2,4,8,16,32,64,128,256} on plain Sternberg -- the ONLY thing varying
across runs is H, via the SAME per-run `flat_units` override
`scripts/run_perf_matched_baselines.py`'s `flat_gru_2x` arm already uses
(`training/train.py::train_one`, "if 'flat_units' in run: ...").

Not enumerated in `run_grid.py`'s auto-generated `CELLS` (a supplementary
sweep, not part of the 5-bit ablation battery) -- same pattern as
`run_bioinit_arm.py`/`run_perf_matched_baselines.py`. Run_id convention:
`M00000_H{H}_s{seed}` (parses fine under `generate_activity_logs.py`'s
`_RE_5BIT`/`_parse_run_id`, which matches on the leading 5 bits via
`re.match` and keeps the full suffixed string as `model_id` -- same as
every other suffixed arm). This script does not itself generate activity
logs or compute geometry metrics; see `scripts/analyze_capacity_curve.py`.

Usage:
  python scripts/run_capacity_curve.py --seeds 1 --budget 8h
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

from brainalign_wm.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from run_grid import (  # noqa: E402
    MANIFEST, REPORT, RESULTS, ROOT,
    build_resolved_config, config_hash, git_commit, load_completed, parse_budget, resolve_train_fn,
    resolved_config_path, write_report,
)

H_VALUES = [2, 4, 8, 16, 32, 64, 128, 256]
_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_capacity_curve] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    """Seed-major ordering (breadth before depth): a complete pass over all
    8 H values is produced before any seed's replicate count increments,
    same convention as every other campaign script here."""
    runs = []
    for seed in seeds:
        for H in H_VALUES:
            model_id = f"M00000_H{H}"
            runs.append({
                "model_id": model_id, "S": 0, "M": 0, "P": 0, "T": 0, "D": 0,
                "flat_units": H, "seed": seed, "run_id": f"{model_id}_s{seed}",
            })
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="8h", help="wall-clock budget, e.g. 8h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--tier", type=str, default="dev", choices=["smoke", "dev", "full"],
                     help="compute tier (default: dev, an exploratory sweep -- Phase 11 does the real Stage 1/2 runs)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    completed = load_completed(MANIFEST)
    commit = git_commit()

    import yaml

    full_cfg = load_config(args.config)
    runs = enumerate_runs(seeds)
    cfg = {"steps": full_cfg.get("tiers", {}).get(args.tier, {}).get("steps", 20000)}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get(args.tier, {}), args.tier)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path("capacity").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_capacity_curve] tier={args.tier} H={H_VALUES} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={n_already} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_capacity_curve] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_capacity_curve] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_capacity_curve] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_capacity_curve] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[run_capacity_curve] <<< {run['run_id']} status={rec.get('status')} "
              f"wall_s={rec['wall_clock_s']} acc={rec.get('accuracy')}", flush=True)

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
