#!/usr/bin/env python3
"""Step-count convergence pilot with early stopping.

Runs a specified architecture at multiple step counts to determine
where performance plateaus. Stops early when all 3 loads reach acc >= 0.999.

Usage:
    python scripts/step_count_pilot.py --cell M00000 --budget 4h
    python scripts/step_count_pilot.py --cell M11111 --budget 4h
    python scripts/step_count_pilot.py --cell M00000 --steps 10000 30000 50000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import get_path, load_config  # noqa: E402

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"

# Cell definitions matching run_grid.py
CELLS = {
    "M00000": {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0},
    "M11111": {"model_id": "M11111", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1},
    "M01111": {"model_id": "M01111", "S": 0, "M": 1, "P": 1, "T": 1, "D": 1},
    "M10111": {"model_id": "M10111", "S": 1, "M": 0, "P": 1, "T": 1, "D": 1},
    "M11011": {"model_id": "M11011", "S": 1, "M": 1, "P": 0, "T": 1, "D": 1},
    "M11101": {"model_id": "M11101", "S": 1, "M": 1, "P": 1, "T": 0, "D": 1},
    "M11110": {"model_id": "M11110", "S": 1, "M": 1, "P": 1, "T": 1, "D": 0},
    "M10000": {"model_id": "M10000", "S": 1, "M": 0, "P": 0, "T": 0, "D": 0},
    "M01000": {"model_id": "M01000", "S": 0, "M": 1, "P": 0, "T": 0, "D": 0},
    "M00100": {"model_id": "M00100", "S": 0, "M": 0, "P": 1, "T": 0, "D": 0},
    "M00010": {"model_id": "M00010", "S": 0, "M": 0, "P": 0, "T": 1, "D": 0},
    "M00001": {"model_id": "M00001", "S": 0, "M": 0, "P": 0, "T": 0, "D": 1},
    "M10010": {"model_id": "M10010", "S": 1, "M": 0, "P": 0, "T": 1, "D": 0},
    "M00011": {"model_id": "M00011", "S": 0, "M": 0, "P": 0, "T": 1, "D": 1},
    "M10001": {"model_id": "M10001", "S": 1, "M": 0, "P": 0, "T": 0, "D": 1},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=str, default="M00000", choices=list(CELLS.keys()),
                     help="architecture to test")
    ap.add_argument("--steps", type=int, nargs="+", default=[10000, 30000, 50000, 75000, 100000, 150000],
                     help="step counts to test")
    ap.add_argument("--budget", type=str, default="4h", help="wall-clock budget")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    args = ap.parse_args(argv)

    budget_s = float(args.budget.rstrip("h")) * 3600 if args.budget.endswith("h") else float(args.budget.rstrip("s"))

    from brainalign_wm.training.train import train_one
    full_cfg = load_config()
    smoke_cfg = full_cfg["tiers"]["smoke"]

    cell = CELLS[args.cell]

    completed = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if rec.get("status") == "completed":
                    completed.add(rec["run_id"])
            except json.JSONDecodeError:
                continue

    t0 = time.time()
    results = []
    prev_acc = None

    for steps in args.steps:
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[pilot] budget reached ({elapsed/3600:.2f}h); stopping.")
            break

        run_id = f"{args.cell}_pilot_{steps // 1000}k"
        if run_id in completed:
            print(f"[pilot] {run_id} already completed; skipping.")
            continue

        run = {**cell, "seed": args.seed, "run_id": run_id}
        cfg = {**smoke_cfg, "steps": steps}

        print(f"[pilot] >>> {run_id} ({steps:,} steps)", flush=True)
        r0 = time.time()
        try:
            result = train_one(run, cfg)
            result.setdefault("status", "completed")
        except Exception as e:
            result = {"status": "error", "error": f"{type(e).__name__}: {e}"}
            print(f"[pilot] !!! {run_id} errored: {result['error']}", flush=True)

        wall = time.time() - r0
        acc = result.get("accuracy", {})
        earlystopped = "early_stop" in str(result)

        rec = {
            **run,
            "config_hash": "pilot",
            "git": "pilot",
            "tier": "pilot",
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **result,
            "wall_clock_s": round(wall, 1),
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[pilot] <<< {run_id}: {result['status']} | "
              f"acc={acc.get('load1','?')}/{acc.get('load2','?')}/{acc.get('load3','?')} | "
              f"{wall:.1f}s", flush=True)

        results.append({"steps": steps, "acc": acc, "wall": wall, "status": result["status"]})

        # Check for plateau: if accuracy barely changed from previous step count, stop
        if prev_acc is not None and all(
            abs(acc.get(f"load{i}", 0) - prev_acc.get(f"load{i}", 0)) < 0.025 for i in (1, 2, 3)
        ):
            # Only stop early if we're already above 0.9 on all loads
            if all(acc.get(f"load{i}", 0) >= 0.9 for i in (1, 2, 3)):
                print(f"[pilot] plateau detected (delta < 0.025 on all loads, all >= 0.9); stopping.", flush=True)
                break

        # Check for perfect accuracy
        if all(acc.get(f"load{i}", 0) >= 0.999 for i in (1, 2, 3)):
            print(f"[pilot] perfect accuracy reached at {steps:,} steps; stopping.", flush=True)
            break

        prev_acc = acc

    # Summary
    total_wall = time.time() - t0
    print(f"\n[pilot] {args.cell} -- {len(results)} runs in {total_wall/3600:.2f}h")
    print(f"{'steps':>8} {'load1':>6} {'load2':>6} {'load3':>6} {'wall(s)':>8} {'status'}")
    print("-" * 55)
    for r in results:
        a = r["acc"]
        print(f"{r['steps']:>8} {a.get('load1','?'):>6} {a.get('load2','?'):>6} "
              f"{a.get('load3','?'):>6} {r['wall']:>8.1f} {r['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
