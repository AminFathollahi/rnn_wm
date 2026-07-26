#!/usr/bin/env python3
"""Phase 2 acceptance check: trials/second for the S=0 and S=1 cells,
before vs. after the throughput fixes (comments.txt §5 Phase 2). Measures
MARGINAL per-step cost (two step counts, subtract) to remove fixed one-time
setup cost (ImageTokenBank/ResNet loading, cudnn warmup) from the estimate --
a naive single-run wall-clock conflates the two, and setup cost does not
shrink with the fix while per-step cost does, so it would bias the ratio.
Not a permanent regression test -- a one-off benchmark, invoked fresh (own
process) per (run, steps) pair so no run benefits from another's warm cache;
output pasted into PHASE_LOG.md."""
import argparse
import shutil
import time

import torch
import yaml
from pathlib import Path

from brainalign_wm.training.train import train_one

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "s0": {"run_id": "bench_s0", "model_id": "M00000", "S": 0, "M": 0, "seed": 0, "P": 0},
    "s1": {"run_id": "bench_s1", "model_id": "M10000", "S": 1, "M": 0, "seed": 0, "P": 0},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", choices=RUNS)
    ap.add_argument("steps", type=int)
    args = ap.parse_args()
    run = RUNS[args.cell]
    # A stale checkpoint from a prior benchmark run would make train_one
    # RESUME instead of training from scratch, silently truncating the
    # measured step count -- always start clean.
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    t0 = time.time()
    train_one(run, {"steps": args.steps})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"RESULT cell={args.cell} steps={args.steps} wall_s={time.time() - t0:.4f}")


if __name__ == "__main__":
    main()
