#!/usr/bin/env python3
"""Phase 11.1 (comments.txt §11, BLOCKING GO/NO-GO gate): the de-leaked
Sternberg task (all of F1-F3's leak fixes now in place, Phase 0-2) is
genuinely harder than the pre-audit numbers suggested and nobody yet
knows whether it is solvable at all. Before any Stage 1/2/3 grid run,
train S=0 and S=1 (vanilla: M=0,P=0,T=0,D=0), seed 0, up to 200k steps,
train-to-criterion.

  GO    -> S=0 meets the §3 criterion within 200k steps.
  NO-GO -> it does not. STOP. Report. Per comments.txt, the task must be
           made learnable BEFORE any grid runs (encode_steps, lure_fraction,
           a load-2 stage, maintain_steps -- in that order; NEVER a
           probe-time cue, NEVER a lowered gate).

Run_id: `M{s}0000_pilot_s{seed}` -- distinct from `run_grid.py`'s own
`M00000_s{seed}`/`M10000_s{seed}` (reserved for the real Stage 1 grid,
whose `max_steps` per 11.2 is 1.5x the slowest PILOT's steps_to_criterion,
a different -- and not yet known -- ceiling than this pilot's fixed 200k).
Reusing the Stage-1 run_id here would make `run_grid.py`'s
`load_completed` skip the real Stage-1 run later, silently substituting
this pilot's result for it.

Usage:
  python scripts/run_phase11_pilot.py --s 0 --seed 0
  python scripts/run_phase11_pilot.py --s 1 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import (  # noqa: E402
    MANIFEST, RESULTS, ROOT,
    build_resolved_config, config_hash, git_commit, resolve_train_fn, resolved_config_path,
)

PILOT_STEPS = 200_000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s", type=int, required=True, choices=[0, 1], help="S=0 flat or S=1 hierarchical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=PILOT_STEPS)
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    commit = git_commit()

    import yaml

    full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    model_id = f"M{args.s}0000_pilot"
    run = {
        "model_id": model_id, "S": args.s, "M": 0, "P": 0, "T": 0, "D": 0,
        "seed": args.seed, "run_id": f"{model_id}_s{args.seed}",
    }
    cfg = {"steps": args.steps}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get("full", {}), "full")
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path(f"phase11_pilot_s{args.s}").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    print(f"[phase11-pilot] run_id={run['run_id']} steps_ceiling={args.steps} config_hash={cfg_hash[:12]}", flush=True)

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": "full", "started": started, "phase": "11.1_pilot"}
    try:
        result = train_one(run, cfg)
        rec.update(result)
        rec.setdefault("status", "completed")
    except Exception as e:  # noqa: BLE001 -- isolation is the point, same as run_grid.py
        rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
        print(f"[phase11-pilot] !!! {run['run_id']} errored: {rec['error']}", flush=True)
    rec["wall_clock_s"] = round(time.time() - t0, 1)
    rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with MANIFEST.open("a") as f:
        f.write(json.dumps(rec) + "\n")

    print(f"\n[phase11-pilot] {json.dumps(rec, default=str, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
