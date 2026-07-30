#!/usr/bin/env python3
"""Phase 11.1 (comments.txt §11, BLOCKING GO/NO-GO gate): the de-leaked
Sternberg task (all of F1-F3's leak fixes now in place, Phase 0-2) is
genuinely harder than the pre-audit numbers suggested and nobody yet
knows whether it is solvable at all. Before any Stage 1/2/3 grid run,
train S=0 and S=1 on the **vanilla tanh substrate**, M=P=T=D=0, seed 0,
train-to-criterion.

SUBSTRATE (Phase 12 audit finding F1 -- the first run of this pilot was
INVALID and its GO verdict does not transfer): §11.1's word "vanilla" was
originally read here as "no mechanisms on (M=P=T=D=0)" and the run dict
carried no `substrate` key, so `train_one` fell through to
`config.yaml`'s `model.substrate: gru` and the pilot trained a GRU
(`results/resolved_config_phase11_pilot_s0.yaml` records `substrate: gru`).
But §4 defines Stage 1 -- which this pilot gates -- as the vanilla tanh RNN
(~25k recurrent synapses at H=128), and gatedness is being promoted to its
own arm precisely because an ungated tanh core maintaining item identity
across a 25-tick delay is the harder problem. A GRU's GO says nothing about
whether the vanilla substrate can clear the gate. `--substrate` now defaults
to vanilla and is written into the run dict explicitly.

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
    ap.add_argument("--substrate", choices=["vanilla", "gru"], default="vanilla",
                    help="F1: Stage 1 (which this pilot gates) is the vanilla tanh RNN, §4")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    commit = git_commit()

    import yaml

    full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    # run_id carries the substrate: the invalid GRU pilot already occupies
    # `M{s}0000_pilot_s{seed}` in the manifest and its checkpoints are still
    # on disk (needed for F5's at-criterion-vs-max_steps geometry check), so
    # a re-run must not resume from them or overwrite them.
    model_id = f"M{args.s}0000_pilot_{args.substrate}"
    run = {
        "model_id": model_id, "S": args.s, "M": 0, "P": 0, "T": 0, "D": 0,
        "substrate": args.substrate,
        "seed": args.seed, "run_id": f"{model_id}_s{args.seed}",
    }
    cfg = {"steps": args.steps}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get("full", {}), "full",
                                         model_overrides={"substrate": args.substrate})
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path(f"phase11_pilot_{args.substrate}_s{args.s}").write_text(
        yaml.safe_dump(resolved_cfg, sort_keys=True)
    )

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
