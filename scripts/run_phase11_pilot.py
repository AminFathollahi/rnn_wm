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

INIT/SUPERVISION DIAGNOSTIC (comments.txt §13.4): `--supervision` and
`--recurrent-init-spectral-radius` isolate, respectively, the training
signal and the vanilla recurrent init's spectral radius -- both left
implicit in the original pilot above, and both flagged by the 2026-08-01
advisor audit as alternative sufficient causes of the flat-vanilla NO-GO.
`--diagnostic` switches to a self-documenting `run_id` encoding structure,
init, and signal (e.g. `VANFLAT_INIT100_SUP_s0`) instead of the historical
`M{s}0000_pilot_{substrate}_s{seed}`; without it, the original command
above always resumes/reproduces exactly, regardless of what `--supervision`/
`--recurrent-init-spectral-radius` happen to be set to.

Non-vanilla diagnostic runs (comments.txt §14.1) drop the `INITnnn` segment
entirely instead of inheriting the vanilla default, since the recurrent-init
radius knob is inert on the GRU path: `FLATGRU_LEGACY_s0`, not
`FLATGRU_INIT062_LEGACY_s0`.

Usage:
  python scripts/run_phase11_pilot.py --s 0 --seed 0 --diagnostic \
      --supervision SUP --recurrent-init-spectral-radius 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainalign_wm.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from run_grid import (  # noqa: E402
    MANIFEST, RESULTS, ROOT,
    build_resolved_config, config_hash, git_commit, resolve_train_fn, resolved_config_path,
)

PILOT_STEPS = 200_000


def build_diagnostic_model_id(s: int, substrate: str, radius: float | None, supervision: str) -> str:
    """Self-documenting diagnostic run_id prefix (comments.txt §13.4/§14.1).

    The init-radius segment is vanilla-specific -- `_build_model` never reads
    `recurrent_init_spectral_radius` on the GRU path, so labeling a non-vanilla
    run with an `INITnnn` token would misstate a vanilla-only measurement as
    if it applied to a different substrate.
    """
    structure = "FLAT" if s == 0 else "HIER"
    if substrate == "vanilla":
        init_label = "INIT062" if radius is None else f"INIT{round(radius * 100):03d}"
        return f"VAN{structure}_{init_label}_{supervision.upper()}"
    return f"{structure}{substrate.upper()}_{supervision.upper()}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s", type=int, required=True, choices=[0, 1], help="S=0 flat or S=1 hierarchical")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=PILOT_STEPS)
    ap.add_argument("--substrate", choices=["vanilla", "gru"], default="vanilla",
                    help="F1: Stage 1 (which this pilot gates) is the vanilla tanh RNN, §4")
    ap.add_argument("--supervision", choices=["legacy", "SUP", "RL"], default="legacy",
                    help="training signal; 'legacy' (this script's own default) preserves the original "
                         "pilot's exact behavior -- CE during warmup only, REINFORCE after")
    ap.add_argument("--recurrent-init-spectral-radius", type=float, default=None,
                    help="vanilla recurrent init override; omitted (default) preserves the existing "
                         "uniform(-1/sqrt(H),1/sqrt(H)) draw, whose spectral radius measures ~0.616 at H=128")
    ap.add_argument("--diagnostic", action="store_true",
                    help="use the self-documenting VANFLAT/VANHIER_INITnnn_SIGNAL_s{seed} run_id for vanilla, or "
                         "FLAT/HIER{SUBSTRATE}_SIGNAL_s{seed} (no INITnnn segment) for other substrates "
                         "(comments.txt §13.4/§14.1) -- instead of the historical "
                         "M{s}0000_pilot_{substrate}_s{seed}. Pass this even for a diagnostic arm whose values "
                         "happen to match the historical defaults (the current-init/legacy control), so it gets "
                         "its own run_id rather than colliding with the original pilot's")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    commit = git_commit()

    import yaml

    full_cfg = load_config(args.config)
    if not args.diagnostic:
        # run_id carries the substrate: the invalid GRU pilot already
        # occupies `M{s}0000_pilot_s{seed}` in the manifest and its
        # checkpoints are still on disk (needed for F5's
        # at-criterion-vs-max_steps geometry check), so a re-run must not
        # resume from them or overwrite them.
        model_id = f"M{args.s}0000_pilot_{args.substrate}"
        resolved_config_name = f"phase11_pilot_{args.substrate}_s{args.s}"
    else:
        # Diagnostic naming (comments.txt §13.4): self-documenting so the
        # manifest never needs a side table to say what a row varied.
        model_id = build_diagnostic_model_id(
            args.s, args.substrate, args.recurrent_init_spectral_radius, args.supervision
        )
        resolved_config_name = f"{model_id.lower()}_s{args.seed}"
    run_id = f"{model_id}_s{args.seed}"
    run = {
        "model_id": model_id, "S": args.s, "M": 0, "P": 0, "T": 0, "D": 0,
        "substrate": args.substrate, "supervision": args.supervision,
        "seed": args.seed, "run_id": run_id,
    }
    if args.recurrent_init_spectral_radius is not None:
        run["recurrent_init_spectral_radius"] = args.recurrent_init_spectral_radius
    cfg = {"steps": args.steps}
    resolved_cfg = build_resolved_config(full_cfg, full_cfg.get("tiers", {}).get("full", {}), "full", run=run)
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path(resolved_config_name).write_text(
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
