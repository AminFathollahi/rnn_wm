#!/usr/bin/env python3
"""Train selected architectures on the six-task diet from a random
initialization -- as opposed to `run_continuations.py`, which starts every
run from an already-finished working-memory checkpoint. This arm answers
what the diet does to a network that never had a working-memory-only
history to begin with.

Isolated per-run execution, same convention as `run_identity_catch.py`,
`run_perf_matched_baselines.py` and the other arm-specific launchers: one
`train_one` call per run, in-process, sequential. Resumable at two levels,
also matching those scripts: a completed run_id is skipped on the next
invocation, and `train_one` itself resumes a partial run from its own
periodic checkpoint.

Run_id convention, read by `training/generate_activity_logs.py` the same
way it reads every other tagged cell: `M{SMPTD}_multitask_{supervision}_s{seed}`
(the manifest `model_id` stays `M{SMPTD}_multitask`, unsuffixed by signal).

Usage:
  python scripts/run_multitask_from_init.py --supervision SUP
  python scripts/run_multitask_from_init.py --supervision SUP --execute
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
    MANIFEST, REPORT, RESULTS,
    build_resolved_config, build_run_id, config_hash, git_commit, load_completed, resolve_train_fn,
    resolved_config_path, write_report,
)

SELECTED_MODELS = ("M00000", "M10000", "M00100", "M11111")
RUN_TAG = "multitask"
# Stated explicitly on every run dict rather than left to a config default --
# both cells this arm trains are GRU at the default recurrent init.
_SUBSTRATE = {"substrate": "gru", "recurrent_init_spectral_radius": None}

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_multitask_from_init] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds, models=SELECTED_MODELS, supervision: str = "SUP") -> list[dict]:
    """Seed-major ordering (breadth before depth), matching every other
    arm-specific launcher in this repository."""
    runs = []
    for seed in seeds:
        for model_id in models:
            if len(model_id) != 6 or not model_id.startswith("M") or any(c not in "01" for c in model_id[1:]):
                raise ValueError(f"invalid model id {model_id!r}; expected M<5 binary digits>")
            bits = {name: int(value) for name, value in zip("SMPTD", model_id[1:])}
            tagged = f"{model_id}_{RUN_TAG}"
            runs.append({
                **bits, **_SUBSTRATE,
                "model_id": tagged,
                "seed": seed,
                "diet": "multitask",
                "supervision": supervision,
                "run_id": build_run_id(tagged, seed, supervision),
            })
    return runs


def report_collisions(runs: list[dict], manifest: Path = MANIFEST) -> list[str]:
    """Run_ids in `runs` that already exist on a manifest row *not* produced
    by a prior invocation of this same arm (i.e. not diet=='multitask'):
    a genuine collision, not a resumable row of our own."""
    if not manifest.exists():
        return []
    by_id: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("run_id"):
            by_id[rec["run_id"]] = rec  # last row wins
    wanted = {r["run_id"] for r in runs}
    return sorted(rid for rid in wanted if rid in by_id and by_id[rid].get("diet") != "multitask")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds (0..N-1)")
    ap.add_argument("--models", default=",".join(SELECTED_MODELS),
                     help=f"comma-separated model ids to train (default: {','.join(SELECTED_MODELS)})")
    ap.add_argument("--supervision", type=str, required=True, choices=["SUP", "RL"],
                     help="training signal, required with no default -- the run dict carries no signal "
                          "of its own, so an omitted flag would fall through to config.yaml's legacy value")
    ap.add_argument("--tier", type=str, default="full", choices=["smoke", "dev", "full"])
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    runs = enumerate_runs(range(args.seeds), models, args.supervision)
    run_ids = [r["run_id"] for r in runs]
    collisions = report_collisions(runs)

    print(f"[run_multitask_from_init] enumerated {len(runs)} runs, {len(set(run_ids))} unique run_ids, "
          f"{len(collisions)} colliding with a non-multitask manifest row", file=sys.stderr, flush=True)
    if collisions:
        raise SystemExit(f"[run_multitask_from_init] refusing to proceed: run_id(s) already used by another "
                          f"arm: {collisions}")

    if not args.execute:
        print(json.dumps(runs, indent=2))
        return 0

    RESULTS.mkdir(exist_ok=True)
    completed = load_completed(MANIFEST, args.tier)
    pending = [r for r in runs if r["run_id"] not in completed]
    commit = git_commit()

    full_cfg = load_config(args.config)
    tier_cfg = dict(full_cfg.get("tiers", {}).get(args.tier, {}))
    cfg = {"steps": tier_cfg.get("steps", 20000)}
    gates_cfg = full_cfg.get("gates") or {}
    if args.tier == "full" and gates_cfg.get("max_steps"):
        cfg["steps"] = int(gates_cfg["max_steps"])
        if gates_cfg.get("max_steps_if_criterion_unmet") is not None:
            cfg["max_steps_if_criterion_unmet"] = int(gates_cfg["max_steps_if_criterion_unmet"])

    import yaml

    resolved_cfg = build_resolved_config(full_cfg, {**tier_cfg, **cfg}, args.tier,
                                         model_overrides=_SUBSTRATE,
                                         run={"supervision": args.supervision, "diet": "multitask", **_SUBSTRATE})
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path(f"multitask_from_init_{args.supervision}").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    print(f"[run_multitask_from_init] tier={args.tier} supervision={args.supervision} steps={cfg['steps']} "
          f"models={list(models)} runs={len(runs)} already_completed={len(runs) - len(pending)} "
          f"config_hash={cfg_hash[:12]}", flush=True)

    if not pending:
        print(f"[run_multitask_from_init] nothing left: all {len(runs)} runs already completed.", flush=True)
        write_report(MANIFEST, REPORT, 0.0, 0.0)
        return 0

    train_one = resolve_train_fn(force_scaffold=False)
    t0 = time.time()
    for run in pending:
        if _STOP:
            print("[run_multitask_from_init] stop requested; exiting loop.", flush=True)
            break
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
        print(f"[run_multitask_from_init] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_multitask_from_init] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"completed {run['run_id']}", flush=True)

    write_report(MANIFEST, REPORT, 0.0, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
