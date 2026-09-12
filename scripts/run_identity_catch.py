#!/usr/bin/env python3
"""Identity-report sub-experiment: train selected architectures at
`task.identity_catch_fraction > 0`, where a fraction of trials replace the
match/non-match probe with a query for the category held at serial
position 1.

The comparison it supports is each cell's maintenance-epoch alignment WITH
vs. WITHOUT the delayed identity report, at comparable match/non-match
accuracy. It separates "the recognition objective never needed accessible
item identity" from "the network cannot preserve it".

Separate CLI from `run_grid.py` because `identity_catch_fraction` is a
per-run override (`train_one`'s `run` dict), not a global config.yaml edit.

Run_id convention, consumed by `training/generate_activity_logs.py::
_run_id_extras` and the variant guard in `analysis/run_all.py::
_is_ablation_or_catch_variant`: `M{SMPTD}_idcatch_{supervision}_s{seed}`
(the manifest `model_id` stays `M{SMPTD}_idcatch`, unsuffixed by signal;
`run_grid.build_run_id` adds the signal to the run_id only, so the two
training-signal passes never collide on checkpoint directory, metrics
CSV, or manifest key).

Usage:
  python scripts/run_identity_catch.py --supervision SUP --seeds 4 --budget 10h
  python scripts/run_identity_catch.py --supervision SUP --cells M00000,M10010 \
      --seeds 8 --tier full --budget 40h
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
    build_resolved_config, build_run_id, config_hash, git_commit, load_completed, parse_budget, resolve_train_fn,
    resolved_config_path, write_report,
)

DEFAULT_CELLS = ["M00000", "M11111"]
IDENTITY_CATCH_FRACTION = 0.12
RUN_ID_SUFFIX = "idcatch"

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_identity_catch] caught signal {signum}; will stop after the current run.", flush=True)


def parse_cells(spec: str) -> list[dict]:
    """'M00000,M10010' -> one cell dict per id, with the S/M/P/T/D bits the
    id encodes. Raises on anything that is not M followed by five bits, so a
    typo cannot silently enumerate a different architecture."""
    cells = []
    for raw in spec.split(","):
        model_id = raw.strip()
        if len(model_id) != 6 or model_id[0] != "M" or any(c not in "01" for c in model_id[1:]):
            raise ValueError(f"cell {model_id!r} is not of the form M<5 binary digits>, e.g. M10010")
        cells.append({"model_id": model_id, **{k: int(v) for k, v in zip("SMPTD", model_id[1:])}})
    return cells


def enumerate_runs(seeds: list[int], cells: list[dict], supervision: str) -> list[dict]:
    """Seed-major ordering (breadth before depth), same convention as
    `run_grid.enumerate_runs`. `model_id` stays unsuffixed by signal so the
    two signal passes are recognized as the same architecture; `run_id`
    gains the signal via `run_grid.build_run_id`, the same namespacing the
    main battery uses, so the two passes never collide on checkpoint
    directory, metrics CSV, or manifest key."""
    runs = []
    for seed in seeds:
        for cell in cells:
            model_id = f"{cell['model_id']}_{RUN_ID_SUFFIX}"
            runs.append({
                **{k: cell[k] for k in "SMPTD"},
                "model_id": model_id,
                "seed": seed,
                "run_id": build_run_id(model_id, seed, supervision),
                "supervision": supervision,
                "identity_catch_fraction": IDENTITY_CATCH_FRACTION,
            })
    return runs


def recorded_supervision(manifest: Path) -> dict[str, str]:
    """run_id -> the training signal its last manifest row recorded. The
    run_id now carries the signal (`build_run_id`), so two signal passes
    never share a run_id; this is a manifest-consistency check for a run_id
    whose recorded `supervision` field disagrees with what its own id
    encodes -- `main` refuses to resume such a row rather than silently
    continuing it under the wrong label."""
    out: dict[str, str] = {}
    if not manifest.exists():
        return out
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("run_id") and rec.get("supervision"):
            out[rec["run_id"]] = rec["supervision"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=4, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--cells", type=str, default=",".join(DEFAULT_CELLS),
                    help=f"comma-separated model ids to train (default: {','.join(DEFAULT_CELLS)})")
    ap.add_argument("--supervision", type=str, required=True, choices=["SUP", "RL"],
                    help="training signal, required with no default: the run dict carries no signal of its "
                         "own, so an omitted flag would fall through to config.yaml's legacy value and "
                         "produce runs that are not comparable with either campaign arm")
    ap.add_argument("--budget", type=str, default="10h", help="wall-clock budget, e.g. 10h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--tier", type=str, default="dev", choices=["smoke", "dev", "full"],
                     help="compute tier (default: dev, a sanity-check tier -- the full study uses --tier full)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    runs = enumerate_runs(seeds, parse_cells(args.cells), args.supervision)
    completed = load_completed(MANIFEST, args.tier)
    prior_supervision = recorded_supervision(MANIFEST)
    clashes = [r["run_id"] for r in runs
               if prior_supervision.get(r["run_id"], args.supervision) != args.supervision]
    if clashes:
        raise SystemExit(
            f"[run_identity_catch] {len(clashes)} run id(s) already trained under a different signal "
            f"(e.g. {clashes[0]}: {prior_supervision[clashes[0]]} on record, {args.supervision} requested). "
            "Resuming would continue those weights under the new signal and mislabel the result."
        )
    commit = git_commit()

    import yaml

    full_cfg = load_config(args.config)
    tier_cfg = dict(full_cfg.get("tiers", {}).get(args.tier, {}))
    cfg = {"steps": tier_cfg.get("steps", 20000)}
    # Same analysis budget as the main campaign, so these runs are
    # comparable with the equal-duration checkpoints already on disk.
    gates_cfg = full_cfg.get("gates") or {}
    if args.tier == "full" and gates_cfg.get("max_steps"):
        cfg["steps"] = int(gates_cfg["max_steps"])
        if gates_cfg.get("max_steps_if_criterion_unmet") is not None:
            cfg["max_steps_if_criterion_unmet"] = int(gates_cfg["max_steps_if_criterion_unmet"])
        print(f"[run_identity_catch] analysis budget: steps={cfg['steps']} from gates.max_steps; "
              f"unmet-criterion ceiling={cfg.get('max_steps_if_criterion_unmet', cfg['steps'])}.", flush=True)
    resolved_cfg = build_resolved_config(full_cfg, {**tier_cfg, **cfg}, args.tier,
                                     run={"supervision": args.supervision})
    # The catch fraction is a per-run override, so the resolved config would
    # otherwise record the config file's 0.0 for runs that trained at 0.12.
    resolved_cfg["task"] = {**resolved_cfg["task"], "identity_catch_fraction": IDENTITY_CATCH_FRACTION}
    cfg_hash = config_hash(resolved_cfg)
    resolved_config_path(f"identity_{args.supervision}").write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_identity_catch] tier={args.tier} supervision={args.supervision} seeds={seeds} "
          f"cells={[c['model_id'] for c in parse_cells(args.cells)]} budget={budget_s/3600:.2f}h "
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
