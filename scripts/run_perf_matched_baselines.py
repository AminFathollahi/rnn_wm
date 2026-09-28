#!/usr/bin/env python3
"""Train capacity- and regularization-matched flat-GRU controls."""
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
    build_resolved_config, build_run_id, config_hash, git_commit, load_completed, parse_budget, resolve_train_fn,
    resolved_config_path, write_report,
)

ARMS = [
    {"suffix": "2x"},
    {"suffix": "l1", "l1_weight_penalty": 0.001},
    {"suffix": "dropout", "core_dropout_p": 0.1},
]

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_perf_matched_baselines] caught signal {signum}; will stop after the current run.", flush=True)


def enumerate_runs(seeds: list[int], base_flat_units: int, supervision: str) -> list[dict]:
    runs = []
    for seed in seeds:
        for arm in ARMS:
            extra = {k: v for k, v in arm.items() if k != "suffix"}
            if arm["suffix"] == "2x":
                extra["flat_units"] = base_flat_units * 2
            model_id = f"M00000_{arm['suffix']}"
            runs.append({
                "model_id": model_id, "S": 0, "M": 0, "P": 0, "T": 0, "D": 0,
                "seed": seed, "run_id": build_run_id(model_id, seed, supervision),
                "supervision": supervision, **extra,
            })
    return runs


def resolved_configs(full_cfg: dict, tier: str, tier_cfg: dict, runs: list[dict]) -> dict[str, dict]:
    configs = {}
    for run in runs:
        model_overrides = {key: run[key] for key in ("flat_units", "core_dropout_p") if key in run}
        resolved = build_resolved_config(full_cfg, tier_cfg, tier, model_overrides=model_overrides, run=run)
        if "l1_weight_penalty" in run:
            resolved["train"] = {**resolved["train"], "l1_weight_penalty": run["l1_weight_penalty"]}
        configs[run["model_id"]] = resolved
    return configs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="8h", help="wall-clock budget, e.g. 8h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--supervision", required=True, choices=["SUP", "RL"], help="training signal")
    ap.add_argument("--tier", type=str, default="dev", choices=["smoke", "dev", "full"],
                     help="compute tier (default: dev, a sanity-check tier -- the full study uses --tier full)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    completed = load_completed(MANIFEST, args.tier)
    commit = git_commit()

    import yaml

    full_cfg = load_config(args.config)
    base_flat_units = int(full_cfg.get("model", {}).get("flat_units", 256))
    runs = enumerate_runs(seeds, base_flat_units, args.supervision)
    tier_cfg = dict(full_cfg.get("tiers", {}).get(args.tier, {}))
    cfg = {"steps": tier_cfg.get("steps", 20000)}
    gates = full_cfg.get("gates") or {}
    if args.tier == "full" and gates.get("max_steps"):
        cfg["steps"] = int(gates["max_steps"])
        if gates.get("max_steps_if_criterion_unmet") is not None:
            cfg["max_steps_if_criterion_unmet"] = int(gates["max_steps_if_criterion_unmet"])
    resolved = resolved_configs(full_cfg, args.tier, {**tier_cfg, **cfg}, runs)
    hashes = {model_id: config_hash(value) for model_id, value in resolved.items()}
    for model_id, value in resolved.items():
        arm = model_id.removeprefix("M00000_")
        resolved_config_path(f"perf_{args.supervision}_{arm}").write_text(yaml.safe_dump(value, sort_keys=True))

    train_one = resolve_train_fn(force_scaffold=False)
    n_already = sum(1 for r in runs if r["run_id"] in completed)
    print(f"[run_perf_matched_baselines] tier={args.tier} supervision={args.supervision} seeds={seeds} "
          f"budget={budget_s/3600:.2f}h runs={len(runs)} already_completed={n_already}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_perf_matched_baselines] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_perf_matched_baselines] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": hashes[run["model_id"]], "git": commit, "tier": args.tier, "started": started}
        print(f"[run_perf_matched_baselines] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated, same as run_grid.py
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 -- isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_perf_matched_baselines] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
