#!/usr/bin/env python3
"""Orchestrator for the 2x2x2 factorial training grid.

Enumerates the eight factorial cells (M000-M111) crossed with a configurable
number of random seeds and executes `brainalign_wm.training.train.train_one`
for each (cell, seed) pair. Design properties:

  * seed-major run ordering, so a complete pass over all eight cells is
    produced before any seed's replicate count is incremented (breadth
    before depth);
  * resumable: completed runs recorded in `results/manifest.jsonl` are
    skipped on subsequent invocations, and `train_one` itself resumes each
    individual run from its last checkpoint (see `training/train.py`);
  * isolated per-run execution: an exception or failed gate in one run is
    recorded and does not halt the remaining runs;
  * enforces a wall-clock budget and terminates cleanly on SIGINT/SIGTERM;
  * writes `results/manifest.jsonl` and a summary report, `RUN_REPORT.md`.

A `--scaffold` mode substitutes a synthetic stub for `train_one`, allowing
the orchestration logic to be exercised without the model/training
dependencies installed.

Usage:
  python run_grid.py --seeds 8 --budget 48h            # execute the training grid
  python run_grid.py --scaffold --seeds 3 --budget 30m # orchestration-only demonstration
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.jsonl"
REPORT = ROOT / "RUN_REPORT.md"

# The eight factorial cells, ordered by (S, M, L) bits -> M<S><M><L>.
CELLS = [
    {"model_id": f"M{s}{m}{l}", "S": s, "M": m, "L": l}
    for s in (0, 1) for m in (0, 1) for l in (0, 1)
]

_STOP = False


def _handle_signal(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"\n[run_grid] caught signal {signum}; will stop after the current run.", flush=True)


# ----------------------------- pure helpers (unit-tested) -----------------------------

def parse_budget(s: str) -> float:
    """'10h' / '30m' / '600s' / '600' -> seconds (float)."""
    s = str(s).strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s)


def enumerate_runs(seeds: list[int]) -> list[dict]:
    """Seed-major ordering => all 8 cells at seed0, then seed1, ... (breadth-first)."""
    runs = []
    for seed in seeds:
        for cell in CELLS:
            runs.append({**cell, "seed": seed, "run_id": f"{cell['model_id']}_s{seed}"})
    return runs


def load_completed(manifest: Path) -> set[str]:
    done: set[str] = set()
    if not manifest.exists():
        return done
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "completed":
            done.add(rec["run_id"])
    return done


def load_all_records(manifest: Path) -> list[dict]:
    recs = []
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    # keep the last record per run_id
    latest: dict[str, dict] = {}
    for r in recs:
        latest[r.get("run_id", "?")] = r
    return list(latest.values())


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def write_report(manifest: Path, report: Path, budget_s: float, elapsed_s: float) -> None:
    recs = load_all_records(manifest)
    by_status: dict[str, int] = {}
    for r in recs:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    lines = [
        "# Training Grid Report",
        "",
        f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} | "
        f"git {git_commit()} | elapsed {elapsed_s/3600:.2f}h / budget {budget_s/3600:.2f}h_",
        "",
        f"**Runs on record:** {len(recs)}  |  " + ", ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        "",
        "## Per-run",
        "",
        "| run_id | S | M | L | status | gates | acc(load1/2/3) | rung | wall(s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(recs, key=lambda x: x.get("run_id", "")):
        g = r.get("gates", {})
        gates = ",".join(f"{k}:{'P' if v else 'F'}" for k, v in g.items()) or "-"
        acc = r.get("accuracy", {})
        accs = "/".join(str(acc.get(f"load{i}", "-")) for i in (1, 2, 3))
        lines.append(
            f"| {r.get('run_id','?')} | {r.get('S','-')} | {r.get('M','-')} | {r.get('L','-')} | "
            f"{r.get('status','?')} | {gates} | {accs} | {r.get('rung','-')} | {r.get('wall_clock_s','-')} |"
        )
    errs = [r for r in recs if r.get("status") in ("error", "failed")]
    if errs:
        lines += ["", "## Errors / failed gates", ""]
        for r in errs:
            lines.append(f"- **{r.get('run_id')}**: {r.get('status')} — {str(r.get('error',''))[:300]}")
    remaining = [r for r in recs if r.get("status") != "completed"]
    lines += [
        "",
        "## Resume",
        "",
        "Re-run `make run-grid` (or `python run_grid.py ...`) to continue; completed runs are skipped.",
        f"Non-completed on record: {len(remaining)}.",
        "",
    ]
    report.write_text("\n".join(lines))
    print(f"[run_grid] wrote {report}", flush=True)


# ----------------------------- training entrypoint resolution -----------------------------

def _scaffold_train_one(run: dict, cfg: dict) -> dict:
    """Synthetic stand-in for `train_one`, used by `--scaffold` to exercise
    the orchestration logic in isolation from the model and training
    dependencies. Deterministically fakes gates and accuracy from the seed
    and writes a placeholder checkpoint file.
    """
    rng = random.Random(hash((run["run_id"],)) & 0xFFFFFFFF)
    ckpt_dir = RESULTS / "checkpoints" / run["run_id"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "ckpt.json").write_text(json.dumps({"step": cfg.get("steps", 100), **run}))
    time.sleep(cfg.get("scaffold_sleep_s", 0.05))
    # local-learning (L=1) is the risky arm; fake a lower pass-rate for it
    base = 0.98 - (0.15 if run["L"] == 1 else 0.0)
    acc = {f"load{i}": round(max(0.0, base - 0.06 * (i - 1) + rng.uniform(-0.03, 0.03)), 3) for i in (1, 2, 3)}
    gates = {"load1>0.95": acc["load1"] > 0.95, "load3>0.80": acc["load3"] > 0.80}
    rung = 1 if run["L"] == 1 else 0
    return {"status": "completed", "gates": gates, "accuracy": acc, "rung": rung}


def resolve_train_fn(force_scaffold: bool):
    if force_scaffold:
        print("[run_grid] --scaffold: using synthetic stub train_one.", flush=True)
        return _scaffold_train_one
    try:
        from brainalign_wm.training.train import train_one  # type: ignore
        return train_one
    except Exception as e:  # not implemented / not installed yet
        print(f"[run_grid] real train_one unavailable ({e!r}); falling back to scaffold stub.\n"
              f"           Implement brainalign_wm.training.train.train_one, or pass --scaffold.",
              flush=True)
        return _scaffold_train_one


# ----------------------------- main loop -----------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8, help="number of seeds (0..N-1); breadth-first")
    ap.add_argument("--budget", type=str, default="10h", help="wall-clock budget, e.g. 10h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--tier", type=str, default="full", choices=["smoke", "dev", "full"])
    ap.add_argument("--scaffold", action="store_true", help="force the synthetic stub (no deps)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    runs = enumerate_runs(seeds)
    completed = load_completed(MANIFEST)
    commit = git_commit()

    # Minimal config load (YAML optional; scaffold defaults if unavailable).
    cfg = {"steps": 100, "scaffold_sleep_s": 0.05, "tier": args.tier}
    try:
        import yaml  # type: ignore
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        cfg.update(loaded.get("tiers", {}).get(args.tier, {}))
    except Exception:
        pass  # scaffold runs fine without a config

    train_one = resolve_train_fn(args.scaffold)

    print(f"[run_grid] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={len(completed)}", flush=True)

    t0 = time.time()
    for run in runs:
        if _STOP:
            print("[run_grid] stop requested; exiting loop.", flush=True)
            break
        if run["run_id"] in completed:
            continue
        elapsed = time.time() - t0
        if elapsed >= budget_s:
            print(f"[run_grid] budget reached ({elapsed/3600:.2f}h); stopping.", flush=True)
            break

        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        r0 = time.time()
        rec = {**run, "config_hash": abs(hash(json.dumps(cfg, sort_keys=True))) % (10**8),
               "git": commit, "tier": args.tier, "started": started}
        print(f"[run_grid] >>> {run['run_id']}", flush=True)
        try:
            result = train_one(run, cfg)  # isolated
            rec.update(result)
            rec.setdefault("status", "completed")
        except Exception as e:  # noqa: BLE001 — isolation is the point
            rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
            print(f"[run_grid] !!! {run['run_id']} errored: {rec['error']}", flush=True)
        rec["wall_clock_s"] = round(time.time() - r0, 1)
        rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with MANIFEST.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
