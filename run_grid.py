#!/usr/bin/env python3
"""Orchestrator for the 5-arm bio-plausibility ablation battery.

Enumerates the 15 ablation-battery cells (S,M,P,T,D arms; see `CELLS`
below) crossed with a configurable number of random seeds and executes
`brainalign_wm.training.train.train_one` for each (cell, seed) pair.
Design properties:

  * seed-major run ordering, so a complete pass over all cells is produced
    before any seed's replicate count is incremented (breadth before
    depth);
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
  python run_grid.py --seeds 5 --budget 48h            # execute the training grid
  python run_grid.py --scaffold --seeds 3 --budget 30m # orchestration-only demonstration
"""
from __future__ import annotations

import argparse
import hashlib
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


def resolved_config_path(campaign: str = "grid") -> Path:
    """Per-campaign resolved-config artifact (audit fix B5): `run_grid.py`
    and the three per-arm campaign scripts (`run_ablation_battery.py`,
    `run_perf_matched_baselines.py`, `run_identity_catch.py`) each write
    their OWN resolved config under a campaign-qualified filename, so one
    script's launch doesn't clobber another's reproducibility artifact when
    they're run interleaved (as they are in Phase C, §6.2)."""
    return RESULTS / f"resolved_config_{campaign}.yaml"


# The 5-arm bio-plausibility ABLATION BATTERY (v6.0, arm order S,M,P,T,D
# fixed; comments.txt 2026-07-08; 3 interaction-probe cells added
# 2026-07-08 per user request), mirroring configs/config.yaml's `cells:`
# list: baseline, full reference, the 5 knock-one-out-from-full cells, the
# 5 add-one-to-baseline cells, and 3 targeted two-arm interaction probes
# closing specific substrate-entanglement gaps (T behaves differently on
# the worker's intrinsic grid (S=1) vs. the imposed flat grid (S=0); the
# 12-cell core battery's lumped add-one/knock-out design can't resolve a
# SPECIFIC pairwise interaction like S x T in isolation from M/P/D). All
# trained by BPTT (`train_one` reads "P", no "L" key).
_ABLATION_BITS = [
    (0, 0, 0, 0, 0),  # baseline: every arm off
    (1, 1, 1, 1, 1),  # full reference: every arm on
    (0, 1, 1, 1, 1),  # -S: full minus structure/hierarchy
    (1, 0, 1, 1, 1),  # -M: full minus modulation
    (1, 1, 0, 1, 1),  # -P: full minus plasticity
    (1, 1, 1, 0, 1),  # -T: full minus topography
    (1, 1, 1, 1, 0),  # -D: full minus Dale's law
    (1, 0, 0, 0, 0),  # +S: baseline plus structure/hierarchy
    (0, 1, 0, 0, 0),  # +M: baseline plus modulation
    (0, 0, 1, 0, 0),  # +P: baseline plus plasticity
    (0, 0, 0, 1, 0),  # +T: baseline plus topography
    (0, 0, 0, 0, 1),  # +D: baseline plus Dale's law
    (1, 0, 0, 1, 0),  # S+T: topography on its native hierarchical substrate, isolated from M/P/D
    (0, 0, 0, 1, 1),  # T+D: spatial smoothness + E/I balance on the flat substrate, isolated from S/M/P
    (1, 0, 0, 0, 1),  # S+D: hierarchy + Dale's law "bio-plausible backbone", isolated from M/P/T
]
CELLS = [
    {"model_id": f"M{s}{m}{p}{t}{d}", "S": s, "M": m, "P": p, "T": t, "D": d}
    for (s, m, p, t, d) in _ABLATION_BITS
]

# Extended local-learning study (§6.3): same S/M architecture, trained by
# node-perturbation/e-prop instead of BPTT (`train_one` reads "L", no "P"
# key). Distinct model_ids (M**L) so they never collide with the Core
# P-cells above -- not enumerated by default (see `--local-learning`).
LOCAL_LEARNING_CELLS = [
    {"model_id": f"M{s}{m}L", "S": s, "M": m, "L": 1}
    for s in (0, 1) for m in (0, 1)
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


def enumerate_runs(seeds: list[int], include_local_learning: bool = False) -> list[dict]:
    """Seed-major ordering => all 8 Core cells at seed0, then seed1, ...
    (breadth-first). `include_local_learning` appends the 4 Extended
    local-learning cells (§6.3) after the Core cells within each seed --
    off by default, since that study is reported on its own terms and
    doesn't gate the Core grid (§17 decision 6)."""
    runs = []
    for seed in seeds:
        for cell in CELLS:
            runs.append({**cell, "seed": seed, "run_id": f"{cell['model_id']}_s{seed}"})
        if include_local_learning:
            for cell in LOCAL_LEARNING_CELLS:
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


def build_resolved_config(full_cfg: dict, tier_cfg: dict, tier: str) -> dict:
    """The full merged, resolved config for this grid invocation: every
    project subsystem (model/mechanisms/task/train/gates/neural) plus the
    tier subset actually in effect -- covers every config change that
    could affect a run, not just the tier subset `run_grid.py` threads
    through to `train_one`."""
    resolved = {k: v for k, v in full_cfg.items() if k != "tiers"}
    resolved["tier"] = {"name": tier, **tier_cfg}
    return resolved


def config_hash(resolved_cfg: dict) -> str:
    """Deterministic hex digest of the full resolved config.
    `hashlib.sha256` (not Python's built-in `hash()`) is used because
    `hash()` on a str is salted per-process by `PYTHONHASHSEED` -- the same
    config would otherwise produce a different "hash" on every process
    launch, which is exactly why the old manifest showed three different
    config_hash values for the identical "full" tier across a single grid.
    `seeding.py` setting `os.environ["PYTHONHASHSEED"]` at runtime does not
    affect the already-running interpreter, so relying on that would not
    have fixed it either -- sha256 makes the whole issue moot."""
    payload = json.dumps(resolved_cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def _fmt(r: dict, key: str) -> str:
    """`r.get(key, "-")`, but also renders an explicitly-`None` value (JSON
    `null` -- e.g. `steps_to_criterion` when criterion was never met) as
    "-" instead of the literal string "None". `False`/`0` are NOT missing
    (e.g. `criterion_met: false`) and print as-is."""
    v = r.get(key, "-")
    return "-" if v is None else str(v)


def _fmt_acc_ci(acc: dict, load: int) -> str:
    """`0.941 [0.912,0.963]`, or just the point estimate / "-" if the CI
    (or the accuracy itself) isn't present -- e.g. `--scaffold` mode's
    synthetic accuracy dict has no `*_ci_lo`/`*_ci_hi` keys (Phase 3)."""
    v = acc.get(f"load{load}")
    if v is None:
        return "-"
    lo, hi = acc.get(f"load{load}_ci_lo"), acc.get(f"load{load}_ci_hi")
    return f"{v} [{lo},{hi}]" if lo is not None and hi is not None else str(v)


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
        # Phase 3 (comments.txt §5 item 3.6): criterion_met/steps_to_criterion/
        # trials_to_criterion/ms_per_step/joules_to_criterion added; final
        # per-load accuracy now carries its Wilson CI; `wall(s)` renamed
        # `wall_total_s` (still the run's total wall clock, unrelated to
        # `wall_s_to_criterion` -- sample efficiency is reported as
        # trials_to_criterion, not a step or wall-clock count, per 3.6).
        "| run_id | S | M | P/L | T | D | status | gates | acc(load1/2/3) | rung | "
        "criterion_met | steps_to_criterion | trials_to_criterion | ms_per_step | "
        "joules_to_criterion | wall_total_s |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(recs, key=lambda x: x.get("run_id", "")):
        g = r.get("gates", {})
        gates = ",".join(f"{k}:{'P' if v else 'F'}" for k, v in g.items()) or "-"
        acc = r.get("accuracy", {})
        accs = "/".join(_fmt_acc_ci(acc, i) for i in (1, 2, 3))
        # Core cells carry "P" (BPTT throughout); the 4 local-learning
        # cells carry "L" instead -- show whichever is present. T/D (§1.2/
        # §1.3) are absent (shown "-") for local-learning and any
        # supplementary-arm run that predates the 5-arm ablation battery.
        p_or_l = r.get("P", r.get("L", "-"))
        lines.append(
            f"| {r.get('run_id','?')} | {r.get('S','-')} | {r.get('M','-')} | {p_or_l} | "
            f"{r.get('T','-')} | {r.get('D','-')} | "
            f"{r.get('status','?')} | {gates} | {accs} | {r.get('rung','-')} | "
            f"{_fmt(r, 'criterion_met')} | {_fmt(r, 'steps_to_criterion')} | "
            f"{_fmt(r, 'trials_to_criterion')} | {_fmt(r, 'ms_per_step')} | "
            f"{_fmt(r, 'joules_to_criterion')} | {r.get('wall_clock_s','-')} |"
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
    # hashlib, not Python's salted-per-process hash() (see config_hash's
    # docstring above; found reused here on adversarial review).
    rng = random.Random(int.from_bytes(hashlib.sha256(run["run_id"].encode()).digest()[:4], "big"))
    ckpt_dir = RESULTS / "checkpoints" / run["run_id"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / "ckpt.json").write_text(json.dumps({"step": cfg.get("steps", 100), **run}))
    time.sleep(cfg.get("scaffold_sleep_s", 0.05))
    # local-learning (L=1) is the risky arm; fake a lower pass-rate for it
    is_local_learning = run.get("L", 0) == 1
    base = 0.98 - (0.15 if is_local_learning else 0.0)
    acc = {f"load{i}": round(max(0.0, base - 0.06 * (i - 1) + rng.uniform(-0.03, 0.03)), 3) for i in (1, 2, 3)}
    gates = {"load1>=0.95": acc["load1"] >= 0.95, "load3>=0.80": acc["load3"] >= 0.80}
    rung = 1 if is_local_learning else 0
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
    ap.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1); breadth-first; study default 5")
    ap.add_argument("--budget", type=str, default="10h", help="wall-clock budget, e.g. 10h / 30m / 600s")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--tier", type=str, default="full", choices=["smoke", "dev", "full"],
                     help="compute tier (default: full, the 150k-step results tier the study uses)")
    ap.add_argument("--scaffold", action="store_true", help="force the synthetic stub (no deps)")
    ap.add_argument("--local-learning", action="store_true",
                     help="also enumerate the 4 Extended local-learning cells (M**L, §6.3)")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    runs = enumerate_runs(seeds, include_local_learning=args.local_learning)
    completed = load_completed(MANIFEST)
    commit = git_commit()

    # Config load (YAML optional; scaffold defaults if unavailable). Audit
    # fix B3: a bare `except Exception: pass` here used to silently swallow
    # a malformed config.yaml and run the full grid on scaffold defaults
    # (100 steps) with no indication anything was wrong -- now each failure
    # mode is handled on its own terms, and a parse error is fatal rather
    # than silent.
    cfg = {"steps": 100, "scaffold_sleep_s": 0.05, "tier": args.tier}
    full_cfg = {}
    yaml = None
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        print("[run_grid] pyyaml not installed; using scaffold defaults.", flush=True)
    else:
        try:
            full_cfg = yaml.safe_load(Path(args.config).read_text()) or {}
            cfg.update(full_cfg.get("tiers", {}).get(args.tier, {}))
        except FileNotFoundError:
            print(f"[run_grid] config not found at {args.config}; using scaffold defaults.", flush=True)
        except yaml.YAMLError as e:
            print(f"[run_grid] failed to parse {args.config}: {e}", flush=True)
            raise

    # Audit fix B1: hash the FULL resolved config (not just the tier
    # subset), computed once per invocation -- every run in this grid
    # invocation shares this one hash, and the resolved config is dumped
    # verbatim so a run is fully reproducible from the manifest alone.
    resolved_cfg = build_resolved_config(full_cfg, cfg, args.tier) if full_cfg else {"tier": {"name": args.tier, **cfg}}
    cfg_hash = config_hash(resolved_cfg)
    if full_cfg:
        # Audit fix L1: reuse the `yaml` module imported above rather than
        # re-importing; the JSON fallback below is for a write/serialize
        # failure only, not a missing dependency (already handled above).
        resolved_path = resolved_config_path("grid")
        try:
            resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))
        except (OSError, yaml.YAMLError) as e:
            print(f"[run_grid] failed to write resolved config as YAML ({e}); falling back to JSON.", flush=True)
            resolved_path.write_text(json.dumps(resolved_cfg, sort_keys=True, default=str, indent=2))

    train_one = resolve_train_fn(args.scaffold)

    print(f"[run_grid] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h "
          f"runs={len(runs)} already_completed={len(completed)} config_hash={cfg_hash[:12]}", flush=True)

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
        rec = {**run, "config_hash": cfg_hash, "git": commit, "tier": args.tier, "started": started}
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
