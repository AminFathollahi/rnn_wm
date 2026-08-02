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
  * writes `results/manifest.jsonl` and a summary report,
    `results/RUN_REPORT.md`;
  * `--workers N` (§12.3): runs N `train_one` calls concurrently via
    `ProcessPoolExecutor` (`run_grid_loop`, shared with
    `scripts/run_stage1_grid.py`). N=8 measured best on this machine
    (Appendix A / executor.md's Phase 12.3 entry: largest N keeping
    per-process ms/step under 1.5x the N=1 value); every manifest row
    records the `workers` value in force so wall-clock/energy DVs are
    never compared across rows with different N.

A `--scaffold` mode substitutes a synthetic stub for `train_one`, allowing
the orchestration logic to be exercised without the model/training
dependencies installed.

Usage:
  python run_grid.py --seeds 5 --budget 48h --workers 8 --supervision RL # execute the training grid, 8-way concurrent
  python run_grid.py --scaffold --seeds 3 --budget 30m --supervision RL  # orchestration-only demonstration

`--supervision {SUP,RL}` is required, with no default (comments.txt §16
item 16.4): `CELLS` carry no `supervision` key, so an omitted flag would
silently fall through to config.yaml's `train.supervision: legacy`, which
is not one of the study's two preregistered levels.
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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.jsonl"
REPORT = RESULTS / "RUN_REPORT.md"


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
# `substrate` and `recurrent_init_spectral_radius` are stated on every cell
# rather than left to inherit `configs/config.yaml`'s `model.*` defaults.
# `train_one` reads both from the run dict when present (`train.py:1567,1571`),
# and the manifest records the run dict -- so an omitted key is a `null`
# manifest field that says nothing about what actually trained. That exact
# fall-through has invalidated two conclusions here (F1, and D20, where
# `M10000_pilot_s0`'s `substrate: null` hid a positive S=1 GRU result for
# weeks). The values equal today's config defaults; the point is that they are
# recorded per run and survive a later edit to the config.
_SUBSTRATE = {"substrate": "gru", "recurrent_init_spectral_radius": None}

CELLS = [
    {"model_id": f"M{s}{m}{p}{t}{d}", "S": s, "M": m, "P": p, "T": t, "D": d, **_SUBSTRATE}
    for (s, m, p, t, d) in _ABLATION_BITS
]

# Extended local-learning study (§6.3): same S/M architecture, trained by
# node-perturbation/e-prop instead of BPTT (`train_one` reads "L", no "P"
# key). Distinct model_ids (M**L) so they never collide with the Core
# P-cells above -- not enumerated by default (see `--local-learning`).
LOCAL_LEARNING_CELLS = [
    {"model_id": f"M{s}{m}L", "S": s, "M": m, "L": 1, **_SUBSTRATE}
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


def build_run_id(model_id: str, seed: int, supervision: str | None) -> str:
    """`supervision` is part of the run identity (advisor.md D32), because
    the checkpoint directory, the metrics CSV and the manifest key are all
    derived from the run_id. Without it the second supervision pass of the
    same grid either gets skipped wholesale by `load_completed` or -- worse,
    because it is silent and produces a plausible-looking row -- resumes the
    first pass's `ckpt.pt` (`train.py`'s resume path), continuing an
    SUP-trained network under RL and labelling the result `supervision: RL`.

    `None` reproduces the historical id exactly. Existing run directories are
    never renamed; ids already on disk keep the old form."""
    return f"{model_id}_{supervision}_s{seed}" if supervision else f"{model_id}_s{seed}"


def enumerate_runs(
    seeds: list[int], include_local_learning: bool = False, supervision: str | None = None,
    cells: list[str] | None = None,
) -> list[dict]:
    """Seed-major ordering => all 15 Core cells at seed0, then seed1, ...
    (breadth-first). `include_local_learning` appends the 4 Extended
    local-learning cells (§6.3) after the Core cells within each seed --
    off by default, since that study is reported on its own terms and
    doesn't gate the Core grid (§17 decision 6).

    `supervision` (comments.txt §16 item 16.4 / advisor.md D24): every
    enumerated cell's own run dict, not the config default. `CELLS` carry
    no `supervision` key, so without this every battery run used to fall
    through to `config.yaml`'s `train.supervision: legacy` -- a pre-Phase-7
    signal that is not one of the study's preregistered levels ({SUP, RL}).
    `None` (the default) preserves that historical fall-through exactly, so
    any other caller of `enumerate_runs` is unaffected; `main` below never
    passes `None`, because its own `--supervision` flag is required. It is
    also part of the run_id -- see `build_run_id`.

    `cells` (comments.txt §18.5): restrict to these `model_id`s. The campaign
    runs the full 15 under `SUP` but only the 7 S=0 cells under `RL`, plus an
    8-cell S=1 failure arm at a lower seed count, so the grid needs a subset
    filter; a second orchestrator would need its own copy of the resume,
    concurrency and manifest logic. An unknown id raises rather than
    silently enumerating nothing."""
    if cells is not None:
        known = {c["model_id"] for c in CELLS} | {c["model_id"] for c in LOCAL_LEARNING_CELLS}
        unknown = [c for c in cells if c not in known]
        if unknown:
            raise ValueError(f"unknown model_id(s) {unknown}; known: {sorted(known)}")
    selected = [c for c in CELLS if cells is None or c["model_id"] in cells]
    selected_local = [c for c in LOCAL_LEARNING_CELLS if cells is None or c["model_id"] in cells]

    runs = []
    for seed in seeds:
        for cell in selected + (selected_local if include_local_learning else []):
            run = {**cell, "seed": seed, "run_id": build_run_id(cell["model_id"], seed, supervision)}
            if supervision is not None:
                run["supervision"] = supervision
            runs.append(run)
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


def build_resolved_config(
    full_cfg: dict, tier_cfg: dict, tier: str, model_overrides: dict | None = None, run: dict | None = None,
) -> dict:
    """The full merged, resolved config for this grid invocation: every
    project subsystem (model/mechanisms/task/train/gates/neural) plus the
    tier subset actually in effect -- covers every config change that
    could affect a run, not just the tier subset `run_grid.py` threads
    through to `train_one`.

    `model_overrides` (Phase 12 audit finding F1): per-run `model.*` keys
    that `train_one` applies from the run dict (currently `substrate`) are
    invisible here otherwise, so the written audit trail records
    config.yaml's DEFAULT while something else actually trained. That is
    exactly how the Phase 11.1 pilot ran a GRU while
    `resolved_config_phase11_pilot_s0.yaml` said -- correctly about the
    file, wrongly about the run -- `substrate: gru`, and nobody noticed for
    a 24-hour campaign. Pass the override whenever it is constant across
    the runs this resolved config covers; a campaign whose cells DISAGREE
    on a model key must write one resolved config per distinct value, not
    a single misleading one.

    `run` (D20): the same run dict passed to `train_one`. Substrate,
    supervision, and the recurrent-init spectral radius are recorded
    explicitly here using `train_one`'s own fallback order (`run` value if
    present, else the config default), regardless of whether the caller
    remembered to name them in `model_overrides` -- a defect class (F1, and
    the supervision key that silently inherited `legacy` for a full round)
    has now hit two of these three keys because an omission was invisible
    here. The S/M/P/T/D cell selector is recorded too (`model.cell`): it
    picks the model class itself, and two runs differing only in it used to
    resolve to the same config and config_hash. `run=None` (every existing
    caller) reproduces the prior output exactly, reading straight from
    `full_cfg`'s own defaults."""
    resolved = {k: v for k, v in full_cfg.items() if k != "tiers"}
    if model_overrides:
        resolved["model"] = {**resolved.get("model", {}), **model_overrides}
    run = run or {}
    resolved_model = resolved.get("model", {})
    resolved_train = resolved.get("train", {})
    resolved["model"] = {
        **resolved_model,
        "substrate": run.get("substrate", resolved_model.get("substrate", "gru")),
        "recurrent_init_spectral_radius": run.get(
            "recurrent_init_spectral_radius", resolved_model.get("recurrent_init_spectral_radius")
        ),
    }
    resolved["train"] = {
        **resolved_train,
        "supervision": run.get("supervision", resolved_train.get("supervision", "legacy")),
    }
    if run:
        # The S/M/P/T/D cell selector picks the model class itself (flat vs
        # hierarchical core, plasticity, etc.) but was previously absent
        # here, so two runs differing only in structure could resolve to
        # the same config and the same config_hash -- discovered when a
        # flat and a hierarchical vanilla run collided on one hash.
        resolved["model"]["cell"] = {k: run[k] for k in ("S", "M", "P", "T", "D") if k in run}
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
    `null` -- e.g. a milestone never reached) as "-" instead of the literal
    string "None". `False`/`0` are NOT missing (e.g. `matched: false`) and
    print as-is."""
    v = r.get(key, "-")
    return "-" if v is None else str(v)


def _fmt_milestone(r: dict, prefix: str) -> str:
    """Phase 12 (§3.3): milestone keys are named `<prefix>_<key>_<threshold>`
    (e.g. `steps_to_load1_0.83`), and the threshold comes from config.yaml,
    not this file -- so look up by prefix instead of hardcoding the
    threshold here, the same root-cause reason train.py iterates
    `criterion`'s own keys instead of `task_loads`."""
    for k, v in r.items():
        if k.startswith(prefix):
            return "-" if v is None else str(v)
    return "-"


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
        # Phase 12 (§3): the old single graded criterion_met/steps_to_criterion
        # split into Gate A's `matched` (inclusion, §3.1) and per-milestone
        # steps/trials_to_<key>_<threshold> (§3.3, milestones, not a stop
        # rule -- `_fmt_milestone` looks these up by prefix since the
        # threshold suffix comes from config.yaml, not this file). `wall(s)`
        # is `wall_total_s`, the run's total wall clock -- with concurrency
        # (§12.3) it is NOT comparable across rows with different `workers`
        # (its own column here), which is why steps_to_*/trials_to_* (not
        # wall_s_to_*/joules_to_*) are the primary efficiency DVs.
        "| run_id | S | M | P/L | T | D | status | gates | acc(load1/2/3) | rung | "
        "matched | steps_to_load1 | steps_to_load3 | ms_per_step | "
        "wall_total_s | workers |",
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
            f"{_fmt(r, 'matched')} | {_fmt_milestone(r, 'steps_to_load1_')} | "
            f"{_fmt_milestone(r, 'steps_to_load3_')} | {_fmt(r, 'ms_per_step')} | "
            f"{_fmt(r, 'wall_clock_s')} | {_fmt(r, 'workers')} |"
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


# ----------------------------- concurrent execution (Phase 12.3) -----------------------------

def _pool_initializer() -> None:
    """Runs once per worker process, before that process imports torch/numpy
    (those imports happen later, inside `resolve_train_fn`/`train_one`,
    called from `_worker_entry` -- not at module load) -- so this actually
    governs the BLAS/OpenMP thread count each worker's torch ends up using.
    32 CPUs / N worker processes x (OpenMP + Python) oversubscribes at the
    default of one OpenMP thread per core; comments.txt §12.3 fixes it at 2
    per worker."""
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"


def _worker_entry(run: dict, cfg: dict, force_scaffold: bool) -> dict:
    """Module-level (picklable) `ProcessPoolExecutor` entry point (§12.3):
    resolves `train_one` itself, inside the worker process. A closure over
    an already-resolved `train_one` is not picklable across a process
    boundary, so each worker re-resolves it from scratch -- cheap (an
    import, not a load) and the only correct option."""
    train_one = resolve_train_fn(force_scaffold)
    return train_one(run, cfg)


def run_grid_loop(
    runs: list[dict], completed: set, cfg: dict, cfg_hash: str, commit: str, tier: str,
    budget_s: float, t0: float, manifest: Path, force_scaffold: bool, workers: int,
    log_prefix: str, extra_rec_fields: dict | None = None,
) -> None:
    """Shared execution loop for `run_grid.py` and `scripts/run_stage1_grid.py`
    (§12.3): one `train_one` call per OS process (`ProcessPoolExecutor`,
    NOT threads -- the whole point is escaping the GIL that the Python
    overhead the audit measured actually lives in; every run is otherwise
    bit-identical to running it alone, which is why this is a pure
    scheduling change, not a hack). Manifest appends are serialised HERE,
    in the parent, as futures complete -- never from a worker. `completed`
    is read once by the caller before this is invoked, same resume
    semantics as the old sequential loop. `workers=1` still goes through
    the pool (not a separate code path), so its manifest rows are produced
    by exactly the same mechanism as any other N -- the property the
    12.3 test locks in. A worker that raises is caught per-future and
    recorded as a `status: error` row; it does not stop the others or
    poison the pool. SIGINT/SIGTERM/budget: stop SUBMITTING new futures,
    let in-flight ones finish, then return."""
    extra_rec_fields = extra_rec_fields or {}
    pending = [r for r in runs if r["run_id"] not in completed]
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_pool_initializer) as ex:
        in_flight: dict = {}
        next_idx = 0

        def _try_submit() -> None:
            nonlocal next_idx
            while (
                next_idx < len(pending)
                and len(in_flight) < workers
                and not _STOP
                and (time.time() - t0) < budget_s
            ):
                run = pending[next_idx]
                next_idx += 1
                started = datetime.now(timezone.utc).isoformat(timespec="seconds")
                print(f"{log_prefix} >>> {run['run_id']}", flush=True)
                fut = ex.submit(_worker_entry, run, cfg, force_scaffold)
                in_flight[fut] = (run, time.time(), started)

        _try_submit()
        while in_flight:
            done, _ = futures_wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                run, r0, started = in_flight.pop(fut)
                rec = {
                    **run, "config_hash": cfg_hash, "git": commit, "tier": tier,
                    "started": started, "workers": workers, **extra_rec_fields,
                }
                try:
                    result = fut.result()
                    rec.update(result)
                    rec.setdefault("status", "completed")
                except Exception as e:  # noqa: BLE001 -- isolation is the point
                    rec.update({"status": "error", "error": f"{type(e).__name__}: {e}"})
                    print(f"{log_prefix} !!! {run['run_id']} errored: {rec['error']}", flush=True)
                rec["wall_clock_s"] = round(time.time() - r0, 1)
                rec["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with manifest.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            if _STOP:
                print(f"{log_prefix} stop requested; letting {len(in_flight)} in-flight run(s) finish.", flush=True)
            elif (time.time() - t0) >= budget_s:
                print(f"{log_prefix} budget reached; letting {len(in_flight)} in-flight run(s) finish.", flush=True)
            else:
                _try_submit()


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
    ap.add_argument("--cells", type=str, default=None,
                     help="comments.txt §18.5: comma-separated model_ids to restrict the grid to, e.g. "
                          "M00000,M01111. Default: every Core cell. The campaign runs all 15 under SUP "
                          "but only the 7 S=0 cells under RL, plus an 8-cell S=1 failure arm at 2 seeds. "
                          "An unknown model_id is an error, not an empty grid.")
    ap.add_argument("--workers", type=int, default=1,
                     help="§12.3: concurrent training processes (ProcessPoolExecutor, one run per "
                          "process). N=8 measured 5.9x aggregate throughput for a 1.36x per-process "
                          "slowdown (Appendix A, 12.3 RESULT); keep N FIXED for a whole stage since "
                          "wall_s_to_*/joules_to_* are not comparable across rows with different "
                          "`workers` (every manifest row records it).")
    ap.add_argument("--supervision", type=str, required=True, choices=["SUP", "RL"],
                     help="comments.txt §16 item 16.4 / advisor.md D24: the study's two preregistered "
                          "training signals. Required, with no default, so the battery cannot launch "
                          "under an implicit choice -- `CELLS` carry no `supervision` key, so an "
                          "omitted flag used to fall through to config.yaml's `train.supervision: "
                          "legacy`, a pre-Phase-7 hybrid that is not one of this study's levels. "
                          "'legacy' is deliberately not an allowed choice here.")
    args = ap.parse_args(argv)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    RESULTS.mkdir(exist_ok=True)
    budget_s = parse_budget(args.budget)
    seeds = list(range(args.seeds))
    cells = [c.strip() for c in args.cells.split(",") if c.strip()] if args.cells else None
    runs = enumerate_runs(seeds, include_local_learning=args.local_learning,
                          supervision=args.supervision, cells=cells)
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

    # Gate B governs the campaign's run length. `train_one` reads its ceiling
    # from `cfg["steps"]` (`train.py:1540`) and never looks at
    # `gates.max_steps`, so without this the battery would run to the tier's
    # `steps` (150,000 at --tier full) while the config, the manifest and the
    # §12 amendment all say Gate B is 80,000: the wrong equal-duration
    # snapshot for every geometry DV, at 1.9x the authorized budget, silently.
    # Only the campaign launcher resolves this -- the pilots and Stage 1 pass
    # their own explicit `steps` and must keep it.
    # `full` only: `smoke` and `dev` exist to run short, and Gate B is a
    # results-tier commitment, not a global one.
    gate_b = (full_cfg.get("gates") or {}).get("max_steps") if args.tier == "full" else None
    if gate_b:
        cfg["steps"] = int(gate_b)
        print(f"[run_grid] Gate B: steps={cfg['steps']} from gates.max_steps "
              f"(tier '{args.tier}' default was {full_cfg.get('tiers', {}).get(args.tier, {}).get('steps')}).",
              flush=True)

    # Audit fix B1: hash the FULL resolved config (not just the tier
    # subset), computed once per invocation -- every run in this grid
    # invocation shares this one hash, and the resolved config is dumped
    # verbatim so a run is fully reproducible from the manifest alone.
    resolved_cfg = (
        build_resolved_config(full_cfg, cfg, args.tier, model_overrides=dict(_SUBSTRATE),
                              run={"supervision": args.supervision})
        if full_cfg else {"tier": {"name": args.tier, **cfg}}
    )
    cfg_hash = config_hash(resolved_cfg)
    if full_cfg:
        # Audit fix L1: reuse the `yaml` module imported above rather than
        # re-importing; the JSON fallback below is for a write/serialize
        # failure only, not a missing dependency (already handled above).
        # Namespaced by supervision for the same reason the run_id is (D32):
        # the two passes resolve to DIFFERENT configs and different
        # `config_hash`es, and one file cannot be the audit trail for both --
        # whichever pass ran last would leave the other pass's manifest rows
        # citing a hash that matches nothing on disk.
        resolved_path = resolved_config_path(f"grid_{args.supervision}")
        try:
            resolved_path.write_text(yaml.safe_dump(resolved_cfg, sort_keys=True))
        except (OSError, yaml.YAMLError) as e:
            print(f"[run_grid] failed to write resolved config as YAML ({e}); falling back to JSON.", flush=True)
            resolved_path.write_text(json.dumps(resolved_cfg, sort_keys=True, default=str, indent=2))

    print(f"[run_grid] tier={args.tier} seeds={seeds} budget={budget_s/3600:.2f}h workers={args.workers} "
          f"runs={len(runs)} already_completed={len(completed)} config_hash={cfg_hash[:12]}", flush=True)

    t0 = time.time()
    run_grid_loop(
        runs, completed, cfg, cfg_hash, commit, args.tier, budget_s, t0, MANIFEST,
        args.scaffold, args.workers, "[run_grid]",
    )

    write_report(MANIFEST, REPORT, budget_s, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
