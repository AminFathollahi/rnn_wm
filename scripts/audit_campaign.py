#!/usr/bin/env python3
"""Mechanical pre-launch/post-launch audit of the campaign's artifacts.

This exists because the mechanical half of the advisor's adversarial battery
is a set of predicates over `manifest.jsonl` x resolved configs x metrics
CSVs, and it was being recomputed by hand (or by a wave of agents) every
round.  Each check below corresponds to a defect that has actually happened
in this repository at least once, so a new defect class becomes a permanent
regression check here rather than a paragraph a future agent has to remember
to re-read.

Covered: provenance/identity (D32), the fall-through class (F1/D20/D24),
tier poisoning (the `M11011_SUP_s0` smoke row, D42), resume counters (D25),
orphaned artifacts, checkpoint-to-log provenance (D33), storage arithmetic
(D36), and a completed run's CSV trace actually reaching the resolved
`gates.max_steps` (D35).

NOT covered, and deliberately left to a human/advisor: gate integrity (is
this the right gate?), estimator validity (D37), confound enumeration
(D17/D28), seed degeneracy, human-side validity, and the claim ladder.  A
clean run here means the artifacts are self-consistent, not that the science
is sound.

    $ python scripts/audit_campaign.py
    $ python scripts/audit_campaign.py --tier full --seeds 8

Exit code is 1 if any new VIOLATION is found, 0 otherwise.  WARNINGs never
change the exit code -- they are things to look at, not things that are
wrong.

`--baseline results/audit_baseline.json` (comments.txt §21.5, D-number per
entry) lists violations that are known, accepted, and permanent -- e.g. the
D42 smoke-tier row kept on purpose (D15: archive, never delete) and D25's
three historical resume-counter corrections.  Fixing any of those means
deleting or rewriting history, so without a baseline this script exits 1
forever and the one NEW violation that actually matters arrives
indistinguishable from the five that are supposed to be there (this is how
`already_completed=26` went unnoticed for three days).  A baseline entry
prints as `KNOWN`; anything not in it prints as `VIOLATION` and exits 1.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.jsonl"
METRICS_DIR = RESULTS / "metrics"
CHECKPOINTS_DIR = RESULTS / "checkpoints"
ACTIVITY_DIR = RESULTS / "activity_logs"

# A campaign run id: five mechanism bits, a supervision level, a seed.
CAMPAIGN_RE = re.compile(r"^M[01]{5}(L)?_(SUP|RL)_s\d+$")

violations: list[str] = []
warnings: list[str] = []


def violation(check: str, msg: str) -> None:
    violations.append(f"[{check}] {msg}")


def warn(check: str, msg: str) -> None:
    warnings.append(f"[{check}] {msg}")


def load_rows() -> list[dict]:
    if not MANIFEST.exists():
        violation("manifest", f"{MANIFEST} does not exist")
        return []
    rows = []
    for i, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            violation("manifest", f"line {i} is not valid JSON ({e})")
    return rows


def check_tier_poisoning(rows: list[dict], tier: str) -> None:
    """A diagnostic run and a campaign run share a run_id by design; only the
    tier tells them apart.  `M11011_SUP_s0` was completed at `--tier smoke`
    (100 steps, chance accuracy) and would have been skipped as done in every
    later full-tier pass."""
    for r in rows:
        rid = r.get("run_id", "")
        if r.get("status") != "completed" or not CAMPAIGN_RE.match(rid):
            continue
        if r.get("tier") != tier:
            violation(
                "tier-poisoning",
                f"{rid} is completed at tier={r.get('tier')!r}, not {tier!r} "
                f"(wall_clock_s={r.get('wall_clock_s')}, config_hash="
                f"{str(r.get('config_hash'))[:12]}) -- it will be skipped as done "
                f"unless load_completed filters on tier",
            )


def check_config_hash_agreement(rows: list[dict], tier: str) -> None:
    """All completed rows of one campaign tier should cite one resolved
    config per supervision level.  A minority hash means some cells were
    trained under a different config than the rest -- not comparable at
    matched duration, which is what C1-C4 rest on."""
    by_sup: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r.get("status") == "completed" and CAMPAIGN_RE.match(r.get("run_id", "")) and r.get("tier") == tier:
            by_sup[str(r.get("supervision"))][str(r.get("config_hash"))[:12]] += 1
    for sup, hashes in by_sup.items():
        if len(hashes) > 1:
            warn("config-hash", f"supervision={sup} completed rows cite {len(hashes)} distinct config hashes: {dict(hashes)}")
        on_disk = RESULTS / f"resolved_config_grid_{sup}.yaml"
        if not on_disk.exists():
            warn("config-hash", f"supervision={sup} has completed rows but {on_disk.name} is not on disk")


def check_required_fields(rows: list[dict]) -> None:
    """D20: every run dict must state substrate, supervision and the
    recurrent init explicitly.  A null field is not evidence of absence
    (D22), but on a NEW row it is evidence the fall-through defect is back."""
    for r in rows:
        rid = r.get("run_id", "")
        if not CAMPAIGN_RE.match(rid):
            continue
        for key in ("substrate", "supervision"):
            if r.get(key) is None:
                violation("required-fields", f"{rid} ({r.get('status')}) has {key}=null")
        # A null spectral radius is only a defect where the radius means
        # something.  `configs/config.yaml:65` sets it for the vanilla
        # substrate only; a GRU run uses the default uniform draw and records
        # null legitimately (`run_grid.py:102-111`).  Warning on those made
        # 132 of this script's 132 warnings noise, which is how an audit tool
        # gets ignored.
        if r.get("substrate") == "vanilla" and r.get("recurrent_init_spectral_radius") is None:
            warn("required-fields", f"{rid} is vanilla with recurrent_init_spectral_radius=null (D19)")


def check_id_disjointness(seeds: int) -> None:
    """D32: run ids must be disjoint across every factor the campaign
    measures, or the second pass resumes the first pass's weights."""
    sys.path.insert(0, str(ROOT))
    try:
        import run_grid as rg
    except Exception as e:  # pragma: no cover - import guard
        warn("id-disjointness", f"could not import run_grid ({e}); skipped")
        return
    ids = {}
    for sup in ("SUP", "RL"):
        ids[sup] = {r["run_id"] for r in rg.enumerate_runs(list(range(seeds)), supervision=sup)}
    overlap = ids["SUP"] & ids["RL"]
    if overlap:
        violation("id-disjointness", f"{len(overlap)} run_ids appear under both supervision levels: {sorted(overlap)[:5]}")
    else:
        print(f"  ok: {len(ids['SUP'])} SUP + {len(ids['RL'])} RL run_ids, disjoint")


def check_orphans(rows: list[dict]) -> None:
    """A killed pass leaves metrics CSVs and checkpoints with no manifest row
    at all, so every manifest-based audit is blind to it."""
    known = {r.get("run_id") for r in rows}
    for csv_path in sorted(METRICS_DIR.glob("*.csv")):
        rid = csv_path.stem
        if not CAMPAIGN_RE.match(rid) or rid in known:
            continue
        last_step = "?"
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                last_step = row.get("step", "?")
        ckpts = sorted(p.name for p in (CHECKPOINTS_DIR / rid).glob("*.pt")) if (CHECKPOINTS_DIR / rid).is_dir() else []
        violation(
            "orphan",
            f"{rid} reached step {last_step} with checkpoints {ckpts or 'NONE'} but has NO manifest row "
            f"-- interrupted pass; the work resumes but is invisible to every manifest-based audit and to RUN_REPORT.md",
        )


def check_duplicate_completions(rows: list[dict]) -> None:
    """The same run completed twice under different configs means one of the
    two artifacts on disk is not the one the manifest describes."""
    seen: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("status") == "completed":
            seen[r.get("run_id", "")].append(r)
    for rid, rs in seen.items():
        hashes = {str(r.get("config_hash"))[:12] for r in rs}
        if len(rs) > 1 and len(hashes) > 1:
            warn("duplicate", f"{rid} completed {len(rs)}x under {len(hashes)} configs {sorted(hashes)}; the last write wins on disk")


def check_resume_counters(rows: list[dict]) -> None:
    """D25: milestone counters restarting at a resume boundary inflated Gate
    B's floor by 2x.  Trust the CSV, not the row -- this recomputes the row's
    claim from the trace it was supposed to come from."""
    for r in rows:
        rid = r.get("run_id", "")
        claimed = r.get("steps_to_load1_0.83")
        if r.get("status") != "completed" or claimed is None:
            continue
        csv_path = METRICS_DIR / f"{rid}.csv"
        if not csv_path.exists():
            warn("resume-counters", f"{rid} claims steps_to_load1_0.83={claimed} but {csv_path.name} is missing")
            continue
        run = 0
        derived = None
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                acc = row.get("train_acc_load1") or ""
                if not acc:
                    continue
                run = run + 1 if float(acc) >= 0.83 else 0
                if run >= 3:
                    derived = int(row["step"])
                    break
        if derived is not None and derived != claimed:
            violation("resume-counters", f"{rid}: manifest says steps_to_load1_0.83={claimed}, CSV trace says {derived}")


def check_activity_log_provenance(rows: list[dict]) -> None:
    """D33: every activity-derived DV reads the log, and the log is built
    from one named checkpoint.  A log with no completed run behind it is an
    analysis input with no provenance."""
    if not ACTIVITY_DIR.exists():
        warn("activity-logs", f"{ACTIVITY_DIR} does not exist (symlink to the external volume not mounted?)")
        return
    done = {r.get("run_id") for r in rows if r.get("status") == "completed"}
    for log in sorted(ACTIVITY_DIR.glob("*.parquet")):
        rid = log.stem
        for suffix in ("_at_criterion", "_chance", "_reflection_shuffled"):
            rid = rid[: -len(suffix)] if rid.endswith(suffix) else rid
        if CAMPAIGN_RE.match(rid) and rid not in done:
            violation("activity-logs", f"{log.name} exists but {rid} has no completed manifest row")


def _campaign_runs(rg) -> list[dict]:
    """The four real launch commands (`executor.md` §18.8), replicated
    exactly via `enumerate_runs` rather than re-derived as a seed x cell
    arithmetic, so a storage/count projection can never disagree with what
    actually launches again (§21.5b): 120 SUP (15 cells x 8 seeds) + 56 RL
    flat (7 S=0 cells x 8 seeds) + 32 RL local-learning (4 cells x 8 seeds)
    + 16 RL S=1 failure arm (8 cells x 2 seeds) = 224, not 240."""
    return (
        rg.enumerate_runs(list(range(8)), supervision="SUP")
        + rg.enumerate_runs(
            list(range(8)), supervision="RL",
            cells=["M00000", "M01111", "M01000", "M00100", "M00010", "M00001", "M00011"],
        )
        + rg.enumerate_runs(
            list(range(8)), supervision="RL", include_local_learning=True,
            cells=["M00L", "M01L", "M10L", "M11L"],
        )
        + rg.enumerate_runs(
            list(range(2)), supervision="RL",
            cells=["M11111", "M10111", "M11011", "M11101", "M11110", "M10000", "M10010", "M10001"],
        )
    )


def check_storage() -> None:
    """D36's arithmetic, on the real 224-run composition rather than a
    seeds-generic 15-cells-x2-supervision count.  An S=1 log stores
    h_worker(196) + h_manager(24) against a flat run's 128, so ~1.7x; every
    M=1 run additionally writes a SECOND log for H2's reflection-shuffle
    control, at the same size class as its own S bit.  The prior version
    assumed 120 SUP + 120 RL = 240 runs (the campaign is 224: RL only runs 7
    S=0 cells at full seed count, plus the local-learning and S=1-failure
    arms at their own seed counts) -- conservative in the safe direction,
    but a safety number that's wrong is still wrong (§21.5b)."""
    if not ACTIVITY_DIR.exists():
        return
    logs = [p for p in ACTIVITY_DIR.glob("*.parquet")]
    if not logs:
        warn("storage", "no activity logs to measure yet; storage projection skipped")
        return
    sys.path.insert(0, str(ROOT))
    import run_grid as rg

    runs = _campaign_runs(rg)
    biggest = max(logs, key=lambda p: p.stat().st_size)
    per_log_gb = biggest.stat().st_size / 1e9
    factor_sum = 0.0
    n_s1 = 0
    for r in runs:
        size_factor = 1.7 if r.get("S") else 1.0
        n_s1 += 1 if r.get("S") else 0
        factor_sum += size_factor  # this run's own primary log
        if r.get("M"):
            factor_sum += size_factor  # its reflection-shuffle partner log (D36)
    projected = per_log_gb * factor_sum
    free_gb = shutil.disk_usage(ACTIVITY_DIR).free / 1e9
    line = (f"largest log {biggest.name} = {per_log_gb:.2f} GB; {len(runs)} campaign runs "
            f"({n_s1} S=1 x1.7 + {len(runs) - n_s1} S=0) + a reflection-shuffle partner log for "
            f"every M=1 run = {projected:.0f} GB against {free_gb:.0f} GB free")
    if projected > free_gb:
        violation("storage", line + " -- does not fit")
    else:
        print(f"  ok: {line}")

    presymlink = RESULTS / "activity_logs.bak_presymlink"
    if presymlink.exists():
        size_gb = sum(p.stat().st_size for p in presymlink.rglob("*") if p.is_file()) / 1e9
        warn("storage", f"{presymlink} is {size_gb:.1f} GB, pending manual deletion since 2026-08-03 "
             "-- NOT deleted (D36/§20.11: the user's call, not this script's)")


def check_workers_consistency(rows: list[dict], tier: str) -> None:
    """D51: `wall_s_to_*`/`joules_to_*` are not comparable across rows with
    different `workers` (run_grid.py:705-709).  A completed campaign is
    expected to span exactly one `workers` value; more than one means a
    cross-cell wall-clock comparison would silently mix scheduling regimes.
    Scoped to `tier` like `check_tier_poisoning` -- a smoke-tier probe run at
    a different `--workers` (e.g. the D42 row) says nothing about the
    campaign's own wall-clock comparability."""
    at_tier = [r for r in rows if r.get("status") == "completed" and r.get("tier") == tier and CAMPAIGN_RE.match(r.get("run_id", ""))]
    values = {r.get("workers") for r in at_tier}
    if len(values) > 1:
        named = {v: sorted(r["run_id"] for r in at_tier if r.get("workers") == v) for v in values}
        warn("workers-consistency", f"completed tier={tier!r} campaign rows carry {len(values)} distinct "
             f"`workers` values, not wall-clock comparable across them: {named}")


def check_max_steps_completion(rows: list[dict]) -> None:
    """D35: `gates.max_steps` is read by no training code path except through
    the tier's resolved config, so a launcher regression can silently run a
    different ceiling than every artifact claims. This check would have
    caught D35 by itself: a `completed` row's `accuracy_at_max_steps` must be
    populated, and its metrics CSV must actually reach the resolved
    `gates.max_steps` -- not just claim to."""
    max_steps_by_sup: dict[str, int | None] = {}
    for sup in ("SUP", "RL"):
        cfg_path = RESULTS / f"resolved_config_grid_{sup}.yaml"
        if not cfg_path.exists():
            continue
        m = re.search(r"^\s*max_steps:\s*(\d+)\s*$", cfg_path.read_text(), re.MULTILINE)
        max_steps_by_sup[sup] = int(m.group(1)) if m else None
    for r in rows:
        rid = r.get("run_id", "")
        if r.get("status") != "completed" or not CAMPAIGN_RE.match(rid):
            continue
        if r.get("accuracy_at_max_steps") is None:
            violation("max-steps", f"{rid} is completed but accuracy_at_max_steps is null")
        expected = max_steps_by_sup.get(str(r.get("supervision")))
        if expected is None:
            continue
        csv_path = METRICS_DIR / f"{rid}.csv"
        if not csv_path.exists():
            warn("max-steps", f"{rid} claims completion but {csv_path.name} is missing")
            continue
        last_step = None
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("step"):
                    last_step = int(row["step"])
        if last_step != expected:
            violation(
                "max-steps",
                f"{rid} is completed but its CSV's last step is {last_step}, not the "
                f"resolved gates.max_steps={expected} -- ran a different ceiling than "
                f"the artifact claims (D35)",
            )


def load_baseline(path: Path) -> dict[str, str]:
    """`message -> reason`.  `{}` if the file doesn't exist -- nothing is
    known yet, so every violation is new."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {entry["message"]: entry["reason"] for entry in data.get("known_violations", [])}


def partition_against_baseline(all_violations: list[str], baseline: dict[str, str]) -> tuple[list[str], list[str]]:
    known = [v for v in all_violations if v in baseline]
    new = [v for v in all_violations if v not in baseline]
    return known, new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="full", help="campaign tier these run_ids belong to (default: full)")
    ap.add_argument("--seeds", type=int, default=8, help="seed count the campaign is authorized for (default: 8)")
    ap.add_argument("--baseline", type=Path, default=RESULTS / "audit_baseline.json",
                     help="known/accepted violations that don't affect the exit code (default: %(default)s)")
    args = ap.parse_args(argv)

    rows = load_rows()
    print(f"audit_campaign: {len(rows)} manifest rows, tier={args.tier!r}, seeds={args.seeds}\n")

    for name, fn in [
        ("tier poisoning", lambda: check_tier_poisoning(rows, args.tier)),
        ("config hash agreement", lambda: check_config_hash_agreement(rows, args.tier)),
        ("required fields", lambda: check_required_fields(rows)),
        ("run_id disjointness", lambda: check_id_disjointness(args.seeds)),
        ("orphaned artifacts", lambda: check_orphans(rows)),
        ("duplicate completions", lambda: check_duplicate_completions(rows)),
        ("resume counters", lambda: check_resume_counters(rows)),
        ("activity log provenance", lambda: check_activity_log_provenance(rows)),
        ("storage arithmetic", lambda: check_storage()),
        ("max_steps completion (D35)", lambda: check_max_steps_completion(rows)),
        ("workers consistency (D51)", lambda: check_workers_consistency(rows, args.tier)),
    ]:
        print(f"- {name}")
        fn()

    baseline = load_baseline(args.baseline)
    known, new = partition_against_baseline(violations, baseline)

    print()
    for w in warnings:
        print(f"WARNING  {w}")
    for v in known:
        print(f"KNOWN    {v}  [{baseline[v]}]")
    for v in new:
        print(f"VIOLATION {v}")
    print(f"\n{len(new)} violation(s), {len(known)} known/accepted (baseline={args.baseline}), {len(warnings)} warning(s)")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main())
