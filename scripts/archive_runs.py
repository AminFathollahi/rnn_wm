#!/usr/bin/env python3
"""Rename legacy run artifacts and manifest identities without deleting data.

With no ``--run-id`` arguments, the script selects completed full-tier SUP
campaign runs whose metrics end at 80,000 steps and appends
``_budget80000``. Explicit run IDs and a custom suffix can be used to archive
an incompatible partial run. The default is always a dry run; filesystem and
manifest changes require ``--apply``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.jsonl"
CANONICAL_SUP_RE = re.compile(r"^M[01]{5}_SUP_s[0-7]$")


def _load_manifest() -> list[dict]:
    rows = []
    for line_no, line in enumerate(MANIFEST.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{MANIFEST}: line {line_no} is invalid JSON: {exc}") from exc
    return rows


def _artifact_pairs(run_id: str, archived_run_id: str) -> list[tuple[Path, Path]]:
    return [
        (
            RESULTS / "metrics" / f"{run_id}.csv",
            RESULTS / "metrics" / f"{archived_run_id}.csv",
        ),
        (
            RESULTS / "checkpoints" / run_id,
            RESULTS / "checkpoints" / archived_run_id,
        ),
        (
            RESULTS / f"resolved_config_{run_id.lower()}.yaml",
            RESULTS / f"resolved_config_{archived_run_id.lower()}.yaml",
        ),
    ]


def _last_metric_step(run_id: str, archived_run_id: str) -> int | None:
    source, archived = _artifact_pairs(run_id, archived_run_id)[0]
    path = source if source.exists() else archived
    if not path.exists():
        return None
    last_step = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("step"):
                last_step = int(row["step"])
    return last_step


def _discover_completed_80000(rows: list[dict], suffix: str) -> list[str]:
    completed = {
        row["run_id"]
        for row in rows
        if row.get("status") == "completed"
        and row.get("tier") == "full"
        and CANONICAL_SUP_RE.fullmatch(row.get("run_id", ""))
    }
    selected = []
    for run_id in sorted(completed):
        archived_run_id = f"{run_id}_{suffix}"
        if _last_metric_step(run_id, archived_run_id) == 80000:
            selected.append(run_id)
    return selected


def _validate_plan(rows: list[dict], run_ids: list[str], suffix: str, expected_step: int | None) -> dict[str, str]:
    manifest_ids = {row.get("run_id") for row in rows}
    mapping = {run_id: f"{run_id}_{suffix}" for run_id in run_ids}
    for source_id, archived_id in mapping.items():
        if source_id not in manifest_ids:
            raise SystemExit(f"{source_id}: no manifest row found")
        if archived_id in manifest_ids:
            raise SystemExit(f"{archived_id}: archive identity already exists in the manifest")
        step = _last_metric_step(source_id, archived_id)
        if expected_step is not None and step != expected_step:
            raise SystemExit(f"{source_id}: metrics end at step {step}, expected {expected_step}")
        for source, archived in _artifact_pairs(source_id, archived_id):
            if source.exists() and archived.exists():
                raise SystemExit(f"refusing collision: both {source} and {archived} exist")
    return mapping


def _rewrite_manifest(rows: list[dict], mapping: dict[str, str]) -> None:
    rewritten = []
    for row in rows:
        record = dict(row)
        if record.get("run_id") in mapping:
            record["run_id"] = mapping[record["run_id"]]
        rewritten.append(json.dumps(record, sort_keys=False))

    fd, temporary_name = tempfile.mkstemp(prefix="manifest.archive.", suffix=".jsonl", dir=MANIFEST.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write("\n".join(rewritten) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, MANIFEST)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="run ID to archive; repeat for more than one (default: discover completed 80,000-step SUP runs)",
    )
    parser.add_argument("--suffix", default="budget80000", help="archive suffix without the leading underscore")
    parser.add_argument("--expected-step", type=int, default=None, help="require every selected metrics CSV to end here")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="print the plan without changing files (the default)")
    mode.add_argument("--apply", action="store_true", help="perform the planned renames and manifest rewrite")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST}")
    if not args.suffix or "/" in args.suffix or "\\" in args.suffix:
        raise SystemExit("--suffix must be a non-empty filename component")

    rows = _load_manifest()
    run_ids = sorted(set(args.run_id)) if args.run_id else _discover_completed_80000(rows, args.suffix)
    if not run_ids:
        print("archive_runs: no matching runs")
        return 0
    mapping = _validate_plan(rows, run_ids, args.suffix, args.expected_step)

    mode_name = "APPLY" if args.apply else "DRY RUN"
    print(f"archive_runs: {mode_name}; {len(mapping)} run(s)")
    for source_id, archived_id in mapping.items():
        row_count = sum(row.get("run_id") == source_id for row in rows)
        print(f"  {source_id} -> {archived_id}  ({row_count} manifest row(s))")
        for source, archived in _artifact_pairs(source_id, archived_id):
            state = "rename" if source.exists() else ("already moved" if archived.exists() else "absent")
            print(f"    [{state}] {source.relative_to(ROOT)} -> {archived.relative_to(ROOT)}")

    if not args.apply:
        print("No changes made. Re-run with --apply to execute this exact plan.")
        return 0

    for source_id, archived_id in mapping.items():
        for source, archived in _artifact_pairs(source_id, archived_id):
            if source.exists():
                source.rename(archived)
    _rewrite_manifest(rows, mapping)
    print(f"Archived {len(mapping)} run(s); no artifact was deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
