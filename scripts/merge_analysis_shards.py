#!/usr/bin/env python3
"""Merge per-run outputs written by the post-training analysis runner."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from brainalign_wm.config import get_path


TABLES = (
    "alignment_by_session",
    "alignment_results",
    "alignment_probe_by_region",
    "alignment_by_population",
    "reflection_shuffle_lesion",
    "dynamics_persistence",
    "encoding_results",
    "dpca_results",
)
REQUIRED = (
    "alignment_by_session",
    "alignment_results",
    "alignment_probe_by_region",
    "alignment_by_population",
)


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", prefix=f".{destination.stem}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _has_data(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--run-list", type=Path, required=True)
    parser.add_argument(
        "--suffix", default="",
        help="shard/output filename suffix, e.g. '_at_criterion' for a non-default-checkpoint pass "
             "(matches run_all.py's own --checkpoint tag) -- keeps that pass's merged tables out of "
             "the default checkpoint's results/<name>.csv.",
    )
    args = parser.parse_args(argv)

    run_ids = [line.strip() for line in args.run_list.read_text().splitlines() if line.strip()]
    if not run_ids:
        raise SystemExit("cannot merge an empty analysis run list")
    missing = []
    for run_id in run_ids:
        shard = args.checkpoint_dir / run_id
        if not (shard / ".complete").is_file():
            missing.append(run_id)
            continue
        missing_tables = [name for name in REQUIRED if not _has_data(shard / f"{name}{args.suffix}.csv")]
        if missing_tables:
            missing.append(f"{run_id} ({', '.join(missing_tables)})")
    if missing:
        raise SystemExit("cannot merge incomplete analysis shards: " + "; ".join(missing))

    results = get_path("results")
    for name in TABLES:
        parts = []
        for run_id in run_ids:
            path = args.checkpoint_dir / run_id / f"{name}{args.suffix}.csv"
            if path.is_file():
                parts.append(pd.read_csv(path))
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True, sort=False)
        if "run_id" in merged:
            sort_columns = [column for column in ("run_id", "session", "region", "subpop") if column in merged]
            merged = merged.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        destination = results / f"{name}{args.suffix}.csv"
        _atomic_csv(merged, destination)
        print(f"[merge_analysis_shards] wrote {destination} ({len(merged)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
