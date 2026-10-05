#!/usr/bin/env python3
"""Build measurement-validation tables from existing activity logs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brainalign_wm.analysis.alignment_distribution import distribution_rows
from brainalign_wm.analysis.dynamics_and_persistence import persistence_index_for_session
from brainalign_wm.analysis.full_delay import FullDelayView, comparison_rows, window_qc_rows
from brainalign_wm.analysis.measurement_validation import (
    epoch_method_rows,
    maintenance_task_structure_rows,
    probe_task_structure_rows,
    sensitivity_curve,
)
from brainalign_wm.analysis.run_all import (
    REGIONS,
    _is_ablation_or_catch_variant,
    _load_completed_runs,
    _valid_model_trials,
)
from brainalign_wm.analysis.stats import compare_distributions
from brainalign_wm.config import DEFAULT_CONFIG_PATH, get_path, load_config
from brainalign_wm.training.logging_schema import read_log

TABLE_FILES = {
    "task_structure": "task_structure_baselines.csv",
    "sensitivity": "maintenance_sensitivity.csv",
    "epoch_method": "epoch_method.csv",
    "persistence": "dynamics_persistence_corrected.csv",
    "full_delay": "maintenance_full_delay.csv",
    "distribution": "alignment_distributions.csv",
}
STATE_VERSION = 3
PRIMARY_COLUMNS = (
    "run_id", "model_id", "S", "M", "P", "T", "D", "seed", "supervision",
    "epoch", "region", "method", "predictor", "mixing_weight", "status",
)


@dataclass(frozen=True)
class WorkUnit:
    table: str
    run_id: str | None = None
    region: str | None = None
    mixing_weight: float | None = None

    @property
    def key(self) -> str:
        payload = json.dumps(
            {"version": STATE_VERSION, **asdict(self)}, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def label(self) -> str:
        fields = [self.table]
        if self.run_id is not None:
            fields.append(self.run_id)
        fields.append(self.region or "pooled")
        if self.mixing_weight is not None:
            fields.append(f"weight={self.mixing_weight:g}")
        return "/".join(fields)


def _core_records(manifest: Path, selected: set[str] | None = None) -> list[dict]:
    records = _load_completed_runs(manifest)
    if selected is not None:
        records = [record for record in records if record["run_id"] in selected]
        missing = selected - {record["run_id"] for record in records}
        if missing:
            raise ValueError(f"no active completed manifest row for: {', '.join(sorted(missing))}")
        return sorted(records, key=lambda record: record["run_id"])
    return sorted(
        (
            record for record in records
            if all(name in record for name in ("S", "M", "P", "T", "D"))
            and not _is_ablation_or_catch_variant(record)
        ),
        key=lambda record: record["run_id"],
    )


def _log_path(log_dir: Path, run_id: str, checkpoint: str) -> Path:
    suffix = "" if checkpoint == "ckpt.pt" else "_" + Path(checkpoint).stem.removeprefix("ckpt_")
    return log_dir / f"{run_id}{suffix}.parquet"


def enumerate_units(
    tables: Iterable[str], records: list[dict], regions: list[str | None], weights: list[float],
) -> list[WorkUnit]:
    units = []
    for table in tables:
        if table == "sensitivity":
            units.extend(
                WorkUnit(table, region=region, mixing_weight=weight)
                for region in regions for weight in weights
            )
        elif table == "persistence":
            units.extend(WorkUnit(table, run_id=record["run_id"]) for record in records)
        elif table == "full_delay":
            units.append(WorkUnit(table))
            units.extend(WorkUnit(table, run_id=record["run_id"], region=region) for record in records for region in regions)
        else:
            units.extend(WorkUnit(table, run_id=record["run_id"], region=region) for record in records for region in regions)
    return units


def _metadata(record: dict) -> dict:
    names = ("model_id", "S", "M", "P", "T", "D", "seed", "supervision")
    return {name: record.get(name) for name in names}


def _persistence_row(run_id: str, model_df, dataset) -> dict:
    model_df = _valid_model_trials(model_df, dataset)
    sessions = sorted(model_df["session"].unique()) if "session" in model_df.columns else []
    model_values, neural_values = [], []
    used = 0
    for session_id in sessions:
        result = persistence_index_for_session(model_df, dataset, session_id, None, dataset.bin_ms)
        if result is None:
            continue
        model_values.extend(np.asarray(result["model_index_standardized"]).ravel())
        neural_values.extend(np.asarray(result["neural_index_standardized"]).ravel())
        used += 1
    model_values = np.asarray(model_values, dtype=float)
    neural_values = np.asarray(neural_values, dtype=float)
    model_values = model_values[np.isfinite(model_values)]
    neural_values = neural_values[np.isfinite(neural_values)]
    row = {
        "run_id": run_id,
        "region": "pooled",
        "status": "ok" if used and len(model_values) >= 2 and len(neural_values) >= 2 else "insufficient_sessions",
        "n_sessions": used,
        "n_model_units": len(model_values),
        "n_neural_units": len(neural_values),
    }
    if row["status"] == "ok":
        row.update({f"persistence_standardized_{key}": value for key, value in compare_distributions(model_values, neural_values).items()})
        row["model_persistence_standardized_mean"] = float(model_values.mean())
        row["neural_persistence_standardized_mean"] = float(neural_values.mean())
    return row


def execute_unit(
    unit: WorkUnit,
    records: dict[str, dict],
    dataset,
    log_dir: Path,
    checkpoint: str,
    n_boot: int,
    seed: int,
    full_dataset=None,
) -> list[dict]:
    if unit.table == "sensitivity":
        return sensitivity_curve(dataset, unit.region, [unit.mixing_weight], seed=seed, n_boot=n_boot)
    if unit.table == "full_delay" and unit.run_id is None:
        return window_qc_rows(dataset, dataset.bin_ms)
    record = records[unit.run_id]
    model_df = read_log(_log_path(log_dir, unit.run_id, checkpoint))
    if unit.table == "task_structure":
        rows = probe_task_structure_rows(unit.run_id, model_df, dataset, unit.region)
        rows += maintenance_task_structure_rows(unit.run_id, model_df, dataset, unit.region)
    elif unit.table == "epoch_method":
        rows = epoch_method_rows(unit.run_id, model_df, dataset, unit.region)
    elif unit.table == "full_delay":
        rows = comparison_rows(
            unit.run_id, model_df, dataset, unit.region, n_boot=n_boot, seed=seed,
            full_dataset=full_dataset,
        )
    elif unit.table == "distribution":
        rows = distribution_rows(
            unit.run_id, model_df, dataset, unit.region, n_boot=n_boot, seed=seed,
        )
    else:
        rows = [_persistence_row(unit.run_id, model_df, dataset)]
    metadata = _metadata(record)
    return [{**row, **metadata} for row in rows]


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, default=_json_value, allow_nan=True, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _publish(table: str, state_dir: Path, out_dir: Path, tag: str) -> Path:
    rows = []
    for path in sorted((state_dir / table).glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    name = Path(TABLE_FILES[table])
    out_path = out_dir / f"{name.stem}{tag}{name.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    order = ("run_id", "epoch", "region", "method", "predictor", "mixing_weight", "session")
    rows.sort(key=lambda row: tuple(str(row.get(key, "")) for key in order))
    names = {key for row in rows for key in row}
    columns = [name for name in PRIMARY_COLUMNS if name in names]
    columns.extend(sorted(names - set(columns)))
    with tempfile.NamedTemporaryFile(
        "w", dir=out_path.parent, prefix=f".{out_path.name}.", newline="", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        if columns:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in columns} for row in rows)
    os.replace(temporary, out_path)
    return out_path


def run_units(
    units: list[WorkUnit], state_dir: Path, out_dir: Path, tag: str,
    executor: Callable[[WorkUnit], list[dict]], overwrite: bool = False, fail_fast: bool = False,
) -> tuple[int, int, int]:
    completed = skipped = failed = 0
    for index, unit in enumerate(units, 1):
        path = state_dir / unit.table / f"{unit.key}.json"
        if path.exists() and not overwrite:
            print(f"[{index}/{len(units)}] cached {unit.key} {unit.label()}", flush=True)
            skipped += 1
            continue
        print(f"[{index}/{len(units)}] running {unit.key} {unit.label()}", flush=True)
        try:
            rows = executor(unit)
            with _locked(state_dir / ".lock"):
                _write_json(path, {"unit": asdict(unit), "rows": rows})
                output = _publish(unit.table, state_dir, out_dir, tag)
            print(f"[{index}/{len(units)}] wrote {len(rows)} row(s) to {output}", flush=True)
            completed += 1
        except Exception as exc:
            print(
                f"[{index}/{len(units)}] failed {unit.key}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            failed += 1
            if fail_fast:
                raise
    return completed, skipped, failed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--out-dir", type=Path, default=get_path("results"))
    parser.add_argument("--checkpoint", default="ckpt.pt")
    parser.add_argument("--tables", nargs="+", choices=TABLE_FILES, default=list(TABLE_FILES))
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--regions", nargs="+", default=[region or "pooled" for region in REGIONS])
    parser.add_argument("--weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--max-sessions-per-dataset", type=int)
    parser.add_argument("--unit", help="execute only the work-unit key shown by --dry-run")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    regions = [None if region.lower() == "pooled" else region for region in args.regions]
    manifest = get_path("results") / "manifest.jsonl"
    records = _core_records(manifest, set(args.runs) if args.runs else None)
    log_dir = get_path("activity_logs")
    available = [record for record in records if _log_path(log_dir, record["run_id"], args.checkpoint).exists()]
    missing_logs = len(records) - len(available)
    if args.max_runs is not None:
        available = available[:args.max_runs]
    units = enumerate_units(args.tables, available, regions, args.weights)
    if args.unit:
        units = [unit for unit in units if unit.key == args.unit]
        if not units:
            print(f"unknown work-unit key: {args.unit}", file=sys.stderr)
            return 2

    cap = args.max_sessions_per_dataset or "all"
    checkpoint_tag = "" if args.checkpoint == "ckpt.pt" else "_" + Path(args.checkpoint).stem.removeprefix("ckpt_")
    cap_tag = "" if cap == "all" else f"_sessions{cap}"
    tag = checkpoint_tag + cap_tag
    namespace = f"v{STATE_VERSION}_{Path(args.checkpoint).stem}_sessions-{cap}_seed-{args.seed}_boot-{args.n_boot}"
    state_dir = args.out_dir / ".measurement_outputs" / namespace
    if args.dry_run:
        for unit in units:
            status = "cached" if (state_dir / unit.table / f"{unit.key}.json").exists() else "pending"
            print(f"{unit.key}\t{status}\t{unit.label()}")
        print(
            f"{len(units)} work unit(s); {missing_logs} completed run(s) omitted "
            "because the activity log is absent"
        )
        return 0
    if not units:
        print("no runnable work units")
        return 0

    cfg = load_config(args.config)
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    dataset = DandiSternbergTierA(
        cfg["paths"]["data_root"],
        datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"],
        min_session_accuracy=cfg["neural"]["min_session_accuracy"],
        bin_ms=cfg["neural"]["bin_ms"],
        max_sessions_per_dataset=args.max_sessions_per_dataset,
    )
    by_run = {record["run_id"]: record for record in available}
    full_dataset = FullDelayView(dataset) if "full_delay" in args.tables else None
    executor = lambda unit: execute_unit(
        unit, by_run, dataset, log_dir, args.checkpoint, args.n_boot, args.seed, full_dataset
    )
    completed, skipped, failed = run_units(
        units, state_dir, args.out_dir, tag, executor, args.overwrite, args.fail_fast
    )
    print(f"completed={completed} cached={skipped} failed={failed}")
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
