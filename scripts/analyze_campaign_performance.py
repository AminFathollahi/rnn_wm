#!/usr/bin/env python3
"""Write active 224-run campaign performance tables from the training manifest."""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from brainalign_wm.config import get_path


CORE_MODEL = re.compile(r"M[01]{5}$")
LOCAL_MODEL = re.compile(r"M[01]{2}L$")


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", prefix=f".{destination.stem}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    results = get_path("results")
    latest: dict[str, dict] = {}
    for line in (results / "manifest.jsonl").read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[record.get("run_id", "")] = record

    rows = []
    for run_id, record in latest.items():
        model_id = str(record.get("model_id", ""))
        supervision = record.get("supervision")
        is_core = CORE_MODEL.fullmatch(model_id) and supervision in {"SUP", "RL"}
        is_local = LOCAL_MODEL.fullmatch(model_id) and supervision == "RL"
        if record.get("status") != "completed" or record.get("archived", False) or not (is_core or is_local):
            continue
        accuracy = record.get("accuracy") or {}
        rows.append({
            "run_id": run_id,
            "analysis_group": "core" if is_core else "local_learning",
            "model_id": model_id,
            "supervision": supervision,
            "seed": record.get("seed"),
            "S": record.get("S"), "M": record.get("M"), "P": record.get("P"),
            "T": record.get("T"), "D": record.get("D"),
            "local_learning_rung": record.get("local_learning_rung"),
            "load1_accuracy": accuracy.get("load1"),
            "load2_accuracy": accuracy.get("load2"),
            "load3_accuracy": accuracy.get("load3"),
            "criterion_met_at_budget": record.get("criterion_met_at_budget"),
            "criterion_met_by_stop": record.get("criterion_met_by_stop"),
        })

    runs = pd.DataFrame(rows).sort_values(["analysis_group", "supervision", "model_id", "seed"], kind="stable")
    if len(runs) != 224:
        raise SystemExit(f"expected 224 active campaign runs, found {len(runs)}")
    _atomic_csv(runs, results / "performance_by_run.csv")

    grouped = runs.groupby(["analysis_group", "supervision", "model_id"], dropna=False)
    summary = grouped.agg(
        n_runs=("run_id", "size"),
        load1_mean=("load1_accuracy", "mean"), load1_sd=("load1_accuracy", "std"),
        load2_mean=("load2_accuracy", "mean"), load2_sd=("load2_accuracy", "std"),
        load3_mean=("load3_accuracy", "mean"), load3_sd=("load3_accuracy", "std"),
        criterion_at_budget_rate=("criterion_met_at_budget", "mean"),
        criterion_by_stop_rate=("criterion_met_by_stop", "mean"),
    ).reset_index()
    _atomic_csv(summary, results / "performance_summary.csv")
    print(f"[analyze_campaign_performance] wrote 224 runs and {len(summary)} grouped rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
