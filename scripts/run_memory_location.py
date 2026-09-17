#!/usr/bin/env python3
"""Evaluate hidden-state and synaptic-trace interventions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from brainalign_wm.analysis.memory_location import (
    carrier_deviation, evaluate_lesion_accuracy, summarize_interventions, summarize_selectivity,
)
from brainalign_wm.config import get_path
from brainalign_wm.tasks.generator import TaskGenerator
from brainalign_wm.tasks.image_token_bank import ImageTokenBank
from brainalign_wm.training.generate_activity_logs import _load_checkpoint
from brainalign_wm.training.train import _build_model, _load_full_config

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"
OUT_PATH = RESULTS / "memory_location.csv"
SUMMARY_PATH = RESULTS / "memory_location_summary.csv"
SELECTIVITY_PATH = RESULTS / "memory_location_selectivity.csv"
SELECTIVITY_SUMMARY_PATH = RESULTS / "memory_location_selectivity_summary.csv"
PLASTIC_MODEL_IDS = ("M00100", "M01111", "M10111", "M11101", "M11110", "M11111")

# target, epoch, label, hold the synaptic trace at its pre-tick value, corruption mode
CONDITIONS = [
    (None, "maintain", "control", True, "noise"),
    ("activity", "maintain", "delay_activity", True, "noise"),
    ("synaptic", "maintain", "delay_synaptic", True, "noise"),
    ("activity", "fixation", "outside_activity", True, "noise"),
    ("synaptic", "fixation", "outside_synaptic", True, "noise"),
    ("activity", "maintain", "delay_activity_free_trace", False, "noise"),
    ("synaptic", "maintain", "delay_synaptic_erased", True, "erase"),
]


def _completed_plastic_runs(manifest: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[r["run_id"]] = r
    return [
        r for r in latest.values()
        if r.get("model_id") in PLASTIC_MODEL_IDS and r.get("status") == "completed" and not r.get("archived", False)
    ]


ALIGNMENT_COLUMNS = ("maintenance_signed_raw_alignment", "maintenance_normalized_alignment")


def _maintenance_alignment_lookup() -> dict:
    """Both the signed estimate and the normalized one. The normalized column
    clips negative alignment to zero, so on its own it has almost no variance
    across these checkpoints."""
    path = RESULTS / "alignment_results.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        return {}
    cols = [c for c in ALIGNMENT_COLUMNS if c in df.columns]
    return df.set_index("run_id")[cols].to_dict("index")


def run_one(record: dict, cfg: dict, image_bank: ImageTokenBank, task_gen: TaskGenerator, device, n_trials: int, eval_seed: int, conditions=None) -> list[dict]:
    S, M, P = record["S"], record["M"], record["P"]
    front_end, core, heads = _build_model(cfg, S, M, P, device)
    _load_checkpoint(front_end, core, heads, record["run_id"], device)
    front_end.eval(); core.eval(); heads.eval()

    rows = []
    for target, epoch, label, hold_trace, mode in (conditions or CONDITIONS):
        acc = evaluate_lesion_accuracy(
            front_end, core, heads, S, M, P, task_gen, image_bank, cfg, device,
            target=target, lesion_epoch=epoch, n_trials=n_trials, eval_seed=eval_seed,
            hold_trace=hold_trace, mode=mode,
        )
        for load in cfg["task"]["loads"]:
            rows.append({
                "run_id": record["run_id"], "model_id": record["model_id"], "S": S, "M": M, "P": P,
                "seed": record["seed"], "supervision": record.get("supervision"), "condition": label,
                "load": load, "accuracy": acc[f"load{load}"],
                "ci_lo": acc[f"load{load}_ci_lo"], "ci_hi": acc[f"load{load}_ci_hi"],
            })
    return rows


def selectivity_rows(record: dict, cfg: dict, image_bank: ImageTokenBank, task_gen: TaskGenerator, device, eval_seed: int) -> list[dict]:
    """Measures, for every lesion condition, how far BOTH carriers move from
    their unlesioned trajectory -- the evidence for or against the claim
    that each lesion disrupts one carrier and leaves the other alone."""
    S, M, P = record["S"], record["M"], record["P"]
    front_end, core, heads = _build_model(cfg, S, M, P, device)
    _load_checkpoint(front_end, core, heads, record["run_id"], device)
    front_end.eval(); core.eval(); heads.eval()

    rows = []
    for target, epoch, label, hold_trace, mode in CONDITIONS:
        if target is None:
            continue
        for load in cfg["task"]["loads"]:
            row = carrier_deviation(
                front_end, core, heads, S, M, P, task_gen, image_bank, cfg, device,
                target=target, lesion_epoch=epoch, load=load, eval_seed=eval_seed,
                hold_trace=hold_trace, mode=mode,
            )
            row.update({"run_id": record["run_id"], "model_id": record["model_id"], "S": S,
                        "supervision": record.get("supervision"), "condition": label})
            rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-trials", type=int, default=150)
    ap.add_argument("--model-ids", nargs="+", default=None, help="restrict to these model_ids (default: all six)")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--selectivity-only", action="store_true", help="measure carrier deviations, skip the accuracy sweep")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict to these condition labels; rows already saved for them are replaced, the rest are kept")
    args = ap.parse_args(argv)

    conditions = CONDITIONS
    if args.conditions:
        unknown = set(args.conditions) - {c[2] for c in CONDITIONS}
        if unknown:
            ap.error(f"unknown condition labels: {sorted(unknown)}")
        conditions = [c for c in CONDITIONS if c[2] in args.conditions]

    if args.summarize_only:
        summary = summarize_interventions(pd.read_csv(OUT_PATH))
        summary.to_csv(SUMMARY_PATH, index=False)
        print(f"wrote {len(summary)} rows -> {SUMMARY_PATH}")
        if SELECTIVITY_PATH.exists():
            selectivity = summarize_selectivity(pd.read_csv(SELECTIVITY_PATH))
            selectivity.to_csv(SELECTIVITY_SUMMARY_PATH, index=False)
            print(selectivity.to_string(index=False))
            print(f"wrote {len(selectivity)} rows -> {SELECTIVITY_SUMMARY_PATH}")
        return 0

    cfg = _load_full_config()
    device = torch.device("cpu")
    records = _completed_plastic_runs(MANIFEST)
    if args.model_ids:
        records = [r for r in records if r["model_id"] in args.model_ids]
    if not records:
        print("no completed plastic runs found")
        return 0

    image_bank = ImageTokenBank(
        stimuli_root=Path(cfg["paths"]["stimuli"]), categories=cfg["task"]["categories"],
        feature_cache_path=Path(cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(cfg, image_bank, seed=0)
    alignment = _maintenance_alignment_lookup()

    all_rows = []
    for i, record in enumerate(sorted(records, key=lambda r: r["run_id"])):
        print(f"[{i+1}/{len(records)}] {record['run_id']}", flush=True)
        if args.selectivity_only:
            all_rows.extend(selectivity_rows(record, cfg, image_bank, task_gen, device, eval_seed=1_000_000 + i))
            continue
        all_rows.extend(run_one(record, cfg, image_bank, task_gen, device, args.n_trials, eval_seed=1_000_000 + i, conditions=conditions))

    df = pd.DataFrame(all_rows)
    if args.selectivity_only:
        df.to_csv(SELECTIVITY_PATH, index=False)
        summarize_selectivity(df).to_csv(SELECTIVITY_SUMMARY_PATH, index=False)
        print(f"wrote {len(df)} rows -> {SELECTIVITY_PATH}")
        return 0
    if args.conditions and OUT_PATH.exists():
        kept = pd.read_csv(OUT_PATH)
        kept = kept[~kept["condition"].isin(args.conditions)]
        df = pd.concat([kept, df], ignore_index=True).sort_values(["run_id", "condition", "load"], ignore_index=True)
    for col in ALIGNMENT_COLUMNS:
        df[col] = df["run_id"].map(lambda r: alignment.get(r, {}).get(col))
    df.to_csv(OUT_PATH, index=False)
    summarize_interventions(df).to_csv(SUMMARY_PATH, index=False)
    print(f"wrote {len(df)} rows -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
