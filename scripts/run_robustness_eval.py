#!/usr/bin/env python3
"""Evaluate alignment as a predictor of held-out robustness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from brainalign_wm.analysis.robustness import (
    accuracy_matched, correlation_table, longer_delay_cfg, novel_load_cfg,
)
from brainalign_wm.config import get_path
from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate
from brainalign_wm.tasks.generator import TaskGenerator
from brainalign_wm.tasks.image_token_bank import ImageTokenBank
from brainalign_wm.training.generate_activity_logs import _load_checkpoint
from brainalign_wm.training.train import _build_model, _load_full_config, evaluate_accuracy

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"
OUT_PATH = RESULTS / "robustness.csv"
SUMMARY_PATH = RESULTS / "robustness_summary.csv"
NOVEL_LOAD = 4
DELAY_FACTOR = 3.0
PREDICTORS = ["maintenance_signed_raw_alignment", "maintenance_normalized_alignment"]
OUTCOMES = ["longer_delay_mean_acc", "longer_delay_drop", f"novel_load{NOVEL_LOAD}_acc"]


def _battery_accuracy() -> pd.DataFrame:
    latest: dict[str, dict] = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        latest[r["run_id"]] = r
    rows = []
    for r in latest.values():
        if r.get("status") != "completed" or r.get("archived", False):
            continue
        if not all(k in r for k in "SMPTD"):
            continue
        acc = r.get("accuracy") or {}
        if not all(f"load{l}" in acc for l in (1, 2, 3)):
            continue
        rows.append({
            "run_id": r["run_id"], "model_id": r["model_id"], "S": r["S"], "M": r["M"], "P": r["P"],
            "T": r["T"], "D": r["D"], "seed": r["seed"], "supervision": r.get("supervision"),
            "mean_acc": float(np.mean([acc["load1"], acc["load2"], acc["load3"]])),
        })
    return pd.DataFrame(rows)


def _with_alignment(df: pd.DataFrame) -> pd.DataFrame:
    align_path = RESULTS / "alignment_results.csv"
    align = pd.read_csv(align_path)[["run_id", "maintenance_normalized_alignment", "maintenance_signed_raw_alignment"]]
    return df.merge(align, on="run_id", how="inner")


def evaluate_one(record: dict, cfg: dict, image_bank: ImageTokenBank, device, n_trials: int, eval_seed: int) -> dict:
    S, M, P = record["S"], record["M"], record["P"]
    front_end, core, heads = _build_model(cfg, S, M, P, device)
    _load_checkpoint(front_end, core, heads, record["run_id"], device)
    front_end.eval(); core.eval(); heads.eval()
    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None

    task_gen = TaskGenerator(cfg, image_bank, seed=0)
    delay_cfg = longer_delay_cfg(cfg, DELAY_FACTOR)
    delay_acc = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, delay_cfg, device, n_trials, eval_seed)
    novel_cfg = novel_load_cfg(cfg, NOVEL_LOAD)
    novel_acc = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, novel_cfg, device, n_trials, eval_seed + 1)

    delay_mean = float(np.mean([delay_acc[f"load{l}"] for l in cfg["task"]["loads"]]))
    return {
        "longer_delay_maintain_steps": delay_cfg["task"]["maintain_steps"],
        "longer_delay_mean_acc": delay_mean,
        "longer_delay_drop": record["mean_acc"] - delay_mean,
        f"novel_load{NOVEL_LOAD}_acc": novel_acc[f"load{NOVEL_LOAD}"],
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Pooled, then split by training signal. The two signals produce very
    different alignment values, so a pooled correlation can carry a sign that
    holds in neither group."""
    tables = [("all", df)] + sorted(df.groupby("supervision"), key=lambda kv: str(kv[0]))
    out = []
    for name, group in tables:
        table = correlation_table(group, PREDICTORS, OUTCOMES)
        table.insert(0, "group", name)
        out.append(table)
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tol", type=float, default=0.02, help="accuracy-matching tolerance around the median")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    if args.summarize_only:
        summary = summarize(pd.read_csv(OUT_PATH))
        summary.to_csv(SUMMARY_PATH, index=False)
        print(summary.to_string(index=False))
        return 0

    cfg = _load_full_config()
    device = torch.device("cpu")
    df = _with_alignment(_battery_accuracy())
    matched, median = accuracy_matched(df, tol=args.tol)
    print(f"{len(matched)}/{len(df)} runs within +/-{args.tol} of median accuracy {median:.4f} "
          f"({matched['model_id'].nunique()} of {df['model_id'].nunique()} architectures)")

    image_bank = ImageTokenBank(
        stimuli_root=Path(cfg["paths"]["stimuli"]), categories=cfg["task"]["categories"],
        feature_cache_path=Path(cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )

    rows = []
    for i, record in enumerate(matched.to_dict("records")):
        print(f"[{i+1}/{len(matched)}] {record['run_id']}", flush=True)
        row = dict(record)
        row["accuracy_match_tol"] = args.tol
        row["accuracy_match_median"] = median
        row.update(evaluate_one(record, cfg, image_bank, device, args.n_trials, eval_seed=2_000_000 + i))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(out)} rows -> {OUT_PATH}")

    summary = summarize(out)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
