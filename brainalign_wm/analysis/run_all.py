"""Runs the alignment and statistical-inference pipeline over whatever
training runs have completed so far (reads `results/manifest.jsonl`; does
not require the full grid to have finished).

For each completed run: generates (or reuses a cached) post-training
activity log by replaying the real Tier A sessions through the trained
checkpoint (see `training/generate_activity_logs.py`), builds a
crossnobis representational dissimilarity matrix from per-trial
maintenance-epoch activity over the coarse condition set (load,
probe-in-set, correct -- the condition fields reliably available across
all replayed trials), and compares it against a crossnobis RDM built the
same way from the real per-trial neural data, reporting alignment both raw
and as a fraction of the neural noise ceiling. A linear mixed-effects
model of alignment on the factorial design (S, M, L, plus an accuracy
covariate) is fit once enough cells are available.

Region-specific (MTL vs. MFC) alignment is not computed by this first
pass -- the `NeuralDataset.noise_ceiling`/`.rates` interface filters by a
single canonical region (e.g. 'hippocampus'), not a region family, so
family-level aggregation needs a small adapter-side extension left for a
later pass; only the pooled, all-regions comparison is computed here.

Results are written to `results/alignment_results.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# crossnobis needs multiple trials per condition per cross-validation fold;
# a condition with too few trials gets a fold with zero members, and
# `crossnobis_rdm` fills those undefined entries with 0.0 rather than NaN
# (see `rdm.py`), which silently injects fake zero-distance entries and
# corrupts the whole-RDM correlation. Filtering to conditions with at least
# this many trials avoids that failure mode; verified necessary in practice
# (some coarse conditions, e.g. an easy load with an error trial, occur
# only once in a session) -- see DECISIONS.md.
MIN_TRIALS_PER_CONDITION = 8


def _filter_min_trials(patterns: np.ndarray, labels: list[tuple], min_count: int) -> tuple[np.ndarray, list[tuple]]:
    counts: dict[tuple, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    keep = [i for i, l in enumerate(labels) if counts[l] >= min_count]
    return patterns[keep], [labels[i] for i in keep]


def _load_completed_runs(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    latest: dict[str, dict] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec.get("run_id", "?")] = rec
    return [r for r in latest.values() if r.get("status") == "completed"]


def _per_trial_model_patterns(df: pd.DataFrame) -> tuple[np.ndarray, list[tuple]]:
    """df: one run's activity log. Returns per-trial maintenance-epoch mean
    activity ([n_trials, n_units]) and matching coarse condition labels."""
    maintain = df[df.epoch == "maintain"]
    is_flat = maintain["h_flat"].iloc[0] is not None
    patterns, labels = [], []
    for trial_id, sub in maintain.groupby("trial_id"):
        if is_flat:
            vec = np.stack(sub["h_flat"].to_numpy()).mean(axis=0)
        else:
            vec = np.concatenate(
                [np.stack(sub["h_worker"].to_numpy()).mean(axis=0), np.stack(sub["h_manager"].to_numpy()).mean(axis=0)]
            )
        row0 = sub.iloc[0]
        patterns.append(vec)
        labels.append((int(row0.load), bool(row0.in_set), bool(row0.correct)))
    return np.array(patterns), labels


def _per_trial_neural_patterns(dandi_data, bin_ms: int) -> tuple[np.ndarray, list[tuple]]:
    """Per-trial mean maintenance-epoch firing rate, all units pooled.
    Returns ([n_trials, n_units], coarse condition labels), row order
    matching `dandi_data.trials()`."""
    rates = dandi_data.rates(None, bin_ms, ["maintain"])  # [n_units, n_trials, n_bins]
    patterns = rates.mean(axis=2).T  # [n_trials, n_units]
    trials = dandi_data.trials()
    labels = [(int(r.load), bool(r.probe_in_set), bool(r.correct)) for r in trials.itertuples()]
    return patterns, labels


def align_one_run(run_id: str, dandi_data, neural_rdm: np.ndarray, neural_labels: list[tuple],
                   ceiling_lower: float, ceiling_upper: float, force_regenerate: bool = False) -> dict:
    from brainalign_wm.analysis.rdm import crossnobis_rdm
    from brainalign_wm.analysis.rsa import compare_rdms, normalized_alignment
    from brainalign_wm.training.generate_activity_logs import generate_activity_log
    from brainalign_wm.training.logging_schema import read_log

    log_path = ROOT / "results" / "activity_logs" / f"{run_id}.parquet"
    if force_regenerate or not log_path.exists():
        log_path = generate_activity_log(run_id, dandi_data)
    df = read_log(log_path)
    model_patterns, model_labels = _per_trial_model_patterns(df)
    model_patterns, model_labels = _filter_min_trials(model_patterns, model_labels, MIN_TRIALS_PER_CONDITION)
    if len(set(model_labels)) < 3:
        return {"run_id": run_id, "status": "too_few_conditions"}

    n_folds = max(2, min(4, min(model_labels.count(l) for l in set(model_labels))))
    model_rdm, model_conds = crossnobis_rdm(model_patterns, model_labels, n_folds=n_folds)

    shared = [c for c in model_conds if c in set(neural_labels)]
    if len(shared) < 3:
        return {"run_id": run_id, "status": "insufficient_shared_conditions"}
    m_idx = [model_conds.index(c) for c in shared]
    m_sub = model_rdm[np.ix_(m_idx, m_idx)]

    neural_conds_sorted = sorted(set(neural_labels), key=str)
    n_idx = [neural_conds_sorted.index(c) for c in shared]
    n_sub = neural_rdm[np.ix_(n_idx, n_idx)]

    raw = compare_rdms(m_sub, n_sub)
    return {
        "run_id": run_id, "status": "ok", "raw_alignment": raw,
        "noise_ceiling_upper": ceiling_upper, "noise_ceiling_lower": ceiling_lower,
        "normalized_alignment": normalized_alignment(raw, ceiling_upper),
        "n_shared_conditions": len(shared), "n_model_trials": len(model_patterns),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--regenerate", action="store_true", help="force regeneration of activity logs")
    ap.add_argument(
        "--max-sessions-per-dataset", type=int, default=None,
        help="bound the number of sessions loaded per dataset; `dandi_nwb.rates()` recomputes "
             "spike histograms with no caching (unlike sim_brain's), so the full Tier A pool "
             "(~1800 units, thousands of trials) is currently impractically slow -- see DECISIONS.md.",
    )
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    completed = _load_completed_runs(ROOT / "results" / "manifest.jsonl")
    if not completed:
        print("[run_all] no completed runs found in results/manifest.jsonl; run `make run-grid` first.")
        return 1
    print(f"[run_all] {len(completed)} completed run(s) found: {[r['run_id'] for r in completed]}")

    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA
    from brainalign_wm.analysis.rdm import crossnobis_rdm

    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"], bin_ms=cfg["neural"]["bin_ms"],
        max_sessions_per_dataset=args.max_sessions_per_dataset,
    )
    print(f"[run_all] Tier A loaded: {len(dandi_data.sessions())} sessions, {len(dandi_data.units())} units, "
          f"regions={dandi_data.regions()}")

    neural_patterns, neural_labels = _per_trial_neural_patterns(dandi_data, dandi_data.bin_ms)
    neural_patterns, neural_labels = _filter_min_trials(neural_patterns, neural_labels, MIN_TRIALS_PER_CONDITION)
    n_folds = max(2, min(4, min(neural_labels.count(l) for l in set(neural_labels))))
    neural_rdm, neural_conds = crossnobis_rdm(neural_patterns, neural_labels, n_folds=n_folds)
    # `crossnobis_rdm` returns conds sorted by str(); reuse that exact ordering downstream
    ceiling_lower, ceiling_upper = dandi_data.noise_ceiling(None, "maintain")
    print(f"[run_all] neural RDM built over {len(neural_conds)} conditions, "
          f"noise ceiling lower={ceiling_lower:.3f} upper={ceiling_upper:.3f}")

    rows = []
    for rec in completed:
        run_id = rec["run_id"]
        print(f"[run_all] aligning {run_id} ...")
        result = align_one_run(run_id, dandi_data, neural_rdm, neural_labels, ceiling_lower, ceiling_upper,
                                force_regenerate=args.regenerate)
        if result.get("status") != "ok":
            print(f"[run_all]   skipped ({result.get('status')})")
            continue
        rows.append({
            "run_id": run_id, "model_id": rec["model_id"], "S": rec["S"], "M": rec["M"], "L": rec["L"],
            "seed": rec["seed"], "accuracy_load1": rec.get("accuracy", {}).get("load1"),
            "accuracy_load3": rec.get("accuracy", {}).get("load3"), **result,
        })

    if not rows:
        print("[run_all] no alignable runs (insufficient shared conditions or missing stimulus cache coverage).")
        return 1

    df = pd.DataFrame(rows)
    out_path = ROOT / "results" / "alignment_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[run_all] wrote {out_path}\n")
    print(df.to_string(index=False))

    from brainalign_wm.analysis.stats import mixed_effects_alignment

    sub = df.rename(columns={"normalized_alignment": "align_score", "accuracy_load3": "accuracy"})
    if sub["S"].nunique() < 2 or sub["M"].nunique() < 2 or sub["L"].nunique() < 2 or len(sub) < 6:
        print(f"\n[run_all] insufficient factor coverage for mixed-effects inference "
              f"(S levels={sub['S'].nunique()}, M levels={sub['M'].nunique()}, L levels={sub['L'].nunique()}, "
              f"n={len(sub)}); re-run once more cells complete.")
    else:
        res = mixed_effects_alignment(sub)
        print(f"\n[run_all] mixed-effects fit ({res['method']}):")
        for k, v in res["params"].items():
            print(f"    {k}: {v:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
