#!/usr/bin/env python
"""Estimate the architectural contrasts of the alignment battery at the levels
at which the design replicates.

The training seed is the unit of replication for every architectural
comparison, and contrasts are paired by seed.  The patient is the unit for the
neural measurement, so maintenance-alignment intervals resample patients
(sessions are nested within patients).  Supervised and reinforcement runs are
estimated separately and never pooled.

Primary estimates are the equal-budget checkpoint on the discovery patients.
The confirmation patients are evaluated once, after the split was frozen and
committed, and are never used to reselect a contrast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brainalign_wm.analysis.contrasts import (
    CORE_CELL,
    build_panels,
    contrast_specifications,
    enabled_arm_count,
    load_patient_folds,
    paired_seed_contrast,
)
from brainalign_wm.analysis.stats import fdr_correct

PRIMARY_FAMILIES = ("add_one", "knock_one_out")


def _load_checkpoint(results_dir: Path, checkpoint: str):
    suffix = "" if checkpoint == "equal_budget" else "_at_criterion"
    run_path = results_dir / f"alignment_results{suffix}.csv"
    session_path = results_dir / f"alignment_by_session{suffix}.csv"
    if not run_path.exists():
        return None, None
    runs = pd.read_csv(run_path)
    sessions = pd.read_csv(session_path) if session_path.exists() else None
    return runs, sessions


def _arm_count_correlation(panel, boot_values, signal, rng) -> dict | None:
    """Exploratory only: rank correlation between a cell's alignment and how
    many arms it enables.  The number of enabled arms is not a scale -- the
    five arms are different interventions, not equal units -- so this is a
    description of the battery, not a per-arm effect."""
    from scipy.stats import spearmanr

    rows = panel.runs
    sel = rows[rows["signal"] == signal]
    # Only the battery's own cells carry an arm count. A control variant of a
    # cell (tagged model_id) and the local-learning family are separate
    # designs, not extra points on this description of the battery.
    cells = sorted(c for c in sel["model_id"].unique() if CORE_CELL.match(c))
    if len(cells) < 4:
        return None
    positions = np.arange(len(rows))
    columns = [positions[rows.index.get_indexer(sel[sel["model_id"] == c].index)] for c in cells]
    counts = np.array([enabled_arm_count(c) for c in cells], dtype=float)
    point = panel.point_values()
    rho = float(spearmanr(counts, [point[c].mean() for c in columns]).statistic)
    replicates = np.empty(boot_values.shape[0])
    for b in range(boot_values.shape[0]):
        means = [boot_values[b, c][rng.integers(0, len(c), len(c))].mean() for c in columns]
        replicates[b] = spearmanr(counts, means).statistic
    return {
        "effect": rho,
        "ci_lo": float(np.percentile(replicates, 2.5)),
        "ci_hi": float(np.percentile(replicates, 97.5)),
        "bootstrap_se": float(replicates.std(ddof=1)),
        "p_value": float(np.nan),
        "n_seeds": len(cells),
        "mde": float(np.nan),
        "residual_accuracy_load3": float(np.nan),
        "max_accuracy_load3_spread": float(np.nan),
    }


def collect_contrasts(
    results_dir: Path, folds: dict, n_boot: int, accuracy_tolerance: float, seed: int
) -> pd.DataFrame:
    records = []
    for checkpoint in ("equal_budget", "gate_a_criterion"):
        runs, sessions = _load_checkpoint(results_dir, checkpoint)
        if runs is None:
            continue
        fold_names = ("all", "discovery", "confirmation") if sessions is not None else ("all",)
        for fold in fold_names:
            fold_patients = None if fold == "all" else {p for p, f in folds.items() if f == fold}
            panels = build_panels(runs, sessions, fold_patients)
            for panel in panels.values():
                if panel.session_values is None and fold != "all":
                    # No per-patient decomposition of this measure, so a fold of
                    # patients would only duplicate the whole-cohort rows.
                    continue
                rng = np.random.default_rng(seed)
                boot_values = panel.bootstrap_values(n_boot, rng)
                n_patients = (
                    len(np.unique(panel.session_patient))
                    if panel.session_patient is not None
                    else 0
                )
                for signal in sorted(panel.runs["signal"].unique()):
                    for matching, tolerance in (
                        ("equal_budget", None),
                        (f"load3_within_{accuracy_tolerance:g}", accuracy_tolerance),
                    ):
                        for family, label, arms, weights in contrast_specifications():
                            stats = paired_seed_contrast(
                                panel, boot_values, weights, signal, rng, tolerance
                            )
                            if stats is None:
                                continue
                            records.append(
                                dict(
                                    checkpoint=checkpoint,
                                    fold=fold,
                                    training_signal=signal,
                                    dv=panel.dv,
                                    family=family,
                                    contrast=label,
                                    arms=arms,
                                    matching=matching,
                                    cells="|".join(
                                        f"{'+' if w > 0 else '-'}{c}" for c, w in weights.items()
                                    ),
                                    cluster_unit=panel.cluster_unit,
                                    n_patients=n_patients,
                                    **stats,
                                )
                            )
                    correlation = _arm_count_correlation(panel, boot_values, signal, rng)
                    if correlation is not None:
                        records.append(
                            dict(
                                checkpoint=checkpoint,
                                fold=fold,
                                training_signal=signal,
                                dv=panel.dv,
                                family="exploratory_arm_count_correlation",
                                contrast="spearman_alignment_vs_enabled_arm_count",
                                arms="all",
                                matching="equal_budget",
                                cells="cell_means",
                                cluster_unit=panel.cluster_unit,
                                n_patients=n_patients,
                                **correlation,
                            )
                        )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table["fdr_reject_within_primary"] = False
    primary = table["family"].isin(PRIMARY_FAMILIES)
    keys = ["checkpoint", "fold", "training_signal", "dv", "matching"]
    for _, group in table[primary].groupby(keys):
        table.loc[group.index, "fdr_reject_within_primary"] = fdr_correct(
            group["p_value"].to_numpy()
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--patient-folds", type=Path, default=Path("configs/patient_folds.json"))
    parser.add_argument("--out", type=Path, default=Path("results/alignment_contrasts.csv"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--accuracy-tolerance",
        type=float,
        default=0.05,
        help="maximum load-3 accuracy spread across the cells of a seed-paired contrast "
        "for that seed to enter the matched-performance sensitivity analysis",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    folds = load_patient_folds(args.patient_folds)
    table = collect_contrasts(
        args.results_dir, folds, args.n_boot, args.accuracy_tolerance, args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows to {args.out}")

    headline = table[
        (table["checkpoint"] == "equal_budget")
        & (table["fold"] == "discovery")
        & (table["dv"] == "maintenance_signed_alignment")
        & (table["matching"] == "equal_budget")
        & table["family"].isin(PRIMARY_FAMILIES)
    ]
    for signal, group in headline.groupby("training_signal"):
        print(f"\nmaintenance alignment, {signal}, discovery patients, equal budget")
        print(
            group[
                ["contrast", "effect", "ci_lo", "ci_hi", "n_seeds", "mde", "residual_accuracy_load3"]
            ].to_string(index=False, float_format=lambda v: f"{v: .4f}")
        )


if __name__ == "__main__":
    main()
