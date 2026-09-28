#!/usr/bin/env python
"""Estimate the architectural contrasts of the alignment battery at the levels
at which the design replicates.

The training seed is the unit of replication for every architectural
comparison, and contrasts are paired by seed.  The patient is the unit for the
neural measurement, so maintenance-alignment intervals resample patients
(sessions are nested within patients).  Supervised and reinforcement runs are
estimated separately and never pooled, and every contrast is reported under
both signals: one a signal does not license keeps its row and carries the
reason in `status`.

For a hierarchical network the worker and the manager are estimated
separately against every named region and never pooled into one vector, so a
flat-against-hierarchical comparison runs twice -- flat against worker and
flat against manager -- at the width both populations were subsampled to.

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
    STAT_FIELDS,
    build_panels,
    contrast_specifications,
    enabled_arm_count,
    is_hierarchical,
    load_patient_folds,
    paired_seed_contrast,
)
from brainalign_wm.analysis.stats import fdr_correct

PRIMARY_FAMILIES = ("add_one", "knock_one_out")
# Reported for every contrast whether or not the signal trained the cells it
# needs; a signal that did not is a coverage fact, not a reason to omit a row.
TRAINING_SIGNALS = ("SUP", "RL")
POPULATIONS = ("all", "worker", "manager")
SLICE_KEYS = ["checkpoint", "fold", "region", "population", "match_units"]


def _load_checkpoint(results_dir: Path, checkpoint: str):
    suffix = "" if checkpoint == "equal_budget" else "_at_criterion"
    run_path = results_dir / f"alignment_results{suffix}.csv"
    session_path = results_dir / f"alignment_by_session{suffix}.csv"
    probe_path = results_dir / f"alignment_probe_by_region{suffix}.csv"
    if not run_path.exists():
        return None, None, None
    runs = pd.read_csv(run_path)
    sessions = pd.read_csv(session_path) if session_path.exists() else None
    probe = pd.read_csv(probe_path) if probe_path.exists() else None
    return runs, sessions, probe


def _slices(sessions: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    """(region, population, width) combinations present in a session table."""
    if sessions is None or not len(sessions):
        return [("pooled", "all", "native")]
    regions = sorted(sessions["region"].astype(str).unique())
    widths = ["native"]
    if "match_target_units" in sessions.columns:
        widths += [
            str(int(width)) for width in sorted(sessions["match_target_units"].dropna().unique())
        ]
    return [(region, population, width) for region in regions for population in POPULATIONS for width in widths]


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
        "status": "ok",
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


DV_NAMES = (
    "maintenance_signed_alignment",
    "maintenance_signed_alignment_equal_patient",
    "probe_alignment",
)


def _flagged_rows(common: dict, status: str, dvs=DV_NAMES) -> list[dict]:
    """Rows for contrasts that carry no observations, so that a gap in
    coverage is stated in the table rather than absent from it."""
    blank = {field: float("nan") for field in STAT_FIELDS}
    return [
        dict(
            common, dv=dv, training_signal=signal, family=family, contrast=label, arms=arms,
            matching="equal_budget",
            cells="|".join(f"{'+' if w > 0 else '-'}{c}" for c, w in weights.items()),
            n_runs_by_cell="", n_seeds=0, n_sessions_in_support=0,
            hierarchical_cells=",".join(sorted(c for c in weights if is_hierarchical(c))),
            complete_session_support=False, status=status, **blank,
        )
        for dv in dvs
        for signal in TRAINING_SIGNALS
        for family, label, arms, weights in contrast_specifications()
    ]


def collect_contrasts(
    results_dir: Path, folds: dict, n_boot: int, accuracy_tolerance: float, seed: int,
    regions: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    records = []
    for checkpoint in ("equal_budget", "gate_a_criterion"):
        runs, sessions, probe = _load_checkpoint(results_dir, checkpoint)
        if runs is None:
            continue
        fold_names = ("all", "discovery", "confirmation") if sessions is not None else ("all",)
        for region, population, match_units in _slices(sessions):
            if regions is not None and region not in regions:
                continue
            for fold in fold_names:
                fold_patients = None if fold == "all" else {p for p, f in folds.items() if f == fold}
                panels = build_panels(
                    runs, sessions, fold_patients, region=region, population=population,
                    probe_df=probe, match_units=match_units,
                )
                slice_common = dict(
                    checkpoint=checkpoint, fold=fold, region=region, population=population,
                    match_units=match_units, estimand="", cluster_unit="", n_patients=0,
                )
                if all(len(panel.runs) == 0 for panel in panels.values()):
                    records.extend(
                        _flagged_rows(slice_common, "population_not_measured_in_region")
                    )
                    continue
                for panel in panels.values():
                    if len(panel.runs) == 0:
                        records.extend(
                            _flagged_rows(
                                slice_common, "measure_not_available_for_population",
                                dvs=(panel.dv.split("__")[0],),
                            )
                        )
                        continue
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
                    common = dict(
                        checkpoint=checkpoint, fold=fold, region=region, population=population,
                        match_units=match_units, dv=panel.dv.split("__")[0],
                        estimand=panel.estimand, cluster_unit=panel.cluster_unit,
                        n_patients=n_patients,
                    )
                    for signal in TRAINING_SIGNALS:
                        for matching, tolerance in (
                            ("equal_budget", None),
                            (f"load3_within_{accuracy_tolerance:g}", accuracy_tolerance),
                        ):
                            for family, label, arms, weights in contrast_specifications():
                                stats = paired_seed_contrast(
                                    panel, boot_values, weights, signal, rng, tolerance
                                )
                                records.append(
                                    dict(
                                        **common,
                                        training_signal=signal,
                                        family=family,
                                        contrast=label,
                                        arms=arms,
                                        matching=matching,
                                        cells="|".join(
                                            f"{'+' if w > 0 else '-'}{c}" for c, w in weights.items()
                                        ),
                                        **stats,
                                    )
                                )
                        correlation = _arm_count_correlation(panel, boot_values, signal, rng)
                        if correlation is not None:
                            records.append(
                                dict(
                                    **common,
                                    training_signal=signal,
                                    family="exploratory_arm_count_correlation",
                                    contrast="spearman_alignment_vs_enabled_arm_count",
                                    arms="all",
                                    matching="equal_budget",
                                    cells="cell_means",
                                    **correlation,
                                )
                            )
    table = pd.DataFrame.from_records(records)
    if table.empty:
        return table
    table["fdr_reject_within_primary"] = False
    primary = table["family"].isin(PRIMARY_FAMILIES) & table["status"].eq("ok")
    keys = SLICE_KEYS + ["training_signal", "dv", "matching"]
    for _, group in table[primary].groupby(keys):
        table.loc[group.index, "fdr_reject_within_primary"] = fdr_correct(
            group["p_value"].to_numpy()
        )
    return table


def coverage_table(table: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast, with each training signal's status and seed
    count side by side, so a contrast only one signal licenses is visible."""
    keys = SLICE_KEYS + ["dv", "matching", "family", "contrast", "arms", "cells"]
    wide = table.pivot_table(
        index=keys, columns="training_signal", values=["status", "n_seeds"],
        aggfunc="first", observed=True,
    )
    wide.columns = [f"{signal}_{field}" for field, signal in wide.columns]
    wide = wide.reset_index()
    status_columns = [f"{signal}_status" for signal in TRAINING_SIGNALS if f"{signal}_status" in wide]
    wide["licensed_signals"] = wide[status_columns].eq("ok").sum(axis=1)
    return wide


def estimand_comparison(table: pd.DataFrame) -> pd.DataFrame:
    """Equal-session against equal-patient weighting of the same per-session
    values, contrast by contrast, with whether the change moves the sign or
    the exclusion of zero from the interval."""
    keys = SLICE_KEYS + ["training_signal", "matching", "family", "contrast", "arms", "cells"]
    pair = table[
        table["dv"].isin(
            ["maintenance_signed_alignment", "maintenance_signed_alignment_equal_patient"]
        )
    ]
    wide = pair.pivot_table(
        index=keys, columns="dv", values=["effect", "ci_lo", "ci_hi", "status", "n_seeds"],
        aggfunc="first", observed=True,
    )
    short = {"maintenance_signed_alignment": "equal_session",
             "maintenance_signed_alignment_equal_patient": "equal_patient"}
    wide.columns = [f"{short[dv]}_{field}" for field, dv in wide.columns]
    wide = wide.reset_index()
    both_ok = wide.get("equal_session_status", pd.Series(dtype=object)).eq("ok") & wide.get(
        "equal_patient_status", pd.Series(dtype=object)
    ).eq("ok")
    wide["effect_difference"] = wide["equal_patient_effect"] - wide["equal_session_effect"]
    excludes = lambda lo, hi: (lo > 0) | (hi < 0)
    session_excludes = excludes(wide["equal_session_ci_lo"], wide["equal_session_ci_hi"])
    patient_excludes = excludes(wide["equal_patient_ci_lo"], wide["equal_patient_ci_hi"])
    wide["sign_flips"] = both_ok & (
        np.sign(wide["equal_session_effect"]) != np.sign(wide["equal_patient_effect"])
    )
    wide["zero_exclusion_flips"] = both_ok & (session_excludes != patient_excludes)
    return wide[both_ok | wide["equal_session_status"].notna()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--patient-folds", type=Path, default=Path("configs/patient_folds.json"))
    parser.add_argument("--out", type=Path, default=Path("results/alignment_contrasts.csv"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--regions", nargs="+", default=None,
        help="limit to these region labels (default: every region in the session table)",
    )
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
        args.results_dir, folds, args.n_boot, args.accuracy_tolerance, args.seed,
        tuple(args.regions) if args.regions else None,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {len(table)} rows to {args.out}")

    coverage_path = args.out.with_name(f"{args.out.stem}_coverage.csv")
    coverage = coverage_table(table)
    coverage.to_csv(coverage_path, index=False)
    print(f"wrote {len(coverage)} rows to {coverage_path}")

    estimand_path = args.out.with_name(f"{args.out.stem}_estimand_comparison.csv")
    comparison = estimand_comparison(table)
    comparison.to_csv(estimand_path, index=False)
    print(f"wrote {len(comparison)} rows to {estimand_path}")
    licensed = comparison[comparison["equal_session_status"].eq("ok")]
    print(
        f"\nequal-session vs equal-patient weighting over {len(licensed)} licensed contrasts: "
        f"{int(licensed['sign_flips'].sum())} sign flip(s), "
        f"{int(licensed['zero_exclusion_flips'].sum())} change(s) in whether the interval excludes zero, "
        f"max |difference| {licensed['effect_difference'].abs().max():.6f}"
    )

    headline = table[
        (table["checkpoint"] == "equal_budget")
        & (table["fold"] == "discovery")
        & (table["dv"] == "maintenance_signed_alignment")
        & (table["matching"] == "equal_budget")
        & table["family"].isin(PRIMARY_FAMILIES)
        & table["status"].eq("ok")
    ]
    for (region, population, signal), group in headline.groupby(
        ["region", "population", "training_signal"]
    ):
        print(f"\nmaintenance alignment, {region}, {population} population, {signal}, discovery patients")
        print(
            group[
                ["match_units", "contrast", "effect", "ci_lo", "ci_hi", "n_seeds", "mde"]
            ].to_string(index=False, float_format=lambda v: f"{v: .4f}")
        )


if __name__ == "__main__":
    main()
