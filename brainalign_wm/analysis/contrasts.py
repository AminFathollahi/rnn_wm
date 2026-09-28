"""Architectural contrasts estimated at the levels at which the design replicates.

What is being estimated
-----------------------
The similarity between a network and a recording is computed separately for
every (seed, session, region, population) cell and only then averaged.  A
network's alignment is therefore the mean alignment of independently trained
networks, not the alignment of an ensemble: averaging seed-wise RDMs before
comparing them would estimate the latter, which is a different quantity.

Two units of replication are respected at once.  The training seed is the unit
for the architectural comparison: every contrast pairs a cell against its
reference seed by seed.  The patient is the unit for the neural measurement --
57 patients supply the 65 sessions and sessions from one patient are not
independent -- so every interval resamples patients, carrying all of a
patient's sessions together.

Two neural weightings of the same per-session values are reported, as two
named estimands rather than one right and one wrong answer:

* equal-session -- the plain mean over a run's sessions, which weights a
  patient by how many sessions that patient contributed;
* equal-patient -- sessions averaged within a patient first, then patients
  averaged with equal weight.

They coincide only when every patient contributed the same number of
sessions.  Each one's bootstrap resamples at the level its point estimate
weights, so interval and point estimate answer the same question.

Probe alignment is a third estimand and is never relabelled as either of the
above: it is one similarity per (seed, region, population) against a
pseudopopulation pooled across sessions, so it has no per-patient
decomposition and its interval resamples seeds alone.

Named regions stay separate throughout; no subject or run is ever assigned
its best-scoring region.  Runs trained under different supervision signals
are never pooled: every estimate is produced within one training signal.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Model ids encode the five architectural arms as bits in this order.
ARM_ORDER = ("S", "M", "P", "T", "D")
ARM_NAMES = {
    "S": "hierarchy",
    "M": "neuromodulation",
    "P": "fast_plasticity",
    "T": "topography",
    "D": "dale",
}
BASELINE_CELL = "M00000"
FULL_CELL = "M11111"

# Cells that enable exactly two arms, used to probe whether two arms combine
# additively.
TWO_ARM_CELLS = {"M10010": ("S", "T"), "M00011": ("T", "D"), "M10001": ("S", "D")}

# Normal-theory multiplier for a minimum detectable difference at 80% power
# and a two-sided 5% test: z(0.975) + z(0.80).
MDE_MULTIPLIER = 2.8024


def add_one_cell(arm: str) -> str:
    return "M" + "".join("1" if a == arm else "0" for a in ARM_ORDER)


def knock_out_cell(arm: str) -> str:
    return "M" + "".join("0" if a == arm else "1" for a in ARM_ORDER)


def training_signal(run_id: str) -> str:
    """`M10010_SUP_s3` -> `SUP`."""
    return run_id.rsplit("_", 1)[0].rsplit("_", 1)[1]


def assign_patient_folds(patients, confirmation_share: float = 1.0 / 3.0) -> dict:
    """Deterministic, content-free split of patient ids into a discovery set
    and a confirmation set.  Patients are ordered by a hash of their id and the
    first `confirmation_share` of that order is held out, so the assignment
    depends only on the ids, is reproducible from them alone, and splits the
    cohort at the requested proportion exactly rather than in expectation."""
    unique = sorted(set(map(str, patients)))
    ordered = sorted(unique, key=lambda pid: hashlib.sha256(pid.encode("utf-8")).hexdigest())
    n_confirmation = int(round(len(ordered) * confirmation_share))
    held_out = set(ordered[:n_confirmation])
    return {pid: ("confirmation" if pid in held_out else "discovery") for pid in unique}


def write_patient_folds(patients, path: Path, confirmation_share: float = 1.0 / 3.0) -> dict:
    folds = assign_patient_folds(patients, confirmation_share)
    payload = {
        "method": "patients ordered by sha256 of the patient id; first third held out",
        "confirmation_share": confirmation_share,
        "n_patients": len(folds),
        "n_discovery": sum(v == "discovery" for v in folds.values()),
        "n_confirmation": sum(v == "confirmation" for v in folds.values()),
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_patient_folds(path: Path) -> dict:
    return json.loads(Path(path).read_text())["folds"]


# What each dependent variable estimates, carried through to the output so a
# reader never has to infer it from the column name.
ESTIMANDS = {
    "maintenance_signed_alignment": (
        "mean over a run's recording sessions of the per-(seed, session, region, population) "
        "similarity; a patient is weighted by how many sessions it contributed"
    ),
    "maintenance_signed_alignment_equal_patient": (
        "sessions averaged within a patient, then patients averaged with equal weight"
    ),
    "probe_alignment": (
        "one similarity per (seed, region, population) against a pseudopopulation pooled "
        "across sessions; not a mean of per-patient similarities"
    ),
}


@dataclass
class AlignmentPanel:
    """Per-run values of one dependent variable plus the resampling structure
    needed for its interval."""

    dv: str
    cluster_unit: str
    runs: pd.DataFrame  # indexed by run_id: model_id, seed, signal, accuracy_load3, value
    session_values: np.ndarray | None = None  # [session, run] aligned to runs.index
    session_patient: np.ndarray | None = None
    estimand: str = ""
    region: str = "pooled"
    population: str = "all"
    match_units: str = "native"

    def point_values(self) -> np.ndarray:
        return self.runs["value"].to_numpy(dtype=float)

    def bootstrap_values(self, n_boot: int, rng: np.random.Generator) -> np.ndarray:
        """[n_boot, n_run] run-level values under resampling of the neural
        clustering unit.  Constant across replicates when the dependent
        variable has no per-patient decomposition."""
        if self.session_values is None:
            return np.repeat(self.point_values()[None, :], n_boot, axis=0)
        patients = np.unique(self.session_patient)
        rows_by_patient = [np.flatnonzero(self.session_patient == p) for p in patients]
        out = np.empty((n_boot, self.session_values.shape[1]), dtype=float)
        for b in range(n_boot):
            drawn = rng.integers(0, len(patients), len(patients))
            rows = np.concatenate([rows_by_patient[i] for i in drawn])
            out[b] = np.nanmean(self.session_values[rows], axis=0)
        return out


def _population_rows(
    table: pd.DataFrame, region: str, population: str, match_units: str = "native"
) -> pd.DataFrame:
    """Rows of a (run, region, subpop, width) table for one population: a
    hierarchical run (S=1) contributes its named `worker`/`manager`
    subpopulation, a flat run (S=0, one population only) always
    contributes its `subpop="all"` row -- the flat side of a
    flat-vs-population contrast, never a restriction to hierarchical runs.
    `population="all"` reproduces the original pooled, whole-vector rows.

    `match_units` picks the width the rows were scored at: `"native"` is
    each population at its own width, and a numeric string is the common
    width both sides of a contrast were subsampled to.  A population
    already at or under that width has no subsampled row, so its native
    row stands in for it -- that is what makes a common width readable
    from one selection."""
    in_region = table[table["region"] == region]
    if "subpop" not in in_region.columns:
        # A table written before populations were scored separately carries
        # the whole-vector estimate alone; it can stand in for `"all"` and
        # for nothing else.
        if population != "all":
            return in_region.iloc[:0]
        in_region = in_region.assign(subpop="all")
    if "match_target_units" not in in_region.columns:
        if match_units != "native":
            return in_region.iloc[:0]
    elif match_units == "native":
        in_region = in_region[in_region["match_target_units"].isna()]
    else:
        width = float(match_units)
        native = in_region["match_target_units"].isna()
        in_region = in_region[native | in_region["match_target_units"].eq(width)]
        key = [c for c in ("run_id", "session", "subpop") if c in in_region.columns]
        in_region = in_region.sort_values(
            "match_target_units", na_position="last", kind="stable"
        ).drop_duplicates(key, keep="first")
    if population == "all":
        return in_region[in_region["subpop"] == "all"]
    is_hier = in_region["S"].astype(int) == 1
    return in_region[(is_hier & (in_region["subpop"] == population)) | (~is_hier & (in_region["subpop"] == "all"))]


def build_panels(
    run_df: pd.DataFrame,
    session_df: pd.DataFrame | None,
    fold_patients: set | None = None,
    region: str = "pooled",
    population: str = "all",
    probe_df: pd.DataFrame | None = None,
    match_units: str = "native",
) -> dict:
    """One panel per dependent variable, for one (region, population, width)
    slice.

    Maintenance alignment is rebuilt from the per-session table so that a fold
    of patients can be selected and so that patients can be resampled; the run
    table's own column is the mean over all sessions and is used only when the
    per-session table is absent (only possible for the original pooled/"all"
    slice). Probe alignment is computed against a condition-averaged neural
    matrix and has no per-session decomposition, so its interval resamples
    seeds alone; it is read from the run table for the original pooled/"all"
    slice and from `probe_df` (one row per run x region x subpop x width)
    otherwise.

    `region`/`population`/`match_units` default to the original whole-vector
    pooled DV at native width, so every pre-existing caller is unaffected. A
    non-default slice names its dv `"{dv}__{region}__{population}"` so the two
    are never confused in output.
    """
    runs = run_df.set_index("run_id")
    base = pd.DataFrame(
        {
            "model_id": runs["model_id"],
            "seed": runs["seed"].astype(int),
            "signal": [training_signal(r) for r in runs.index],
            "accuracy_load3": runs["accuracy_load3"].astype(float),
        }
    )
    is_default_slice = region == "pooled" and population == "all" and match_units == "native"
    suffix = "" if is_default_slice else f"__{region}__{population}"

    panels = {}

    maintenance = base.copy()
    session_values = session_patient = None
    if session_df is not None and len(session_df):
        sel = _population_rows(session_df, region, population, match_units)
        if fold_patients is not None:
            sel = sel[sel["patient"].astype(str).isin(fold_patients)]
        wide = sel.pivot_table(
            index="session", columns="run_id", values="raw_alignment", aggfunc="mean"
        )
        wide = wide.reindex(columns=[r for r in base.index if r in wide.columns])
        maintenance = base.loc[wide.columns].copy()
        maintenance["value"] = wide.mean(axis=0).to_numpy(dtype=float)
        session_values = wide.to_numpy(dtype=float)
        patient_of_session = (
            sel.drop_duplicates("session").set_index("session")["patient"].astype(str)
        )
        session_patient = patient_of_session.reindex(wide.index).to_numpy()
        maintenance["n_sessions"] = wide.notna().sum(axis=0).to_numpy(dtype=int)
        patient_frame = wide.assign(_patient=session_patient).groupby("_patient", sort=True).mean()
        maintenance["n_patients"] = patient_frame.notna().sum(axis=0).to_numpy(dtype=int)
    elif is_default_slice:
        maintenance["value"] = runs["maintenance_signed_raw_alignment"].astype(float)
    slice_kwargs = dict(region=region, population=population, match_units=match_units)
    dv = f"maintenance_signed_alignment{suffix}"
    panels[dv] = AlignmentPanel(
        dv=dv,
        cluster_unit="patient" if session_values is not None else "seed",
        runs=maintenance,
        session_values=session_values,
        session_patient=session_patient,
        estimand=ESTIMANDS["maintenance_signed_alignment"],
        **slice_kwargs,
    )
    if session_values is not None:
        equal_patient = base.loc[patient_frame.columns].copy()
        equal_patient["value"] = patient_frame.mean(axis=0).to_numpy(dtype=float)
        equal_patient["n_sessions"] = maintenance.loc[equal_patient.index, "n_sessions"]
        equal_patient["n_patients"] = patient_frame.notna().sum(axis=0).to_numpy(dtype=int)
        patient_dv = f"maintenance_signed_alignment_equal_patient{suffix}"
        panels[patient_dv] = AlignmentPanel(
            dv=patient_dv,
            cluster_unit="patient",
            runs=equal_patient,
            session_values=patient_frame.to_numpy(dtype=float),
            session_patient=patient_frame.index.to_numpy(),
            estimand=ESTIMANDS["maintenance_signed_alignment_equal_patient"],
            **slice_kwargs,
        )

    probe = base.copy()
    if is_default_slice or probe_df is None:
        probe["value"] = runs["probe_raw_alignment"].astype(float)
    else:
        sel = _population_rows(probe_df[probe_df["status"] == "ok"], region, population, match_units)
        vals = sel.drop_duplicates("run_id").set_index("run_id")["raw_alignment"].astype(float)
        probe = base.loc[[r for r in base.index if r in vals.index]].copy()
        probe["value"] = vals.reindex(probe.index)
    probe = probe[np.isfinite(probe["value"])]
    dv = f"probe_alignment{suffix}"
    panels[dv] = AlignmentPanel(
        dv=dv, cluster_unit="seed", runs=probe,
        estimand=ESTIMANDS["probe_alignment"], **slice_kwargs,
    )
    return panels


def _cell_columns(panel: AlignmentPanel, cell: str, signal: str) -> pd.Series:
    rows = panel.runs
    sel = rows[(rows["model_id"] == cell) & (rows["signal"] == signal)]
    return pd.Series(np.arange(len(rows))[rows.index.get_indexer(sel.index)], index=sel["seed"].to_numpy())


STAT_FIELDS = (
    "effect", "ci_lo", "ci_hi", "bootstrap_se", "p_value", "mde",
    "residual_accuracy_load3", "max_accuracy_load3_spread", "min_accuracy_load3",
)


def paired_seed_contrast(
    panel: AlignmentPanel,
    boot_values: np.ndarray,
    cell_weights: dict,
    signal: str,
    rng: np.random.Generator,
    accuracy_tolerance: float | None = None,
) -> dict:
    """Seed-paired weighted combination of cells, e.g. {a: +1, b: -1} for a
    difference or {ab: +1, a: -1, b: -1, baseline: +1} for the amount by which
    two arms together exceed the sum of their separate effects.

    The interval is a two-level bootstrap: the neural clustering unit is
    resampled inside `boot_values` and seeds are resampled here, paired, so
    the same seed is drawn for every cell in the combination.

    Always returns a row.  A combination this training signal does not
    license -- a cell it never trained, or too few seeds shared across the
    cells -- comes back with `status` naming the reason and the estimates
    left missing, so a gap in coverage is visible in the table instead of
    being silently absent from it.
    """
    columns = {cell: _cell_columns(panel, cell, signal) for cell in cell_weights}
    coverage = {
        "n_runs_by_cell": "|".join(f"{cell}:{len(columns[cell])}" for cell in sorted(columns)),
        "hierarchical_cells": ",".join(sorted(c for c in cell_weights if is_hierarchical(c))),
        "n_seeds": 0,
        "n_sessions_in_support": 0,
        "complete_session_support": False,
    }
    blank = {field: float("nan") for field in STAT_FIELDS}
    absent = sorted(cell for cell, col in columns.items() if len(col) == 0)
    if absent:
        return {**blank, **coverage, "status": "cells_not_trained_under_signal:" + ",".join(absent)}

    seeds = sorted(set.intersection(*(set(c.index) for c in columns.values())))
    accuracy = panel.runs["accuracy_load3"].to_numpy(dtype=float)
    index = {cell: np.array([columns[cell][s] for s in seeds], dtype=int) for cell in cell_weights}
    gap = np.zeros(0)
    accuracy_by_cell = np.zeros((len(cell_weights), 0))
    if seeds:
        accuracy_by_cell = np.stack([accuracy[index[cell]] for cell in cell_weights])
        gap = accuracy_by_cell.max(axis=0) - accuracy_by_cell.min(axis=0)
        if accuracy_tolerance is not None:
            keep = gap <= accuracy_tolerance
            seeds = [s for s, k in zip(seeds, keep) if k]
            index = {cell: index[cell][keep] for cell in cell_weights}
            gap = gap[keep]
            accuracy_by_cell = accuracy_by_cell[:, keep]
    coverage["n_seeds"] = len(seeds)
    if len(seeds) < 2:
        reason = "too_few_seeds_within_accuracy_tolerance" if accuracy_tolerance is not None else "too_few_paired_seeds"
        return {**blank, **coverage, "status": reason}

    used = np.concatenate([index[cell] for cell in cell_weights])
    if panel.session_values is not None:
        # Seed and neural-observation means commute only over a table with no
        # holes, so the support the difference is actually taken over is
        # reported rather than assumed.
        available = np.isfinite(panel.session_values[:, used])
        shared_rows = available.all(axis=1)
        coverage["n_sessions_in_support"] = int(shared_rows.sum())
        coverage["complete_session_support"] = bool(available.all())
    for column in ("n_sessions", "n_patients"):
        if column in panel.runs:
            coverage[f"min_observed_{column}"] = int(panel.runs.iloc[used][column].min())

    point = panel.point_values()
    combine = lambda values: sum(w * values[..., index[cell]] for cell, w in cell_weights.items())
    effect = float(np.mean(combine(point)))
    per_seed = combine(boot_values)
    draw = rng.integers(0, len(seeds), size=per_seed.shape)
    replicates = np.take_along_axis(per_seed, draw, axis=1).mean(axis=1)
    se = float(replicates.std(ddof=1))
    above, below = float(np.mean(replicates >= 0.0)), float(np.mean(replicates <= 0.0))
    signed_accuracy = combine(accuracy)
    return {
        "status": "ok",
        "effect": effect,
        "ci_lo": float(np.percentile(replicates, 2.5)),
        "ci_hi": float(np.percentile(replicates, 97.5)),
        "bootstrap_se": se,
        "p_value": float(min(1.0, 2.0 * min(above, below))),
        "mde": MDE_MULTIPLIER * se,
        "residual_accuracy_load3": float(np.mean(signed_accuracy)),
        "max_accuracy_load3_spread": float(np.max(gap)),
        "min_accuracy_load3": float(np.mean(accuracy_by_cell.min(axis=0))),
        **coverage,
    }


def contrast_specifications() -> list:
    """(family, label, arms, cell_weights) for every contrast the battery was
    designed around: adding one arm to the baseline, removing one arm from the
    full model, and for each two-arm cell its comparison against each component
    cell plus the amount by which the pair exceeds the sum of its parts."""
    specs = []
    for arm in ARM_ORDER:
        specs.append(
            ("add_one", f"add_{ARM_NAMES[arm]}", arm, {add_one_cell(arm): 1.0, BASELINE_CELL: -1.0})
        )
    for arm in ARM_ORDER:
        specs.append(
            (
                "knock_one_out",
                f"remove_{ARM_NAMES[arm]}",
                arm,
                {FULL_CELL: 1.0, knock_out_cell(arm): -1.0},
            )
        )
    for cell, arms in TWO_ARM_CELLS.items():
        pair = "+".join(ARM_NAMES[a] for a in arms)
        label_arms = "".join(arms)
        specs.append(
            ("two_arm_vs_baseline", f"add_{pair}", label_arms, {cell: 1.0, BASELINE_CELL: -1.0})
        )
        for arm in arms:
            specs.append(
                (
                    "two_arm_vs_component",
                    f"{pair}_over_{ARM_NAMES[arm]}",
                    label_arms,
                    {cell: 1.0, add_one_cell(arm): -1.0},
                )
            )
        weights = {cell: 1.0, BASELINE_CELL: 1.0}
        for arm in arms:
            weights[add_one_cell(arm)] = -1.0
        specs.append(("two_arm_excess_over_additive", f"{pair}_excess", label_arms, weights))
    return specs


# The five arm bits are the first five characters after the "M"; a control
# variant of a cell carries a trailing tag ("M10010_w128", "M00000_idcatch")
# that names what was varied, and leaves the bits themselves untouched.
_ARM_BITS = re.compile(r"^M([01]{5})")
CORE_CELL = re.compile(r"^M[01]{5}$")


def enabled_arm_count(model_id: str) -> int:
    bits = _ARM_BITS.match(model_id)
    if bits is None:
        raise ValueError(f"{model_id!r} does not start with the five arm bits")
    return sum(int(bit) for bit in bits.group(1))


def is_hierarchical(model_id: str) -> bool:
    """Whether a cell has the worker/manager split, read from its first arm
    bit.  A contrast between two cells that both lack it has no separate
    worker or manager side: each population slice reads the same flat
    vectors, so its rows repeat the pooled ones."""
    bits = _ARM_BITS.match(model_id)
    return bits is not None and bits.group(1)[0] == "1"
