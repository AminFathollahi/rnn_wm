"""Architectural contrasts estimated at the levels at which the design replicates.

Two units of replication are respected at once.  The training seed is the unit
for the architectural comparison: every contrast pairs a cell against its
reference seed by seed.  The patient is the unit for the neural measurement:
the maintenance alignment of a run is a mean over recording sessions, 57
patients supply the 65 sessions, and sessions from one patient are not
independent, so every interval resamples patients (carrying all of a patient's
sessions together) rather than sessions.

Runs trained under different supervision signals are never pooled: every
estimate is produced within one training signal.
"""
from __future__ import annotations

import hashlib
import json
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
    and a confirmation set.  The assignment depends only on the patient id, so
    it is reproducible from the ids alone and cannot drift with the data."""
    folds = {}
    scale = float(1 << 32)
    for patient in sorted(set(map(str, patients))):
        digest = hashlib.sha256(patient.encode("utf-8")).digest()
        unit = int.from_bytes(digest[:4], "big") / scale
        folds[patient] = "confirmation" if unit < confirmation_share else "discovery"
    return folds


def write_patient_folds(patients, path: Path, confirmation_share: float = 1.0 / 3.0) -> dict:
    folds = assign_patient_folds(patients, confirmation_share)
    payload = {
        "method": "sha256 of the patient id, first 4 bytes as a fraction of 2**32",
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


@dataclass
class AlignmentPanel:
    """Per-run values of one dependent variable plus the resampling structure
    needed for its interval."""

    dv: str
    cluster_unit: str
    runs: pd.DataFrame  # indexed by run_id: model_id, seed, signal, accuracy_load3, value
    session_values: np.ndarray | None = None  # [session, run] aligned to runs.index
    session_patient: np.ndarray | None = None

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
            out[b] = self.session_values[rows].mean(axis=0)
        return out


def build_panels(
    run_df: pd.DataFrame,
    session_df: pd.DataFrame | None,
    fold_patients: set | None = None,
) -> dict:
    """One panel per dependent variable.

    Maintenance alignment is rebuilt from the per-session table so that a fold
    of patients can be selected and so that patients can be resampled; the run
    table's own column is the mean over all sessions and is used only when the
    per-session table is absent.  Probe alignment is computed against a
    condition-averaged neural matrix and has no per-session decomposition, so
    its interval resamples seeds alone.
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

    panels = {}

    maintenance = base.copy()
    session_values = session_patient = None
    if session_df is not None and len(session_df):
        pooled = session_df[(session_df["region"] == "pooled") & (session_df["subpop"] == "all")]
        if fold_patients is not None:
            pooled = pooled[pooled["patient"].astype(str).isin(fold_patients)]
        wide = pooled.pivot_table(
            index="session", columns="run_id", values="raw_alignment", aggfunc="mean"
        )
        wide = wide.reindex(columns=[r for r in base.index if r in wide.columns])
        maintenance = base.loc[wide.columns].copy()
        maintenance["value"] = wide.mean(axis=0).to_numpy(dtype=float)
        session_values = wide.to_numpy(dtype=float)
        patient_of_session = (
            pooled.drop_duplicates("session").set_index("session")["patient"].astype(str)
        )
        session_patient = patient_of_session.reindex(wide.index).to_numpy()
    else:
        maintenance["value"] = runs["maintenance_signed_raw_alignment"].astype(float)
    panels["maintenance_signed_alignment"] = AlignmentPanel(
        dv="maintenance_signed_alignment",
        cluster_unit="patient" if session_values is not None else "seed",
        runs=maintenance,
        session_values=session_values,
        session_patient=session_patient,
    )

    probe = base.copy()
    probe["value"] = runs["probe_raw_alignment"].astype(float)
    probe = probe[np.isfinite(probe["value"])]
    panels["probe_alignment"] = AlignmentPanel(
        dv="probe_alignment", cluster_unit="seed", runs=probe
    )
    return panels


def _cell_columns(panel: AlignmentPanel, cell: str, signal: str) -> pd.Series:
    rows = panel.runs
    sel = rows[(rows["model_id"] == cell) & (rows["signal"] == signal)]
    return pd.Series(np.arange(len(rows))[rows.index.get_indexer(sel.index)], index=sel["seed"].to_numpy())


def paired_seed_contrast(
    panel: AlignmentPanel,
    boot_values: np.ndarray,
    cell_weights: dict,
    signal: str,
    rng: np.random.Generator,
    accuracy_tolerance: float | None = None,
) -> dict | None:
    """Seed-paired weighted combination of cells, e.g. {a: +1, b: -1} for a
    difference or {ab: +1, a: -1, b: -1, baseline: +1} for the amount by which
    two arms together exceed the sum of their separate effects.

    The interval is a two-level bootstrap: patients are resampled inside
    `boot_values` (the neural unit) and seeds are resampled here, paired, so
    the same seed is drawn for every cell in the combination.
    """
    columns = {cell: _cell_columns(panel, cell, signal) for cell in cell_weights}
    seeds = sorted(set.intersection(*(set(c.index) for c in columns.values())))
    accuracy = panel.runs["accuracy_load3"].to_numpy(dtype=float)
    index = {cell: np.array([columns[cell][s] for s in seeds], dtype=int) for cell in cell_weights}
    if seeds:
        accuracy_by_cell = np.stack([accuracy[index[cell]] for cell in cell_weights])
        gap = accuracy_by_cell.max(axis=0) - accuracy_by_cell.min(axis=0)
        if accuracy_tolerance is not None:
            keep = gap <= accuracy_tolerance
            seeds = [s for s, k in zip(seeds, keep) if k]
            index = {cell: index[cell][keep] for cell in cell_weights}
            gap = gap[keep]
    if len(seeds) < 2:
        return None
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
        "effect": effect,
        "ci_lo": float(np.percentile(replicates, 2.5)),
        "ci_hi": float(np.percentile(replicates, 97.5)),
        "bootstrap_se": se,
        "p_value": float(min(1.0, 2.0 * min(above, below))),
        "n_seeds": len(seeds),
        "mde": MDE_MULTIPLIER * se,
        "residual_accuracy_load3": float(np.mean(signed_accuracy)),
        "max_accuracy_load3_spread": float(np.max(gap)),
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


def enabled_arm_count(model_id: str) -> int:
    return sum(int(bit) for bit in model_id[1:])
