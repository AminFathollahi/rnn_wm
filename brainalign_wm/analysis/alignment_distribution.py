"""Distributional summaries of maintenance alignment."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from brainalign_wm.analysis.measurement_validation import patient_cluster_bootstrap_ci
from brainalign_wm.analysis.run_all import (
    _aggregate_maintenance,
    _maintenance_alignment_for_run,
    _maintenance_condition_fn_stratified,
    _session_has_identity_repeats,
    _valid_model_trials,
)


def _session_schema(trials) -> tuple[str, str]:
    if _session_has_identity_repeats(trials):
        return "held_items", "load_stratified_identity"
    return "held_items", "load_stratified_category"


def _relation(value: float, pooled: float) -> str:
    if not np.isfinite(value) or not np.isfinite(pooled):
        return "indeterminate"
    if np.isclose(value, 0.0):
        return "neutral"
    if np.isclose(pooled, 0.0):
        return "pooled_neutral"
    return "carries" if np.sign(value) == np.sign(pooled) else "opposes"


def _usable(row: dict) -> bool:
    return (
        row.get("status") == "ok"
        and np.isfinite(row.get("raw_alignment", np.nan))
        and np.isfinite(row.get("noise_ceiling_upper", np.nan))
    )


def session_distribution_rows(run_id: str, model_df, dataset, region) -> list[dict]:
    model_df = _valid_model_trials(model_df, dataset)
    trials = dataset.trials()
    maintain = model_df[model_df["epoch"] == "maintain"]
    hierarchical = len(maintain) > 0 and maintain["h_flat"].iloc[0] is None
    subpops = ("all", "worker", "manager") if hierarchical else ("all",)
    loads = sorted(int(load) for load in trials["load"].dropna().unique())
    rows = []
    for load in [None, *loads]:
        values = _maintenance_alignment_for_run(
            run_id, model_df, dataset, region, subpops=subpops, load_value=load,
        )
        for value in values:
            session = value["session"]
            session_trials = trials[trials["session"] == session]
            field, schema = _session_schema(session_trials)
            selected_trials = session_trials if load is None else session_trials[session_trials["load"] == load]
            selected_model = maintain[maintain["session"] == session]
            if load is not None:
                selected_model = selected_model[selected_model["load"] == load]
            condition_fn = _maintenance_condition_fn_stratified(session_trials)
            conditions = {condition_fn(row) for row in selected_trials.itertuples()}
            rows.append({
                **value,
                "analysis_level": "session",
                "epoch": "maintain",
                "method": "per_session_load_stratified",
                "load": "all" if load is None else load,
                "condition_field": field,
                "condition_schema": schema,
                "patient": value.get("patient", dataset.patient_of(session)),
                "n_trials": len(selected_trials),
                "n_model_trials": selected_model["trial_id"].nunique(),
                "n_condition_levels": len(conditions),
            })
    return rows


def _group_rows(rows: list[dict], dimensions: tuple[str, ...]):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in dimensions)].append(row)
    return sorted(groups.items(), key=lambda item: tuple(map(str, item[0])))


def _summary_row(
    rows: list[dict], dimensions: tuple[str, ...], values: tuple, pooled: float,
    pooled_count: int, n_boot: int, seed: int,
) -> dict:
    ok = [row for row in rows if _usable(row)]
    row = {
        "analysis_level": "distribution_summary",
        "epoch": "maintain",
        "method": "per_session_load_stratified",
        "crossing": "+".join(dimensions) if dimensions else "overall",
        "load": "all",
        "condition_field": "all",
        "condition_schema": "all",
        "patient": "all",
        "n_sessions_attempted": len(rows),
        "n_sessions_ok": len(ok),
        "n_patients_attempted": len({item["patient"] for item in rows}),
        "n_patients_ok": len({item["patient"] for item in ok}),
        "n_trials": sum(int(item["n_trials"]) for item in rows),
        "n_model_trials": sum(int(item["n_model_trials"]) for item in rows),
        "status": "ok" if ok else "no_usable_sessions",
        "pooled_raw_alignment": pooled,
    }
    row.update(dict(zip(dimensions, values)))
    levels = np.array([item["n_condition_levels"] for item in rows], dtype=float)
    row.update({
        "n_condition_levels_sum": int(levels.sum()),
        "n_condition_levels_min": int(levels.min()),
        "n_condition_levels_median": float(np.median(levels)),
        "n_condition_levels_max": int(levels.max()),
    })
    shared = np.array([item.get("n_shared_conditions", np.nan) for item in ok], dtype=float)
    shared = shared[np.isfinite(shared)]
    row["n_shared_conditions_sum"] = int(shared.sum()) if len(shared) else 0
    if not ok:
        row["relation_to_pooled"] = "indeterminate"
        return row
    aggregate = _aggregate_maintenance(ok)
    raw = np.array([item["raw_alignment"] for item in ok], dtype=float)
    normalized = np.array([item["normalized_alignment"] for item in ok], dtype=float)
    patients = np.array([item["patient"] for item in ok])
    low, high = patient_cluster_bootstrap_ci(raw, patients, n_boot=n_boot, seed=seed)
    row.update({
        "raw_alignment": aggregate["maintenance_signed_raw_alignment"],
        "noise_ceiling_upper": aggregate["maintenance_noise_ceiling_upper"],
        "normalized_alignment": aggregate["maintenance_normalized_alignment"],
        "raw_ci_lower": low,
        "raw_ci_upper": high,
        "raw_std": float(raw.std(ddof=1)) if len(raw) > 1 else 0.0,
        "raw_min": float(raw.min()),
        "raw_q25": float(np.percentile(raw, 25)),
        "raw_median": float(np.median(raw)),
        "raw_q75": float(np.percentile(raw, 75)),
        "raw_max": float(raw.max()),
        "session_normalized_mean": float(normalized.mean()),
        "session_normalized_std": float(normalized.std(ddof=1)) if len(normalized) > 1 else 0.0,
        "session_normalized_min": float(normalized.min()),
        "session_normalized_q25": float(np.percentile(normalized, 25)),
        "session_normalized_median": float(np.median(normalized)),
        "session_normalized_q75": float(np.percentile(normalized, 75)),
        "session_normalized_max": float(normalized.max()),
        "relation_to_pooled": _relation(float(raw.mean()), pooled),
    })
    if shared.size:
        row.update({
            "n_shared_conditions_min": int(shared.min()),
            "n_shared_conditions_median": float(np.median(shared)),
            "n_shared_conditions_max": int(shared.max()),
        })
    if all(item["load"] == "all" for item in rows) and pooled_count:
        row["raw_contribution_to_pooled"] = float(raw.sum() / pooled_count)
    return row


def summarize_distribution(rows: list[dict], n_boot: int = 2000, seed: int = 0) -> list[dict]:
    summaries = []
    subpops = sorted({row.get("subpop", "all") for row in rows})
    for subpop in subpops:
        selected = [row for row in rows if row.get("subpop", "all") == subpop]
        pooled_rows = [row for row in selected if row["load"] == "all"]
        pooled_ok = [row for row in pooled_rows if _usable(row)]
        pooled = (
            _aggregate_maintenance(pooled_ok)["maintenance_signed_raw_alignment"]
            if pooled_ok else float("nan")
        )
        specs = (
            (pooled_rows, ()),
            (pooled_rows, ("condition_field", "condition_schema")),
            (pooled_rows, ("patient",)),
            (pooled_rows, ("condition_field", "condition_schema", "patient")),
            ([row for row in selected if row["load"] != "all"], ("load",)),
            ([row for row in selected if row["load"] != "all"], ("load", "condition_field", "condition_schema")),
            ([row for row in selected if row["load"] != "all"], ("load", "patient")),
            ([row for row in selected if row["load"] != "all"], ("load", "condition_field", "condition_schema", "patient")),
        )
        for source, dimensions in specs:
            for values, group in _group_rows(source, dimensions):
                summary = _summary_row(
                    group, dimensions, values, pooled, len(pooled_ok), n_boot, seed,
                )
                summary.update({
                    "run_id": group[0]["run_id"],
                    "region": group[0]["region"],
                    "subpop": subpop,
                })
                summaries.append(summary)
    return summaries


def distribution_rows(
    run_id: str, model_df, dataset, region, n_boot: int = 2000, seed: int = 0,
) -> list[dict]:
    sessions = session_distribution_rows(run_id, model_df, dataset, region)
    return sessions + summarize_distribution(sessions, n_boot=n_boot, seed=seed)
