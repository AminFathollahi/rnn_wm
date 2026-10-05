"""Variable-duration maintenance rates."""
from __future__ import annotations

from math import ceil

import numpy as np


def window_rate(spikes: np.ndarray, start: float, stop: float, bin_ms: int = 50) -> tuple[float, int]:
    if bin_ms <= 0:
        raise ValueError("bin_ms must be positive")
    duration = stop - start
    if not np.isfinite(duration) or duration <= 0:
        return float("nan"), 0
    bin_s = bin_ms / 1000
    edges = start + np.arange(ceil(duration / bin_s) + 1) * bin_s
    edges[-1] = stop
    left = np.searchsorted(spikes, edges[:-1], side="left")
    right = np.searchsorted(spikes, edges[1:], side="left")
    return float((right - left).sum() / duration), len(edges) - 1


class FullDelayView:
    def __init__(self, dataset):
        self.dataset = dataset
        self.bin_ms = dataset.bin_ms
        self._cache = {}
        self._all_cache = {}

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def rates(self, region, bin_ms: int, epochs: list[str]) -> np.ndarray:
        if epochs != ["maintain"]:
            return self.dataset.rates(region, bin_ms, epochs)
        key = (region, bin_ms)
        if key not in self._cache:
            self._cache[key] = self._maintenance_rates(region, bin_ms)
        return self._cache[key]

    def _maintenance_rates(self, region, bin_ms: int) -> np.ndarray:
        if bin_ms not in self._all_cache:
            units = self.dataset.units()
            self._all_cache[bin_ms] = (units, self._rates_for_units(units, bin_ms))
        all_units, rates = self._all_cache[bin_ms]
        if region is None:
            return rates
        index = {unit: position for position, unit in enumerate(all_units)}
        return rates[[index[unit] for unit in self.dataset.units(region)]]

    def _rates_for_units(self, units, bin_ms: int) -> np.ndarray:
        from brainalign_wm.analysis.pseudopopulation import _unit_session

        trials = self.dataset.trials()
        out = np.zeros((len(units), len(trials), 1), dtype=float)
        sessions = trials["session"].to_numpy()
        starts = trials["t_maintain"].to_numpy(dtype=float)
        stops = trials["t_probe"].to_numpy(dtype=float)
        for unit_index, unit in enumerate(units):
            session = _unit_session(unit)
            # A handful of recorded units store their spike times out of
            # order; `window_rate` locates the window edges by binary
            # search, which silently miscounts unless the times ascend.
            spikes = np.sort(np.asarray(self.dataset.spike_times(unit), dtype=float))
            for trial_index in np.flatnonzero(sessions == session):
                out[unit_index, trial_index, 0] = window_rate(
                    spikes, starts[trial_index], stops[trial_index], bin_ms,
                )[0]
        return out

    def noise_ceiling(self, region, epoch: str) -> tuple[float, float]:
        from brainalign_wm.analysis.rsa import noise_ceiling_from_dataset

        return noise_ceiling_from_dataset(self, region, epoch)


def window_qc_rows(dataset, bin_ms: int) -> list[dict]:
    trials = dataset.trials().copy()
    trials["duration_s"] = trials["t_probe"] - trials["t_maintain"]
    trials["n_bins"] = np.ceil(trials["duration_s"] / (bin_ms / 1000)).astype(int)
    excluded = dataset.excluded_trials() if hasattr(dataset, "excluded_trials") else trials.iloc[0:0]
    loads = sorted(set(trials["load"]) | set(excluded.get("load", [])))
    rows = []
    for load in [None, *loads]:
        selected = trials if load is None else trials[trials["load"] == load]
        rejected = excluded if load is None or "load" not in excluded else excluded[excluded["load"] == load]
        durations = selected["duration_s"].to_numpy(dtype=float)
        bins = selected["n_bins"].to_numpy(dtype=float)
        row = {
            "analysis_level": "window_qc",
            "load": "all" if load is None else int(load),
            "status": "ok" if len(selected) else "no_valid_trials",
            "n_trials": len(selected),
            "n_excluded": len(rejected),
        }
        if len(rejected) and "exclusion_reason" in rejected:
            counts = rejected["exclusion_reason"].value_counts().sort_index()
            row["excluded_by_reason"] = ";".join(f"{name}:{count}" for name, count in counts.items())
        if len(selected):
            row["duration_s_mean"] = float(durations.mean())
            row["n_bins_mean"] = float(bins.mean())
            for name, value in zip(
                ("min", "q25", "median", "q75", "max"),
                np.percentile(durations, (0, 25, 50, 75, 100)),
            ):
                row[f"duration_s_{name}"] = float(value)
            for name, value in zip(
                ("min", "q25", "median", "q75", "max"),
                np.percentile(bins, (0, 25, 50, 75, 100)),
            ):
                row[f"n_bins_{name}"] = float(value)
        rows.append(row)
    return rows


def comparison_rows(
    run_id: str, model_df, dataset, region, n_boot: int = 2000, seed: int = 0, full_dataset=None,
) -> list[dict]:
    from brainalign_wm.analysis.run_all import (
        _aggregate_maintenance,
        _maintenance_alignment_for_run,
        _valid_model_trials,
    )

    def paired_difference(sample: list[dict], name: str) -> float:
        values = {}
        for source in ("fixed", "full_delay"):
            rows = [
                {
                    "raw_alignment": row[f"{source}_raw_alignment"],
                    "noise_ceiling_upper": row[f"{source}_noise_ceiling_upper"],
                }
                for row in sample
            ]
            values[source] = _aggregate_maintenance(rows)[f"maintenance_{name}"]
        return values["full_delay"] - values["fixed"]

    def paired_ci(sample: list[dict], name: str) -> tuple[float, float]:
        patients = sorted({row["patient"] for row in sample})
        grouped = {patient: [row for row in sample if row["patient"] == patient] for patient in patients}
        rng = np.random.default_rng(seed)
        draws = np.empty(n_boot)
        for index in range(n_boot):
            selected = rng.choice(patients, size=len(patients), replace=True)
            draws[index] = paired_difference([row for patient in selected for row in grouped[patient]], name)
        return tuple(np.percentile(draws, (2.5, 97.5)))

    model_df = _valid_model_trials(model_df, dataset)
    maintain = model_df[model_df["epoch"] == "maintain"]
    hierarchical = len(maintain) > 0 and maintain["h_flat"].iloc[0] is None
    subpops = ("all", "worker", "manager") if hierarchical else ("all",)
    fixed = _maintenance_alignment_for_run(run_id, model_df, dataset, region, subpops=subpops)
    full = _maintenance_alignment_for_run(
        run_id, model_df, full_dataset or FullDelayView(dataset), region, subpops=subpops,
    )
    fixed_by = {(row.get("session"), row.get("subpop", "all")): row for row in fixed}
    full_by = {(row.get("session"), row.get("subpop", "all")): row for row in full}
    rows = []
    for session, subpop in sorted(set(fixed_by) | set(full_by)):
        fixed_row = fixed_by.get((session, subpop), {})
        full_row = full_by.get((session, subpop), {})
        paired = fixed_row.get("status") == full_row.get("status") == "ok"
        row = {
            "analysis_level": "session",
            "run_id": run_id,
            "session": session,
            "patient": fixed_row.get("patient", full_row.get("patient")),
            "region": region or "pooled",
            "subpop": subpop,
            "status": "ok" if paired else "unpaired",
            "fixed_status": fixed_row.get("status", "missing"),
            "full_delay_status": full_row.get("status", "missing"),
        }
        for source, values in (("fixed", fixed_row), ("full_delay", full_row)):
            for name in (
                "raw_alignment", "normalized_alignment", "noise_ceiling_lower",
                "noise_ceiling_upper", "n_shared_conditions",
            ):
                if name in values:
                    row[f"{source}_{name}"] = values[name]
        if paired:
            row["raw_difference"] = full_row["raw_alignment"] - fixed_row["raw_alignment"]
            row["normalized_difference"] = full_row["normalized_alignment"] - fixed_row["normalized_alignment"]
        rows.append(row)

    for subpop in subpops:
        selected = [row for row in rows if row["subpop"] == subpop]
        paired = [row for row in rows if row["subpop"] == subpop and row["status"] == "ok"]
        summary = {
            "analysis_level": "run",
            "run_id": run_id,
            "region": region or "pooled",
            "subpop": subpop,
            "status": "ok" if paired else "no_paired_sessions",
            "n_sessions": len(paired),
            "n_sessions_fixed": sum(row["fixed_status"] == "ok" for row in selected),
            "n_sessions_full_delay": sum(row["full_delay_status"] == "ok" for row in selected),
            "n_sessions_paired": len(paired),
            "n_patients": len({row["patient"] for row in paired}),
        }
        if paired:
            fixed_rows = [
                {
                    "raw_alignment": row["fixed_raw_alignment"],
                    "noise_ceiling_upper": row["fixed_noise_ceiling_upper"],
                }
                for row in paired
            ]
            full_rows = [
                {
                    "raw_alignment": row["full_delay_raw_alignment"],
                    "noise_ceiling_upper": row["full_delay_noise_ceiling_upper"],
                }
                for row in paired
            ]
            fixed_agg = _aggregate_maintenance(fixed_rows)
            full_agg = _aggregate_maintenance(full_rows)
            for name in ("raw_alignment", "noise_ceiling_upper", "normalized_alignment"):
                summary[f"fixed_{name}"] = fixed_agg[f"maintenance_{name}"]
                summary[f"full_delay_{name}"] = full_agg[f"maintenance_{name}"]
            for source in ("fixed", "full_delay"):
                conditions = np.array([row[f"{source}_n_shared_conditions"] for row in paired], dtype=float)
                summary[f"{source}_n_shared_conditions_min"] = int(conditions.min())
                summary[f"{source}_n_shared_conditions_median"] = float(np.median(conditions))
                summary[f"{source}_n_shared_conditions_max"] = int(conditions.max())
            summary["raw_difference"] = summary["full_delay_raw_alignment"] - summary["fixed_raw_alignment"]
            summary["normalized_difference"] = (
                summary["full_delay_normalized_alignment"] - summary["fixed_normalized_alignment"]
            )
            for name in ("raw_difference", "normalized_difference"):
                metric = name.removesuffix("_difference") + "_alignment"
                low, high = paired_ci(paired, metric)
                summary[f"{name}_ci_lower"] = low
                summary[f"{name}_ci_upper"] = high
        rows.append(summary)
    return rows
