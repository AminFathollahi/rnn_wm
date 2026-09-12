#!/usr/bin/env python3
"""Neural alignment for the extended-tier variant runs (bio-plausible
ablation arms, identity-report sub-experiment, performance-matched
baselines) -- deliberately a SEPARATE script from `analysis/run_all.py`'s
main pipeline, not a relaxation of `run_all.py::
_is_ablation_or_catch_variant`'s guard.

Why separate rather than folding these runs into `run_all.py`'s existing
loop: that guard exists to keep these variants (same S/M/P/T/D bits as a
core battery cell, materially different trained representation) OUT of the
pooled S x M x P x T x D mixed-effects regression AND out of the chance-
control gate's `headline_df`-driven logic (which assumes `model_id[:-1]`
is a valid core architecture string, e.g. "M1111" from "M11111" --
"M11111_energ" from "M11111_energy" is not, and would either crash
`chance_control_check` or silently corrupt the chance-vs-trained dedup).
Relaxing the guard in `run_all.py` itself would need touching both of
those call sites (and any future one) to re-exclude variants at the
pooling step instead of the skip step -- more surface area for exactly the
kind of silent-pooling bug the guard was written to prevent. Calling
`align_one_run` directly here, against a list of (variant run_id, base
core run_id) pairs, gets the comparison DV (probe/maintenance alignment at
matched accuracy, vs. the base cell) with zero risk to the already-
validated core pipeline.

For the identity-report family this also produces the memory-demand
contrast: maintenance alignment with vs. without the delayed identity
report, per architecture, paired by seed, with a two-level bootstrap
interval (patients resampled for the neural measurement, seeds resampled
for the architectural one) and an accuracy-matched sensitivity analysis.

Usage:
  python scripts/align_extended_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from brainalign_wm.config import get_path, load_config
from brainalign_wm.analysis.run_all import _aggregate_maintenance, align_one_run

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"
OUT_PATH = RESULTS / "alignment_extended_variants.csv"
DEMAND_CONTRAST_PATH = RESULTS / "alignment_identity_demand_contrast.csv"

# Variant run_id family -> the base core cell it is compared against. Every
# variant run_id is "{base_model_id}_{family}_s{seed}"; the base run shares
# the architecture and the seed. `None` means the base cell varies with the
# variant's own S/M/P/T/D bits.
FAMILIES = {
    "energy": "M11111", "noise": "M11111", "pbwm": "M11111",
    "idcatch": None,
    "2x": "M00000", "l1": "M00000", "dropout": "M00000",
    "ei": "M00000", "dynsyn": "M00000", "lowrank": "M00000",
    "local289": None,
}
IDENTITY_FAMILY = "idcatch"
# Largest load-3 accuracy gap a seed pair may show and still count as
# behaviourally matched in the sensitivity analysis.
ACCURACY_TOLERANCE = 0.05


def _completed_runs() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    seen = {}
    with MANIFEST.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "completed":
                seen[rec["run_id"]] = rec
    return seen


def _base_run_id(base_model_id: str, rec: dict, known: dict) -> str:
    """The completed run the variant is compared against: same architecture,
    same seed, same training signal. Campaign run ids carry the signal
    (`M00000_SUP_s0`); older ids do not, so both spellings are tried and the
    signal-qualified one wins."""
    seed = rec["seed"]
    supervision = rec.get("supervision")
    candidates = [f"{base_model_id}_{supervision}_s{seed}"] if supervision else []
    candidates.append(f"{base_model_id}_s{seed}")
    for candidate in candidates:
        if candidate in known:
            return candidate
    return candidates[-1]


def _variant_runs() -> list[dict]:
    known = _completed_runs()
    out = []
    for run_id, rec in known.items():
        if "P" not in rec:
            continue  # local-learning cell (M**L, carries "L" not "P") -- not a variant of a core cell
        model_id = rec["model_id"]
        if model_id == f"M{rec['S']}{rec['M']}{rec['P']}{rec['T']}{rec['D']}":
            continue  # a plain core battery cell, not a variant
        family = model_id.split("_", 1)[1] if "_" in model_id else ""
        if family not in FAMILIES:
            print(f"[align_extended_variants]   skipping {run_id} (unrecognized family {family!r})")
            continue
        base = FAMILIES[family] or f"M{rec['S']}{rec['M']}{rec['P']}{rec['T']}{rec['D']}"
        base_rec = known.get(_base_run_id(base, rec, known), {})
        out.append({**rec, "family": family, "base_model_id": base,
                    "base_run_id": _base_run_id(base, rec, known),
                    "base_accuracy": base_rec.get("accuracy", {})})
    return out


def _session_alignment(result: dict) -> dict[str, tuple[float, str]]:
    """session -> (signed raw maintenance alignment, patient), pooled region."""
    return {
        row["session"]: (float(row["raw_alignment"]), str(row.get("patient", row["session"])))
        for row in result["maintenance"]
        if row.get("region") == "pooled" and row.get("status") == "ok"
        and row.get("subpop", "all") == "all"
    }


def demand_contrast(
    pairs: list[dict], session_alignment: dict[str, dict], n_boot: int = 2000, seed: int = 0,
) -> list[dict]:
    """Maintenance alignment with the delayed identity report minus the same
    architecture and seed without it, one row per architecture.

    The interval resamples the two units the design actually replicates over:
    patients (carrying all of a patient's sessions together, since the
    alignment of one run is a mean over recording sessions and sessions from
    one patient are not independent) and, paired, training seeds. Reported
    over all seed pairs and again over the pairs whose load-3 accuracy agrees
    within `ACCURACY_TOLERANCE`, so a difference cannot be a behavioural
    difference in disguise."""
    rows = []
    by_architecture: dict[str, list[dict]] = {}
    for pair in pairs:
        by_architecture.setdefault(pair["base_model_id"], []).append(pair)
    rng = np.random.default_rng(seed)
    for architecture, arch_pairs in sorted(by_architecture.items()):
        usable = [p for p in arch_pairs
                  if session_alignment.get(p["run_id"]) and session_alignment.get(p["base_run_id"])]
        for matched_only in (False, True):
            selected = usable
            if matched_only:
                selected = [p for p in usable if p["accuracy_gap_load3"] is not None
                            and p["accuracy_gap_load3"] <= ACCURACY_TOLERANCE]
            row = {"architecture": architecture, "matched_on_accuracy": matched_only,
                   "accuracy_tolerance": ACCURACY_TOLERANCE if matched_only else None,
                   "n_seeds": len(selected)}
            if len(selected) < 2:
                rows.append({**row, "status": "insufficient_seed_pairs"})
                continue
            sessions = sorted(set.intersection(*[
                set(session_alignment[p["run_id"]]) & set(session_alignment[p["base_run_id"]])
                for p in selected
            ]))
            if not sessions:
                rows.append({**row, "status": "no_shared_sessions"})
                continue
            with_demand = np.array([[session_alignment[p["run_id"]][s][0] for s in sessions] for p in selected])
            without_demand = np.array([[session_alignment[p["base_run_id"]][s][0] for s in sessions] for p in selected])
            patients = np.array([session_alignment[selected[0]["run_id"]][s][1] for s in sessions])
            unique_patients = np.unique(patients)
            columns_by_patient = [np.flatnonzero(patients == p) for p in unique_patients]
            replicates = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                drawn_patients = rng.integers(0, len(unique_patients), len(unique_patients))
                columns = np.concatenate([columns_by_patient[i] for i in drawn_patients])
                per_seed = with_demand[:, columns].mean(axis=1) - without_demand[:, columns].mean(axis=1)
                drawn_seeds = rng.integers(0, len(per_seed), len(per_seed))
                replicates[b] = per_seed[drawn_seeds].mean()
            per_seed_point = with_demand.mean(axis=1) - without_demand.mean(axis=1)
            effect = float(per_seed_point.mean())
            above, below = float(np.mean(replicates >= 0.0)), float(np.mean(replicates <= 0.0))
            rows.append({
                **row, "status": "ok", "n_sessions": len(sessions), "n_patients": len(unique_patients),
                "alignment_with_demand": float(with_demand.mean()),
                "alignment_without_demand": float(without_demand.mean()),
                "effect": effect,
                "ci_lo": float(np.percentile(replicates, 2.5)),
                "ci_hi": float(np.percentile(replicates, 97.5)),
                "bootstrap_se": float(replicates.std(ddof=1)),
                "p_value": float(min(1.0, 2.0 * min(above, below))),
                "max_accuracy_gap_load3": max(
                    (p["accuracy_gap_load3"] for p in selected if p["accuracy_gap_load3"] is not None),
                    default=None,
                ),
            })
    return rows


def main() -> int:
    cfg = load_config()
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"], bin_ms=cfg["neural"]["bin_ms"],
    )

    variants = _variant_runs()
    base_run_ids = sorted({v["base_run_id"] for v in variants})
    all_run_ids = base_run_ids + [v["run_id"] for v in variants]
    print(f"[align_extended_variants] {len(variants)} variant run(s), {len(base_run_ids)} base core run(s) to align")

    rows = []
    session_alignment: dict[str, dict] = {}
    for run_id in all_run_ids:
        print(f"[align_extended_variants] aligning {run_id} ...", flush=True)
        try:
            result = align_one_run(run_id, dandi_data)
        except FileNotFoundError as e:
            print(f"[align_extended_variants]   skipped ({e})")
            continue
        ok = [r for r in result["maintenance"] if r.get("region") == "pooled" and r.get("status") == "ok"]
        maint_agg = _aggregate_maintenance(ok) if ok else {}
        session_alignment[run_id] = _session_alignment(result)
        probe_pooled = next((r for r in result["probe"] if r.get("region") == "pooled"), {})
        rows.append({"run_id": run_id, **maint_agg,
                      "probe_raw_alignment": probe_pooled.get("raw_alignment"),
                      "probe_normalized_alignment": probe_pooled.get("normalized_alignment"),
                      "probe_status": probe_pooled.get("status")})

    align_by_run = {r["run_id"]: r for r in rows}
    out_rows = []
    for v in variants:
        variant_align = align_by_run.get(v["run_id"], {})
        base_align = align_by_run.get(v["base_run_id"], {})
        variant_acc3 = v.get("accuracy", {}).get("load3")
        base_acc3 = v.get("base_accuracy", {}).get("load3")
        out_rows.append({
            "run_id": v["run_id"], "family": v["family"], "base_run_id": v["base_run_id"],
            "base_model_id": v["base_model_id"],
            "seed": v["seed"], "accuracy_load1": v.get("accuracy", {}).get("load1"),
            "accuracy_load3": variant_acc3,
            "base_accuracy_load1": v.get("base_accuracy", {}).get("load1"),
            "base_accuracy_load3": base_acc3,
            "accuracy_gap_load3": (
                abs(variant_acc3 - base_acc3) if variant_acc3 is not None and base_acc3 is not None else None
            ),
            "variant_maintenance_signed_raw_alignment": variant_align.get("maintenance_signed_raw_alignment"),
            "base_maintenance_signed_raw_alignment": base_align.get("maintenance_signed_raw_alignment"),
            "variant_maintenance_normalized_alignment": variant_align.get("maintenance_normalized_alignment"),
            "base_maintenance_normalized_alignment": base_align.get("maintenance_normalized_alignment"),
            "variant_probe_normalized_alignment": variant_align.get("probe_normalized_alignment"),
            "base_probe_normalized_alignment": base_align.get("probe_normalized_alignment"),
        })
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\n[align_extended_variants] wrote {OUT_PATH} ({len(out_df)} rows)")
    if len(out_df):
        print(out_df.to_string(index=False))

    identity_pairs = [r for r in out_rows if r["family"] == IDENTITY_FAMILY]
    if identity_pairs:
        contrast_rows = demand_contrast(identity_pairs, session_alignment)
        contrast_df = pd.DataFrame(contrast_rows)
        contrast_df.to_csv(DEMAND_CONTRAST_PATH, index=False)
        print(f"\n[align_extended_variants] wrote {DEMAND_CONTRAST_PATH} ({len(contrast_df)} rows)")
        print(contrast_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
