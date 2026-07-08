#!/usr/bin/env python3
"""Neural alignment for the Extended-tier variant runs (bio-plausible
ablation battery, item 5; identity-catch sub-experiment, item 6;
performance-matched baselines, item 4.1) -- deliberately a SEPARATE
script from `analysis/run_all.py`'s main pipeline, not a relaxation of
`run_all.py::_is_ablation_or_catch_variant`'s guard.

Why separate rather than folding these runs into `run_all.py`'s existing
loop: that guard exists to keep these variants (same S/M/P/T/D bits as a
Core ablation-battery cell, materially different trained representation)
OUT of the pooled S x M x P x T x D mixed-effects regression AND out of
the H5 chance-control gate's `headline_df`-driven logic (which assumes
`model_id[:-1]` is a valid Core architecture string, e.g. "M1111" from
"M11111" -- "M11111_energ" from "M11111_energy" is not, and would either
crash `chance_control_check` or silently corrupt the chance-vs-trained
dedup). Relaxing the guard in
`run_all.py` itself would need touching both of those call sites (and any
future one) to re-exclude variants at the pooling step instead of the
skip step -- more surface area for exactly the kind of silent-pooling bug
the guard was written to prevent. Calling `align_one_run` directly here,
against a hand-built list of (variant run_id, base Core run_id) pairs,
gets item 5/6's actual comparison DV (probe/maintenance alignment at
matched accuracy, vs. the base cell) with zero risk to the already-
validated Core pipeline.

Usage:
  python scripts/align_extended_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from brainalign_wm.analysis.run_all import ROOT, _aggregate_maintenance, align_one_run

MANIFEST = ROOT / "results" / "manifest.jsonl"
OUT_PATH = ROOT / "results" / "alignment_extended_variants.csv"

# (variant run_id prefix family, base Core run_id it should be compared
# against) -- every ablation/idcatch/perf-baseline run_id is
# "{base_model_id}_{suffix}_s{seed}"; the base Core run_id it's compared
# against shares the same seed. Re-anchored to the 5-arm ablation
# battery's full-reference (M11111) / baseline (M00000) cells, v6.0 pivot.
FAMILIES = {
    "energy": "M11111", "noise": "M11111", "pbwm": "M11111",   # item 5 (§4.4)
    "idcatch": None,                                             # item 6 (§9.4a): base varies (M00000 or M11111)
    "2x": "M00000", "l1": "M00000", "dropout": "M00000",         # item 4.1
}


def _variant_runs() -> list[dict]:
    if not MANIFEST.exists():
        return []
    seen = {}
    with MANIFEST.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == "completed":
                seen[rec["run_id"]] = rec
    out = []
    for run_id, rec in seen.items():
        if "P" not in rec:
            continue  # local-learning cell (M**L, carries "L" not "P") -- not a variant of a Core cell
        model_id = rec["model_id"]
        if model_id == f"M{rec['S']}{rec['M']}{rec['P']}{rec['T']}{rec['D']}":
            continue  # a plain Core ablation-battery cell, not a variant
        suffix = model_id.split("_", 1)[1] if "_" in model_id else ""
        base = FAMILIES.get(suffix)
        if base is None and suffix == "idcatch":
            base = f"M{rec['S']}{rec['M']}{rec['P']}{rec['T']}{rec['D']}"  # M00000_idcatch -> M00000, M11111_idcatch -> M11111
        if base is None:
            print(f"[align_extended_variants]   skipping {run_id} (unrecognized suffix {suffix!r})")
            continue
        out.append({**rec, "family": suffix, "base_model_id": base, "base_run_id": f"{base}_s{rec['seed']}"})
    return out


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA

    dandi_data = DandiSternbergTierA(
        cfg["paths"]["data_root"], datasets=tuple(cfg["neural"]["datasets_tierA"]),
        min_firing_hz=cfg["neural"]["min_firing_hz"],
        min_isolation_distance=cfg["neural"].get("min_isolation_distance", 20.0),
        bin_ms=cfg["neural"]["bin_ms"],
    )

    variants = _variant_runs()
    base_run_ids = sorted({v["base_run_id"] for v in variants})
    all_run_ids = base_run_ids + [v["run_id"] for v in variants]
    print(f"[align_extended_variants] {len(variants)} variant run(s), {len(base_run_ids)} base Core run(s) to align")

    rows = []
    for run_id in all_run_ids:
        print(f"[align_extended_variants] aligning {run_id} ...", flush=True)
        try:
            result = align_one_run(run_id, dandi_data)
        except FileNotFoundError as e:
            print(f"[align_extended_variants]   skipped ({e})")
            continue
        ok = [r for r in result["maintenance"] if r.get("region") == "pooled" and r.get("status") == "ok"]
        maint_agg = _aggregate_maintenance(ok) if ok else {}
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
        out_rows.append({
            "run_id": v["run_id"], "family": v["family"], "base_run_id": v["base_run_id"],
            "seed": v["seed"], "accuracy_load1": v.get("accuracy", {}).get("load1"),
            "accuracy_load3": v.get("accuracy", {}).get("load3"),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
