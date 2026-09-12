#!/usr/bin/env python3
"""Estimate paired effects in the flat width and connectivity study."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from brainalign_wm.config import get_path

RESULTS = get_path("results")
DETAIL_PATH = RESULTS / "alignment_extended_variants_detail.csv"
OUT_PATH = RESULTS / "flat_control_contrasts.csv"


def design_cell(run_id: str) -> dict | None:
    """Decode the width, connectivity, topography, and seed from a run id."""
    match = re.match(r"^(M000[01]0)(?:_([^_]+))?_SUP_s(\d+)$", run_id)
    if not match:
        return None
    model_id, tag, seed = match.groups()
    if tag is None:
        width, connectivity = 128, "dense"
    else:
        cells = {
            "local128": (128, "local"),
            "random128": (128, "random"),
            "dense289": (289, "dense"),
            "local289native": (289, "local"),
            "random289": (289, "random"),
        }
        if tag not in cells:
            return None
        width, connectivity = cells[tag]
    return {"width": width, "connectivity": connectivity, "topography": int(model_id[4]), "seed": int(seed)}


def add_design(df: pd.DataFrame) -> pd.DataFrame:
    """Attach design factors and retain only valid factorial rows."""
    records = []
    for row in df.to_dict("records"):
        cell = design_cell(str(row.get("run_id", "")))
        if cell is not None and row.get("status") == "ok":
            records.append({**row, **cell})
    return pd.DataFrame(records)


def paired_contrasts(df: pd.DataFrame, value: str = "raw_alignment") -> pd.DataFrame:
    """Return seed-paired main comparisons and their two-way interactions."""
    group_cols = [c for c in ("epoch", "region", "subpop", "session", "patient") if c in df.columns]
    keys = group_cols + ["seed"]
    point = df.groupby(keys + ["width", "connectivity", "topography"], dropna=False)[value].mean().reset_index()
    rows = []

    def compare(label: str, subset: pd.DataFrame, left: dict, right: dict, factors: dict) -> None:
        left_df = subset
        right_df = subset
        for name, val in left.items():
            left_df = left_df[left_df[name] == val]
        for name, val in right.items():
            right_df = right_df[right_df[name] == val]
        paired = left_df.merge(right_df, on=keys, suffixes=("_left", "_right"))
        effects = paired[f"{value}_left"] - paired[f"{value}_right"]
        rows.append({
            **factors,
            "contrast": label,
            "n_pairs": len(effects),
            "effect": float(effects.mean()) if len(effects) else np.nan,
            "sd": float(effects.std(ddof=1)) if len(effects) > 1 else np.nan,
        })

    def interaction(label: str, subset: pd.DataFrame, terms: list[tuple[float, dict]], factors: dict) -> None:
        merged = None
        for index, (sign, filters) in enumerate(terms):
            term = subset
            for name, val in filters.items():
                term = term[term[name] == val]
            term = term[keys + [value]].rename(columns={value: f"value_{index}"})
            merged = term if merged is None else merged.merge(term, on=keys)
        effects = sum(sign * merged[f"value_{index}"] for index, (sign, _) in enumerate(terms)) if len(merged) else []
        rows.append({
            **factors,
            "contrast": label,
            "n_pairs": len(effects),
            "effect": float(np.mean(effects)) if len(effects) else np.nan,
            "sd": float(np.std(effects, ddof=1)) if len(effects) > 1 else np.nan,
        })

    for values, subset in point.groupby(group_cols, dropna=False) if group_cols else [((), point)]:
        context = dict(zip(group_cols, values if isinstance(values, tuple) else (values,)))
        for topo in (0, 1):
            for conn in ("dense", "random", "local"):
                compare("width_289_minus_128", subset, {"width": 289, "connectivity": conn, "topography": topo},
                        {"width": 128, "connectivity": conn, "topography": topo}, {**context, "topography": topo, "connectivity": conn})
            for width in (128, 289):
                compare("dense_minus_random", subset, {"width": width, "connectivity": "dense", "topography": topo},
                        {"width": width, "connectivity": "random", "topography": topo}, {**context, "topography": topo, "width": width})
                compare("random_minus_local", subset, {"width": width, "connectivity": "random", "topography": topo},
                        {"width": width, "connectivity": "local", "topography": topo}, {**context, "topography": topo, "width": width})
        for width in (128, 289):
            for conn in ("dense", "random", "local"):
                compare("topography_on_minus_off", subset, {"width": width, "connectivity": conn, "topography": 1},
                        {"width": width, "connectivity": conn, "topography": 0}, {**context, "width": width, "connectivity": conn})
        for conn in ("dense", "random", "local"):
            interaction("width_by_topography", subset, [
                (1, {"width": 289, "connectivity": conn, "topography": 1}),
                (-1, {"width": 289, "connectivity": conn, "topography": 0}),
                (-1, {"width": 128, "connectivity": conn, "topography": 1}),
                (1, {"width": 128, "connectivity": conn, "topography": 0}),
            ], {**context, "connectivity": conn})
        for width in (128, 289):
            for first, second in (("dense", "random"), ("random", "local")):
                interaction("connectivity_by_topography", subset, [
                    (1, {"width": width, "connectivity": first, "topography": 1}),
                    (-1, {"width": width, "connectivity": first, "topography": 0}),
                    (-1, {"width": width, "connectivity": second, "topography": 1}),
                    (1, {"width": width, "connectivity": second, "topography": 0}),
                ], {**context, "width": width, "connectivity_pair": f"{first}-{second}"})
        for topo in (0, 1):
            for first, second in (("dense", "random"), ("random", "local")):
                interaction("width_by_connectivity", subset, [
                    (1, {"width": 289, "connectivity": first, "topography": topo}),
                    (-1, {"width": 128, "connectivity": first, "topography": topo}),
                    (-1, {"width": 289, "connectivity": second, "topography": topo}),
                    (1, {"width": 128, "connectivity": second, "topography": topo}),
                ], {**context, "topography": topo, "connectivity_pair": f"{first}-{second}"})
    return pd.DataFrame(rows)


def main() -> int:
    detail = pd.read_csv(DETAIL_PATH)
    design = add_design(detail)
    if design.empty:
        raise SystemExit("no completed flat-control rows found")
    out = pd.concat([
        paired_contrasts(design, "raw_alignment").assign(measure="raw_alignment"),
        paired_contrasts(design, "normalized_alignment").assign(measure="normalized_alignment"),
    ], ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
