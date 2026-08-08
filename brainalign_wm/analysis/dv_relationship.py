"""Builds a per-(cell, seed) table of the two headline DVs -- task
performance and brain alignment -- plus internal network-organization
metrics, and quantifies how they relate to each other across the
ablation battery.

Sources (each optional; a run missing one source just gets NaN there,
not dropped):
    results/manifest.jsonl          -> accuracy, and the S/M/P/T/D spine
    results/alignment_results.csv   -> rsa_alignment (probe-epoch primary)
    results/network_properties.jsonl -> modularity_q, small_worldness,
                                         mixed_selectivity
    results/attractor_properties.jsonl -> n_fixed_points, n_stable_fixed_points,
                                         max_eig_modulus (analysis/attractors.py)
    results/dynamics_persistence.csv -> persistence_index

Only rows belonging to the 5-arm ablation battery are included (a
completed manifest record with S/M/P/T/D all present and a model_id that
reconstructs to exactly those five bits); local-learning cells and
ablation/identity-catch/perf-matched-baseline variants are excluded, same
filtering `analysis/run_all.py` applies to its own headline table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DV_CSV = ROOT / "results" / "dv_relationship.csv"


def _manifest_spine() -> pd.DataFrame:
    """One row per completed ablation-battery run: run_id, model_id,
    S, M, P, T, D, seed, accuracy (load3, matching the accuracy proxy
    `run_all.py` already uses for its own regressions)."""
    from brainalign_wm.analysis.run_all import _is_ablation_or_catch_variant, _load_completed_runs

    completed = _load_completed_runs(ROOT / "results" / "manifest.jsonl")
    rows = []
    for rec in completed:
        if "P" not in rec or "T" not in rec or "D" not in rec:
            continue  # local-learning cell, or a pre-5-arm-pivot legacy record
        if _is_ablation_or_catch_variant(rec):
            continue  # supplementary arm / identity-catch run, not a battery cell
        rows.append({
            "run_id": rec["run_id"], "cell": rec["model_id"],
            "S": rec["S"], "M": rec["M"], "P": rec["P"], "T": rec["T"], "D": rec["D"],
            "seed": rec["seed"], "accuracy": rec.get("accuracy", {}).get("load3"),
        })
    return pd.DataFrame(rows)


def _load_alignment() -> pd.DataFrame:
    """rsa_alignment per run_id: probe-epoch normalized alignment is the
    primary DV (invariant to region/session-schema noise the maintenance
    path is exposed to); falls back to maintenance-epoch alignment only
    for a run where the probe column itself is entirely absent."""
    path = ROOT / "results" / "alignment_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "rsa_alignment"])
    df = pd.read_csv(path)
    if "probe_normalized_alignment" in df.columns and df["probe_normalized_alignment"].notna().any():
        df["rsa_alignment"] = df["probe_normalized_alignment"]
    elif "maintenance_normalized_alignment" in df.columns:
        df["rsa_alignment"] = df["maintenance_normalized_alignment"]
    else:
        df["rsa_alignment"] = np.nan
    return df[["run_id", "rsa_alignment"]]


def _load_network_properties() -> pd.DataFrame:
    """modularity_q/small_worldness/mixed_selectivity per run_id. S=0 uses
    the flat core's own values directly; S=1 has two recurrent components
    (worker, manager) with no single natural aggregate, so this averages
    them -- a simplification, not a claim that worker and manager organize
    identically."""
    path = ROOT / "results" / "network_properties.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "modularity_q", "small_worldness", "mixed_selectivity"])
    latest: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["run_id"]] = rec

    rows = []
    for run_id, rec in latest.items():
        if rec.get("S") == 0:
            mod_q, small_w = rec.get("flat_modularity_q"), rec.get("flat_small_worldness")
        else:
            mods = [v for v in (rec.get("worker_modularity_q"), rec.get("manager_modularity_q")) if v is not None]
            sws = [v for v in (rec.get("worker_small_worldness"), rec.get("manager_small_worldness")) if v is not None]
            mod_q = float(np.mean(mods)) if mods else None
            small_w = float(np.mean(sws)) if sws else None
        rows.append({
            "run_id": run_id, "modularity_q": mod_q, "small_worldness": small_w,
            "mixed_selectivity": rec.get("mixed_selectivity_mean"),
        })
    return pd.DataFrame(rows)


def _load_attractor_properties() -> pd.DataFrame:
    """n_fixed_points/n_stable_fixed_points/max_eig_modulus per run_id
    (`analysis/attractors.py`'s whole-system pooled fields -- already
    computed over the joint [worker;manager] state for S=1, so unlike
    `_load_network_properties` there is no separate worker/manager value
    to average here)."""
    path = ROOT / "results" / "attractor_properties.jsonl"
    cols = ["run_id", "n_fixed_points", "n_stable_fixed_points", "max_eig_modulus"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    latest: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["run_id"]] = rec
    rows = [{"run_id": run_id, **{c: rec.get(c) for c in cols[1:]}} for run_id, rec in latest.items()]
    return pd.DataFrame(rows, columns=cols)


def _load_persistence() -> pd.DataFrame:
    path = ROOT / "results" / "dynamics_persistence.csv"
    if not path.exists():
        return pd.DataFrame(columns=["run_id", "persistence_index"])
    df = pd.read_csv(path)
    if "h6_model_persistence_mean" not in df.columns:
        return pd.DataFrame(columns=["run_id", "persistence_index"])
    return df[["run_id", "h6_model_persistence_mean"]].rename(columns={"h6_model_persistence_mean": "persistence_index"})


def build_dv_table() -> pd.DataFrame:
    df = _manifest_spine()
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "cell", "seed", "S", "M", "P", "T", "D", "accuracy", "rsa_alignment",
            "modularity_q", "small_worldness", "mixed_selectivity", "persistence_index",
            "n_fixed_points", "n_stable_fixed_points", "max_eig_modulus",
        ])
    df = df.merge(_load_alignment(), on="run_id", how="left")
    df = df.merge(_load_network_properties(), on="run_id", how="left")
    df = df.merge(_load_attractor_properties(), on="run_id", how="left")
    df = df.merge(_load_persistence(), on="run_id", how="left")
    cols = [
        "cell", "seed", "S", "M", "P", "T", "D", "accuracy", "rsa_alignment",
        "modularity_q", "small_worldness", "mixed_selectivity", "persistence_index",
        "n_fixed_points", "n_stable_fixed_points", "max_eig_modulus",
    ]
    return df[cols]


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Pearson + Spearman r with a percentile-bootstrap CI, resampling
    (x, y) pairs with replacement. Each row of the DV table is already one
    (cell, seed) replicate, so a plain row-level resample is the correct
    unit here (no further clustering needed)."""
    from scipy.stats import pearsonr, spearmanr

    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 4:
        return None
    pearson_r = float(pearsonr(x, y)[0])
    spearman_r = float(spearmanr(x, y)[0])
    rng = np.random.RandomState(seed)
    boot_pearson, boot_spearman = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        boot_pearson.append(pearsonr(xb, yb)[0])
        boot_spearman.append(spearmanr(xb, yb)[0])
    if not boot_pearson:
        return {"n": n, "pearson_r": pearson_r, "pearson_ci": (float("nan"), float("nan")),
                "spearman_r": spearman_r, "spearman_ci": (float("nan"), float("nan"))}
    return {
        "n": n, "pearson_r": pearson_r,
        "pearson_ci": (float(np.percentile(boot_pearson, 2.5)), float(np.percentile(boot_pearson, 97.5))),
        "spearman_r": spearman_r,
        "spearman_ci": (float(np.percentile(boot_spearman, 2.5)), float(np.percentile(boot_spearman, 97.5))),
    }


PAIRS = [
    ("accuracy", "rsa_alignment"),
    ("accuracy", "modularity_q"),
    ("accuracy", "small_worldness"),
    ("accuracy", "mixed_selectivity"),
    ("accuracy", "persistence_index"),
    ("accuracy", "n_fixed_points"),
    ("accuracy", "n_stable_fixed_points"),
    ("accuracy", "max_eig_modulus"),
    ("rsa_alignment", "modularity_q"),
    ("rsa_alignment", "small_worldness"),
    ("rsa_alignment", "mixed_selectivity"),
    ("rsa_alignment", "persistence_index"),
    ("rsa_alignment", "n_fixed_points"),
    ("rsa_alignment", "n_stable_fixed_points"),
    ("rsa_alignment", "max_eig_modulus"),
]


def compute_correlations(df: pd.DataFrame) -> dict:
    out = {}
    for a, b in PAIRS:
        if a not in df.columns or b not in df.columns:
            continue
        result = _bootstrap_corr(df[a].to_numpy(dtype=float), df[b].to_numpy(dtype=float))
        out[f"{a}<->{b}"] = result
    return out


def mixed_effects_fit(df: pd.DataFrame) -> dict | None:
    """rsa_alignment ~ accuracy, with a random intercept per seed."""
    from brainalign_wm.analysis.stats import mixed_effects_alignment

    sub = df.dropna(subset=["rsa_alignment", "accuracy", "seed"])
    if len(sub) < 4 or sub["seed"].nunique() < 2:
        return None
    return mixed_effects_alignment(sub, formula="rsa_alignment ~ accuracy", extra_vc_col="cell")


def run_dv_relationship(out_path: Path = DV_CSV) -> dict:
    df = build_dv_table()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[dv_relationship] wrote {out_path} ({len(df)} rows)")

    correlations = compute_correlations(df)
    for pair, result in correlations.items():
        if result is None:
            print(f"[dv_relationship]   {pair}: too few paired observations")
            continue
        print(
            f"[dv_relationship]   {pair}: n={result['n']} "
            f"pearson r={result['pearson_r']:.3f} CI={tuple(round(c, 3) for c in result['pearson_ci'])}  "
            f"spearman r={result['spearman_r']:.3f} CI={tuple(round(c, 3) for c in result['spearman_ci'])}"
        )

    fit = mixed_effects_fit(df)
    if fit is not None:
        print(f"[dv_relationship] rsa_alignment ~ accuracy + (1|seed) fit ({fit['method']}):")
        for k, v in fit["params"].items():
            print(f"    {k}: {v:.4f}")
    else:
        print("[dv_relationship] insufficient rows/seed coverage for the mixed-effects fit.")

    return {"table": df, "correlations": correlations, "mixed_effects_fit": fit}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(DV_CSV))
    args = ap.parse_args(argv)
    run_dv_relationship(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
