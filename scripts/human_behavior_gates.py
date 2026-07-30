#!/usr/bin/env python3
"""Derive the behavioral gates from the actual human Sternberg performance
in the recordings this study aligns against.

The gates decide which model runs are "behaviourally matched" to the humans
whose single neurons are the alignment target. Setting them from a paper
about a different task (n-back, different stimuli, different subjects) is a
worse choice than reading them off the very trials that produced the neural
data -- so this script does the latter.

Reads `response_accuracy` x `loads` per trial straight out of the NWB files
via `h5py` (same access path as `neural/adapters/dandi_nwb.py`; no pynwb),
pools across sessions within a dataset, and reports per-load accuracy with a
Wilson 95% CI, per session and pooled.

    python scripts/human_behavior_gates.py
    python scripts/human_behavior_gates.py --datasets 000469 000673 001187

Writes `results/human_behavior.csv` (one row per session x load) so the
numbers are auditable and the gates are reproducible from the raw data.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def wilson_ci(n_correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Used instead of the normal approximation
    because accuracies here sit near the ceiling, where the normal interval
    runs past 1.0 and understates the lower bound -- exactly the bound a
    gate depends on."""
    if n == 0:
        return (0.0, 0.0)
    p = n_correct / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def dedupe_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 12 (§3.1/9.8): 001187 is an MTL re-release of recordings also
    present in 000673 -- 19 session identifiers appear in both, with
    identical trial counts and accuracies (e.g. `sub-10_ses-1_P68CS` is
    68/70 in both). Any pooled statistic must count each underlying session
    once, not once per release that happens to include it. Collapse on
    (session, load), keeping the first occurrence -- order doesn't matter
    since duplicates are identical by construction, only presence does."""
    return df.drop_duplicates(subset=["session", "load"], keep="first").reset_index(drop=True)


def session_quantiles(df: pd.DataFrame, load: int) -> dict:
    """n, min, q05, q10, q25, median of PER-SESSION accuracy (not pooled
    trial-level accuracy) for one load. Caller is responsible for having
    deduplicated `df` first if it spans datasets that can overlap."""
    accs = df.loc[df.load == load, "accuracy"].to_numpy(dtype=float)
    if accs.size == 0:
        return {"n": 0, "min": float("nan"), "q05": float("nan"), "q10": float("nan"),
                "q25": float("nan"), "median": float("nan")}
    return {
        "n": int(accs.size),
        "min": float(np.min(accs)),
        "q05": float(np.percentile(accs, 5)),
        "q10": float(np.percentile(accs, 10)),
        "q25": float(np.percentile(accs, 25)),
        "median": float(np.median(accs)),
    }


def collect(data_root: Path, datasets: list[str]) -> pd.DataFrame:
    import h5py

    from brainalign_wm.neural.adapters.dandi_nwb import _decode, _trials_group, find_wm_sessions

    rows = []
    for ds in datasets:
        ds_root = data_root / ds
        if not ds_root.exists():
            print(f"[human] {ds}: not found at {ds_root}; skipping.")
            continue
        sessions = find_wm_sessions(ds_root)
        print(f"[human] {ds}: {len(sessions)} WM sessions")
        for path in sessions:
            try:
                with h5py.File(path, "r") as h:
                    identifier = _decode(h["identifier"][()])
                    tr = h[_trials_group(h)]
                    loads = np.asarray(tr["loads"][:], dtype=int)
                    correct = np.asarray(tr["response_accuracy"][:], dtype=float)
            except Exception as e:  # a malformed/atypical file is data, not a crash
                print(f"[human]   !! {path.name}: {type(e).__name__}: {e}")
                continue
            for load in sorted(set(loads.tolist())):
                sel = loads == load
                n, k = int(sel.sum()), int(correct[sel].sum())
                lo, hi = wilson_ci(k, n)
                rows.append(
                    {
                        "dataset": ds, "session": identifier, "load": load,
                        "n_trials": n, "n_correct": k,
                        "accuracy": k / n if n else float("nan"),
                        "ci_lo": lo, "ci_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=str, default=cfg["paths"]["data_root"])
    ap.add_argument("--datasets", type=str, nargs="+",
                    default=cfg["neural"]["datasets_tierA"] + cfg["neural"]["datasets_tierB"])
    args = ap.parse_args(argv)

    df = collect(Path(args.data_root), args.datasets)
    if df.empty:
        print("[human] no sessions read; cannot derive gates.")
        return 1

    out = ROOT / "results" / "human_behavior.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n[human] wrote {out}  ({len(df)} session x load rows)\n")

    # ---- Phase 12 (§3.1/9.8): dedup on session identifier BEFORE anything
    # pooled. 19 identifiers are shared between 000673 and 001187 (001187 is
    # an MTL re-release of overlapping recordings) -- naive pooling double-
    # counts them (111 rows) and gives q10=0.8222, not the correct 0.8344.
    df_dedup = dedupe_sessions(df)
    n_dup = len(df[df.load == 1]) - len(df_dedup[df_dedup.load == 1])
    print(f"[human] load-1 rows: {len(df[df.load == 1])} raw -> "
          f"{len(df_dedup[df_dedup.load == 1])} deduplicated ({n_dup} duplicate sessions dropped)")

    # ---- GATE A (§3.1): load 1 only, the one load every dataset measured.
    # Pooling load 2/3 across datasets is invalid: only 000469 ran them, and
    # it is also the lowest-performing dataset, so a pooled graded profile
    # puts load3 ABOVE load2 -- a Simpson's-paradox artefact of unequal
    # coverage, not a fact about humans. Load 2/3 are reported from 000469
    # alone, labelled single-dataset, not pooled.
    q_load1 = session_quantiles(df_dedup, 1)
    single_ds = "000469"
    df_single = df[df.dataset == single_ds]  # no cross-dataset dup risk within one dataset
    q_load2 = session_quantiles(df_single, 2)
    q_load3 = session_quantiles(df_single, 3)

    print("\nPer-session accuracy quantiles (Gate A basis, §3.1)")
    print(f"{'load':>6} {'source':>22} {'n':>4} {'min':>7} {'q05':>7} {'q10':>7} {'q25':>7} {'median':>7}")
    print("-" * 70)
    for load, q, src in ((1, q_load1, "dedup pool (3 ds)"), (2, q_load2, f"{single_ds} only"),
                         (3, q_load3, f"{single_ds} only")):
        print(f"{load:>6} {src:>22} {q['n']:>4} {q['min']:>7.4f} {q['q05']:>7.4f} "
              f"{q['q10']:>7.4f} {q['q25']:>7.4f} {q['median']:>7.4f}")

    print("\nPer dataset (trial-pooled, for reference)")
    for (ds, load), g in df.groupby(["dataset", "load"]):
        n, k = int(g.n_trials.sum()), int(g.n_correct.sum())
        print(f"  {ds} load{load}: {k/n:.4f}  (n={n}, {len(g)} sessions)")

    gate_a_load1 = math.floor(q_load1["q10"] * 100) / 100
    gate_b_load3 = math.floor(q_load3["q10"] * 100) / 100
    print("\n" + "=" * 70)
    print("PROPOSED GATES (paste into configs/config.yaml)")
    print("=" * 70)
    print("gates:")
    print("  # Gate A (§3.1): behavioural matching / inclusion. Load 1 only,")
    print(f"  # deduplicated pooled q10 = {q_load1['q10']:.4f} (n={q_load1['n']} sessions).")
    print("  # Does NOT stop training.")
    print("  criterion:")
    print(f"    load1: {gate_a_load1:.2f}")
    print("  consecutive_evals: 3")
    print("  # §3.3 efficiency DVs.")
    print("  extra_milestones:")
    print(f"    load3: {gate_b_load3:.2f}          # {single_ds} load-3 q10 = {q_load3['q10']:.4f}")
    print("  max_steps: null        # Gate B (§3.2), set by 12.6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
