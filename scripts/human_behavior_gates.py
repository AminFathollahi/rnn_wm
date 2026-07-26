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

    print("Per-load, pooled over all sessions and datasets")
    print(f"{'load':>5} {'sessions':>9} {'trials':>8} {'accuracy':>9} {'95% CI':>16} {'per-session min':>16}")
    print("-" * 70)
    pooled = {}
    for load, g in df.groupby("load"):
        n, k = int(g.n_trials.sum()), int(g.n_correct.sum())
        acc = k / n
        lo, hi = wilson_ci(k, n)
        pooled[int(load)] = {"acc": acc, "lo": lo, "hi": hi, "n": n,
                             "min_session": float(g.accuracy.min()),
                             "median_session": float(g.accuracy.median())}
        print(f"{load:>5} {len(g):>9} {n:>8} {acc:>9.4f} {f'[{lo:.3f},{hi:.3f}]':>16} {g.accuracy.min():>16.4f}")

    print("\nPer dataset")
    for (ds, load), g in df.groupby(["dataset", "load"]):
        n, k = int(g.n_trials.sum()), int(g.n_correct.sum())
        print(f"  {ds} load{load}: {k/n:.4f}  (n={n}, {len(g)} sessions)")

    # ---- DO NOT POOL ACROSS DATASETS FOR A LOAD-GRADED PROFILE ----
    # Only some datasets ran every load (here: only 000469 has load 2). The
    # pooled numbers above therefore show load3 ABOVE load2, which is
    # backwards for a WM task -- a Simpson's-paradox artefact of load 2
    # being measured only in the lowest-performing dataset. A load-graded
    # gate must come from datasets that measured the whole ladder.
    loads_all = sorted(df.load.unique().tolist())
    complete = [ds for ds, g in df.groupby("dataset") if sorted(g.load.unique().tolist()) == loads_all]
    print(f"\nDatasets covering the full load ladder {loads_all}: {complete or 'NONE'}")
    for ds, g in df.groupby("dataset"):
        print(f"  {ds}: loads {sorted(g.load.unique().tolist())}"
              f"{'  <- complete' if ds in complete else '  (partial -- not usable for a graded profile)'}")
    if not complete:
        print("\n[human] no dataset covers every load; cannot derive a graded gate. "
              "Report per-load gates only for loads with full coverage.")
        return 1

    prof = {}
    src = df[df.dataset.isin(complete)]
    for load, g in src.groupby("load"):
        n, k = int(g.n_trials.sum()), int(g.n_correct.sum())
        lo, hi = wilson_ci(k, n)
        prof[int(load)] = {"acc": k / n, "lo": lo, "hi": hi, "n": n,
                           "median_session": float(g.accuracy.median())}
    accs = [prof[l]["acc"] for l in sorted(prof)]
    monotone = all(a >= b for a, b in zip(accs, accs[1:]))
    print(f"\nGraded profile from {complete}: "
          + ", ".join(f"load{l} {prof[l]['acc']:.4f}" for l in sorted(prof))
          + f"   monotone_decreasing={monotone}")
    if not monotone:
        print("[human] WARNING: profile is not monotone in load. Investigate before using as a gate.")

    print("\n" + "=" * 70)
    print("PROPOSED GATES (paste into configs/config.yaml)")
    print("=" * 70)
    print("gates:")
    print("  # Derived from human Sternberg performance in the very sessions this")
    print("  # study aligns against -- scripts/human_behavior_gates.py,")
    print("  # results/human_behavior.csv. Not a literature estimate.")
    print(f"  # Graded profile from {complete} (the only dataset(s) covering every load);")
    print("  # pooling across datasets is invalid here -- see the script's comment.")
    print("  convergence:   # stops training: match the median human session, per load")
    for load in sorted(prof):
        print(f"    load{load}: {math.floor(prof[load]['median_session'] * 100) / 100:.2f}"
              f"    # median human session {prof[load]['median_session']:.4f}")
    print("  inclusion:     # admits a run to the representational analyses:")
    print("                 # human lower 95% CI bound, per load")
    for load in sorted(prof):
        print(f"    load{load}: {math.floor(prof[load]['lo'] * 100) / 100:.2f}"
              f"    # human {prof[load]['acc']:.4f} "
              f"[{prof[load]['lo']:.3f},{prof[load]['hi']:.3f}], n={prof[load]['n']}")
    print("\n  # Cross-check, ALL datasets, loads with full coverage only:")
    for load in sorted(pooled):
        n_ds = df[df.load == load].dataset.nunique()
        if n_ds == df.dataset.nunique():
            print(f"  #   load{load}: {pooled[load]['acc']:.4f} "
                  f"[{pooled[load]['lo']:.3f},{pooled[load]['hi']:.3f}], n={pooled[load]['n']} "
                  f"({n_ds} datasets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
