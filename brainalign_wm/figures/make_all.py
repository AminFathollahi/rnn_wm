"""Renders the manuscript figures to vector-format PDFs, from whatever
results are currently available (`results/manifest.jsonl`,
`results/alignment_results.csv`). Each figure degrades gracefully -- it is
skipped with a message, not a crash, if its inputs are not yet present.

Implemented so far:
  F2 -- behavior and gates: per-cell accuracy at each load, gate thresholds,
        and the local-learning rung reached.
  F3 -- main alignment result: per-cell noise-ceiling-normalized alignment.

Not yet implemented: F1 (design schematic), F4 (reflective-gate causal
effects), F5 (single-neuron and population geometry), F6 (dynamic vs.
stable coding, oblique sweep), F7 (persistent activity, lesions), F8
(cross-dataset replication) -- these require analyses (§9.5-9.8 scope)
beyond the pooled RSA computed by `analysis/run_all.py` so far.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CELL_ORDER = ["M000", "M001", "M010", "M011", "M100", "M101", "M110", "M111"]


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    latest: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[rec.get("run_id", "?")] = rec
    return pd.DataFrame(latest.values())


def make_f2_behavior(manifest_df: pd.DataFrame, out_path: Path, gates_cfg: dict) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    completed = manifest_df[manifest_df.status == "completed"] if len(manifest_df) else manifest_df
    if len(completed) == 0:
        print("[figures] F2: no completed runs yet; skipping.")
        return False

    completed = completed.copy()
    completed["load1"] = completed["accuracy"].apply(lambda a: a.get("load1") if isinstance(a, dict) else None)
    completed["load2"] = completed["accuracy"].apply(lambda a: a.get("load2") if isinstance(a, dict) else None)
    completed["load3"] = completed["accuracy"].apply(lambda a: a.get("load3") if isinstance(a, dict) else None)
    agg = completed.groupby("model_id")[["load1", "load2", "load3"]].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(agg))
    width = 0.25
    for i, load in enumerate(("load1", "load2", "load3")):
        ax.bar([xi + (i - 1) * width for xi in x], agg[load], width=width, label=load)
    ax.axhline(gates_cfg["load1_acc"], color="C0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(gates_cfg["load3_acc"], color="C2", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("F2: behavior and gates (mean accuracy per cell, seeds pooled)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f3_alignment(alignment_csv: Path, out_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not alignment_csv.exists():
        print("[figures] F3: no alignment_results.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(alignment_csv)
    if len(df) == 0:
        print("[figures] F3: alignment_results.csv is empty; skipping.")
        return False

    agg = df.groupby("model_id")[["normalized_alignment", "raw_alignment", "noise_ceiling_upper"]].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(agg))
    ax.bar(x, agg["normalized_alignment"], color="C0", label="normalized alignment (fraction of noise ceiling)")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="noise ceiling")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("alignment / noise ceiling")
    ax.set_ylim(0, 1.15)
    ax.set_title("F3: main alignment result (pooled regions, seeds averaged)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    manifest_df = _load_manifest(ROOT / "results" / "manifest.jsonl")

    made_any = False
    made_any |= make_f2_behavior(manifest_df, out_dir / "F2_behavior_gates.pdf", cfg["gates"])
    made_any |= make_f3_alignment(ROOT / "results" / "alignment_results.csv", out_dir / "F3_alignment.pdf")

    if not made_any:
        print("[figures] no figures produced -- no completed runs or alignment results available yet.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
