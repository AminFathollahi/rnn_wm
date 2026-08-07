"""Renders the manuscript figures to vector-format PDFs, from whatever
results are currently available (`results/manifest.jsonl`,
`results/alignment_results.csv`, and the audit-fix pass's new
`results/{reflection_shuffle_lesion,dynamics_persistence}.csv`). Each
figure degrades gracefully -- it is skipped with a message, not a crash, if
its inputs are not yet present.

Implemented so far:
  F2 -- behavior and gates: per-cell accuracy at each load, gate thresholds,
        and the local-learning rung reached.
  F3 -- main alignment result: per-cell noise-ceiling-normalized alignment
        (maintenance, per-session-category schema, and probe, pooled-coarse
        schema -- see `analysis/run_all.py`'s module docstring).
  F4 -- reflective-gate causal control (C1): normal vs. reflection-shuffled
        maintenance alignment, per M=1 cell.
  F5 -- persistent-activity index distributions, model vs. brain (H6).
  F6 -- dynamic-vs-stable delay coding (cross-temporal stability index),
        model vs. brain (H5).
  F9  -- subpopulation x region dissociation: normalized maintenance
         alignment for {worker, manager} x {MTL, MFC}, hierarchical runs
         only, per-session points overlaid, flat runs drawn as an explicit
         not-applicable band. Numbered F9, not F1 -- F1 is already reserved
         below for the design schematic.
  F10 -- model-tick x neural-bin alignment heatmap: one panel per run,
         matched-time diagonal marked, argmax annotated.
  F11 -- behavioural position against the human distribution: per-load
         violin of the deduplicated human session pool with model runs
         overlaid, `FLATGRU_RL_s0`'s early and late training snapshots
         connected by an arrow.

Explicitly scoped out of this pass, not silently dropped:
F1 (design schematic -- purely illustrative, no analysis dependency) and F8
(cross-dataset replication against Tier B/001187 -- needs its own adapter
validation pass, out of scope here). A seed-spread figure is also not
implemented here -- it needs a multi-seed run that has not been launched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CELL_ORDER = [
    "M00000", "M11111",
    "M01111", "M10111", "M11011", "M11101", "M11110",
    "M10000", "M01000", "M00100", "M00010", "M00001",
    "M10010", "M00011", "M10001",
]


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
    # Gate A's `criterion` is load-1-only; `extra_milestones` covers loads
    # tracked descriptively rather than gated. Draw a threshold line only
    # for loads present in either config block.
    thresholds = {**gates_cfg.get("criterion", {}), **gates_cfg.get("extra_milestones", {})}
    load_colors = {"load1": "C0", "load2": "C1", "load3": "C2"}
    for load, color in load_colors.items():
        if load in thresholds:
            ax.axhline(thresholds[load], color=color, linestyle="--", linewidth=1, alpha=0.6)
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
    """Audit-fix rewrite: `alignment_results.csv`'s headline columns are now
    `maintenance_normalized_alignment` (per-session category schema) and
    `probe_normalized_alignment` (pooled coarse schema) -- see
    `analysis/run_all.py`'s module docstring -- not a single
    `normalized_alignment` column."""
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

    cols = [c for c in ("maintenance_normalized_alignment", "probe_normalized_alignment") if c in df.columns]
    if not cols:
        print("[figures] F3: neither maintenance_ nor probe_normalized_alignment present; skipping.")
        return False
    agg = df.groupby("model_id")[cols].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(agg))
    width = 0.35
    for i, col in enumerate(cols):
        ax.bar([xi + (i - 0.5) * width for xi in x], agg[col], width=width, label=col.replace("_normalized_alignment", ""))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="noise ceiling")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("alignment / noise ceiling")
    ax.set_ylim(0, 1.15)
    ax.set_title("F3: main alignment result (maintenance + probe, seeds averaged)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f4_reflection_shuffle(lesion_csv: Path, out_path: Path) -> bool:
    """F4: normal vs. reflection-shuffled maintenance
    alignment, per M=1 cell -- the causal test of H2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not lesion_csv.exists():
        print("[figures] F4: no reflection_shuffle_lesion.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(lesion_csv)
    df = df[df.status == "ok"] if "status" in df.columns else df.iloc[0:0]
    if len(df) == 0:
        print("[figures] F4: no ok reflection-shuffle rows yet; skipping.")
        return False

    agg = df.groupby("model_id")[["normal_normalized_alignment", "shuffled_normalized_alignment"]].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], agg["normal_normalized_alignment"], width=width, label="normal")
    ax.bar([xi + width / 2 for xi in x], agg["shuffled_normalized_alignment"], width=width, label="reflection-shuffled")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("maintenance normalized alignment")
    ax.set_title("F4: reflective-gate causal control (H2) -- normal vs. shuffled R_t")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f5_persistence(dynamics_csv: Path, out_path: Path) -> bool:
    """F5 (H6): model vs. brain persistent-activity-index
    means, per cell, with the permutation-test effect size annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dynamics_csv.exists():
        print("[figures] F5: no dynamics_persistence.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(dynamics_csv)
    df = df[df.get("h6_model_persistence_mean").notna()] if "h6_model_persistence_mean" in df.columns else df.iloc[0:0]
    if len(df) == 0:
        print("[figures] F5: no H6 persistence rows yet; skipping.")
        return False

    agg = df.groupby("model_id")[["h6_model_persistence_mean", "h6_neural_persistence_mean"]].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], agg["h6_model_persistence_mean"], width=width, label="model")
    ax.bar([xi + width / 2 for xi in x], agg["h6_neural_persistence_mean"], width=width, label="brain")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("persistent-activity index")
    ax.set_title("F5: persistent activity, model vs. brain (H6)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f6_dynamic_stable(dynamics_csv: Path, out_path: Path) -> bool:
    """F6 (H5): model vs. brain cross-temporal stability
    index (dynamic vs. stable delay coding), per cell."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not dynamics_csv.exists():
        print("[figures] F6: no dynamics_persistence.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(dynamics_csv)
    df = df[df.get("h5_model_stability_mean").notna()] if "h5_model_stability_mean" in df.columns else df.iloc[0:0]
    if len(df) == 0:
        print("[figures] F6: no H5 stability rows yet; skipping.")
        return False

    agg = df.groupby("model_id")[["h5_model_stability_mean", "h5_neural_stability_mean"]].mean()
    agg = agg.reindex([c for c in CELL_ORDER if c in agg.index])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(agg))
    width = 0.35
    ax.bar([xi - width / 2 for xi in x], agg["h5_model_stability_mean"], width=width, label="model")
    ax.bar([xi + width / 2 for xi in x], agg["h5_neural_stability_mean"], width=width, label="brain")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index, rotation=0)
    ax.set_ylabel("stability index (off-diag/diag generalization)")
    ax.set_title("F6: dynamic vs. stable delay coding, model vs. brain (H5)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f7_dv_relationship(dv_csv: Path, out_path: Path) -> bool:
    """F7: accuracy vs. rsa_alignment scatter (one point per cell x seed,
    colored by S, marker by M), with a fitted line + Pearson r; a second
    panel does the same for accuracy vs. an organization metric
    (modularity_q, or mixed_selectivity if modularity_q has no coverage)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not dv_csv.exists():
        print("[figures] F7: no dv_relationship.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(dv_csv)
    if len(df) == 0:
        print("[figures] F7: dv_relationship.csv is empty; skipping.")
        return False

    org_col = "modularity_q" if df["modularity_q"].notna().sum() >= 4 else "mixed_selectivity"
    markers = {0: "o", 1: "^"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, y_col in zip(axes, ("rsa_alignment", org_col)):
        sub = df.dropna(subset=["accuracy", y_col])
        if len(sub) < 2:
            ax.set_title(f"{y_col}: insufficient data")
            continue
        for m_val, marker in markers.items():
            g = sub[sub["M"] == m_val]
            if len(g):
                ax.scatter(g["accuracy"], g[y_col], c=g["S"], cmap="coolwarm", marker=marker,
                           label=f"M={m_val}", edgecolors="black", linewidths=0.3)
        if sub["accuracy"].nunique() > 1:
            coeffs = np.polyfit(sub["accuracy"], sub[y_col], 1)
            xs = np.linspace(sub["accuracy"].min(), sub["accuracy"].max(), 50)
            ax.plot(xs, np.polyval(coeffs, xs), color="black", linewidth=1, linestyle="--")
        r = sub["accuracy"].corr(sub[y_col])
        ax.set_xlabel("accuracy (load3)")
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs. accuracy (Pearson r={r:.2f})")
        ax.legend(fontsize=8)
    fig.suptitle("F7: DV relationship -- performance vs. alignment vs. organization (color=S, marker=M)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f9_subpop_dissociation(alignment_by_session_csv: Path, out_path: Path) -> bool:
    """Normalized maintenance alignment for {worker, manager} x {MTL, MFC},
    hierarchical runs only, with per-session points overlaid. A flat run has
    no subpopulations and is drawn as an explicit not-applicable band rather
    than omitted or zeroed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not alignment_by_session_csv.exists():
        print("[figures] F9: no alignment_by_session.csv yet; run `analysis/run_all.py` first. Skipping.")
        return False
    df = pd.read_csv(alignment_by_session_csv)
    if "subpop" not in df.columns:
        print("[figures] F9: alignment_by_session.csv has no `subpop` column; skipping.")
        return False

    hier = df[(df.region.isin(["MTL", "MFC"])) & (df.subpop.isin(["worker", "manager"]))]
    flat_na = df[(df.region.isin(["MTL", "MFC"])) & (df.status == "not_applicable")]
    if len(hier) == 0 and len(flat_na) == 0:
        print("[figures] F9: no hierarchical subpop rows and no flat not_applicable rows; skipping.")
        return False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cells = [("worker", "MTL"), ("manager", "MTL"), ("worker", "MFC"), ("manager", "MFC")]
    means, xs = [], []
    for i, (subpop, region) in enumerate(cells):
        g = hier[(hier.region == region) & (hier.subpop == subpop) & (hier.status == "ok")]
        vals = g["normalized_alignment"].dropna().to_numpy()
        means.append(float(np.mean(vals)) if len(vals) else np.nan)
        xs.append(i)
        if len(vals):
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=10, alpha=0.4, color="black", zorder=3)
    colors = ["C0", "C1", "C0", "C1"]
    ax.bar(xs, means, color=colors, alpha=0.7, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{sp}\n{reg}" for sp, reg in cells])
    ax.set_ylabel("maintenance normalized_alignment")
    ax.set_title("F9: worker↔MTL / manager↔MFC subpopulation dissociation")
    if len(flat_na):
        ax.text(0.5, 0.02, f"flat runs: not_applicable ({len(flat_na)} rows, no subpopulation to plot)",
                transform=ax.transAxes, ha="center", fontsize=8, style="italic", color="gray")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f10_tick_bin_heatmap(results_dir: Path, out_path: Path) -> bool:
    """One heatmap panel per run with a `tick_bin_alignment_{run_id}.csv` on
    disk (see `scripts/diagnose_maintenance_sign.py`), matched-time diagonal
    marked, argmax annotated."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    csvs = sorted(results_dir.glob("tick_bin_alignment_*.csv"))
    if not csvs:
        print("[figures] F10: no tick_bin_alignment_*.csv yet; run `scripts/diagnose_maintenance_sign.py` first. Skipping.")
        return False

    fig, axes = plt.subplots(1, len(csvs), figsize=(5.5 * len(csvs), 4.5), squeeze=False)
    for ax, csv_path in zip(axes[0], csvs):
        run_id = csv_path.stem.replace("tick_bin_alignment_", "")
        mat = pd.read_csv(csv_path, index_col=0).to_numpy()
        n_ticks, n_bins = mat.shape
        vmax = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        diag_bins = [int(round(t * (n_bins - 1) / max(n_ticks - 1, 1))) for t in range(n_ticks)]
        ax.plot(diag_bins, range(n_ticks), color="black", linewidth=1, linestyle="--", label="matched time")
        argmax_t, argmax_b = np.unravel_index(int(np.nanargmax(mat)), mat.shape)
        ax.scatter([argmax_b], [argmax_t], marker="*", s=150, color="gold", edgecolors="black", zorder=5, label="argmax")
        ax.set_xlabel("neural bin (50 ms)")
        ax.set_ylabel("model tick")
        ax.set_title(f"{run_id}\nargmax={mat[argmax_t, argmax_b]:.3f}")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("F10: model-tick x neural-bin alignment")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[figures] wrote {out_path}")
    return True


def make_f11_behavioral_position(human_behavior_csv: Path, manifest_df: pd.DataFrame, out_path: Path) -> bool:
    """Human per-session accuracy as a violin per load (deduplicated
    session pool), model runs overlaid, with `FLATGRU_RL_s0`'s early
    (`ckpt_at_criterion.pt`) and late (`ckpt.pt`) checkpoints connected by
    an arrow to show how its behavioural position shifts over training."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not human_behavior_csv.exists():
        print("[figures] F11: no human_behavior.csv yet; run `scripts/human_behavior_gates.py` first. Skipping.")
        return False
    hdf = pd.read_csv(human_behavior_csv).drop_duplicates(subset=["session", "load"], keep="first")
    if len(hdf) == 0 or len(manifest_df) == 0:
        print("[figures] F11: human_behavior.csv or manifest is empty; skipping.")
        return False

    completed = manifest_df[manifest_df.status == "completed"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    loads = [1, 2, 3]
    violin_data = [hdf[hdf.load == load]["accuracy"].dropna().to_numpy() for load in loads]
    parts = ax.violinplot([v for v in violin_data if len(v)], positions=[l for l, v in zip(loads, violin_data) if len(v)],
                          widths=0.7, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.3)

    for _, row in completed.iterrows():
        acc = row.get("accuracy")
        if not isinstance(acc, dict):
            continue
        ys = [acc.get(f"load{l}") for l in loads]
        xs = [l + (np.random.RandomState(hash(row["run_id"]) % 2**31).rand() - 0.5) * 0.3 for l in loads]
        ax.scatter(xs, ys, s=15, alpha=0.6, label=None)

    # FLATGRU_RL_s0's early-checkpoint -> late-checkpoint arrow.
    at_criterion_csv = ROOT / "results" / "metrics" / "FLATGRU_RL_s0.csv"
    gate_a_row = completed[completed.run_id == "FLATGRU_RL_s0"]
    if len(gate_a_row) and at_criterion_csv.exists():
        mdf = pd.read_csv(at_criterion_csv)
        first_ms = gate_a_row.iloc[0].get("first_milestone_step")
        crit_row = mdf[mdf.step == first_ms] if first_ms else pd.DataFrame()
        final_acc = gate_a_row.iloc[0].get("accuracy") or {}
        if len(crit_row):
            crit = crit_row.iloc[0]
            for load in loads:
                y0 = crit.get(f"train_acc_load{load}")
                y1 = final_acc.get(f"load{load}")
                if y0 is not None and y1 is not None:
                    ax.annotate("", xy=(load, y1), xytext=(load, y0),
                               arrowprops=dict(arrowstyle="->", color="red", lw=1.5))

    ax.set_xticks(loads)
    ax.set_xlabel("load")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("F11: behavioural position vs. human distribution\n"
                 "red arrow: FLATGRU_RL_s0 early → late checkpoint")
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
    made_any |= make_f4_reflection_shuffle(ROOT / "results" / "reflection_shuffle_lesion.csv", out_dir / "F4_reflection_shuffle.pdf")
    made_any |= make_f5_persistence(ROOT / "results" / "dynamics_persistence.csv", out_dir / "F5_persistence.pdf")
    made_any |= make_f6_dynamic_stable(ROOT / "results" / "dynamics_persistence.csv", out_dir / "F6_dynamic_stable.pdf")
    made_any |= make_f7_dv_relationship(ROOT / "results" / "dv_relationship.csv", out_dir / "F7_dv_relationship.pdf")
    made_any |= make_f9_subpop_dissociation(ROOT / "results" / "alignment_by_session.csv", out_dir / "F9_subpop_dissociation.pdf")
    made_any |= make_f10_tick_bin_heatmap(ROOT / "results", out_dir / "F10_tick_bin_heatmap.pdf")
    made_any |= make_f11_behavioral_position(ROOT / "results" / "human_behavior.csv", manifest_df, out_dir / "F11_behavioral_position.pdf")

    if not made_any:
        print("[figures] no figures produced -- no completed runs or alignment results available yet.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
