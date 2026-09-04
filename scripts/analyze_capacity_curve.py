#!/usr/bin/env python3
"""Item 9.1 (comments.txt §5): post-hoc analysis of `run_capacity_curve.py`'s
sweep. Reads `results/manifest.jsonl` for the `M00000_H*` runs, loads each
run's checkpoint, computes final accuracy (`evaluate_accuracy`, reusing the
exact function `training/train.py` itself uses -- no reimplementation) and
Phase 8's `mean_speed`/`pca_participation_ratio` geometry metrics on a fresh
eval rollout of the delay ("maintain") period, aggregates across seeds, and
writes `results/capacity_curve.csv` + `results/figures/capacity_curve.png`
with the knee (elbow of load3 accuracy vs H) marked.

Deliberately does NOT reuse `generate_activity_logs.py`'s parquet-log path
(`_run_id_extras`'s `_2x`/`_pbwm`/`_idcatch`/`_bioinit` suffix convention has
no entry for `_H{H}`, and `dandi_data` there is real-neural-alignment
stimulus data this purely-model-side analysis doesn't need) -- instead rolls
out fresh eval trials directly with the SAME building blocks `train.py`'s
own `evaluate_accuracy`/`_run_trial` use (`_build_model`, `_step_core`,
`_init_state`, `_image_features_all_ticks`), capturing the S=0 raw
recurrent state `state["h"]` (matches `h_flat`'s existing convention: see
`generate_activity_logs.py`, `h_flat=state["h"][0].tolist()`).

Usage:
  python scripts/analyze_capacity_curve.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import MANIFEST, RESULTS, load_all_records  # noqa: E402

from brainalign_wm.analysis.geometry import mean_speed, pca_participation_ratio  # noqa: E402
from brainalign_wm.training.train import (  # noqa: E402
    ROOT as MODEL_ROOT, _build_model, _image_features_all_ticks, _init_state, _load_full_config, _step_core,
    evaluate_accuracy,
)

OUT_CSV = RESULTS / "capacity_curve.csv"
OUT_FIG = RESULTS / "figures" / "capacity_curve.png"
_RE_H = re.compile(r"^M00000_H(\d+)$")

N_EVAL_TRIALS = 100
GEOMETRY_LOAD = 3  # hardest/most memory-demanding load -- the delay-period manifold of interest


def _rollout_hidden_states(front_end, core, heads, task_gen, image_bank, cfg, device, load: int, n_trials: int, eval_seed: int) -> np.ndarray:
    """Eval-mode-only rollout (no BPTT, no loss) capturing S=0's raw
    recurrent state at every maintain-epoch tick. Mirrors `_run_trial`'s
    tick loop and `evaluate_accuracy`'s own per-load trial-batch
    generation, but this function's only job is the [n_trials, T_maintain,
    H] single-trial ensemble Phase 8's geometry functions need (C3: no
    trial-averaging)."""
    m = cfg["model"]
    master_rng = np.random.RandomState(eval_seed)
    trials = []
    for _ in range(n_trials):
        trial_seed = int(master_rng.randint(0, 2**31 - 1))
        rng = np.random.RandomState(trial_seed)
        steps = task_gen.sternberg.generate_trial(
            rng, loads=[load], lure_fraction=cfg["task"]["lure_fraction"],
            maintain_steps=cfg["task"]["maintain_steps"], trial_id=-1, split="test",
            identity_catch_fraction=0.0,
        )
        trials.append(steps)
    B = len(trials)
    T = len(trials[0])
    state = _init_state(core, 0, 0, B, device)
    all_v = _image_features_all_ticks(image_bank, trials, m["feature_dim"], device)
    maintain_h = []
    with torch.no_grad():
        for i in range(T):
            ts_list = [trials[b][i] for b in range(B)]
            epoch = ts_list[0].epoch
            v_t = all_v[i]
            c_t = torch.tensor([ts.c_t for ts in ts_list], dtype=torch.float32, device=device)
            z_t = front_end(v_t, c_t)
            _, new_state, _ = _step_core(core, 0, 0, 0, z_t, state, t=i, gate_bias=None, training=False)
            if epoch == "maintain":
                maintain_h.append(new_state["h"].cpu().numpy().copy())
            state = new_state
    if not maintain_h:
        return np.zeros((B, 0, m["flat_units"]))
    return np.stack(maintain_h, axis=1)  # [B, T_maintain, H]


def _find_knee(H_vals: list[int], acc_vals: list[float]) -> int:
    """Simplest-workable elbow heuristic (comments.txt mandates no specific
    algorithm, just that the knee be identified and reported): the smallest
    H whose accuracy is within 2 percentage points of the accuracy ceiling
    reached by the largest H tested -- the point past which more capacity
    buys essentially nothing. Falls back to the largest H if accuracy never
    plateaus (monotonically improving all the way to H=256)."""
    ceiling = acc_vals[-1]
    for H, acc in zip(H_vals, acc_vals):
        if acc >= ceiling - 0.02:
            return H
    return H_vals[-1]


def main() -> int:
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.utils.device import get_device

    full_cfg = _load_full_config()
    device = get_device()
    records = [r for r in load_all_records(MANIFEST) if r.get("status") == "completed"]
    by_h: dict[int, list[dict]] = {}
    for rec in records:
        m5 = _RE_H.match(rec.get("model_id", ""))
        if m5:
            by_h.setdefault(int(m5.group(1)), []).append(rec)

    if not by_h:
        RESULTS.mkdir(exist_ok=True)
        pd.DataFrame(columns=[
            "H", "n_seeds", "acc_load1_mean", "acc_load2_mean", "acc_load3_mean",
            "matched_frac", "mean_speed_mean", "pca_participation_ratio_mean",
        ]).to_csv(OUT_CSV, index=False)
        print(f"[analyze_capacity_curve] no completed M00000_H* runs found in {MANIFEST} -- "
              f"wrote header-only {OUT_CSV}. Run scripts/run_capacity_curve.py first.", flush=True)
        return 0

    image_bank = ImageTokenBank(
        stimuli_root=Path(full_cfg["paths"]["stimuli"]), categories=full_cfg["task"]["categories"],
        feature_cache_path=Path(full_cfg["paths"]["feature_cache"]) / "image_token_bank.npy",
        seed=0,
    )

    rows = []
    for H in sorted(by_h):
        per_seed = {"acc_load1": [], "acc_load2": [], "acc_load3": [], "matched": [],
                    "mean_speed": [], "pca_participation_ratio": []}
        for rec in by_h[H]:
            run_id, seed = rec["run_id"], rec["seed"]
            cfg_H = {**full_cfg, "model": {**full_cfg["model"], "flat_units": H}}
            front_end, core, heads = _build_model(cfg_H, S=0, M=0, P=0, device=device)
            ckpt_path = RESULTS / "checkpoints" / run_id / "ckpt.pt"
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            front_end.load_state_dict(ck["front_end"])
            core.load_state_dict(ck["core"])
            heads.load_state_dict(ck["heads"])
            front_end.eval(); core.eval(); heads.eval()

            task_gen = TaskGenerator(cfg_H, image_bank, seed=seed)
            acc = evaluate_accuracy(
                front_end, core, heads, S=0, P=0, reflective_gate=None, task_gen=task_gen,
                image_bank=image_bank, cfg=cfg_H, device=device,
                n_trials=N_EVAL_TRIALS, eval_seed=910_000_000 + seed,
            )
            per_seed["acc_load1"].append(acc["load1"])
            per_seed["acc_load2"].append(acc["load2"])
            per_seed["acc_load3"].append(acc["load3"])
            per_seed["matched"].append(bool(rec.get("matched", False)))

            Z = _rollout_hidden_states(
                front_end, core, heads, task_gen, image_bank, cfg_H, device,
                load=GEOMETRY_LOAD, n_trials=N_EVAL_TRIALS, eval_seed=920_000_000 + seed,
            )
            if Z.shape[0] >= 2 and Z.shape[1] >= 2:
                per_seed["mean_speed"].append(mean_speed(Z))
                per_seed["pca_participation_ratio"].append(pca_participation_ratio(Z.reshape(-1, Z.shape[-1])))

        row = {
            "H": H, "n_seeds": len(by_h[H]),
            "acc_load1_mean": float(np.mean(per_seed["acc_load1"])), "acc_load1_std": float(np.std(per_seed["acc_load1"])),
            "acc_load2_mean": float(np.mean(per_seed["acc_load2"])), "acc_load2_std": float(np.std(per_seed["acc_load2"])),
            "acc_load3_mean": float(np.mean(per_seed["acc_load3"])), "acc_load3_std": float(np.std(per_seed["acc_load3"])),
            "matched_frac": float(np.mean(per_seed["matched"])),
            "mean_speed_mean": float(np.mean(per_seed["mean_speed"])) if per_seed["mean_speed"] else float("nan"),
            "pca_participation_ratio_mean": float(np.mean(per_seed["pca_participation_ratio"])) if per_seed["pca_participation_ratio"] else float("nan"),
        }
        rows.append(row)
        print(f"[analyze_capacity_curve] H={H}: {row}", flush=True)

    df = pd.DataFrame(rows).sort_values("H")
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[analyze_capacity_curve] wrote {OUT_CSV}", flush=True)

    knee = _find_knee(df["H"].tolist(), df["acc_load3_mean"].tolist())
    print(f"[analyze_capacity_curve] knee (smallest H within 2pp of the load3-accuracy ceiling): H={knee}", flush=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax1.errorbar(df["H"], df["acc_load3_mean"], yerr=df["acc_load3_std"], marker="o", label="load3 accuracy")
    ax1.errorbar(df["H"], df["acc_load1_mean"], yerr=df["acc_load1_std"], marker="o", label="load1 accuracy", alpha=0.6)
    ax1.axvline(knee, color="red", linestyle="--", linewidth=1, label=f"knee (H={knee})")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("H (flat_units)")
    ax1.set_ylabel("accuracy-at-criterion")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.set_title("9.1: accuracy vs H")

    ax2.plot(df["H"], df["pca_participation_ratio_mean"], marker="o", color="C2")
    ax2.axvline(knee, color="red", linestyle="--", linewidth=1)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("H (flat_units)")
    ax2.set_ylabel("PCA participation ratio (delay period, load3)")
    ax2.set_title("9.1: geometry vs H")

    fig.suptitle("Phase 9.1 capacity curve (M00000 architecture, Sternberg)")
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"[analyze_capacity_curve] wrote {OUT_FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
