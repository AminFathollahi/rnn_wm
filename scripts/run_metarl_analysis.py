#!/usr/bin/env python3
"""Phase 7 acceptance check (comments.txt §5, items 7.1/7.2): demonstration
METARL runs and their decoding analyses -- NOT a full Stage-1/2 grid run
(that's Phase 11's job), just enough training signal to show the mechanism.

7.1: trains one S=0 METARL run, then decodes `n` and `feature` from `h_star`
at each trial-index-within-block (1..K) over many fresh eval blocks, and
plots decode accuracy vs. trial-index.

7.2: trains one S=1 METARL run, decodes `n` from `h_worker` and `h_manager`
SEPARATELY (same trial-index axis), and reports the comparison -- does the
manager carry the inferred task variable more/faster than the worker.

`nback.sequence_length` is overridden down from the global config default
(20) to a shorter value here (see `--sequence-length`) purely for this
run's wall-clock budget -- `ImageTokenBank.sample`'s per-call linear scan
over the stimuli pool (not this phase's fix, see executor.md) dominates
per-step cost far more than the GPU forward/backward, so a shorter n-back
sequence (fewer stimulus-sample calls per block) buys a proportionally
faster demonstration run without touching the config default every other
n-back consumer uses. `metarl_block_size` (K) is NOT shortened -- that is
the spec's actual "BLOCKS of K=20" parameter, not a free efficiency knob.

    python scripts/run_metarl_analysis.py --steps-s0 3000 --steps-s1 2000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding
from brainalign_wm.tasks.image_token_bank import ImageTokenBank
from brainalign_wm.tasks.nback import NBackGenerator
from brainalign_wm.training.train import MetaRLAdapter, _build_model, run_metarl_block, sample_metarl_block

ROOT = Path(__file__).resolve().parent.parent


def train_metarl(cfg: dict, S: int, steps: int, batch_size: int, seed: int, device, image_bank, nback_gen):
    m = cfg["model"]
    _front_end, core, heads = _build_model(cfg, S=S, M=0, P=0, device=device)
    adapter = MetaRLAdapter(m["feature_dim"], m["action_dim"], m["bottleneck"]).to(device)
    params = list(adapter.parameters()) + list(core.parameters()) + list(heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    block_size = int(cfg["train"]["metarl_block_size"])
    sequence_length = int(cfg["nback"]["sequence_length"])

    for step in range(steps):
        batch, _n, _f = sample_metarl_block(
            nback_gen, cfg, seed=seed, step_idx=step, batch_size=batch_size, block_size=block_size,
        )
        optimizer.zero_grad()
        out = run_metarl_block(
            adapter, core, heads, S, 0, 0, image_bank, batch, sequence_length=sequence_length,
            feature_dim=m["feature_dim"], n_actions=m["action_dim"], device=device, mode="bptt",
        )
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        if (step + 1) % 200 == 0:
            print(f"[metarl S={S}] step {step + 1}/{steps} loss={out['loss'].item():.4f}", flush=True)
    return core, heads, adapter


def eval_decoding(
    cfg: dict, S: int, core, heads, adapter, image_bank, nback_gen, n_blocks: int, seed: int, device,
    eval_chunk: int = 32,
) -> dict:
    """Fresh eval blocks (split="test"), decoding `n` and `feature` at each
    of the K trial-index positions from `h_star` (S=0/S=1) and, for S=1,
    `h_worker`/`h_manager` separately. Reuses `cross_temporal_decoding`
    (StratifiedKFold + StandardScaler + LogisticRegression) with block-
    position substituted for its usual within-trial timebin axis -- only
    the DIAGONAL (decode at position k, test at position k) is kept; no
    cross-position generalization matrix is needed for this analysis."""
    block_size = int(cfg["train"]["metarl_block_size"])
    sequence_length = int(cfg["nback"]["sequence_length"])
    m = cfg["model"]

    h_star_chunks, h_worker_chunks, h_manager_chunks, n_all, feat_all = [], [], [], [], []
    remaining, chunk_idx = n_blocks, 0
    while remaining > 0:
        bsz = min(eval_chunk, remaining)
        batch, n_labels, feature_labels = sample_metarl_block(
            nback_gen, cfg, seed=seed, step_idx=1_000_000 + chunk_idx, batch_size=bsz, block_size=block_size,
            split="test",
        )
        with torch.no_grad():
            out = run_metarl_block(
                adapter, core, heads, S, 0, 0, image_bank, batch, sequence_length=sequence_length,
                feature_dim=m["feature_dim"], n_actions=m["action_dim"], device=device, mode="eval",
            )
        h_star_chunks.append(out["h_star"])
        n_all.extend(n_labels)
        feat_all.extend(feature_labels)
        if S == 1:
            h_worker_chunks.append(out["h_worker"])
            h_manager_chunks.append(out["h_manager"])
        remaining -= bsz
        chunk_idx += 1

    h_star_all = np.concatenate(h_star_chunks, axis=0)  # [n_blocks, K, D]
    n_all = np.array(n_all)
    feat_all = np.array(feat_all)

    result = {
        "n_decode": np.diag(cross_temporal_decoding(h_star_all, n_all, n_folds=4, seed=0)),
        "feature_decode": np.diag(cross_temporal_decoding(h_star_all, feat_all, n_folds=4, seed=0)),
    }
    if S == 1:
        h_worker_all = np.concatenate(h_worker_chunks, axis=0)
        h_manager_all = np.concatenate(h_manager_chunks, axis=0)
        result["n_decode_worker"] = np.diag(cross_temporal_decoding(h_worker_all, n_all, n_folds=4, seed=0))
        result["n_decode_manager"] = np.diag(cross_temporal_decoding(h_manager_all, n_all, n_folds=4, seed=0))
    return result


def make_plot(n_decode, feature_decode, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(n_decode)
    x = np.arange(1, K + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, n_decode, marker="o", label="decode n (3-way chance=0.33)")
    ax.plot(x, feature_decode, marker="s", label="decode feature (2-way chance=0.50)")
    ax.axhline(1 / 3, color="C0", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0.5, color="C1", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("trial index within block")
    ax.set_ylabel("decoding accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Phase 7 item 7.1: METARL task-variable decoding vs. trial index")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[metarl] wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-s0", type=int, default=3000)
    ap.add_argument("--steps-s1", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-blocks", type=int, default=300)
    ap.add_argument("--sequence-length", type=int, default=8, help="overrides nback.sequence_length for this run's wall-clock budget only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full_cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    full_cfg = {**full_cfg, "nback": {**full_cfg["nback"], "sequence_length": args.sequence_length}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_bank = ImageTokenBank(
        stimuli_root=ROOT / full_cfg["paths"]["stimuli"], categories=full_cfg["task"]["categories"],
        feature_cache_path=ROOT / full_cfg["paths"]["feature_cache"] / "image_token_bank.npy", seed=0,
    )
    nback_gen = NBackGenerator(full_cfg, image_bank)

    print("=== 7.1: S=0 METARL run ===", flush=True)
    core0, heads0, adapter0 = train_metarl(
        full_cfg, S=0, steps=args.steps_s0, batch_size=args.batch_size, seed=args.seed, device=device,
        image_bank=image_bank, nback_gen=nback_gen,
    )
    result0 = eval_decoding(
        full_cfg, S=0, core=core0, heads=heads0, adapter=adapter0, image_bank=image_bank, nback_gen=nback_gen,
        n_blocks=args.eval_blocks, seed=args.seed + 500_000, device=device,
    )
    print("\nn_decode:      ", np.round(result0["n_decode"], 3).tolist())
    print("feature_decode:", np.round(result0["feature_decode"], 3).tolist())
    make_plot(result0["n_decode"], result0["feature_decode"], ROOT / "results" / "figures" / "metarl_decoding_7_1.png")

    print("\n=== 7.2: S=1 METARL run ===", flush=True)
    core1, heads1, adapter1 = train_metarl(
        full_cfg, S=1, steps=args.steps_s1, batch_size=args.batch_size, seed=args.seed, device=device,
        image_bank=image_bank, nback_gen=nback_gen,
    )
    result1 = eval_decoding(
        full_cfg, S=1, core=core1, heads=heads1, adapter=adapter1, image_bank=image_bank, nback_gen=nback_gen,
        n_blocks=args.eval_blocks, seed=args.seed + 500_000, device=device,
    )
    print("\nn_decode (h_star):   ", np.round(result1["n_decode"], 3).tolist())
    print("n_decode (h_worker): ", np.round(result1["n_decode_worker"], 3).tolist())
    print("n_decode (h_manager):", np.round(result1["n_decode_manager"], 3).tolist())
    print(
        "\nmean n_decode -- worker: %.3f, manager: %.3f (%s carries n more strongly)"
        % (
            float(np.mean(result1["n_decode_worker"])),
            float(np.mean(result1["n_decode_manager"])),
            "manager" if np.mean(result1["n_decode_manager"]) > np.mean(result1["n_decode_worker"]) else "worker",
        )
    )


if __name__ == "__main__":
    main()
