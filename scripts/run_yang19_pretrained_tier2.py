#!/usr/bin/env python3
"""Item 10.1 Tier 2 (comments.txt §5, SECONDARY/bonus): Yang's own 20
pretrained networks (`github.com/gyyang/multitask`), as an external,
independently-trained replication check -- NOT retrained or matched to
this repo in any way.

LOADING: comments.txt's own recipe -- do not install TF1/Python2; TF2
reads TF1 checkpoints directly via `tf.train.load_checkpoint(path)`
(every variable as a numpy array), and the leaky-RNN forward pass
    h_{t+1} = (1-alpha) h_t + alpha * f(W_rec h_t + W_in u_t + b)
is reimplemented in torch below (verified against the checkpoint's own
`hp.json`: rnn_type=LeakyRNN, activation=softplus, alpha=0.2, n_rnn=256,
n_input=85, n_output=33). No TF forward pass is ever run -- TF is only
used as a checkpoint reader.

EXTERNAL ASSETS (not committed to this repo -- ~270MB TF wheel + ~40MB
pretrained-model zip, neither belongs in git):
  1. `pip install tensorflow-cpu` (checkpoint reading only).
  2. Yang's 20 pretrained models: Google Drive folder linked from
     github.com/gyyang/multitask's README (`gdown --folder <url>`),
     unzip `train_all.zip` -> one directory (`0/`, `1/`, ... `39/`) per
     model, each with `hp.json` + `model.ckpt.*`.
  3. `git clone https://github.com/gyyang/multitask` -- only `task.py`
     is used here (pure numpy stimulus generation, no TF import at
     module level), for FAITHFUL trial generation instead of guessing
     at Yang's exact stimulus encoding.

WHAT THIS COMPUTES (comments.txt's explicit "CAN compare" list): the
task-agnostic geometry metric PROFILE -- `pca_participation_ratio`
(reused from `brainalign_wm/analysis/geometry.py`, the same function
Phase 8 already uses) on Yang's own hidden-state activity while it
performs its own tasks (dm1, contextdm1, multidm -- comments.txt's own
suggested set). Explicitly NOT attempted: the stimulus-matched
maintenance-epoch DV (comments.txt is explicit this cannot be faked by
projecting images into Yang's 85-d input space).

    python scripts/run_yang19_pretrained_tier2.py \\
        --model-dir /tmp/yang19_pretrained/.../train_all/0 \\
        --task-py /tmp/multitask_repo/task.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import get_path  # noqa: E402
from brainalign_wm.analysis.geometry import pca_participation_ratio  # noqa: E402


def load_task_module(task_py_path: str):
    spec = importlib.util.spec_from_file_location("yang_task", task_py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_weights(model_dir: Path) -> dict:
    import tensorflow as tf

    reader = tf.train.load_checkpoint(str(model_dir / "model.ckpt"))
    kernel = reader.get_tensor("rnn/leaky_rnn_cell/kernel")  # [n_input+n_rnn, n_rnn]
    bias = reader.get_tensor("rnn/leaky_rnn_cell/bias")  # [n_rnn]
    w_out = reader.get_tensor("output/weights")  # [n_rnn, n_output]
    b_out = reader.get_tensor("output/biases")  # [n_output]
    return {"kernel": kernel, "bias": bias, "w_out": w_out, "b_out": b_out}


def leaky_rnn_rollout(x: np.ndarray, weights: dict, alpha: float, n_rnn: int) -> np.ndarray:
    """`x`: [T, B, n_input] (Yang's own `Trial.x` layout). Returns hidden
    states `[T, B, n_rnn]`. `W_in = kernel[:n_input]`, `W_rec =
    kernel[n_input:]` -- TF's `BasicRNNCell`-style convention: pre-activation
    = concat([input, h]) @ kernel + bias (verified: kernel.shape[0] ==
    n_input + n_rnn)."""
    n_input = x.shape[-1]
    kernel = torch.as_tensor(weights["kernel"], dtype=torch.float32)
    w_in, w_rec = kernel[:n_input], kernel[n_input:n_input + n_rnn]
    bias = torch.as_tensor(weights["bias"], dtype=torch.float32)
    T, B, _ = x.shape
    h = torch.zeros(B, n_rnn)
    hs = []
    x_t_all = torch.as_tensor(x, dtype=torch.float32)
    for t in range(T):
        pre = x_t_all[t] @ w_in + h @ w_rec + bias
        h = (1 - alpha) * h + alpha * torch.nn.functional.softplus(pre)
        hs.append(h)
    return torch.stack(hs, dim=0).numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--task-py", type=str, required=True)
    ap.add_argument("--tasks", type=str, nargs="+", default=["dm1", "contextdm1", "multidm"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    model_dir = Path(args.model_dir)
    hp = json.loads((model_dir / "hp.json").read_text())
    weights = load_weights(model_dir)
    assert weights["kernel"].shape == (hp["n_input"] + hp["n_rnn"], hp["n_rnn"]), (
        f"unexpected kernel shape {weights['kernel'].shape} for n_input={hp['n_input']}, n_rnn={hp['n_rnn']}"
    )

    task_mod = load_task_module(args.task_py)
    config = dict(hp)
    config["rng"] = np.random.RandomState(args.seed)

    results = {}
    for task in args.tasks:
        gen_fn = getattr(task_mod, task)
        trial = gen_fn(config, mode="random", batch_size=args.batch_size)
        hs = leaky_rnn_rollout(trial.x, weights, alpha=hp["alpha"], n_rnn=hp["n_rnn"])
        # Pool every (trial x timebin) state vector during the stimulus/delay
        # period (before the go epoch) -- same "single-trial, pooled" PR
        # convention `participation_ratio_by_load` already uses for our own
        # networks (item 8.2).
        go_start = trial.epochs.get("go1", (None, None))[0]
        if go_start is None:
            go_start = trial.epochs["stim1"][1]
        go_on = int(np.asarray(go_start).reshape(-1)[0])  # scalar in "random" mode (uniform across the batch)
        pre_go = hs[:go_on].reshape(-1, hs.shape[-1])
        pr = pca_participation_ratio(pre_go)
        results[task] = {"pca_participation_ratio": pr, "n_timebins": int(go_on), "n_trials": args.batch_size}
        print(f"[yang19-tier2] task={task:14s} pre-go PR={pr:.3f} (T={go_on}, B={args.batch_size})", flush=True)

    out = get_path("results") / "yang19_pretrained_tier2_geometry.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model_dir": str(model_dir), "hp_subset": {
        k: hp[k] for k in ("n_rnn", "n_input", "n_output", "activation", "rnn_type", "alpha")
    }, "results": results}, indent=2))
    print(f"[yang19-tier2] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
