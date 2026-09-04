#!/usr/bin/env python3
"""Measure real peak GPU memory for one training step of a battery cell and
compare it against `run_grid._run_mib`'s prediction.

Not a training run. For each cell this builds the model and optimiser from
`results/resolved_config_grid_SUP.yaml`, runs ONE forward + backward +
`optimizer.step()` on a full `batch_size: 128` batch of the longest trial the
curriculum reaches (load 3, 76 ticks), and reads
`torch.cuda.max_memory_allocated()`. No manifest row, checkpoint, or metrics
CSV is written -- there is nothing to archive.

Every cell runs in its own fresh subprocess so one cell's CUDA allocator
state cannot flatter the next (`--cell` is the internal single-cell worker
mode; invoking the script with no arguments is the orchestrator that spawns
one subprocess per cell and prints the acceptance table).

D41's checkpointing fix has never executed on this GPU (D49, comments.txt
§21.1): every manifest row for the four S=1-plastic cells predates the
commit that landed it. This script is the cheap way to find out whether the
fix's derivation matches the card before spending any GPU-hours on it.

    PY=/home/amin/miniconda3/envs/wm_dynamics/bin/python
    $PY scripts/probe_peak_memory.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_grid  # noqa: E402
from brainalign_wm.config import get_path, load_config  # noqa: E402

RESOLVED_CONFIG = get_path("results") / "resolved_config_grid_SUP.yaml"
CARD_CAPACITY_MIB = 12227

# D49's cell list (comments.txt §21.1): the S=1 plastic worst case that
# actually crashed, the S=1 non-plastic `_BASE_MIB`-branch cell at its
# largest, the S=0 plastic survivor that calibrates the formula against a
# real completion, and the cheapest cell as the floor.
DEFAULT_CELLS = ["M11111", "M11011", "M01111", "M00000"]


def _bits(model_id: str) -> tuple[int, int, int, int, int]:
    digits = model_id[1:]
    if len(digits) != 5 or not digits.isdigit():
        raise ValueError(f"expected a model_id like 'M11111', got {model_id!r}")
    s, m, p, t, d = (int(c) for c in digits)
    return s, m, p, t, d


def load_full_cfg(checkpointing: bool | None = None) -> dict:
    cfg = load_config(RESOLVED_CONFIG)
    if checkpointing is not None:
        cfg = {**cfg, "mechanisms": {**cfg["mechanisms"], "plastic_gradient_checkpointing": checkpointing}}
    return cfg


def build_load3_batch(task_gen, cfg: dict, batch_size: int, seed: int) -> list:
    """B independent load-3 trials, the same generator call `evaluate_accuracy`
    uses (`train.py:1555`), just fixed at the curriculum's longest load."""
    rng = np.random.RandomState(seed)
    identity_catch_fraction = float(cfg["task"].get("identity_catch_fraction", 0.0))
    batch = []
    for _ in range(batch_size):
        trial_seed = int(rng.randint(0, 2**31 - 1))
        trial_rng = np.random.RandomState(trial_seed)
        steps = task_gen.sternberg.generate_trial(
            trial_rng, loads=[3], lure_fraction=cfg["task"]["lure_fraction"],
            maintain_steps=cfg["task"]["maintain_steps"], trial_id=-1, split="train",
            identity_catch_fraction=identity_catch_fraction,
        )
        batch.append(steps)
    return batch


def probe_one_cell(model_id: str, checkpointing: bool, seed: int = 0) -> dict:
    """Runs in-process; the caller (`main`'s worker mode) is what a fresh
    subprocess invokes. Returns measured peak MiB and the trial length
    actually used, or an `error` field if CUDA raised (e.g. the D49 control:
    `M11111` with checkpointing forced off)."""
    import torch

    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate
    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.training.train import _build_model, _dale_penalty, _gate_width, _run_trial
    from brainalign_wm.utils.seeding import seed_everything

    assert torch.cuda.is_available(), "probe_peak_memory measures CUDA peak memory; no GPU visible"
    device = torch.device("cuda")
    seed_everything(seed)

    full_cfg = load_full_cfg(checkpointing=checkpointing)
    S, M, P, T, D = _bits(model_id)

    image_bank = ImageTokenBank(
        stimuli_root=Path(full_cfg["paths"]["stimuli"]), categories=full_cfg["task"]["categories"],
        feature_cache_path=Path(full_cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(full_cfg, image_bank, seed=seed)
    trial_batch = build_load3_batch(task_gen, full_cfg, int(full_cfg["train"]["batch_size"]), seed=seed)

    front_end, core, heads = _build_model(full_cfg, S, M, P, device)
    reflective_gate = ReflectiveGate(full_cfg["mechanisms"]["reflection_lambda"], full_cfg["mechanisms"]["reflection_beta"]) if M else None
    params = list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters())
    optimizer = torch.optim.Adam(params, lr=full_cfg["train"]["lr"])

    m, t_cfg = full_cfg["model"], full_cfg["train"]
    topo_loss_weight = float(t_cfg.get("topo_loss_weight_on", 0.01)) if T else float(t_cfg.get("topo_loss_weight", 0.0))
    dale_penalty_weight = float(t_cfg.get("dale_penalty_weight_on", 0.01)) if D else float(t_cfg.get("dale_penalty_weight", 0.0))
    checkpoint_plastic = bool(P) and bool(full_cfg["mechanisms"].get("plastic_gradient_checkpointing", True))

    torch.cuda.reset_peak_memory_stats(device)
    try:
        optimizer.zero_grad()
        loss, _correct, _reward, _identity = _run_trial(
            front_end, core, heads, reflective_gate, S, M, P, trial_batch, image_bank,
            m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="bptt",
            signal="ce", value_weight=float(t_cfg["value_loss_weight"]),
            entropy_coef=float(t_cfg.get("entropy_coef", 0.0)),
            topo_loss_weight=topo_loss_weight, flat_grid=tuple(m.get("flat_grid", [16, 16])),
            categories=full_cfg["task"]["categories"], checkpoint_plastic=checkpoint_plastic,
        )
        if dale_penalty_weight > 0:
            loss = loss + dale_penalty_weight * _dale_penalty(core, S, float(m.get("dale_ei_split", 0.8)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as exc:
        return {"model_id": model_id, "checkpointing": checkpointing, "measured_ticks": len(trial_batch[0]),
                "error": f"OutOfMemoryError: {exc}"}

    measured_mib = torch.cuda.max_memory_allocated(device) / 2**20
    return {
        "model_id": model_id, "checkpointing": checkpointing,
        "measured_mib": round(measured_mib, 1), "measured_ticks": len(trial_batch[0]),
    }


def _predicted_mib(model_id: str, checkpointing: bool) -> int:
    s, _m, p, _t, _d = _bits(model_id)
    return run_grid._run_mib({"S": s, "P": p}, load_full_cfg(checkpointing=checkpointing))


def _print_table(rows: list[dict]) -> None:
    header = f"{'cell':<10} {'ckpt':<6} {'predicted_mib':>14} {'measured_mib':>13} {'ratio':>7}  note"
    print(header)
    print("-" * len(header))
    for r in rows:
        predicted = r.get("predicted_mib")
        if "error" in r:
            print(f"{r['model_id']:<10} {str(r['checkpointing']):<6} {predicted:>14} {'OOM':>13} {'>=1':>7}  {r['error'].splitlines()[0][:80]}")
            continue
        measured = r["measured_mib"]
        ratio = r["ratio"]
        note = "OK (conservative)" if ratio <= 1.0 else "MEASURED EXCEEDS PREDICTION -- estimator wrong"
        print(f"{r['model_id']:<10} {str(r['checkpointing']):<6} {predicted:>14} {measured:>13} {ratio:>7.3f}  {note}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", help="internal worker mode: probe exactly this one model_id in this process")
    ap.add_argument("--no-checkpointing", action="store_true",
                     help="internal worker mode: force mechanisms.plastic_gradient_checkpointing=false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cells", nargs="+", default=DEFAULT_CELLS,
                     help=f"orchestrator mode: cells to probe (default {DEFAULT_CELLS})")
    ap.add_argument("--skip-control", action="store_true",
                     help="orchestrator mode: skip the M11111-uncheckpointed OOM control")
    args = ap.parse_args(argv)

    if args.cell:
        result = probe_one_cell(args.cell, checkpointing=not args.no_checkpointing, seed=args.seed)
        print("RESULT " + json.dumps(result))
        return 0

    jobs = [(c, True) for c in args.cells]
    if not args.skip_control:
        jobs.append(("M11111", False))  # D41's control: must OOM or predict > card capacity

    rows = []
    for cell, checkpointing in jobs:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--cell", cell, "--seed", str(args.seed)]
        if not checkpointing:
            cmd.append("--no-checkpointing")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result_line = next((line for line in proc.stdout.splitlines() if line.startswith("RESULT ")), None)
        predicted = _predicted_mib(cell, checkpointing)
        if result_line is None:
            tail = "\n".join(proc.stderr.strip().splitlines()[-5:]) or f"subprocess exited {proc.returncode} with no RESULT line"
            rows.append({"model_id": cell, "checkpointing": checkpointing, "predicted_mib": predicted, "error": tail})
            continue
        result = json.loads(result_line[len("RESULT "):])
        if "error" in result:
            rows.append({**result, "predicted_mib": predicted})
        else:
            rows.append({**result, "predicted_mib": predicted, "ratio": round(result["measured_mib"] / predicted, 3)})

    _print_table(rows)

    over_budget = [r for r in rows if r.get("ratio", 0) > 1.0]
    if over_budget:
        print(f"\n{len(over_budget)} cell(s) measured ABOVE _run_mib's prediction -- "
              "the estimator needs correction, do not adjust launch parameters to fit. See comments.txt §21.1.")
        return 1
    print("\nAll checkpointed cells measured at or under their prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
