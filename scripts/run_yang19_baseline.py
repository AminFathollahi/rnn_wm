#!/usr/bin/env python3
"""Item 10.1 Tier 1 (comments.txt §5): trains THIS repo's canonical
S=0,M=0,P=0 cell on Yang's 20-task suite (`neurogym.envs.collections.
yang19`, `multitask.py::Yang19BatchEnv`), matched against the identical
architecture trained on Sternberg alone.

The Sternberg-only side of the comparison is NOT retrained here -- it
already exists: `M00000_teacher_s0` (Phase 9c prep, this session) is the
exact same S=0,M=0,P=0,T=0,D=0 cell, trained to the real §3 criterion at
tier=full (`results/manifest.jsonl`, `criterion_met=true`,
`steps_to_criterion=70000`). Retraining a second, redundant Sternberg-only
run would just be the same experiment twice; reusing it is the "same
seed, same everything else" match comments.txt asks for, for free.

Only the Yang19-diet side needs new training, via its own loop (below) --
`training/train.py::train_one` is Sternberg-gate-specific (criterion keyed
to load1/2/3, `evaluate_accuracy` computes per-load Sternberg accuracy),
so it cannot run a Yang-19 diet directly. This script reuses `_build_model`
for the CORE (the actual thing under scientific comparison -- same cell,
same width, same optimizer) but builds its own input adapter and output
head, since Yang-19's native action space (Discrete(17), one fixation +
16-direction ring) is dimensionally incompatible with the shared 3-action
head the Sternberg/6-task-diet pipeline uses (`models/heads.py::N_ACTIONS`
is untouched -- this only builds a second, separate `Heads(n_actions=17)`
instance for this comparison, exactly the same "per-task-family adapter,
shared core" convention `multitask.py`'s 6-task diet already established).

Train-to-criterion: no `gates.criterion` exists for Yang-19 (Sternberg-
specific). Defines its own: pooled per-tick decision accuracy (excluding
routine fixation ticks, matching `run_multitask_neurogym_trial`'s own
tick-weighting rationale) over a trailing window must exceed
`--criterion-acc`, sustained for `--consecutive-evals` consecutive
periodic evals -- same persistence-based anti-noise logic as `train.py`'s
Sternberg criterion, applied to Yang-19's own accuracy metric. Snapshots
`ckpt_at_criterion.pt` there (same hybrid-policy convention as `train.py`,
this session's "Train-to-criterion hybrid policy" commit) and keeps
training to `--steps` for an equal-duration `ckpt.pt`.

    python scripts/run_yang19_baseline.py --steps 30000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.tasks.multitask import YANG19_TASKS, Yang19BatchEnv, pick_task  # noqa: E402
from brainalign_wm.training.train import _build_model, _init_state, _step_core  # noqa: E402

RUN_ID = "M00000_yang19_s{seed}"


class Yang19Adapter(nn.Module):
    """Raw-obs(33) + task-one-hot(20) straight to bottleneck width --
    mirrors `multitask.NeuroGymAdapter`'s call shape exactly, just sized
    for Yang-19's own obs/task-count (a separate module instance; nothing
    about the 6-task diet's `NeuroGymAdapter`/`C_DIM_MULTITASK` changes)."""

    def __init__(self, obs_dim: int, n_tasks: int, bottleneck_dim: int):
        super().__init__()
        self.w = nn.Linear(obs_dim + n_tasks, bottleneck_dim)

    def forward(self, obs_t: torch.Tensor, task_onehot: torch.Tensor) -> torch.Tensor:
        return self.w(torch.cat([obs_t, task_onehot], dim=-1))


def task_onehot_vec(task_name: str) -> list[float]:
    v = [0.0] * len(YANG19_TASKS)
    v[YANG19_TASKS.index(task_name)] = 1.0
    return v


def run_yang19_trial(adapter, core, heads, batch_env, B: int, max_ticks: int, device, task_onehot):
    """One trial rollout + BPTT loss, same tick-weighted cross-entropy
    convention as `train.py::run_multitask_neurogym_trial`'s `has_gt`
    branch (every yang19 task provides a `gt`, verified empirically --
    see executor.md). Returns (loss, n_decision_ticks_correct, n_decision_ticks,
    reward_per_trial)."""
    c_t = task_onehot.expand(B, -1)
    state = _init_state(core, 0, 0, B, device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    reward_per_trial = torch.zeros(B, device=device)
    ce_terms = []
    n_correct, n_decision = 0, 0

    for t in range(max_ticks):
        if bool(done.all()):
            break
        active = (~done).float()
        obs_t = torch.as_tensor(batch_env.obs, dtype=torch.float32, device=device)
        z_t = adapter(obs_t, c_t)
        h_star, state, _u = _step_core(core, 0, 0, 0, z_t, state, t=t, gate_bias=None)
        _policy, _value, logits = heads(h_star)
        action = torch.multinomial(torch.softmax(logits, dim=-1).detach(), 1).squeeze(-1)
        obs_np, reward_np, gt_head_np, newly_done_np = batch_env.step(action.cpu().numpy())
        reward_t = torch.as_tensor(reward_np, dtype=torch.float32, device=device) * active
        reward_per_trial = reward_per_trial + reward_t

        gt_t = torch.as_tensor(gt_head_np, dtype=torch.long, device=device)
        ce_per_sample = F.cross_entropy(logits, gt_t, reduction="none")
        # Fixation (gt==0) ticks dominate a trial; down-weight them the same
        # way `run_multitask_neurogym_trial` already does for the 6-task
        # diet, so training doesn't just learn "always fixate".
        per_sample_weight = torch.where(gt_t != 0, 1.0, 0.1) * active
        ce_terms.append((ce_per_sample * per_sample_weight).sum() / per_sample_weight.sum().clamp_min(1e-8))

        with torch.no_grad():
            decision_mask = (gt_t != 0) & (active.bool())
            n_decision += int(decision_mask.sum().item())
            n_correct += int(((action == gt_t) & decision_mask).sum().item())

        done = done | torch.as_tensor(newly_done_np, dtype=torch.bool, device=device)

    loss = torch.stack(ce_terms).mean()
    return loss, n_correct, n_decision, reward_per_trial.detach().cpu().tolist()


@torch.no_grad()
def eval_yang19(adapter, core, heads, tasks, batch_size, max_ticks, device, seed):
    """Fresh-seeded (disjoint from training) per-decision-tick accuracy,
    pooled over `tasks` -- the Yang-19 analog of `evaluate_accuracy`."""
    n_correct, n_decision = 0, 0
    for i, task in enumerate(tasks):
        env = Yang19BatchEnv(task, batch_size=batch_size, seed=900_000_000 + seed * 1000 + i)
        onehot = torch.tensor(task_onehot_vec(task), dtype=torch.float32, device=device)
        c_t = onehot.expand(batch_size, -1)
        state = _init_state(core, 0, 0, batch_size, device)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(max_ticks):
            if bool(done.all()):
                break
            obs_t = torch.as_tensor(env.obs, dtype=torch.float32, device=device)
            z_t = adapter(obs_t, c_t)
            h_star, state, _u = _step_core(core, 0, 0, 0, z_t, state, t=t, gate_bias=None, training=False)
            _policy, _value, logits = heads(h_star)
            action = torch.argmax(logits, dim=-1)
            obs_np, _r, gt_head_np, newly_done_np = env.step(action.cpu().numpy())
            active = ~done
            gt_t = torch.as_tensor(gt_head_np, dtype=torch.long, device=device)
            decision_mask = (gt_t != 0) & active
            n_decision += int(decision_mask.sum().item())
            n_correct += int(((action == gt_t) & decision_mask).sum().item())
            done = done | torch.as_tensor(newly_done_np, dtype=torch.bool, device=device)
    return n_correct / n_decision if n_decision else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30000, help="trials (one Yang-19 trial per step)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--criterion-acc", type=float, default=0.90)
    ap.add_argument("--consecutive-evals", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=100)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    full_cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    m = full_cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _fe, core, _heads = _build_model(full_cfg, S=0, M=0, P=0, device=device)
    from brainalign_wm.models.heads import Heads

    heads = Heads(m["flat_units"], readout_scale=m["readout_scale"], n_actions=17).to(device)
    adapter = Yang19Adapter(33, len(YANG19_TASKS), m["bottleneck"]).to(device)
    params = list(core.parameters()) + list(heads.parameters()) + list(adapter.parameters())
    optimizer = torch.optim.Adam(params, lr=full_cfg["train"]["lr"])

    run_id = RUN_ID.format(seed=args.seed)
    ckpt_dir = ROOT / "results" / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / "results" / "metrics" / f"{run_id}.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    onehots = {t: torch.tensor(task_onehot_vec(t), dtype=torch.float32, device=device) for t in YANG19_TASKS}
    envs = {t: Yang19BatchEnv(t, batch_size=args.batch_size, seed=args.seed * 1_000_003) for t in YANG19_TASKS}

    consecutive = 0
    criterion_met = False
    steps_to_criterion = None
    wall_s_to_criterion = None
    t0 = time.time()

    with open(metrics_path, "w") as mf:
        mf.write("step,task,loss,decision_acc,eval_acc,wall_s\n")

        for step in range(args.steps):
            task = pick_task(args.seed, step, tasks=YANG19_TASKS)
            optimizer.zero_grad()
            loss, n_correct, n_decision, _reward = run_yang19_trial(
                adapter, core, heads, envs[task], args.batch_size, args.max_ticks, device, onehots[task],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            envs[task] = Yang19BatchEnv(task, batch_size=args.batch_size, seed=args.seed * 1_000_003 + step)

            decision_acc = n_correct / n_decision if n_decision else float("nan")
            eval_acc = ""
            if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
                eval_acc_val = eval_yang19(adapter, core, heads, YANG19_TASKS, 64, args.max_ticks, device, args.seed)
                eval_acc = f"{eval_acc_val:.4f}"
                print(f"[yang19] step {step+1}/{args.steps} task={task} loss={loss.item():.4f} "
                      f"pooled_eval_decision_acc={eval_acc_val:.4f}", flush=True)
                if not criterion_met:
                    if eval_acc_val >= args.criterion_acc:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= args.consecutive_evals:
                        criterion_met = True
                        steps_to_criterion = step + 1
                        wall_s_to_criterion = round(time.time() - t0, 1)
                        torch.save(
                            {"step": step + 1, "core": core.state_dict(), "heads": heads.state_dict(),
                             "adapter": adapter.state_dict()},
                            ckpt_dir / "ckpt_at_criterion.pt",
                        )
                        print(f"[yang19] criterion ({args.criterion_acc}) met at step {steps_to_criterion} "
                              f"({args.consecutive_evals} consecutive evals); snapshotting, continuing to "
                              f"{args.steps} for an equal-duration checkpoint.", flush=True)

            mf.write(f"{step+1},{task},{loss.item():.6f},{decision_acc:.4f},{eval_acc},{time.time()-t0:.1f}\n")

    torch.save(
        {"step": args.steps, "core": core.state_dict(), "heads": heads.state_dict(), "adapter": adapter.state_dict()},
        ckpt_dir / "ckpt.pt",
    )
    final_eval_acc = eval_yang19(adapter, core, heads, YANG19_TASKS, 200, args.max_ticks, device, args.seed + 1)
    wall_clock_s = round(time.time() - t0, 1)

    record = {
        "run_id": run_id, "seed": args.seed, "steps": args.steps, "criterion_met": criterion_met,
        "steps_to_criterion": steps_to_criterion, "trials_to_criterion":
            steps_to_criterion * args.batch_size if steps_to_criterion else None,
        "wall_s_to_criterion": wall_s_to_criterion, "final_eval_pooled_decision_acc": final_eval_acc,
        "wall_clock_s": wall_clock_s,
    }
    (ROOT / "results" / f"{run_id}_summary.json").write_text(json.dumps(record, indent=2))
    print(f"\n[yang19] {json.dumps(record, indent=2)}")
    print(f"[yang19] wrote {ckpt_dir}/ckpt.pt, {ckpt_dir}/ckpt_at_criterion.pt "
          f"({'exists' if criterion_met else 'not written -- criterion never met'}), "
          f"{metrics_path}, results/{run_id}_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
