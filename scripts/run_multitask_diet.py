#!/usr/bin/env python3
"""Phase 5 acceptance check (comments.txt §5): a 2,000-step run interleaving
all 6 tasks in the multi-task diet (image Sternberg + 5 NeuroGym tasks),
one task drawn uniformly per training step (item 5.2), reporting per-task
mean reward against a random-policy chance baseline ("learns above chance
on every task in the diet"). Not a permanent regression test -- a one-off
acceptance run, output pasted into executor.md.

    python scripts/run_multitask_diet.py --steps 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import load_config  # noqa: E402
from brainalign_wm.tasks.multitask import (
    C_DIM_MULTITASK,
    DIET_TASKS,
    NEUROGYM_TASKS,
    NeuroGymAdapter,
    NeuroGymBatchEnv,
    obs_dim_for,
    pick_task,
    task_context_vector,
)  # noqa: E402
from brainalign_wm.training.train import (
    _build_model,
    _image_features_all_ticks,
    _init_state,
    _step_core,
    _target_action,
    run_multitask_neurogym_trial,
)  # noqa: E402


def run_sternberg_diet_trial(front_end_mt, core, heads, image_bank, trial_steps_batch, feature_dim, device):
    """The Sternberg slice of the diet: `_run_trial`'s own "ce"-signal
    tick loop, condensed for the M=0/P=0/T=0/D=0 baseline cell (no
    reflective gate, identity-catch, or ablation-battery penalties -- none
    apply to this acceptance run), and using `front_end_mt` (task_vec_dim
    = C_DIM_MULTITASK = 13) instead of the Sternberg-only pipeline's own
    10-dim `FrontEnd` -- the two are separate module instances, so nothing
    about Phases 0-4's tested front end changes."""
    B = len(trial_steps_batch)
    T = len(trial_steps_batch[0])
    state = _init_state(core, 0, 0, B, device)
    all_v = _image_features_all_ticks(image_bank, trial_steps_batch, feature_dim, device)
    probe_i = next(i for i, s in enumerate(trial_steps_batch[0]) if s.epoch == "probe")
    true_in_set = torch.tensor([bool(trial_steps_batch[b][probe_i].in_set) for b in range(B)], device=device)
    last_action = torch.full((B,), -1, dtype=torch.long, device=device)
    total_loss = torch.zeros((), device=device)
    n_terms = 0.0

    for i in range(T):
        ts_list = [trial_steps_batch[b][i] for b in range(B)]
        epoch = ts_list[0].epoch
        v_t = all_v[i]
        c_t = torch.tensor(
            [task_context_vector("sternberg", ts.c_t) for ts in ts_list], dtype=torch.float32, device=device
        )
        z_t = front_end_mt(v_t, c_t)
        h_star, new_state, _u_t = _step_core(core, 0, 0, 0, z_t, state, t=i, gate_bias=None)
        policy, _value, logits = heads(h_star)
        targets = torch.tensor([_target_action(ts.epoch, ts.in_set) for ts in ts_list], device=device)
        tick_weight = 1.0 if epoch == "probe" else 0.1
        total_loss = total_loss + tick_weight * F.cross_entropy(logits, targets)
        n_terms += tick_weight
        if epoch == "probe":
            last_action = torch.argmax(policy, dim=-1)
        state = new_state

    correct = (last_action == 1) == true_in_set
    return total_loss / n_terms, correct.float().tolist()


def chance_reward(task: str, batch_size: int, max_ticks: int, seed: int) -> float:
    """Mean per-trial reward under a uniform-random head-action policy --
    the "chance" baseline each task's trained reward must clear."""
    batch_env = NeuroGymBatchEnv(task, batch_size=batch_size, seed=seed)
    done = np.zeros(batch_size, dtype=bool)
    total = np.zeros(batch_size, dtype=np.float32)
    rng = np.random.RandomState(seed)
    for _ in range(max_ticks):
        if done.all():
            break
        actions = rng.randint(0, 3, size=batch_size)
        _obs, reward, _gt, newly_done = batch_env.step(actions)
        total += reward * (~done)
        done = done | newly_done
    return float(total.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full_cfg = load_config()
    m = full_cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_ticks = int(full_cfg["task"]["multitask_max_ticks"])

    front_end, core, heads = _build_model(full_cfg, S=0, M=0, P=0, device=device)
    from brainalign_wm.models.front_end import FrontEnd

    front_end_mt = FrontEnd(m["feature_dim"], C_DIM_MULTITASK, m["bottleneck"], m["input_noise_sigma"]).to(device)
    adapters = {
        task: NeuroGymAdapter(obs_dim_for(task), C_DIM_MULTITASK, m["bottleneck"]).to(device) for task in NEUROGYM_TASKS
    }

    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    image_bank = ImageTokenBank(
        stimuli_root=Path(full_cfg["paths"]["stimuli"]), categories=full_cfg["task"]["categories"],
        feature_cache_path=Path(full_cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(full_cfg, image_bank, seed=args.seed)

    params = list(front_end_mt.parameters()) + list(core.parameters()) + list(heads.parameters())
    for adapter in adapters.values():
        params += list(adapter.parameters())
    optimizer = torch.optim.Adam(params, lr=full_cfg["train"]["lr"])

    running_reward: dict[str, list[float]] = {t: [] for t in DIET_TASKS}
    for step in range(args.steps):
        task = pick_task(args.seed, step)
        optimizer.zero_grad()
        if task == "sternberg":
            batch = task_gen.sample_batch(step, args.steps, args.batch_size)
            loss, reward = run_sternberg_diet_trial(front_end_mt, core, heads, image_bank, batch, m["feature_dim"], device)
        else:
            batch_env = NeuroGymBatchEnv(task, batch_size=args.batch_size, seed=args.seed * 1_000_003 + step)
            loss, reward = run_multitask_neurogym_trial(
                adapters[task], core, heads, S=0, M=0, P=0, task_name=task, batch_env=batch_env,
                B=args.batch_size, max_ticks=max_ticks, device=device, mode="bptt",
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        running_reward[task].extend(reward)
        if (step + 1) % 200 == 0:
            print(f"[diet] step {step+1}/{args.steps} task={task} loss={float(loss):.4f}", flush=True)

    print("\n=== per-task trained mean reward (last 100 trials seen) vs. chance ===")
    for task in DIET_TASKS:
        trained = running_reward[task][-100:]
        trained_mean = float(np.mean(trained)) if trained else float("nan")
        if task == "sternberg":
            chance = 0.5  # binary in/out judgment, uniform-random guess
        else:
            chance = chance_reward(task, batch_size=200, max_ticks=max_ticks, seed=999)
        print(f"{task:24s} n_trials_seen={len(running_reward[task]):5d} trained_mean_reward={trained_mean:+.4f} chance={chance:+.4f}")


if __name__ == "__main__":
    main()
