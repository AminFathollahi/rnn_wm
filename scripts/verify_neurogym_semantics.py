#!/usr/bin/env python3
"""Verify the six-task training interface, and that each auxiliary task is
learnable through it.

Two independent reports over the same six tasks (working-memory Sternberg
plus the five auxiliary environments):

  --contract  observation wiring, action-space mapping onto the shared
              3-way head, reward contract, per-trial state reset, and
              where within a trial reward arrives relative to the action
              that earned it.
  --learn     a bounded per-task training probe: a few hundred updates on
              one task at a time, reporting mean reward against that task's
              own measured random-policy reward.  Writes nothing outside
              --out; it trains no campaign run and records no manifest row.

    python scripts/verify_neurogym_semantics.py --contract
    python scripts/verify_neurogym_semantics.py --learn --updates 400
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import load_config  # noqa: E402
from brainalign_wm.tasks.multitask import (  # noqa: E402
    DIET_TASKS,
    ENV_GT_TO_HEAD,
    HAS_GT,
    HEAD_TO_ENV,
    NEUROGYM_TASKS,
    TASK_ACTION_DIM,
    NeuroGymAdapter,
    NeuroGymBatchEnv,
    make_env,
    obs_dim_for,
    task_context_vector,
)
from brainalign_wm.tasks.sternberg import context_vector  # noqa: E402
from brainalign_wm.training.train import _build_model, run_multitask_neurogym_trial  # noqa: E402
from scripts.run_multitask_diet import chance_reward  # noqa: E402

FIXATE = 0


def _rollout(task: str, batch: int, seed: int, max_ticks: int, actions) -> dict:
    """One batched rollout under a caller-supplied head-action rule.
    `actions(tick, gt_head)` returns the [B] head actions for that tick."""
    env = NeuroGymBatchEnv(task, batch, seed=seed)
    done = np.zeros(batch, dtype=bool)
    total = np.zeros(batch, dtype=np.float32)
    per_tick_reward, gt_trace, obs_trace = [], [], []
    length = np.zeros(batch, dtype=int)
    gt_head = np.zeros(batch, dtype=np.int64)
    for tick in range(max_ticks):
        if done.all():
            break
        obs_trace.append(env.obs.copy())
        act = actions(tick, gt_head)
        _obs, reward, gt_head_new, newly_done = env.step(act)
        gt_head = gt_head_new if gt_head_new is not None else gt_head
        active = ~done
        total += reward * active
        length += active
        per_tick_reward.append(reward * active)
        gt_trace.append(None if gt_head_new is None else gt_head_new.copy())
        done = done | newly_done
    return {
        "reward": total,
        "ticks": length,
        "per_tick_reward": per_tick_reward,
        "gt": gt_trace,
        "obs": obs_trace,
        "truncated": int((~done).sum()),
    }


def _contract(cfg: dict, batch: int, max_ticks: int) -> int:
    model = cfg["model"]
    failures = []

    def check(ok: bool, message: str) -> None:
        print(f"  {'ok  ' if ok else 'FAIL'} {message}")
        if not ok:
            failures.append(message)

    print("\n== task cue ==")
    cues = {t: tuple(task_context_vector(t)) for t in DIET_TASKS}
    check(len(set(cues.values())) == len(DIET_TASKS), f"{len(DIET_TASKS)} task codes are distinct")
    check(all(len(c) == model["task_vec_dim"] for c in cues.values()),
          f"every task code has the configured cue width {model['task_vec_dim']}")
    epochs = ["fixation", "encode", "maintain", "probe", "feedback", "iti"]
    passthrough = [context_vector(e, encoded_count=2) for e in epochs]
    check(all(task_context_vector("sternberg", c) == c for c in passthrough),
          f"the working-memory cue passes through unchanged at all {len(epochs)} epochs")
    for task in DIET_TASKS:
        print(f"       {task:24s} {list(cues[task])}")

    print("\n== observation wiring ==")
    device = torch.device("cpu")
    front_end, _core, heads = _build_model(cfg, 0, 0, 0, device)
    z = front_end(torch.zeros(4, model["feature_dim"]), torch.zeros(4, model["task_vec_dim"]))
    check(tuple(z.shape) == (4, model["bottleneck"]),
          f"sternberg: image features {model['feature_dim']} + cue {model['task_vec_dim']} "
          f"-> bottleneck {tuple(z.shape)}")
    for task in NEUROGYM_TASKS:
        obs_dim = obs_dim_for(task)
        adapter = NeuroGymAdapter(obs_dim, model["task_vec_dim"], model["bottleneck"])
        za = adapter(torch.zeros(4, obs_dim), torch.zeros(4, model["task_vec_dim"]))
        check(tuple(za.shape) == (4, model["bottleneck"]),
              f"{task}: observation {obs_dim} + cue {model['task_vec_dim']} -> bottleneck {tuple(za.shape)}")

    print("\n== action space ==")
    check(heads.pi.out_features == TASK_ACTION_DIM,
          f"one shared policy head of width {heads.pi.out_features} for every task")
    for task in NEUROGYM_TASKS:
        native = make_env(task).action_space.n
        mapping = HEAD_TO_ENV[task]
        check(set(mapping) == set(range(TASK_ACTION_DIM)) and all(0 <= v < native for v in mapping.values()),
              f"{task}: head {dict(mapping)} into native Discrete({native})")
        collapsed = [h for h in mapping if list(mapping.values()).count(mapping[h]) > 1]
        if collapsed:
            print(f"       head actions {collapsed} share a native action: "
                  f"the head is wider than this task's action space")

    print("\n== reward contract and per-trial reset ==")
    rng = np.random.RandomState(0)
    for task in NEUROGYM_TASKS:
        env_rewards = getattr(make_env(task).unwrapped, "rewards", {})
        draw = rng.randint(0, TASK_ACTION_DIM, size=(max_ticks, batch))
        first = _rollout(task, batch, 11, max_ticks, lambda t, g: draw[t])
        again = _rollout(task, batch, 11, max_ticks, lambda t, g: draw[t])
        other = _rollout(task, batch, 5_000, max_ticks, lambda t, g: draw[t])
        check(np.array_equal(first["reward"], again["reward"]) and np.array_equal(first["ticks"], again["ticks"]),
              f"{task}: a fresh batched environment at the same seed replays the same trials")
        check(not np.array_equal(first["reward"], other["reward"]),
              f"{task}: a different seed draws different trials")
        check(len(set(first["ticks"])) > 0 and first["ticks"].max() <= max_ticks,
              f"{task}: trial lengths {first['ticks'].min()}-{first['ticks'].max()} ticks "
              f"(cap {max_ticks}, {first['truncated']}/{batch} hit it), rewards {env_rewards}")
        has_gt = first["gt"][0] is not None
        check(has_gt == HAS_GT[task], f"{task}: ground-truth targets available = {has_gt}")
        if has_gt:
            seen = {int(v) for trace in first["gt"] for v in trace}
            check(seen <= set(ENV_GT_TO_HEAD[task].values()),
                  f"{task}: observed targets {sorted(seen)} lie in the head's action space")

    print("\n== reward timing and target alignment ==")
    for task in NEUROGYM_TASKS:
        fixating = _rollout(task, batch, 77, max_ticks, lambda t, g: np.full(batch, FIXATE))
        earning = [i for i, r in enumerate(fixating["per_tick_reward"]) if np.any(r != 0)]
        print(f"       {task:24s} holding the fixation action: {fixating['ticks'].max():2d} ticks max, "
              f"reward on ticks {earning or 'none'}, mean {fixating['reward'].mean():+.4f}")
        if not HAS_GT[task]:
            continue
        # The environment reports each tick's target for the observation
        # just acted on, so replaying that target as the action must score
        # far better than holding fixation.  A target reported one tick late
        # would score at or below the fixating policy.
        targets = [t.copy() for t in fixating["gt"]]
        replay = _rollout(task, batch, 77, max_ticks,
                          lambda t, g: targets[t] if t < len(targets) else np.full(batch, FIXATE))
        check(replay["reward"].mean() > fixating["reward"].mean(),
              f"{task}: replaying the reported target scores {replay['reward'].mean():+.4f} "
              f"against {fixating['reward'].mean():+.4f} for holding fixation")
    print("\n       sternberg: one reward per trial, at the feedback epoch, for the action taken "
          "at the probe epoch of the same trial.")

    print(f"\n{len(failures)} failures")
    return 1 if failures else 0


def _learn(cfg: dict, batch: int, updates: int, seed: int, out: Path) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_ticks = cfg["model"], int(cfg["task"]["multitask_max_ticks"])
    report = {}
    for task in NEUROGYM_TASKS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        _front_end, core, heads = _build_model(cfg, 0, 0, 0, device)
        adapter = NeuroGymAdapter(obs_dim_for(task), model["task_vec_dim"], model["bottleneck"]).to(device)
        params = list(core.parameters()) + list(heads.parameters()) + list(adapter.parameters())
        optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
        history, started = [], time.time()
        for step in range(updates):
            env = NeuroGymBatchEnv(task, batch, seed=seed * 1_000_003 + step * batch)
            optimizer.zero_grad()
            loss, reward = run_multitask_neurogym_trial(
                adapter, core, heads, 0, 0, 0, task, env, batch, max_ticks, device,
                value_weight=cfg["train"]["value_loss_weight"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            history.append(float(np.mean(reward)))
        block = max(1, updates // 10)
        chance = chance_reward(task, batch_size=512, max_ticks=max_ticks, seed=999)
        report[task] = {
            "updates": updates,
            "batch_size": batch,
            "chance_reward": round(chance, 4),
            "first_block_reward": round(float(np.mean(history[:block])), 4),
            "final_block_reward": round(float(np.mean(history[-block:])), 4),
            "above_chance": bool(np.mean(history[-block:]) > chance),
            "s_per_update": round((time.time() - started) / updates, 4),
        }
        r = report[task]
        print(f"{task:24s} chance={r['chance_reward']:+.4f} first={r['first_block_reward']:+.4f} "
              f"final={r['final_block_reward']:+.4f} above_chance={r['above_chance']} "
              f"{r['s_per_update']:.3f}s/update", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    (out / "auxiliary_task_learning.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out / 'auxiliary_task_learning.json'}")
    return 0 if all(r["above_chance"] for r in report.values()) else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", action="store_true", help="report the six-task interface contract")
    parser.add_argument("--learn", action="store_true", help="run the bounded per-task learning probe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--updates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("."), help="directory for the learning probe's report")
    args = parser.parse_args(argv)

    cfg = load_config()
    status = 0
    if args.contract or not args.learn:
        status |= _contract(cfg, args.batch_size, int(cfg["task"]["multitask_max_ticks"]))
    if args.learn:
        status |= _learn(cfg, args.batch_size, args.updates, args.seed, args.out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
