#!/usr/bin/env python3
"""Continue matched checkpoints on working memory alone or six tasks."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from brainalign_wm.config import get_path, load_config
from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate
from brainalign_wm.tasks.multitask import (
    DIET_TASKS,
    NEUROGYM_TASKS,
    TASK_CUE_SCHEMA,
    NeuroGymAdapter,
    NeuroGymBatchEnv,
    obs_dim_for,
    pick_task,
    task_context_vector,
    task_cue_metadata,
)
from brainalign_wm.training.train import (
    _build_model,
    _checkpoint_state,
    _dale_penalty,
    _gate_width,
    _init_state,
    _restore_checkpoint_rng,
    _run_trial,
    _step_core,
    final_evaluation,
    run_multitask_neurogym_trial,
)
from brainalign_wm.utils.seeding import seed_everything

RESULTS = get_path("results")
MANIFEST = RESULTS / "manifest.jsonl"
SELECTED_MODELS = ("M00000", "M10000", "M00100", "M11111")
DIET_TAGS = {"wm_only": "contwm", "multitask": "contmulti"}


def enumerate_continuations(
    seeds=range(8), models=SELECTED_MODELS, diets=("wm_only", "multitask"), updates: int = 24_000,
) -> list[dict]:
    """Enumerate paired continuation runs from preselected checkpoints."""
    runs = []
    for seed in seeds:
        for model_id in models:
            if len(model_id) != 6 or not model_id.startswith("M"):
                raise ValueError(f"invalid model id {model_id!r}")
            bits = {name: int(value) for name, value in zip("SMPTD", model_id[1:])}
            source = f"{model_id}_SUP_s{seed}"
            for diet in diets:
                if diet not in DIET_TAGS:
                    raise ValueError(f"unknown continuation diet {diet!r}")
                tag = DIET_TAGS[diet]
                runs.append({
                    **bits,
                    "model_id": f"{model_id}_{tag}",
                    "run_id": f"{model_id}_{tag}_SUP_s{seed}",
                    "source_run_id": source,
                    "seed": seed,
                    "continuation_seed": 10_000 + seed,
                    "diet": diet,
                    "supervision": "SUP",
                    "additional_updates": updates,
                    "optimizer_restart": "adam_fresh",
                })
    return runs


def _task_at(run: dict, step: int) -> str:
    return pick_task(run["continuation_seed"], step) if run["diet"] == "multitask" else "sternberg"


def _expected_exposures(run: dict, updates: int | None = None) -> dict[str, int]:
    stop = run["additional_updates"] if updates is None else updates
    if not 0 <= stop <= run["additional_updates"]:
        raise ValueError(f"invalid continuation update count {stop}")
    exposures = {task: 0 for task in DIET_TASKS}
    for step in range(stop):
        exposures[_task_at(run, step)] += 1
    return exposures


def _resume_progress(saved: dict, run: dict) -> tuple[int, dict[str, int]]:
    continuation = saved.get("continuation", {})
    for key in ("source_run_id", "additional_updates", "optimizer_restart"):
        if continuation.get(key) != run[key]:
            raise RuntimeError(f"continuation checkpoint has incompatible {key}")
    step = int(saved["step"])
    if int(continuation.get("completed_updates", step)) != step:
        raise RuntimeError("continuation checkpoint has inconsistent completed update counts")
    expected = _expected_exposures(run, step)
    exposures = {task: int(continuation.get("exposures", {}).get(task, 0)) for task in DIET_TASKS}
    if exposures != expected:
        raise RuntimeError("continuation checkpoint exposure counts do not match its update step")
    return step, exposures


def _wm_curriculum_position(exposures: dict[str, int], planned: dict[str, int]) -> tuple[int, int]:
    completed = exposures["sternberg"]
    total = planned["sternberg"]
    if not 0 <= completed <= total:
        raise RuntimeError("working-memory exposure count exceeds its continuation budget")
    return completed, total


def _state_digest(modules: list[torch.nn.Module]) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_source(run: dict, cfg: dict, device):
    front_end, core, heads = _build_model(cfg, run["S"], run["M"], run["P"], device)
    source_path = RESULTS / "checkpoints" / run["source_run_id"] / "ckpt.pt"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    source = torch.load(source_path, map_location=device, weights_only=False)
    front_end.load_state_dict(source["front_end"])
    core.load_state_dict(source["core"])
    heads.load_state_dict(source["heads"])
    adapters = {
        task: NeuroGymAdapter(obs_dim_for(task), cfg["model"]["task_vec_dim"], cfg["model"]["bottleneck"]).to(device)
        for task in NEUROGYM_TASKS
    }
    return front_end, core, heads, adapters


def verify_source(run: dict, cfg: dict, device) -> dict:
    """Verify cue and noiseless-forward compatibility before continuation."""
    seed_everything(run["continuation_seed"])
    front_end, core, heads, adapters = _load_source(run, cfg, device)
    from brainalign_wm.tasks.sternberg import context_vector

    base = context_vector("maintain")
    if task_context_vector("sternberg", base) != base:
        raise RuntimeError("working-memory cue changed under the task schema")
    original_noise = front_end.input_noise_sigma
    front_end.input_noise_sigma = 0.0
    features = torch.zeros(2, cfg["model"]["feature_dim"], device=device)
    cue_a = torch.tensor([base, base], dtype=torch.float32, device=device)
    cue_b = torch.tensor([task_context_vector("sternberg", base)] * 2, dtype=torch.float32, device=device)
    with torch.no_grad():
        state_a = front_end(features, cue_a)
        state_b = front_end(features, cue_b)
        initial_a = _init_state(core, run["S"], run["P"], 2, device)
        initial_b = _init_state(core, run["S"], run["P"], 2, device)
        readout_a, _, _ = _step_core(core, run["S"], run["M"], run["P"], state_a, initial_a, 0, None)
        readout_b, _, _ = _step_core(core, run["S"], run["M"], run["P"], state_b, initial_b, 0, None)
        action_a = heads(readout_a)[2]
        action_b = heads(readout_b)[2]
    front_end.input_noise_sigma = original_noise
    if not torch.equal(state_a, state_b) or not torch.equal(action_a, action_b):
        raise RuntimeError("working-memory forward compatibility failed")
    return {
        "source_state_sha256": _state_digest([front_end, core, heads]),
        "adapter_state_sha256": _state_digest(list(adapters.values())),
    }


def _save(path: Path, step: int, modules: tuple, adapters: dict, optimizer, exposures: dict, run: dict) -> None:
    front_end, core, heads = modules
    planned = _expected_exposures(run)
    wm_completed, wm_total = _wm_curriculum_position(exposures, planned)
    state = _checkpoint_state(step, front_end, core, heads, adapters, optimizer, rung=0)
    state["continuation"] = {
        "source_run_id": run["source_run_id"],
        "additional_updates": run["additional_updates"],
        "optimizer_restart": run["optimizer_restart"],
        "completed_updates": step,
        "exposures": dict(exposures),
        "planned_exposures": planned,
        "wm_curriculum": {
            "unit": "sternberg_exposure",
            "completed": wm_completed,
            "total": wm_total,
        },
        "task_cue": task_cue_metadata(),
    }
    temp = path.with_suffix(f"{path.suffix}.partial")
    temp.unlink(missing_ok=True)
    torch.save(state, temp)
    temp.replace(path)


def _auxiliary_evaluation(adapters, core, heads, run, cfg, device, reflective_gate, trials=128) -> dict:
    scores = {}
    batch_size = min(32, trials)
    for task_index, task in enumerate(NEUROGYM_TASKS):
        values = []
        for offset in range(0, trials, batch_size):
            size = min(batch_size, trials - offset)
            env = NeuroGymBatchEnv(task, size, seed=910_000_000 + run["seed"] * 1000 + task_index * 100 + offset)
            with torch.no_grad():
                _, reward = run_multitask_neurogym_trial(
                    adapters[task], core, heads, run["S"], run["M"], run["P"], task, env, size,
                    cfg["task"]["multitask_max_ticks"], device, mode="eval", reflective_gate=reflective_gate,
                )
            values.extend(reward)
        scores[task] = float(np.mean(values))
    return scores


def train_continuation(run: dict, cfg: dict, batch_size: int) -> dict:
    """Train or resume one continuation and return its evaluations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verification = verify_source(run, cfg, device)
    seed_everything(run["continuation_seed"])
    front_end, core, heads, adapters = _load_source(run, cfg, device)
    modules = (front_end, core, heads)
    params = list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters())
    for adapter in adapters.values():
        params.extend(adapter.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["train"]["lr"])
    checkpoint_dir = RESULTS / "checkpoints" / run["run_id"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ckpt.pt"
    exposures = {task: 0 for task in DIET_TASKS}
    planned_exposures = _expected_exposures(run)
    start = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        front_end.load_state_dict(saved["front_end"])
        core.load_state_dict(saved["core"])
        heads.load_state_dict(saved["heads"])
        for task, adapter in adapters.items():
            adapter.load_state_dict(saved["adapters"][task])
        optimizer.load_state_dict(saved["optimizer"])
        start, exposures = _resume_progress(saved, run)
        _restore_checkpoint_rng(saved)

    from brainalign_wm.tasks.generator import TaskGenerator
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    bank = ImageTokenBank(
        Path(cfg["paths"]["stimuli"]), cfg["task"]["categories"],
        Path(cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    generator = TaskGenerator(cfg, bank, seed=run["seed"])
    gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if run["M"] else None
    topo_weight = cfg["train"].get("topo_loss_weight_on", 0.01) if run["T"] else 0.0
    dale_weight = cfg["train"].get("dale_penalty_weight_on", 0.01) if run["D"] else 0.0
    flat_grid = tuple(cfg["model"]["flat_grid"])
    started = time.time()
    for step in range(start, run["additional_updates"]):
        task = _task_at(run, step)
        optimizer.zero_grad()
        if task == "sternberg":
            wm_step, wm_total = _wm_curriculum_position(exposures, planned_exposures)
            batch = generator.sample_batch(wm_step, wm_total, batch_size)
            loss, _, _, _ = _run_trial(
                front_end, core, heads, gate, run["S"], run["M"], run["P"], batch, bank,
                cfg["model"]["feature_dim"], cfg["model"]["action_dim"], _gate_width(run["S"], cfg["model"]),
                device, mode="bptt", signal="ce", topo_loss_weight=topo_weight, flat_grid=flat_grid,
                task_name_for_context="sternberg",
                checkpoint_plastic=bool(run["P"] and cfg["mechanisms"].get("plastic_gradient_checkpointing", True)),
            )
        else:
            env = NeuroGymBatchEnv(task, batch_size, seed=run["continuation_seed"] * 1_000_003 + step)
            loss, _ = run_multitask_neurogym_trial(
                adapters[task], core, heads, run["S"], run["M"], run["P"], task, env, batch_size,
                cfg["task"]["multitask_max_ticks"], device, value_weight=cfg["train"]["value_loss_weight"],
                reflective_gate=gate, topo_loss_weight=topo_weight, flat_grid=flat_grid,
            )
        if dale_weight:
            loss = loss + dale_weight * _dale_penalty(core, run["S"], cfg["model"]["dale_ei_split"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        optimizer.step()
        exposures[task] += 1
        if (step + 1) % cfg["train"]["checkpoint_every"] == 0 or step + 1 == run["additional_updates"]:
            _save(checkpoint_path, step + 1, modules, adapters, optimizer, exposures, run)

    wm = final_evaluation(
        front_end, core, heads, run["S"], run["P"], gate, generator, bank, cfg, device, run["seed"],
        task_name_for_context="sternberg",
    )
    auxiliary = _auxiliary_evaluation(adapters, core, heads, run, cfg, device, gate)
    return {
        **run,
        **verification,
        "status": "completed",
        "batch_size": batch_size,
        "task_cue": task_cue_metadata(TASK_CUE_SCHEMA),
        "completed_updates": sum(exposures.values()),
        "planned_exposures": planned_exposures,
        "exposures": exposures,
        "accuracy": wm,
        "auxiliary_reward": auxiliary,
        "wall_clock_train_s": round(time.time() - started, 1),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--models", default=",".join(SELECTED_MODELS))
    parser.add_argument("--diets", default="wm_only,multitask")
    parser.add_argument("--updates", type=int, default=24_000)
    parser.add_argument("--batch-size", type=int, default=None, help="defaults to the configured training batch size")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    runs = enumerate_continuations(
        range(args.seeds), tuple(args.models.split(",")), tuple(args.diets.split(",")), args.updates
    )
    if not args.execute:
        print(json.dumps(runs, indent=2))
        return 0
    cfg = load_config()
    batch_size = args.batch_size or cfg["train"]["batch_size"]
    completed = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("status") == "completed":
                completed.add(rec["run_id"])
    for run in runs:
        if run["run_id"] in completed:
            continue
        record = train_continuation(run, cfg, batch_size)
        with MANIFEST.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.write(json.dumps(record) + "\n")
        print(f"completed {run['run_id']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
