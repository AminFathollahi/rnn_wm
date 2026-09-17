"""Hidden-state and synaptic-trace interventions for plastic recurrent cells."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch

from brainalign_wm.training.train import (
    _gate_bias_step,
    _gate_width,
    _image_features_batch,
    _init_state,
    _step_core,
)
from scripts.human_behavior_gates import wilson_ci

STATE_KEYS = {0: ("h", "hebb"), 1: ("h_worker", "hebb_worker")}


def _bound(core, S: int, key: str) -> float:
    cell = core.cell if S == 0 else core.worker
    return float(cell.hebb_clip) if "hebb" in key else 1.0


def _corrupt(
    x: torch.Tensor, bound: float, frac: float, gen: torch.Generator, mode: str = "noise",
) -> torch.Tensor:
    if mode == "erase":
        replacement = torch.zeros_like(x)
    elif mode == "noise":
        replacement = (torch.rand(x.shape, generator=gen, device=x.device) * 2 - 1) * bound
    else:
        raise ValueError(f"unknown corruption mode: {mode}")
    return (1.0 - frac) * x + frac * replacement


def lesion_state(
    core, S: int, state: dict, target: str, frac: float, gen: torch.Generator, mode: str = "noise",
) -> dict:
    """Replaces one carrier with uniform noise over its own range, or with
    zero under `mode="erase"`. Erasing removes the carrier's stored content
    without injecting a signal the network never produces: an erased
    synaptic trace leaves the effective recurrent weights equal to the
    trained slow weights, which separates memory loss from the disruption a
    noise-filled trace adds on top of it."""
    if target not in {"activity", "synaptic"}:
        raise ValueError(f"unknown intervention target: {target}")
    h_key, hebb_key = STATE_KEYS[int(bool(S))]
    key = h_key if target == "activity" else hebb_key
    out = dict(state)
    out[key] = _corrupt(state[key], _bound(core, S, key), frac, gen, mode)
    return out


def lesion_step(
    core, S: int, M: int, P: int, z_t, state: dict, t: int, gate_bias, target: Optional[str],
    frac: float, gen: torch.Generator, R_t=None, hold_trace: bool = True, mode: str = "noise",
):
    """One tick with the chosen carrier corrupted at the cell's input.

    `hold_trace` protects the synaptic trace from the corrupted activity
    during an activity lesion by restoring its pre-tick value: without it
    the Hebbian update, which is driven by the activity the lesion just
    replaced with noise, writes that noise into the trace and the two
    carriers are no longer disrupted independently. The protection is not
    free -- these networks keep writing to the trace through the delay, so a
    held trace falls behind the one an unlesioned trial would carry at the
    same tick. There is no mirror-image protection in the synaptic arm: the
    trace reaches the readout only through the activity it drives, so
    shielding the activity from the corrupted trace would remove the effect
    being measured. Both deviations are quantified by `carrier_deviation`."""
    h_key, hebb_key = STATE_KEYS[1 if S else 0]
    if target is None:
        return _step_core(core, S, M, P, z_t, state, t=t, gate_bias=gate_bias, R_t=R_t)

    in_state = lesion_state(core, S, state, target, frac, gen, mode)
    h_star, new_state, u_t = _step_core(core, S, M, P, z_t, in_state, t=t, gate_bias=gate_bias, R_t=R_t)
    if target == "activity" and hold_trace:
        new_state[hebb_key] = state[hebb_key]
    return h_star, new_state, u_t


def run_lesioned_trials(
    front_end, core, heads, reflective_gate, S: int, M: int, P: int,
    trial_steps_batch: list, image_bank, cfg: dict, device,
    target: Optional[str], lesion_epoch: str, frac: float = 1.0, rng_seed: int = 0,
    hold_trace: bool = True, mode: str = "noise", trace: Optional[list] = None,
) -> list[bool]:
    """`trace`, when given, is filled with one `(epoch, activity, synaptic)`
    entry per tick for the selectivity measurement."""
    m = cfg["model"]
    B, T = len(trial_steps_batch), len(trial_steps_batch[0])
    gate_width = _gate_width(S, m)
    gen = torch.Generator(device=device).manual_seed(rng_seed)

    state = _init_state(core, S, P, B, device)
    R_prev = reflective_gate.init_state(B, device) if reflective_gate is not None else None
    prev_policy = torch.full((B, m["action_dim"]), 1.0 / m["action_dim"], device=device)
    prev_value = torch.zeros(B, 1, device=device)
    prev_action_logp = torch.log(prev_policy[:, 0].clamp_min(1e-8)).unsqueeze(-1)
    prev_is_feedback = torch.zeros(B, 1, device=device)
    prev_reward = torch.zeros(B, 1, device=device)

    probe_i = next(i for i, s in enumerate(trial_steps_batch[0]) if s.epoch == "probe")
    true_in_set = torch.tensor([bool(trial_steps_batch[b][probe_i].in_set) for b in range(B)], device=device)
    correct_now = torch.zeros(B, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(T):
            epoch = trial_steps_batch[0][i].epoch
            v_t = _image_features_batch(
                image_bank, [trial_steps_batch[b][i].image_id for b in range(B)], m["feature_dim"], device
            )
            c_t = torch.tensor([trial_steps_batch[b][i].c_t for b in range(B)], dtype=torch.float32, device=device)

            gate_bias = None
            if reflective_gate is not None:
                gate_bias, R_prev = _gate_bias_step(
                    reflective_gate, R_prev, prev_is_feedback, prev_reward, prev_value, prev_action_logp,
                    gate_width, B, device,
                )

            z_t = front_end(v_t, c_t)
            tick_target = target if epoch == lesion_epoch else None
            h_star, state, _ = lesion_step(
                core, S, M, P, z_t, state, t=i, gate_bias=gate_bias, target=tick_target, frac=frac, gen=gen, R_t=R_prev,
                hold_trace=hold_trace, mode=mode,
            )
            if trace is not None:
                h_key, hebb_key = STATE_KEYS[1 if S else 0]
                trace.append((epoch, state[h_key].clone(), state[hebb_key].clone()))
            policy, value, _ = heads(h_star)
            action = torch.argmax(policy, dim=-1)

            if epoch == "probe":
                last_action = action
            prev_policy = policy
            prev_value = value.unsqueeze(-1)
            prev_action_logp = torch.log(policy.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)).unsqueeze(-1)
            prev_is_feedback = torch.full((B, 1), 1.0 if epoch == "feedback" else 0.0, device=device)
            if epoch == "feedback":
                correct_now = (last_action == 1) == true_in_set
                prev_reward = correct_now.float().unsqueeze(-1)

    return correct_now.tolist()


def trial_batch(task_gen, cfg: dict, load: int, size: int, rng: np.random.RandomState) -> list:
    out = []
    for _ in range(size):
        trial_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))
        out.append(task_gen.sternberg.generate_trial(
            trial_rng, loads=[load], lure_fraction=cfg["task"]["lure_fraction"],
            maintain_steps=cfg["task"]["maintain_steps"], trial_id=-1, split="test",
        ))
    return out


def evaluate_lesion_accuracy(
    front_end, core, heads, S: int, M: int, P: int, task_gen, image_bank, cfg: dict, device,
    target: Optional[str], lesion_epoch: str, n_trials: int, eval_seed: int, frac: float = 1.0,
    hold_trace: bool = True, mode: str = "noise",
) -> dict:
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate

    reflective_gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None
    eval_batch_size = int(cfg["train"].get("eval_batch_size", 8))
    master_rng = np.random.RandomState(eval_seed)
    acc = {}
    for load in cfg["task"]["loads"]:
        n_correct, n_done = 0, 0
        while n_done < n_trials:
            bsz = min(eval_batch_size, n_trials - n_done)
            batch = trial_batch(task_gen, cfg, load, bsz, master_rng)
            correct = run_lesioned_trials(
                front_end, core, heads, reflective_gate, S, M, P, batch, image_bank, cfg, device,
                target=target, lesion_epoch=lesion_epoch, frac=frac, rng_seed=eval_seed + n_done,
                hold_trace=hold_trace, mode=mode,
            )
            n_correct += sum(correct)
            n_done += bsz
        acc[f"load{load}"] = round(n_correct / n_trials, 4)
        lo, hi = wilson_ci(n_correct, n_trials)
        acc[f"load{load}_ci_lo"], acc[f"load{load}_ci_hi"] = round(lo, 4), round(hi, 4)
    return acc


def carrier_deviation(
    front_end, core, heads, S: int, M: int, P: int, task_gen, image_bank, cfg: dict, device,
    target: str, lesion_epoch: str, load: int, batch_size: int = 4, eval_seed: int = 0,
    hold_trace: bool = True, mode: str = "noise",
) -> dict:
    """How far each carrier is driven from its unlesioned trajectory.

    Runs one batch of identical trials with and without the lesion and
    returns, for the activity and the synaptic trace separately, the mean
    per-tick deviation ||lesioned - unlesioned|| over the lesioned ticks and
    over the ticks that follow them, each divided by the root-mean-square
    norm the unlesioned carrier reaches across the whole trial. The
    trial-wide denominator keeps the ratio finite where a carrier is
    momentarily near zero -- the synaptic trace starts every trial empty, so
    a per-tick denominator is undefined for a lesion placed before the first
    stimulus. A lesion that disrupts one carrier independently of the other
    shows a large deviation on its target and a small one off it."""
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate

    gate = ReflectiveGate(cfg["mechanisms"]["reflection_lambda"], cfg["mechanisms"]["reflection_beta"]) if M else None
    batch = trial_batch(task_gen, cfg, load, batch_size, np.random.RandomState(eval_seed))
    traces = {}
    for name, tgt in (("control", None), ("lesion", target)):
        traces[name] = []
        run_lesioned_trials(
            front_end, core, heads, gate, S, M, P, batch, image_bank, cfg, device,
            target=tgt, lesion_epoch=lesion_epoch, rng_seed=eval_seed, hold_trace=hold_trace,
            mode=mode, trace=traces[name],
        )

    out = {"target": target, "mode": mode, "lesion_epoch": lesion_epoch, "load": load, "hold_trace": hold_trace}
    lesioned_ticks = [i for i, (epoch, _, _) in enumerate(traces["control"]) if epoch == lesion_epoch]
    windows = {"during": lesioned_ticks, "after": list(range(max(lesioned_ticks) + 1, len(traces["control"])))}
    for carrier, slot in (("activity", 1), ("synaptic", 2)):
        norms = torch.stack([entry[slot].norm() for entry in traces["control"]])
        scale = float(norms.pow(2).mean().sqrt().clamp_min(1e-8))
        for window, ticks in windows.items():
            devs = [
                float((traces["lesion"][i][slot] - traces["control"][i][slot]).norm()) / scale
                for i in ticks
            ]
            out[f"{carrier}_deviation_{window}"] = round(float(np.mean(devs)), 4) if devs else float("nan")
    return out


def summarize_interventions(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["run_id", "load"]
    base = (
        df.loc[df["condition"].eq("control"), keys + ["accuracy"]]
        .rename(columns={"accuracy": "control_accuracy"})
    )
    paired = df.merge(base, on=keys, validate="many_to_one")
    paired["accuracy_drop"] = paired["control_accuracy"] - paired["accuracy"]
    groups = ["S", "supervision", "condition", "load"]
    return (
        paired.groupby(groups, dropna=False)["accuracy_drop"]
        .agg(n_runs="size", mean_drop="mean", median_drop="median", sd_drop="std")
        .reset_index()
    )


def summarize_selectivity(df: pd.DataFrame) -> pd.DataFrame:
    """Per condition, how far the lesioned carrier and the untouched one move
    from their unlesioned trajectories, in units of each carrier's own
    root-mean-square norm. Medians, because the per-checkpoint deviations are
    heavy-tailed. `off_over_on` below 1 means the lesion moved its target more
    than the carrier it was supposed to leave alone; the two carriers are
    coupled after the lesion window, so the `during` rows are the ones that
    bear on independence and the `after` rows measure the coupling."""
    rows = []
    for condition, g in df.groupby("condition", dropna=False):
        on = "activity" if g["target"].iloc[0] == "activity" else "synaptic"
        off = "synaptic" if on == "activity" else "activity"
        for window in ("during", "after"):
            on_v = g[f"{on}_deviation_{window}"].median()
            off_v = g[f"{off}_deviation_{window}"].median()
            rows.append({
                "condition": condition, "target": on, "window": window, "n": len(g),
                "on_target_deviation": round(float(on_v), 4),
                "off_target_deviation": round(float(off_v), 4),
                "off_over_on": round(float(off_v / on_v), 4) if on_v else float("nan"),
            })
    return pd.DataFrame(rows).sort_values(["window", "condition"], ignore_index=True)
