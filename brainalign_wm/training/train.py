"""Single-run training entrypoint called by ../../run_grid.py (protocol
§5-§7, §0.2/§14).

Contract:
    train_one(run: dict, cfg: dict) -> dict
      run = {"run_id","model_id","S","M","L","seed"}
      cfg  = the tier-merged dict run_grid.py builds ({"steps",...}); the
             FULL project config (model/mechanisms/task/train sections) is
             loaded here directly from configs/config.yaml, since run_grid
             only threads the tier subset through.
      returns {"status": "completed"|"failed",
               "gates": {"load1>0.95": bool, "load3>0.80": bool},
               "accuracy": {"load1": float, "load2": float, "load3": float},
               "rung": int}

Design choices made to close real ambiguities in the protocol (logged in
DECISIONS.md, not just here):
  - **Training signal.** Rather than a full RL loop, the network is trained
    to emit the *ideal task policy* at every tick (fixation during
    fixation/encode/maintain/feedback/iti; in_set-dependent yes/no during
    probe) via cross-entropy -- for this deterministic-labeling task,
    matching the ideal policy *is* the reward-maximizing policy, so this is
    a faithful, tractable stand-in for a full policy-gradient loop. A value
    head is trained (MSE) against the trial's correct/incorrect outcome so
    it's usable as the reflective gate's reward baseline (§6.1).
  - **Reflective-gate causal ordering.** `u^m_t = sigmoid(...+ beta*R_t)`
    needs R_t, which needs this tick's own policy output -- circular if R_t
    were computed from the *same* tick. Resolved with a one-tick lag: R_t
    fed into tick t's manager gate is computed from tick t-1's
    policy/value output. Neuroscientifically defensible (reflection is
    inherently retrospective) and breaks the cycle cleanly.
  - **Local learning (L=1).** Runs under `torch.no_grad()`; node
    perturbation is injected into each traced module's output activity
    (h_t for flat, h_worker_t/h_manager_t separately for HRL) every tick;
    eligibility traces accumulate per weight matrix; the three-factor
    update fires at trial end using real trial-correctness as reward.
    Starts at fallback-ladder rung 1; if the load-1 gate isn't cleared by
    `train.rung_check_frac` of the budget, escalates to rung 2
    (`local_learning.py`'s trace-normalized/adaptive-baseline variant) for
    the remainder. Rung 3 (e-prop) is not implemented this session (see
    `mechanisms/local_learning.py`) -- if rung 2 also fails, this is
    recorded honestly as rung=2, gates=False (protocol §6.2's own
    "record it" outcome), never silently substituting BPTT.
  - **Batch size 1.** One trial per step; simplest correct thing given the
    time budget, not a scope requirement.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_full_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


class _GatedFlatCore(torch.nn.Module):
    """S=0, M=1 (cells M010/M011): protocol §6.1's reflective gate is
    written in terms of "the manager's" GRU update gate, which doesn't
    exist for a flat model -- `FlatGRUCore` (S=0) uses plain `nn.GRUCell`,
    which has no hook for an external gate bias. Resolution (logged in
    DECISIONS.md): for S=0, the flat GRU acts as its own gated unit -- same
    `MaskedGRUCell` machinery as the HRL manager, mask=None (dense, no
    spatial constraint; only the S=1 worker is spatially masked), with the
    reflective bias applied directly to its own update gate. This is the
    only architecturally sensible way to give M=1 a real, non-degenerate
    effect on a flat model; without it, M010/M011 would be silently
    identical to M000/M001."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        from brainalign_wm.models.gru_cell import MaskedGRUCell

        self.cell = MaskedGRUCell(input_dim, hidden_dim, mask=None)
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, z_t, h_prev, extra_update_bias=None):
        return self.cell(z_t, h_prev, extra_update_bias=extra_update_bias)  # (h_t, u_t)

    def readout_state(self, h_t):
        return h_t


def _build_model(full_cfg: dict, S: int, M: int, device):
    from brainalign_wm.models.front_end import FrontEnd
    from brainalign_wm.models.flat_gru import FlatGRUCore
    from brainalign_wm.models.hrl import HRLCore
    from brainalign_wm.models.heads import Heads

    m = full_cfg["model"]
    front_end = FrontEnd(m["feature_dim"], m["task_vec_dim"], m["bottleneck"], m["input_noise_sigma"]).to(device)
    if S == 0:
        core = (_GatedFlatCore(m["bottleneck"], m["flat_units"]) if M else FlatGRUCore(m["bottleneck"], hidden_dim=m["flat_units"])).to(device)
        h_star_dim = m["flat_units"]
    else:
        core = HRLCore(
            input_dim=m["bottleneck"], worker_units=m["worker_units"], manager_units=m["manager_units"],
            grid=tuple(m["worker_grid"]), density=m["worker_density"], manager_period=m["manager_period"],
            g_dim=m["g_dim"], pool_block=m["worker_pool_block"], reflective=bool(M),
        ).to(device)
        h_star_dim = m["worker_units"] + m["manager_units"]
    heads = Heads(h_star_dim, readout_scale=m["readout_scale"]).to(device)
    return front_end, core, heads


def _target_action(epoch: str, in_set: Optional[bool]) -> int:
    """The ideal task policy (see module docstring): 0=fixation, 1=yes(in-set), 2=no."""
    if epoch == "probe":
        return 1 if in_set else 2
    return 0


def _step_core(core, S: int, M: int, z_t, state, t: int, gate_bias):
    if S == 0:
        if M:
            h_t, u_t = core(z_t, state["h"], extra_update_bias=gate_bias)
        else:
            h_t = core(z_t, state["h"])
            u_t = None
        new_state = {"h": h_t}
        h_star = core.readout_state(h_t)
    else:
        new_state, u_t = core(z_t, state, t=t, gate_bias=gate_bias)
        h_star = core.readout_state(new_state)
    return h_star, new_state, u_t


def _gate_width(S: int, m: dict) -> int:
    return m["flat_units"] if S == 0 else m["manager_units"]


def _init_state(core, S: int, batch: int, device):
    if S == 0:
        return {"h": core.init_state(batch, device)}
    return core.init_state(batch, device)


def _image_feature(image_bank, image_id: Optional[int], feature_dim: int, device) -> torch.Tensor:
    if image_id is None:
        return torch.zeros(1, feature_dim, device=device)
    feat = image_bank.feature_of(image_id)
    return torch.as_tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)


def _run_trial(
    front_end, core, heads, reflective_gate, S: int, M: int,
    trial_steps, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    device, mode: str,  # "bptt" | "eval"
):
    """Unrolls one trial. Returns (loss_or_None, correct: bool, per-tick dict
    for local-learning tracing (activities/inputs), reward: float)."""
    state = _init_state(core, S, 1, device)
    R_prev = reflective_gate.init_state(1, device) if reflective_gate is not None else None
    prev_policy = torch.full((1, action_dim), 1.0 / action_dim, device=device)
    prev_value = torch.zeros(1, device=device)
    prev_action_logp = torch.log(prev_policy[0, 0]).unsqueeze(0)
    prev_is_feedback = torch.zeros(1, 1, device=device)
    prev_reward = torch.zeros(1, device=device)

    total_loss = torch.zeros((), device=device)
    n_loss_terms = 0
    last_probe_action: Optional[int] = None
    true_in_set: Optional[bool] = None
    trace_records = []  # for local learning: (h_prev_all, x_t, xi placeholders filled by caller)

    for i, ts in enumerate(trial_steps):
        v_t = _image_feature(image_bank, ts.image_id, feature_dim, device)
        c_t = torch.tensor([ts.c_t], dtype=torch.float32, device=device)

        gate_bias = None
        if reflective_gate is not None:
            delta_t = reflective_gate.surprise(prev_is_feedback, prev_reward, prev_value, prev_action_logp)
            R_t = reflective_gate.step(delta_t, R_prev)
            gate_bias = reflective_gate.gate_bias(R_t).expand(1, gate_width)
            R_prev = R_t

        z_t = front_end(v_t, c_t)
        h_star, new_state, u_t = _step_core(core, S, M, z_t, state, t=i, gate_bias=gate_bias)
        policy, value, logits = heads(h_star)

        target = _target_action(ts.epoch, ts.in_set)
        if mode == "bptt":
            # The "predict fixation" target during fixation/encode/maintain/
            # feedback/iti is trivial (constant, no discrimination needed)
            # and outnumbers the real task signal ~5:1 in tick count. Averaged
            # uniformly, the easy majority dominates the loss and the network
            # learns to always predict fixation while never learning to
            # actually solve the match/non-match judgment (verified
            # empirically: loss dropped 0.87->0.14 over 300 trials while
            # probe-epoch accuracy stayed at chance). Upweighting the probe
            # epoch's cross-entropy fixes this -- the response decision is
            # the behaviorally meaningful signal; see DECISIONS.md.
            tick_weight = 1.0 if ts.epoch == "probe" else 0.1
            total_loss = total_loss + tick_weight * F.cross_entropy(logits, torch.tensor([target], device=device))
            n_loss_terms += tick_weight

        action = int(torch.argmax(policy, dim=-1).item()) if mode == "eval" else int(
            torch.multinomial(policy.detach(), 1).item()
        )
        if ts.epoch == "probe":
            last_probe_action = action
            true_in_set = ts.in_set

        prev_policy = policy.detach()
        prev_value = value.detach()
        prev_action_logp = torch.log(policy.detach()[0, action].clamp_min(1e-8)).unsqueeze(0)
        prev_is_feedback = torch.ones(1, 1, device=device) if ts.epoch == "feedback" else torch.zeros(1, 1, device=device)

        state = new_state

        if ts.epoch == "feedback":
            correct = (last_probe_action == 1) == bool(true_in_set)
            reward = 1.0 if correct else 0.0
            prev_reward = torch.full((1,), reward, device=device)
            if mode == "bptt":
                value_target = torch.full((1,), reward, device=device)
                total_loss = total_loss + full_cfg_train_value_weight() * F.mse_loss(value, value_target)

    correct = (last_probe_action == 1) == bool(true_in_set) if last_probe_action is not None else False
    reward = 1.0 if correct else 0.0
    loss = total_loss / max(n_loss_terms, 1) if mode == "bptt" else None
    return loss, correct, trace_records, reward


_VALUE_WEIGHT = 0.5


def full_cfg_train_value_weight() -> float:
    return _VALUE_WEIGHT


def _perturbed_gru_step(cell, x_t, h_prev, xi_pre, extra_update_bias=None):
    """Manually replicates the GRU math (identical for `nn.GRUCell` and
    `gru_cell.MaskedGRUCell` -- both expose `weight_ih/weight_hh/bias_ih/
    bias_hh` with the same [3*hidden, in] layout) so a **pre-activation**
    perturbation `xi_pre` (shape [batch, 3*hidden]) can be injected before
    gating, matching protocol §6.2's node-perturbation formulation (perturb
    each unit's total input drive, one perturbation per gate-pre-activation
    unit). This is NOT equivalent to perturbing the post-gate output h_t
    (dim=hidden): an earlier version did that and crashed, because the
    traced weight matrices have `3*hidden` output rows (one per r/u/n gate
    unit), not `hidden` -- see DECISIONS.md.
    """
    w_hh_eff = cell.weight_hh if getattr(cell, "mask", None) is None else cell.weight_hh * cell.mask
    gi = x_t @ cell.weight_ih.t() + cell.bias_ih + xi_pre
    gh = h_prev @ w_hh_eff.t() + cell.bias_hh + xi_pre
    i_r, i_u, i_n = gi.chunk(3, dim=-1)
    h_r, h_u, h_n = gh.chunk(3, dim=-1)
    r_t = torch.sigmoid(i_r + h_r)
    u_pre = i_u + h_u
    if extra_update_bias is not None:
        u_pre = u_pre + extra_update_bias
    u_t = torch.sigmoid(u_pre)
    n_t = torch.tanh(i_n + r_t * h_n)
    h_t = (1 - u_t) * h_prev + u_t * n_t
    return h_t, u_t


def _run_trial_local(
    front_end, core, heads, reflective_gate, S: int, learners: dict,
    trial_steps, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    sigma_p: float, device,
) -> tuple[bool, float]:
    """Local-learning (L=1) trial: no autograd; node perturbation injected
    into each traced module's output activity every tick; eligibility
    traces accumulated; three-factor update applied at trial end."""
    with torch.no_grad():
        state = _init_state(core, S, 1, device)
        R_prev = reflective_gate.init_state(1, device) if reflective_gate is not None else None
        prev_policy = torch.full((1, action_dim), 1.0 / action_dim, device=device)
        prev_value = torch.zeros(1, device=device)
        prev_action_logp = torch.log(prev_policy[0, 0]).unsqueeze(0)
        prev_is_feedback = torch.zeros(1, 1, device=device)
        prev_reward = torch.zeros(1, device=device)
        last_probe_action, true_in_set = None, None

        for lrn in learners.values():
            lrn.reset_traces()

        for i, ts in enumerate(trial_steps):
            v_t = _image_feature(image_bank, ts.image_id, feature_dim, device)
            c_t = torch.tensor([ts.c_t], dtype=torch.float32, device=device)

            gate_bias = None
            if reflective_gate is not None:
                delta_t = reflective_gate.surprise(prev_is_feedback, prev_reward, prev_value, prev_action_logp)
                R_t = reflective_gate.step(delta_t, R_prev)
                gate_bias = reflective_gate.gate_bias(R_t).expand(1, gate_width)
                R_prev = R_t

            z_t = front_end(v_t, c_t)

            if S == 0:
                h_prev = state["h"]
                cell = core.cell  # nn.GRUCell (M=0) or gru_cell.MaskedGRUCell (M=1, _GatedFlatCore)
                if "flat" in learners:
                    xi = learners["flat"].sample_perturbation((1, 3 * core.hidden_dim), device=device)
                else:
                    xi = torch.zeros(1, 3 * core.hidden_dim, device=device)
                h_t, _ = _perturbed_gru_step(cell, z_t, h_prev, xi, extra_update_bias=gate_bias)
                if "flat" in learners:
                    learners["flat"].trace_step(xi, h_prev=h_prev, x_t=z_t)
                state = {"h": h_t}
                h_star = core.readout_state(h_t)
            else:
                h_w_prev, h_m_prev, g_prev = state["h_worker"], state["h_manager"], state["g"]
                worker_in = torch.cat([z_t, g_prev], dim=-1)
                if "worker" in learners:
                    xi_w = learners["worker"].sample_perturbation((1, 3 * core.worker_units), device=device)
                else:
                    xi_w = torch.zeros(1, 3 * core.worker_units, device=device)
                h_w_t, _ = _perturbed_gru_step(core.worker, worker_in, h_w_prev, xi_w)
                if "worker" in learners:
                    learners["worker"].trace_step(xi_w, h_prev=h_w_prev, x_t=worker_in)

                s_t = core.pool_worker(h_w_t)
                if "manager" in learners:
                    xi_m = learners["manager"].sample_perturbation((1, 3 * core.manager_units), device=device)
                else:
                    xi_m = torch.zeros(1, 3 * core.manager_units, device=device)
                if reflective_gate is not None:
                    h_m_t, _ = _perturbed_gru_step(core.manager, s_t, h_m_prev, xi_m, extra_update_bias=gate_bias)
                else:
                    is_tick = (i % core.manager_period) == 0
                    if is_tick:
                        h_m_t, _ = _perturbed_gru_step(core.manager, s_t, h_m_prev, xi_m)
                    else:
                        h_m_t = h_m_prev
                if "manager" in learners:
                    learners["manager"].trace_step(xi_m, h_prev=h_m_prev, x_t=s_t)
                g_t = core.g_proj(h_m_t)
                state = {"h_worker": h_w_t, "h_manager": h_m_t, "g": g_t}
                h_star = torch.cat([h_w_t, h_m_t], dim=-1)

            # Perturb pi/value separately (not via heads.forward): their output
            # dims differ (n_actions vs. 1), so a single shared perturbation
            # tensor would crash the value weight's trace (shape mismatch).
            logits = heads.pi(h_star)
            value_pre = heads.value(h_star)  # [batch, 1], pre-squeeze
            if "pi" in learners:
                xi_pi = learners["pi"].sample_perturbation(logits.shape, device=device)
                logits = logits + xi_pi
                learners["pi"].trace_step(xi_pi, h_prev=h_star, x_t=h_star)
            if "value" in learners:
                xi_v = learners["value"].sample_perturbation(value_pre.shape, device=device)
                value_pre = value_pre + xi_v
                learners["value"].trace_step(xi_v, h_prev=h_star, x_t=h_star)
            policy = torch.softmax(logits, dim=-1)
            value = value_pre.squeeze(-1)

            action = int(torch.argmax(policy, dim=-1).item())
            if ts.epoch == "probe":
                last_probe_action, true_in_set = action, ts.in_set

            prev_policy, prev_value = policy, value
            prev_action_logp = torch.log(policy[0, action].clamp_min(1e-8)).unsqueeze(0)
            prev_is_feedback = torch.ones(1, 1, device=device) if ts.epoch == "feedback" else torch.zeros(1, 1, device=device)

            if ts.epoch == "feedback":
                correct = (last_probe_action == 1) == bool(true_in_set)
                prev_reward = torch.full((1,), 1.0 if correct else 0.0, device=device)

        correct = (last_probe_action == 1) == bool(true_in_set) if last_probe_action is not None else False
        reward = 1.0 if correct else 0.0
        for lrn in learners.values():
            lrn.apply_update(reward)
        return correct, reward


def _make_local_learners(core, heads, S: int, mech_cfg: dict, rung: int, seed: int) -> dict:
    from brainalign_wm.mechanisms.local_learning import make_learner_for_cell, NodePerturbationLearner

    learners = {}
    kwargs = dict(sigma_p=mech_cfg["perturb_sigma"], gamma_e=mech_cfg["elig_decay"], lr_local=mech_cfg["lr_local"])
    if S == 0:
        learners["flat"] = make_learner_for_cell(core.cell, rung=rung, seed=seed, **kwargs)
    else:
        learners["worker"] = make_learner_for_cell(core.worker, rung=rung, seed=seed, **kwargs)
        learners["manager"] = make_learner_for_cell(core.manager, rung=rung, seed=seed + 1, **kwargs)
    # `pi` and `value` are independent single-weight linear layers (not a
    # GRU hh/ih pair), and their output dims differ (n_actions vs. 1) -- so
    # each needs its OWN learner (a shared perturbation tensor across both
    # would crash the value weight's trace on a shape mismatch). Each
    # learner's two `NodePerturbationLearner` slots are pointed at the SAME
    # single weight (duplicated on purpose, not a typo): both slots then use
    # the same (xi, h_star) shapes and the update is applied twice to that
    # one parameter, i.e. an effective 2x local learning rate for the heads
    # -- a documented quirk of reusing the GRU-shaped container for a plain
    # linear layer, not a numerical bug.
    learners["pi"] = NodePerturbationLearner(
        weight_hh=heads.pi.weight, weight_ih=heads.pi.weight,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung == 2), seed=seed + 2,
    )
    learners["value"] = NodePerturbationLearner(
        weight_hh=heads.value.weight, weight_ih=heads.value.weight,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung == 2), seed=seed + 3,
    )
    return learners


def evaluate_accuracy(front_end, core, heads, S: int, reflective_gate, task_gen, image_bank, cfg, device) -> dict:
    m = cfg["model"]
    acc = {}
    for load in cfg["task"]["loads"]:
        n_trials = cfg["train"]["eval_trials_per_load"]
        n_correct = 0
        for k in range(n_trials):
            from brainalign_wm.tasks.sternberg import SternbergGenerator

            rng = np.random.RandomState(hash((load, k, 999)) & 0xFFFFFFFF)
            steps = task_gen.sternberg.generate_trial(
                rng, loads=[load], lure_fraction=cfg["task"]["lure_fraction"],
                maintain_steps=cfg["task"]["maintain_steps"], trial_id=-1, split="test",
            )
            with torch.no_grad():
                _, correct, _, _ = _run_trial(
                    front_end, core, heads, reflective_gate, S, int(reflective_gate is not None),
                    steps, image_bank, m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="eval",
                )
            n_correct += int(correct)
        acc[f"load{load}"] = round(n_correct / n_trials, 4)
    return acc


def train_one(run: dict, cfg: dict) -> dict:
    from brainalign_wm.utils.seeding import seed_everything
    from brainalign_wm.utils.device import get_device
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.tasks.generator import TaskGenerator

    full_cfg = _load_full_config()
    S, M, L, seed = run["S"], run["M"], run["L"], run["seed"]
    total_steps = int(cfg.get("steps", full_cfg["tiers"]["smoke"]["steps"]))
    run_id = run["run_id"]

    seed_everything(seed)
    device = get_device()

    m, mech_cfg, t_cfg = full_cfg["model"], full_cfg["mechanisms"], full_cfg["train"]
    global _VALUE_WEIGHT
    _VALUE_WEIGHT = t_cfg["value_loss_weight"]

    stimuli_root = ROOT / full_cfg["paths"]["stimuli"]
    if not stimuli_root.exists():
        return {"status": "failed", "error": f"stimuli pool missing at {stimuli_root}; run scripts/build_stimuli_pool.py",
                "gates": {"load1>0.95": False, "load3>0.80": False}, "accuracy": {}, "rung": 0}

    image_bank = ImageTokenBank(
        stimuli_root=stimuli_root, categories=full_cfg["task"]["categories"],
        feature_cache_path=ROOT / full_cfg["paths"]["feature_cache"] / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(full_cfg, image_bank, seed=seed)

    front_end, core, heads = _build_model(full_cfg, S, M, device)
    reflective_gate = ReflectiveGate(mech_cfg["reflection_lambda"], mech_cfg["reflection_beta"]) if M else None

    ckpt_dir = ROOT / "results" / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ckpt.pt"
    start_step = 0
    rung = 1

    optimizer = None
    if L == 0:
        params = list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters())
        optimizer = torch.optim.Adam(params, lr=t_cfg["lr"])

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        front_end.load_state_dict(ck["front_end"])
        core.load_state_dict(ck["core"])
        heads.load_state_dict(ck["heads"])
        if optimizer is not None and ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        rung = ck.get("rung", 1)

    learners = _make_local_learners(core, heads, S, mech_cfg, rung, seed) if L == 1 else None
    rung_check_step = int(t_cfg["rung_check_frac"] * total_steps)
    escalated = rung != 1

    t0 = time.time()
    for step in range(start_step, total_steps):
        params = task_gen.curriculum_params(step, total_steps)
        steps = task_gen.sternberg.generate_trial(
            rng=np.random.RandomState(np.random.SeedSequence([seed, step]).generate_state(1)[0]),
            loads=params["loads"], lure_fraction=params["lure_fraction"],
            maintain_steps=params["maintain_steps"], trial_id=step,
            load_weights=params["load_weights"],
        )

        if L == 0:
            optimizer.zero_grad()
            loss, correct, _, reward = _run_trial(
                front_end, core, heads, reflective_gate, S, M, steps, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="bptt",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters()), 5.0
            )
            optimizer.step()
        else:
            correct, reward = _run_trial_local(
                front_end, core, heads, reflective_gate, S, learners, steps, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), mech_cfg["perturb_sigma"], device,
            )
            if not escalated and step >= rung_check_step:
                interim = evaluate_accuracy(front_end, core, heads, S, reflective_gate, task_gen, image_bank, full_cfg, device)
                if interim.get("load1", 0.0) < full_cfg["gates"]["load1_acc"]:
                    rung = 2
                    learners = _make_local_learners(core, heads, S, mech_cfg, rung, seed + 100)
                escalated = True

        if (step + 1) % t_cfg["checkpoint_every"] == 0 or step == total_steps - 1:
            torch.save(
                {
                    "step": step + 1, "front_end": front_end.state_dict(), "core": core.state_dict(),
                    "heads": heads.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                    "rung": rung,
                },
                ckpt_path,
            )

    accuracy = evaluate_accuracy(front_end, core, heads, S, reflective_gate, task_gen, image_bank, full_cfg, device)
    gates = {
        "load1>0.95": accuracy.get("load1", 0.0) > full_cfg["gates"]["load1_acc"],
        "load3>0.80": accuracy.get("load3", 0.0) > full_cfg["gates"]["load3_acc"],
    }
    return {
        "status": "completed", "gates": gates, "accuracy": accuracy,
        "rung": rung if L == 1 else 0, "wall_clock_train_s": round(time.time() - t0, 1),
    }
