"""Single-run training entrypoint, called by `run_grid.py`.

Contract:
    train_one(run: dict, cfg: dict) -> dict
      run = {"run_id", "model_id", "S", "M", "L", "seed"}
      cfg  = the tier-merged dict `run_grid.py` builds (containing "steps",
             among other tier parameters); the full project configuration
             (model/mechanisms/task/train sections) is loaded here directly
             from configs/config.yaml, since `run_grid.py` only threads the
             tier subset through.
      returns {"status": "completed"|"failed",
               "gates": {"load1>=0.95": bool, "load3>=0.80": bool},
               "accuracy": {"load1": float, "load2": float, "load3": float},
               "rung": int}

Design decisions made to resolve underspecified aspects of the training
procedure (also recorded in DECISIONS.md):
  - **Training signal (matched between L=0 and L=1, audit fix A1a).** The
    L=0/L=1 contrast is only a clean test of "global backprop vs. local
    node-perturbation credit assignment" if both arms train on the SAME
    reward signal; the original design trained L=0 on dense per-tick
    teacher-forcing (imitation of the ideal policy) while L=1 trained on a
    single sparse end-of-trial scalar, confounding credit-assignment
    mechanism with supervision density (L=1 sat at chance -- see
    comments.txt A1a). Fixed with a two-phase regime driven by
    `tasks/curriculum.py`'s existing `phase` field ("warmup"/"ramp"/
    "target"):
      * **warmup** (first `task.curriculum.warmup_frac` of steps): L=0 keeps
        the original dense per-tick cross-entropy (imitation of the ideal
        policy; tractability warmup, not part of the controlled comparison).
        L=1 keeps its normal per-trial node-perturbation update timing, but
        the scalar reward fed to `apply_update` is the *fraction of ticks
        whose greedy action matched the ideal target action* -- a denser,
        smoother proxy than bare correct/incorrect, giving both arms a
        comparably dense (if differently-shaped) training signal during
        the tractability warmup.
      * **ramp + target** (remaining ~80% of steps): L=0 switches to
        REINFORCE with a value baseline, backpropagated through time, on the
        trial's real sparse end-of-trial reward -- the SAME reward L=1 has
        always used. Policy-gradient terms are only collected at
        probe-epoch ticks (the only ticks whose action affects the reward;
        REINFORCE on non-causal ticks would just inject variance). L=1 is
        unchanged in this regime. From this point on, L=0 and L=1 differ
        *only* in credit assignment (global BPTT policy-gradient vs. local
        node perturbation) on an identical reward -- see comments.txt A1a
        and DECISIONS.md. H3 ("L=1 >= BPTT at matched behavior") is only
        meaningful for cells that clear the behavioral gate; see
        `preregistration.md`.
  - **Batching (audit fix A1c).** One training step now runs a batch of `B`
    trials (`train.batch_size`) instead of a single trial. Since tick-count
    is fully determined by `load` (all other per-trial draws vary content,
    not length), a batch draws one shared load from the curriculum's
    distribution and then samples B independent trials at that load (see
    `tasks/generator.py::TaskGenerator.sample_batch`) -- every trial in the
    batch has identical epoch/tick structure, so no padding is needed.
    Discovered while implementing this: the reflective gate's `surprise()`/
    `step()`/`gate_bias()` chain silently mis-broadcasts for batch > 1 unless
    `reward`/`value`/`action_logp` are kept as `[B, 1]` (not `[B]`) alongside
    `is_feedback`/`R_t` -- `[B,1] * [B]` broadcasts to `[B,B]`, not `[B,1]`.
    Fixed by keeping the `[B, 1]` convention consistently through that path.
  - **Reflective-gate causal ordering.** The manager's update-gate equation
    `u^m_t = sigmoid(...+ beta*R_t)` requires R_t, which in turn requires
    the current tick's own policy output -- a circular dependency if R_t
    were computed from that same tick. This is resolved with a one-tick
    lag: the value of R_t fed into tick t's manager gate is computed from
    tick t-1's policy and value output, which both breaks the cycle and is
    defensible on the grounds that reflection is inherently retrospective.
  - **Local learning (L=1).** Runs under `torch.no_grad()`. Node
    perturbation is injected at the pre-activation of each GRU gate unit
    (dimension 3*hidden) via `_perturbed_gru_step`, not at the post-gate
    hidden state; eligibility traces accumulate per weight matrix (now
    per-batch-element -- see `mechanisms/local_learning.py`), and the
    three-factor update fires at trial-batch end using the batch's rewards.
    Training starts at fallback-ladder rung 1; if the load-1 gate is not
    cleared by `train.rung_check_frac` of the step budget, it escalates to
    rung 2 (trace-normalized, genuinely-adaptive-baseline variant in
    `local_learning.py`, see A1b in DECISIONS.md) for the remainder. Rung 3
    (e-prop) is not implemented (see `mechanisms/local_learning.py`); if
    rung 2 also fails to clear the gate, this is recorded honestly as
    rung=2, gates=False, rather than silently substituting backpropagation.
  - **Gate boundary (audit fix B2).** Behavioral gates use `>=` (not the
    original strict `>`), applied consistently here and in
    `preregistration.md`.
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
    """S=0, M=1 (cells M010/M011): the reflective gate is defined in terms
    of the manager's GRU update gate, which does not exist for a flat
    model. `FlatGRUCore` (S=0) uses a plain `nn.GRUCell`, which provides no
    hook for an external gate bias. The resolution (recorded in
    DECISIONS.md) is that for S=0, the flat GRU acts as its own gated unit,
    using the same `MaskedGRUCell` machinery as the HRL manager (mask=None,
    dense -- only the S=1 worker is spatially masked), with the reflective
    bias applied directly to its own update gate. This is the only
    architecturally sensible way to give M=1 a real, non-degenerate effect
    on a flat model; without it, M010/M011 would be silently identical to
    M000/M001."""

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


def _image_features_batch(image_bank, image_ids: list, feature_dim: int, device) -> torch.Tensor:
    """[B, feature_dim] stacked frozen-encoder features; `None` entries
    ("no stimulus this tick") become zero vectors."""
    feats = np.zeros((len(image_ids), feature_dim), dtype=np.float32)
    for b, image_id in enumerate(image_ids):
        if image_id is not None:
            feats[b] = np.asarray(image_bank.feature_of(image_id), dtype=np.float32)
    return torch.as_tensor(feats, dtype=torch.float32, device=device)


def _gate_bias_step(reflective_gate, R_prev, prev_is_feedback, prev_reward, prev_value, prev_action_logp, gate_width, B, device):
    """One reflective-gate tick. All `prev_*` args are `[B, 1]` -- kept
    consistently 2D through this whole path (not `[B]`) because
    `[B,1] * [B]` broadcasts to `[B,B]`, not `[B,1]`, for B>1 (a real bug
    caught while implementing batching; see module docstring)."""
    delta_t = reflective_gate.surprise(prev_is_feedback, prev_reward, prev_value, prev_action_logp)
    R_t = reflective_gate.step(delta_t, R_prev)
    gate_bias = reflective_gate.gate_bias(R_t).expand(B, gate_width)
    return gate_bias, R_t


def _run_trial(
    front_end, core, heads, reflective_gate, S: int, M: int,
    trial_steps_batch: list, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    device, mode: str,  # "bptt" | "eval"
    signal: str = "ce",  # "ce" | "reinforce"; only consulted when mode=="bptt"
    value_weight: float = 0.5,
    entropy_coef: float = 0.0,
):
    """Unrolls a batch of B trials sharing one load (see
    `TaskGenerator.sample_batch` -- every trial has identical tick/epoch
    structure). Returns (loss_or_None, correct: list[bool] len B,
    reward: list[float] len B)."""
    B = len(trial_steps_batch)
    T = len(trial_steps_batch[0])
    state = _init_state(core, S, B, device)
    R_prev = reflective_gate.init_state(B, device) if reflective_gate is not None else None
    prev_policy = torch.full((B, action_dim), 1.0 / action_dim, device=device)
    prev_value = torch.zeros(B, 1, device=device)
    prev_action_logp = torch.log(prev_policy[:, 0].clamp_min(1e-8)).unsqueeze(-1)
    prev_is_feedback = torch.zeros(B, 1, device=device)
    prev_reward = torch.zeros(B, 1, device=device)

    total_loss = torch.zeros((), device=device)
    n_ce_terms = 0.0
    policy_terms: list[tuple[torch.Tensor, torch.Tensor]] = []  # (logp_a [B], value_pred [B]) at probe ticks
    value_only_terms: list[torch.Tensor] = []  # value_pred [B] at feedback tick(s)
    entropy_terms: list[torch.Tensor] = []

    last_probe_action = [None] * B
    true_in_set = [None] * B

    for i in range(T):
        ts_list = [trial_steps_batch[b][i] for b in range(B)]
        epoch = ts_list[0].epoch  # shared across the batch: same load => same schedule
        v_t = _image_features_batch(image_bank, [ts.image_id for ts in ts_list], feature_dim, device)
        c_t = torch.tensor([ts.c_t for ts in ts_list], dtype=torch.float32, device=device)

        gate_bias = None
        if reflective_gate is not None:
            gate_bias, R_prev = _gate_bias_step(
                reflective_gate, R_prev, prev_is_feedback, prev_reward, prev_value, prev_action_logp,
                gate_width, B, device,
            )

        z_t = front_end(v_t, c_t)
        h_star, new_state, u_t = _step_core(core, S, M, z_t, state, t=i, gate_bias=gate_bias)
        policy, value, logits = heads(h_star)  # policy/logits: [B, action_dim]; value: [B]

        if mode == "bptt" and signal == "ce":
            targets = torch.tensor([_target_action(ts.epoch, ts.in_set) for ts in ts_list], device=device)
            # See DECISIONS.md (M7): the trivial "predict fixation" target
            # outnumbers the real decision ~5:1 in tick count and dominates
            # the loss unless upweighted.
            tick_weight = 1.0 if epoch == "probe" else 0.1
            total_loss = total_loss + tick_weight * F.cross_entropy(logits, targets)
            n_ce_terms += tick_weight

        if mode == "eval":
            action = torch.argmax(policy, dim=-1)
        else:
            action = torch.multinomial(policy.detach(), 1).squeeze(-1)  # [B]; sampling policy is load-bearing under REINFORCE (B5)

        if mode == "bptt" and signal == "reinforce" and epoch == "probe":
            logp_a = torch.log(policy.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8))  # keeps graph
            policy_terms.append((logp_a, value))
            probs = policy.clamp_min(1e-8)
            entropy_terms.append(-(probs * torch.log(probs)).sum(dim=-1).mean())

        if epoch == "probe":
            for b in range(B):
                last_probe_action[b] = int(action[b].item())
                true_in_set[b] = ts_list[b].in_set

        prev_policy = policy.detach()
        prev_value = value.detach().unsqueeze(-1)
        prev_action_logp = torch.log(policy.detach().gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)).unsqueeze(-1)
        prev_is_feedback = torch.full((B, 1), 1.0 if epoch == "feedback" else 0.0, device=device)

        state = new_state

        if epoch == "feedback":
            if mode == "bptt" and signal == "ce":
                correct_now = [(last_probe_action[b] == 1) == bool(true_in_set[b]) for b in range(B)]
                value_target = torch.tensor([1.0 if c else 0.0 for c in correct_now], device=device)
                total_loss = total_loss + value_weight * F.mse_loss(value, value_target)
                prev_reward = value_target.unsqueeze(-1)
            elif mode == "bptt" and signal == "reinforce":
                value_only_terms.append(value)  # target added post-loop once reward is final
                prev_reward = torch.tensor(
                    [1.0 if (last_probe_action[b] == 1) == bool(true_in_set[b]) else 0.0 for b in range(B)],
                    device=device,
                ).unsqueeze(-1)
            else:
                prev_reward = torch.tensor(
                    [1.0 if (last_probe_action[b] == 1) == bool(true_in_set[b]) else 0.0 for b in range(B)],
                    device=device,
                ).unsqueeze(-1)

    correct = [
        (last_probe_action[b] == 1) == bool(true_in_set[b]) if last_probe_action[b] is not None else False
        for b in range(B)
    ]
    reward = [1.0 if c else 0.0 for c in correct]

    if mode == "bptt" and signal == "reinforce":
        reward_t = torch.tensor(reward, device=device)
        policy_loss = torch.zeros((), device=device)
        value_loss = torch.zeros((), device=device)
        n_value_terms = 0
        for logp_a, value_pred in policy_terms:
            advantage = (reward_t - value_pred.detach())
            policy_loss = policy_loss + (-logp_a * advantage).mean()
            value_loss = value_loss + F.mse_loss(value_pred, reward_t)
            n_value_terms += 1
        for value_pred in value_only_terms:
            value_loss = value_loss + F.mse_loss(value_pred, reward_t)
            n_value_terms += 1
        n_probe_ticks = max(len(policy_terms), 1)
        mean_entropy = torch.stack(entropy_terms).mean() if entropy_terms else torch.zeros((), device=device)
        total_loss = (
            policy_loss / n_probe_ticks
            + value_weight * value_loss / max(n_value_terms, 1)
            - entropy_coef * mean_entropy
        )
        n_ce_terms = 1.0  # total_loss already fully normalized above

    loss = total_loss / max(n_ce_terms, 1.0) if mode == "bptt" else None
    return loss, correct, reward


def _perturbed_gru_step(cell, x_t, h_prev, xi_pre, extra_update_bias=None):
    """Manually replicates the GRU math (identical for `nn.GRUCell` and
    `gru_cell.MaskedGRUCell` -- both expose `weight_ih`/`weight_hh`/
    `bias_ih`/`bias_hh` with the same [3*hidden, in] layout) so that a
    pre-activation perturbation `xi_pre` (shape [batch, 3*hidden]) can be
    injected before gating: one perturbation term per gate-pre-activation
    unit, representing a perturbation of each unit's total input drive.
    This is not equivalent to perturbing the post-gate output h_t (dimension
    `hidden`): an earlier version did that and failed, because the traced
    weight matrices have `3*hidden` output rows (one per reset/update/
    candidate gate unit), not `hidden` -- see DECISIONS.md.
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
    trial_steps_batch: list, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    sigma_p: float, device, dense_reward: bool = False,
) -> tuple[list, list]:
    """Local-learning (L=1) batch of B trials sharing one load: no autograd,
    node perturbation injected into each traced module's output activity
    every tick, per-sample eligibility traces accumulated, three-factor
    update applied at trial-batch end. `dense_reward=True` (warmup phase
    only, see module docstring/A1a): the reward is the fraction of
    probe-epoch ticks whose greedy action matched the ideal target action,
    rather than bare trial-end correct/incorrect."""
    with torch.no_grad():
        B = len(trial_steps_batch)
        T = len(trial_steps_batch[0])
        state = _init_state(core, S, B, device)
        R_prev = reflective_gate.init_state(B, device) if reflective_gate is not None else None
        prev_value = torch.zeros(B, 1, device=device)
        prev_policy0 = torch.full((B,), 1.0 / action_dim, device=device)
        prev_action_logp = torch.log(prev_policy0.clamp_min(1e-8)).unsqueeze(-1)
        prev_is_feedback = torch.zeros(B, 1, device=device)
        prev_reward = torch.zeros(B, 1, device=device)
        last_probe_action = [None] * B
        true_in_set = [None] * B
        n_probe_matched = [0] * B
        n_probe_total = [0] * B

        for lrn in learners.values():
            lrn.reset_traces()

        for i in range(T):
            ts_list = [trial_steps_batch[b][i] for b in range(B)]
            epoch = ts_list[0].epoch
            v_t = _image_features_batch(image_bank, [ts.image_id for ts in ts_list], feature_dim, device)
            c_t = torch.tensor([ts.c_t for ts in ts_list], dtype=torch.float32, device=device)

            gate_bias = None
            if reflective_gate is not None:
                gate_bias, R_prev = _gate_bias_step(
                    reflective_gate, R_prev, prev_is_feedback, prev_reward, prev_value, prev_action_logp,
                    gate_width, B, device,
                )

            z_t = front_end(v_t, c_t)

            if S == 0:
                h_prev = state["h"]
                cell = core.cell  # nn.GRUCell (M=0) or gru_cell.MaskedGRUCell (M=1, _GatedFlatCore)
                if "flat" in learners:
                    xi = learners["flat"].sample_perturbation((B, 3 * core.hidden_dim), device=device)
                else:
                    xi = torch.zeros(B, 3 * core.hidden_dim, device=device)
                h_t, _ = _perturbed_gru_step(cell, z_t, h_prev, xi, extra_update_bias=gate_bias)
                if "flat" in learners:
                    learners["flat"].trace_step(xi, h_prev=h_prev, x_t=z_t)
                state = {"h": h_t}
                h_star = core.readout_state(h_t)
            else:
                h_w_prev, h_m_prev, g_prev = state["h_worker"], state["h_manager"], state["g"]
                worker_in = torch.cat([z_t, g_prev], dim=-1)
                if "worker" in learners:
                    xi_w = learners["worker"].sample_perturbation((B, 3 * core.worker_units), device=device)
                else:
                    xi_w = torch.zeros(B, 3 * core.worker_units, device=device)
                h_w_t, _ = _perturbed_gru_step(core.worker, worker_in, h_w_prev, xi_w)
                if "worker" in learners:
                    learners["worker"].trace_step(xi_w, h_prev=h_w_prev, x_t=worker_in)

                s_t = core.pool_worker(h_w_t)
                if "manager" in learners:
                    xi_m = learners["manager"].sample_perturbation((B, 3 * core.manager_units), device=device)
                else:
                    xi_m = torch.zeros(B, 3 * core.manager_units, device=device)
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
            # tensor would crash the value weight's trace on a shape mismatch.
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

            action = torch.argmax(policy, dim=-1)

            if epoch == "probe":
                for b in range(B):
                    last_probe_action[b] = int(action[b].item())
                    true_in_set[b] = ts_list[b].in_set
                    n_probe_total[b] += 1
                    if _target_action(ts_list[b].epoch, ts_list[b].in_set) == int(action[b].item()):
                        n_probe_matched[b] += 1

            prev_value = value.unsqueeze(-1)
            prev_action_logp = torch.log(policy.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)).unsqueeze(-1)
            prev_is_feedback = torch.full((B, 1), 1.0 if epoch == "feedback" else 0.0, device=device)

            if epoch == "feedback":
                correct_now = [(last_probe_action[b] == 1) == bool(true_in_set[b]) for b in range(B)]
                prev_reward = torch.tensor([1.0 if c else 0.0 for c in correct_now], device=device).unsqueeze(-1)

        correct = [
            (last_probe_action[b] == 1) == bool(true_in_set[b]) if last_probe_action[b] is not None else False
            for b in range(B)
        ]
        if dense_reward:
            reward = [
                (n_probe_matched[b] / n_probe_total[b]) if n_probe_total[b] > 0 else 0.0
                for b in range(B)
            ]
        else:
            reward = [1.0 if c else 0.0 for c in correct]
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
    # `Heads.pi`/`Heads.value` are plain `Linear` layers (a single weight
    # matrix each, not a GRU hh/ih pair) -- `weight_ih=None` puts the learner
    # in single-weight mode (see `NodePerturbationLearner.__init__`), fixing
    # what used to be a duplicated-slot 2x effective local learning rate for
    # the heads relative to the recurrent core (audit fix B5/B6).
    learners["pi"] = NodePerturbationLearner(
        weight_hh=heads.pi.weight, weight_ih=None,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung == 2), seed=seed + 2,
    )
    learners["value"] = NodePerturbationLearner(
        weight_hh=heads.value.weight, weight_ih=None,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung == 2), seed=seed + 3,
    )
    return learners


def evaluate_accuracy(front_end, core, heads, S: int, reflective_gate, task_gen, image_bank, cfg, device) -> dict:
    m = cfg["model"]
    eval_batch_size = int(cfg["train"].get("eval_batch_size", 8))
    acc = {}
    for load in cfg["task"]["loads"]:
        n_trials = cfg["train"]["eval_trials_per_load"]
        n_correct = 0
        n_done = 0
        while n_done < n_trials:
            bsz = min(eval_batch_size, n_trials - n_done)
            batch = []
            for k in range(bsz):
                from brainalign_wm.tasks.sternberg import SternbergGenerator  # noqa: F401 (imported for parity with prior behavior)

                rng = np.random.RandomState(hash((load, n_done + k, 999)) & 0xFFFFFFFF)
                steps = task_gen.sternberg.generate_trial(
                    rng, loads=[load], lure_fraction=cfg["task"]["lure_fraction"],
                    maintain_steps=cfg["task"]["maintain_steps"], trial_id=-1, split="test",
                )
                batch.append(steps)
            with torch.no_grad():
                _, correct, _ = _run_trial(
                    front_end, core, heads, reflective_gate, S, int(reflective_gate is not None),
                    batch, image_bank, m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="eval",
                )
            n_correct += sum(correct)
            n_done += bsz
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
    value_weight = float(t_cfg["value_loss_weight"])
    entropy_coef = float(t_cfg.get("entropy_coef", 0.0))
    batch_size = int(t_cfg.get("batch_size", 1))

    stimuli_root = ROOT / full_cfg["paths"]["stimuli"]
    if not stimuli_root.exists():
        return {"status": "failed", "error": f"stimuli pool missing at {stimuli_root}; run scripts/build_stimuli_pool.py",
                "gates": {"load1>=0.95": False, "load3>=0.80": False}, "accuracy": {}, "rung": 0}

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
        phase = params["phase"]
        trial_batch = task_gen.sample_batch(step, total_steps, batch_size)

        if L == 0:
            optimizer.zero_grad()
            signal = "ce" if phase == "warmup" else "reinforce"
            loss, correct, reward = _run_trial(
                front_end, core, heads, reflective_gate, S, M, trial_batch, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="bptt",
                signal=signal, value_weight=value_weight, entropy_coef=entropy_coef,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters()), 5.0
            )
            optimizer.step()
        else:
            dense_reward = phase == "warmup"
            correct, reward = _run_trial_local(
                front_end, core, heads, reflective_gate, S, learners, trial_batch, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), mech_cfg["perturb_sigma"], device,
                dense_reward=dense_reward,
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
        "load1>=0.95": accuracy.get("load1", 0.0) >= full_cfg["gates"]["load1_acc"],
        "load3>=0.80": accuracy.get("load3", 0.0) >= full_cfg["gates"]["load3_acc"],
    }
    return {
        "status": "completed", "gates": gates, "accuracy": accuracy,
        "rung": rung if L == 1 else 0, "wall_clock_train_s": round(time.time() - t0, 1),
    }
