"""Single-run training entrypoint, called by `run_grid.py`.

Audit addition (2026-07-13): periodic training-metrics CSV logging.
Every `eval_every` steps, the current training loss and a full
`evaluate_accuracy` call are recorded to `results/metrics/{run_id}.csv`.
This enables convergence analysis, step-count optimization, and
diagnosis of training instability -- all impossible with only the
final checkpoint.

Contract:
    train_one(run: dict, cfg: dict) -> dict
      run = {"run_id", "model_id", "S", "M", "seed", "P"} for the 8 Core
            cells (BPTT throughout, §17 decision 6), or {..., "L": 1}
            instead of "P" for the four Extended local-learning cells
            (M00L/M01L/M10L/M11L, node-perturbation/e-prop, §6.3)
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
procedure:
  - **Training signal, matched between BPTT (L=0) and local-learning
    (L=1) cells.** A two-phase regime driven by `tasks/curriculum.py`'s
    `phase` field ("warmup"/"ramp"/"target") gives both a comparable
    reward density, since comparing credit-assignment mechanisms is only
    meaningful if both train on the same signal:
      * **warmup** (first `task.curriculum.warmup_frac` of steps): BPTT
        cells use dense per-tick cross-entropy (imitation of the ideal
        policy; tractability warmup, not part of the controlled
        comparison). Local-learning cells keep their normal per-trial
        node-perturbation update timing, but the scalar reward fed to
        `apply_update` is the *fraction of ticks whose greedy action
        matched the ideal target action* -- a denser, smoother proxy than
        bare correct/incorrect.
      * **ramp + target** (remaining steps): BPTT cells switch to
        REINFORCE with a value baseline, backpropagated through time, on
        the trial's real sparse end-of-trial reward -- the same reward
        local-learning cells have always used. Policy-gradient terms are
        only collected at probe-epoch ticks (the only ticks whose action
        affects the reward). From this point on, the two arms differ only
        in credit assignment (global BPTT policy-gradient vs. local node
        perturbation) on an identical reward.
  - **Batching.** One training step runs a batch of `B` trials
    (`train.batch_size`) instead of a single trial. Since tick-count is
    fully determined by `load` (all other per-trial draws vary content,
    not length), a batch draws one shared load from the curriculum's
    distribution and then samples B independent trials at that load (see
    `tasks/generator.py::TaskGenerator.sample_batch`) -- every trial in the
    batch has identical epoch/tick structure, so no padding is needed. The
    reflective gate's `surprise()`/`step()`/`gate_bias()` chain requires
    `reward`/`value`/`action_logp`/`is_feedback`/`R_t` to stay `[B, 1]`
    (not `[B]`) throughout: `[B,1] * [B]` broadcasts to `[B,B]`, not
    `[B,1]`, for B>1.
  - **Reflective-gate causal ordering.** The manager's update-gate equation
    `u^m_t = sigmoid(...+ beta*R_t)` requires R_t, which in turn requires
    the current tick's own policy output -- a circular dependency if R_t
    were computed from that same tick. Resolved with a one-tick lag: the
    value of R_t fed into tick t's manager gate is computed from tick
    t-1's policy and value output, which both breaks the cycle and is
    defensible on the grounds that reflection is inherently retrospective.
  - **Local learning (L=1).** Runs under `torch.no_grad()`. Node
    perturbation is injected at the pre-activation of each GRU gate unit
    (dimension 3*hidden) via `_perturbed_gru_step`, not at the post-gate
    hidden state; eligibility traces accumulate per weight matrix
    (per-batch-element -- see `mechanisms/local_learning.py`), and the
    three-factor update fires at trial-batch end using the batch's
    rewards. Training starts at fallback-ladder rung 1; if the load-1 gate
    is not cleared by `train.rung_check_frac` of the step budget, it
    escalates to rung 2 (trace-normalized, adaptive-baseline variant), and
    by `train.rung_check_frac_2` to rung 3 (e-prop, pseudo-derivative
    trace). If rung 3 also fails to clear the gate, this is recorded
    honestly as rung=3, gates=False, rather than silently substituting
    backpropagation.
  - **Gate boundary.** Behavioral gates use `>=` (not `>`), applied
    consistently here and in `preregistration.md`.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]


class _MetricsLogger:
    """Lightweight CSV writer for per-step training metrics.

    Writes header + one row per log call; flushed periodically.
    Created per-run in `results/metrics/{run_id}.csv`.
    """

    FIELDNAMES = [
        "step", "phase", "train_loss", "train_acc_load1", "train_acc_load2", "train_acc_load3",
        "grad_norm", "wall_s",
    ]

    def __init__(self, run_id: str):
        self.path = ROOT / "results" / "metrics" / f"{run_id}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log(self, row: dict) -> None:
        self._writer.writerow({k: row.get(k, "") for k in self.FIELDNAMES})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _load_full_config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())


class _GatedFlatCore(torch.nn.Module):
    """S=0 core for M=1 and/or P=1 (cells M010/M011/M001/M011/M101/M111 --
    i.e. anything but the true M000/M100 baseline): the reflective gate
    (M=1) and Hebbian fast weights (P=1, §6.2) both need hooks `nn.GRUCell`
    doesn't provide. Uses the same `MaskedGRUCell`/`PlasticGRUCell`
    machinery as the HRL manager/worker (mask=None, dense -- only the S=1
    worker is spatially masked), with the reflective bias applied directly
    to its own update gate (without this, M010/M011
    would be silently identical to M000/M001). `FlatGRUCore` (plain
    `nn.GRUCell`) remains the true M=0,P=0 baseline."""

    def __init__(self, input_dim: int, hidden_dim: int, plastic: bool = False, hebb_kwargs: Optional[dict] = None):
        super().__init__()
        from brainalign_wm.models.gru_cell import MaskedGRUCell, PlasticGRUCell

        self.plastic = plastic
        self.cell = (
            PlasticGRUCell(input_dim, hidden_dim, mask=None, **(hebb_kwargs or {}))
            if plastic else MaskedGRUCell(input_dim, hidden_dim, mask=None)
        )
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size: int, device=None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, z_t, h_prev, hebb_prev=None, extra_update_bias=None):
        if self.plastic:
            return self.cell(z_t, h_prev, hebb_prev, extra_update_bias=extra_update_bias)  # (h_t, u_t, hebb_t)
        return self.cell(z_t, h_prev, extra_update_bias=extra_update_bias)  # (h_t, u_t)

    def readout_state(self, h_t):
        return h_t


def _build_model(full_cfg: dict, S: int, M: int, P: int, device, pbwm_gate: bool = False):
    from brainalign_wm.models.front_end import FrontEnd
    from brainalign_wm.models.flat_gru import FlatGRUCore
    from brainalign_wm.models.hrl import HRLCore
    from brainalign_wm.models.heads import Heads

    m, mech = full_cfg["model"], full_cfg["mechanisms"]
    hebb_kwargs = {
        "eta_decay": mech.get("hebb_eta_decay", 0.9),
        "eta_hebb": mech.get("hebb_eta_hebb", 0.05),
        "hebb_clip": mech.get("hebb_clip", 2.0),
    }
    front_end = FrontEnd(m["feature_dim"], m["task_vec_dim"], m["bottleneck"], m["input_noise_sigma"]).to(device)
    if S == 0:
        core = (
            _GatedFlatCore(m["bottleneck"], m["flat_units"], plastic=bool(P), hebb_kwargs=hebb_kwargs)
            if (M or P) else FlatGRUCore(m["bottleneck"], hidden_dim=m["flat_units"])
        ).to(device)
        h_star_dim = m["flat_units"]
    else:
        core = HRLCore(
            input_dim=m["bottleneck"], worker_units=m["worker_units"], manager_units=m["manager_units"],
            grid=tuple(m["worker_grid"]), density=m["worker_density"], manager_period=m["manager_period"],
            g_dim=m["g_dim"], pool_block=m["worker_pool_block"], reflective=bool(M), plastic=bool(P),
            hebb_kwargs=hebb_kwargs, pbwm_gate=pbwm_gate, reflection_beta=mech["reflection_beta"],
        ).to(device)
        h_star_dim = m["worker_units"] + m["manager_units"]
    identity_catch_fraction = float(full_cfg["task"].get("identity_catch_fraction", 0.0))
    n_identity_categories = len(full_cfg["task"]["categories"]) if identity_catch_fraction > 0 else 0
    heads = Heads(h_star_dim, readout_scale=m["readout_scale"], n_identity_categories=n_identity_categories).to(device)
    return front_end, core, heads


def _target_action(epoch: str, in_set: Optional[bool]) -> int:
    """The ideal task policy (see module docstring): 0=fixation, 1=yes(in-set), 2=no."""
    if epoch == "probe":
        return 1 if in_set else 2
    return 0


def _step_core(
    core, S: int, M: int, P: int, z_t, state, t: int, gate_bias, recurrent_noise_sigma: float = 0.0, R_t=None,
    core_dropout_p: float = 0.0, training: bool = True,
):
    """`recurrent_noise_sigma` (bio-plausible ablation battery, §4.4
    "+noise"): Gaussian noise added directly to the persisting recurrent
    state (not just the input bottleneck's existing `sigma_in`) -- 0.0
    (off) for every Core cell; only the "+noise" ablation arm sets it.
    `R_t` (raw reflection signal, ablation-battery arm M111_pbwm only): the
    PBWM manager applies its own per-gate beta, so it needs R_t itself, not
    the pre-scaled `gate_bias` the standard reflective GRU manager uses --
    `HRLCore.forward` ignores whichever of the two doesn't apply.
    `core_dropout_p` (performance-matched baseline "flat_gru_dropout"):
    standard dropout on the flat (S=0) core's hidden state, train-mode
    only (`training=False` at eval, matching normal dropout convention --
    unlike `recurrent_noise_sigma`, which models a persistent biological
    noise source and stays on at eval too). S=0 only: the baseline family
    is explicitly flat-GRU (no S=1 dropout path needed)."""
    if S == 0:
        if M or P:
            if P:
                h_t, u_t, hebb_t = core(z_t, state["h"], state["hebb"], extra_update_bias=gate_bias)
                new_state = {"h": h_t, "hebb": hebb_t}
            else:
                h_t, u_t = core(z_t, state["h"], extra_update_bias=gate_bias)
                new_state = {"h": h_t}
        else:
            h_t = core(z_t, state["h"])
            u_t = None
            new_state = {"h": h_t}
        if recurrent_noise_sigma > 0:
            new_state["h"] = new_state["h"] + torch.randn_like(new_state["h"]) * recurrent_noise_sigma
        if core_dropout_p > 0 and training:
            new_state["h"] = F.dropout(new_state["h"], p=core_dropout_p, training=True)
        h_star = core.readout_state(new_state["h"])
    else:
        new_state, u_t = core(z_t, state, t=t, gate_bias=gate_bias, R_t=R_t)
        if recurrent_noise_sigma > 0:
            for key in ("h_worker", "h_manager"):
                new_state[key] = new_state[key] + torch.randn_like(new_state[key]) * recurrent_noise_sigma
        h_star = core.readout_state(new_state)
    return h_star, new_state, u_t


def _gate_width(S: int, m: dict) -> int:
    return m["flat_units"] if S == 0 else m["manager_units"]


def _dale_penalty(core, S: int, ei_split: float) -> torch.Tensor:
    """Arm D (§1.3, soft Dale's-law penalty): over each recurrent
    `weight_hh` [3*H, H] (flat cell for S=0; worker AND manager for S=1),
    designate the first `ei_split*H` hidden units (the COLUMNS = a unit's
    OUTGOING weights) excitatory and the rest inhibitory, and penalize sign
    violations. A soft penalty (not a hard clamp) is used deliberately --
    hard-clamping weight signs fights the optimizer / the `MaskedGRUCell`
    layout (§1.3).

    AUDIT 2026-07-26: the penalty is now taken over the MASKED (effective)
    recurrent weights. The S=1 worker's locality mask leaves ~4.1% of
    `weight_hh` alive (4,755 of 115,248 entries); the previous unmasked
    version spent ~96% of arm D's gradient on synapses that are multiplied
    by zero in the forward pass and therefore do not exist, which both
    diluted the mean by ~24x and made D substantially weaker on S=1 than
    on S=0 -- an S x D confound in a battery whose whole point is to
    separate the two arms."""
    cells = [core.cell] if S == 0 else [core.worker, core.manager]
    total = torch.zeros((), device=cells[0].weight_hh.device)
    for cell in cells:
        w = cell.weight_hh
        mask = getattr(cell, "mask", None)  # [3H, H] or None (nn.GRUCell / dense manager)
        H = w.shape[1]
        n_e = int(round(ei_split * H))
        viol = torch.cat([F.relu(-w[:, :n_e]), F.relu(w[:, n_e:])], dim=1)
        if mask is None:
            total = total + viol.mean()
        else:
            total = total + viol.mul(mask).sum() / mask.sum().clamp_min(1.0)
    return total


def _init_state(core, S: int, P: int, batch: int, device):
    if S == 0:
        state = {"h": core.init_state(batch, device)}
        if P:
            state["hebb"] = core.cell.init_hebb(batch, device)
        return state
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
    front_end, core, heads, reflective_gate, S: int, M: int, P: int,
    trial_steps_batch: list, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    device, mode: str,  # "bptt" | "eval"
    signal: str = "ce",  # "ce" | "reinforce"; only consulted when mode=="bptt"
    value_weight: float = 0.5,
    entropy_coef: float = 0.0,
    energy_cost_weight: float = 0.0,
    topo_loss_weight: float = 0.0,
    flat_grid: tuple = (16, 16),
    recurrent_noise_sigma: float = 0.0,
    core_dropout_p: float = 0.0,
    categories: Optional[list] = None,
):
    """Unrolls a batch of B trials sharing one load (see
    `TaskGenerator.sample_batch` -- every trial has identical tick/epoch
    structure). Returns (loss_or_None, correct: list[bool] len B,
    reward: list[float] len B, identity_catch: dict|None). `energy_cost_weight`
    (bio-plausible ablation battery, §4.4 "+energy cost"): an L2/mean-firing-rate
    penalty on the readout state `h_star`, added to the training loss -- 0.0 (off)
    for every Core cell; only the "+energy cost" ablation arm sets it.
    `topo_loss_weight`/`flat_grid` (arm T, §1.2): a spatial-smoothness penalty
    on the RECURRENT population's own activity (never `h_star`) -- the S=1
    worker reshaped onto its intrinsic `core.grid` (14x14), or the S=0 flat
    core's hidden state reshaped onto the imposed `flat_grid` (16x16, the
    All-TNNs "topography without hierarchy" test). S owns the spatial
    locality MASK; T owns this smoothness LOSS -- kept orthogonal so their
    ablation effects don't bleed into each other. 0.0 (off) for every Core
    cell; only the "+T"/"-T" ablation-battery cells set it.
    `categories` (§9.4a): the task's category list,
    used to map an identity-catch trial's target category to a class index
    for `heads.identity_aux` -- required whenever `heads.identity_aux is
    not None` (i.e. `task.identity_catch_fraction > 0`), unused otherwise.
    `identity_catch` is `{"correct": int, "total": int}` accumulated over
    catch trials in this batch (identity-report accuracy, §9.4a), or None
    when `heads.identity_aux is None`."""
    B = len(trial_steps_batch)
    T = len(trial_steps_batch[0])
    state = _init_state(core, S, P, B, device)
    R_prev = reflective_gate.init_state(B, device) if reflective_gate is not None else None
    prev_policy = torch.full((B, action_dim), 1.0 / action_dim, device=device)
    prev_value = torch.zeros(B, 1, device=device)
    prev_action_logp = torch.log(prev_policy[:, 0].clamp_min(1e-8)).unsqueeze(-1)
    prev_is_feedback = torch.zeros(B, 1, device=device)
    prev_reward = torch.zeros(B, 1, device=device)

    total_loss = torch.zeros((), device=device)
    n_ce_terms = 0.0
    policy_terms: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (logp_a, value_pred, non_catch mask) [B] at probe ticks
    value_only_terms: list[tuple[torch.Tensor, torch.Tensor]] = []  # (value_pred [B], non_catch mask [B]) at feedback tick(s)
    entropy_terms: list[torch.Tensor] = []
    energy_terms: list[torch.Tensor] = []
    topo_terms: list[torch.Tensor] = []  # arm T (§1.2): spatial-smoothness penalty terms
    identity_aux_terms: list[torch.Tensor] = []  # §9.4a auxiliary identity-report CE loss, catch trials only
    identity_catch = {"correct": 0, "total": 0} if heads.identity_aux is not None else None

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
        h_star, new_state, u_t = _step_core(
            core, S, M, P, z_t, state, t=i, gate_bias=gate_bias, recurrent_noise_sigma=recurrent_noise_sigma,
            R_t=R_prev, core_dropout_p=core_dropout_p, training=(mode == "bptt"),
        )
        policy, value, logits = heads(h_star)  # policy/logits: [B, action_dim]; value: [B]

        if mode == "bptt" and energy_cost_weight > 0:
            energy_terms.append(h_star.pow(2).mean())

        if mode == "bptt" and topo_loss_weight > 0:
            # RECURRENT population activity, not h_star -- S=1's worker
            # lives on its own intrinsic grid (`core.grid`); S=0 has no
            # intrinsic grid, so `flat_grid` imposes one (arbitrary unit
            # ordering; the loss is what makes the map meaningful over
            # training, standard All-TNNs method).
            if S == 0:
                gh, gw = flat_grid
                grid_act = new_state["h"].view(-1, gh, gw)
            else:
                gh, gw = core.grid
                grid_act = new_state["h_worker"].view(-1, gh, gw)
            topo_terms.append(
                torch.diff(grid_act, dim=1).pow(2).mean() + torch.diff(grid_act, dim=2).pow(2).mean()
            )

        # Identity-report catch trials (§9.4a) have no
        # real in/out judgment (`in_set=None`) -- excluded from the policy/
        # value loss below via `non_catch`, so they contribute only through
        # the auxiliary identity-report loss (added after the tick loop
        # would otherwise discard it under "reinforce" -- see end of
        # function). A no-op (all-ones) mask whenever no catch trial is in
        # this batch, i.e. always in Core (`identity_catch_fraction=0`).
        non_catch = torch.tensor([not ts.is_identity_catch for ts in ts_list], dtype=torch.float32, device=device)
        any_catch = bool((non_catch < 1).any())

        if mode == "bptt" and signal == "ce" and (not any_catch or non_catch.sum() > 0):
            targets = torch.tensor([_target_action(ts.epoch, ts.in_set) for ts in ts_list], device=device)
            # The trivial "predict fixation" target
            # outnumbers the real decision ~5:1 in tick count and dominates
            # the loss unless upweighted.
            tick_weight = 1.0 if epoch == "probe" else 0.1
            ce_per_sample = F.cross_entropy(logits, targets, reduction="none")
            total_loss = total_loss + tick_weight * (ce_per_sample * non_catch).sum() / non_catch.sum()
            n_ce_terms += tick_weight

        if mode == "eval":
            action = torch.argmax(policy, dim=-1)
        else:
            action = torch.multinomial(policy.detach(), 1).squeeze(-1)  # [B]; sampling policy is load-bearing under REINFORCE (B5)

        if mode == "bptt" and signal == "reinforce" and epoch == "probe" and (not any_catch or non_catch.sum() > 0):
            logp_a = torch.log(policy.gather(1, action.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8))  # keeps graph
            policy_terms.append((logp_a, value, non_catch))
            probs = policy.clamp_min(1e-8)
            entropy_terms.append(-(probs * torch.log(probs)).sum(dim=-1).mean())

        if heads.identity_aux is not None and epoch == "probe" and any_catch:
            catch_mask = 1.0 - non_catch
            id_logits = heads.identity_logits(h_star)
            target_idx = torch.tensor(
                [categories.index(ts.identity_catch_category) if ts.is_identity_catch else 0 for ts in ts_list],
                device=device,
            )
            if mode == "bptt":
                id_ce_per_sample = F.cross_entropy(id_logits, target_idx, reduction="none")
                identity_aux_terms.append((id_ce_per_sample * catch_mask).sum() / catch_mask.sum())
            id_pred = torch.argmax(id_logits, dim=-1).detach()
            identity_catch["correct"] += int(((id_pred == target_idx).float() * catch_mask).sum().item())
            identity_catch["total"] += int(catch_mask.sum().item())

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
                value_only_terms.append((value, non_catch))  # target added post-loop once reward is final
                prev_reward = torch.tensor(
                    [1.0 if (last_probe_action[b] == 1) == bool(true_in_set[b]) else 0.0 for b in range(B)],
                    device=device,
                ).unsqueeze(-1)
            else:
                prev_reward = torch.tensor(
                    [1.0 if (last_probe_action[b] == 1) == bool(true_in_set[b]) else 0.0 for b in range(B)],
                    device=device,
                ).unsqueeze(-1)

    # NOTE (§9.4a): for an identity-catch trial, `true_in_set[b]` is None
    # (no real in/out judgment was made), so its `correct`/`reward` entry
    # below is not a meaningful match/non-match outcome -- harmless in Core
    # (no catch trials ever occur there); catch-trial performance is
    # reported separately via `identity_catch` above.
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
        for logp_a, value_pred, non_catch_mask in policy_terms:
            advantage = (reward_t - value_pred.detach())
            denom = non_catch_mask.sum().clamp_min(1.0)
            policy_loss = policy_loss + (-logp_a * advantage * non_catch_mask).sum() / denom
            value_loss = value_loss + ((value_pred - reward_t).pow(2) * non_catch_mask).sum() / denom
            n_value_terms += 1
        for value_pred, non_catch_mask in value_only_terms:
            denom = non_catch_mask.sum().clamp_min(1.0)
            value_loss = value_loss + ((value_pred - reward_t).pow(2) * non_catch_mask).sum() / denom
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
    if loss is not None and identity_aux_terms:
        loss = loss + torch.stack(identity_aux_terms).mean()
    if loss is not None and energy_terms:
        loss = loss + energy_cost_weight * torch.stack(energy_terms).mean()
    if loss is not None and topo_terms:
        loss = loss + topo_loss_weight * torch.stack(topo_terms).mean()
    return loss, correct, reward, identity_catch


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
    candidate gate unit), not `hidden`.

    AUDIT 2026-07-26: `xi_pre` used to be added to BOTH `gi` and `gh`, so
    each gate actually received a perturbation of 2*xi (and the candidate
    gate received `(1 + r_t) * xi_n`, i.e. a *state-dependent* magnitude)
    while `NodePerturbationLearner.trace_step` traced plain `xi`. Node
    perturbation estimates the gradient from the correlation between the
    perturbation it applied and the reward it got, so tracing a different
    vector than the one applied miscalibrates every recurrent update, by a
    factor that drifts with r_t for one of the three gates. It is now added
    exactly once, to each gate's own pre-activation -- the quantity the
    trace actually claims to have perturbed.
    """
    w_hh_eff = cell.weight_hh if getattr(cell, "mask", None) is None else cell.weight_hh * cell.mask
    gi = x_t @ cell.weight_ih.t() + cell.bias_ih
    gh = h_prev @ w_hh_eff.t() + cell.bias_hh
    i_r, i_u, i_n = gi.chunk(3, dim=-1)
    h_r, h_u, h_n = gh.chunk(3, dim=-1)
    xi_r, xi_u, xi_n = xi_pre.chunk(3, dim=-1)
    r_t = torch.sigmoid(i_r + h_r + xi_r)
    u_pre = i_u + h_u + xi_u
    if extra_update_bias is not None:
        u_pre = u_pre + extra_update_bias
    u_t = torch.sigmoid(u_pre)
    n_t = torch.tanh(i_n + r_t * h_n + xi_n)
    h_t = (1 - u_t) * h_prev + u_t * n_t
    return h_t, u_t


def _eprop_gru_step(cell, x_t, h_prev, extra_update_bias=None):
    """Rung 3 (e-prop, Bellec et al. 2020, §6.3): an UNPERTURBED GRU step
    (no exploratory noise -- e-prop's eligibility trace comes from each
    unit's own local pseudo-derivative, not a randomly-probed direction)
    that also returns that pseudo-derivative, shape [batch, 3*hidden] --
    the same convention `_perturbed_gru_step`'s `xi_pre` uses, so it plugs
    directly into the SAME `NodePerturbationLearner.trace_step`/
    `apply_update` eligibility-trace and three-factor-update machinery
    rungs 1-2 use (reusing rather than rebuilding that machinery). Each gate's
    pseudo-derivative is that gate's own nonlinearity's true local
    derivative (sigmoid'/tanh' at its own pre-activation) -- GRU gates are
    differentiable, unlike e-prop's original spiking-neuron setting, so no
    surrogate is needed, only the (still local, still not
    backpropagated-through-time) analytic derivative."""
    w_hh_eff = cell.weight_hh if getattr(cell, "mask", None) is None else cell.weight_hh * cell.mask
    gi = x_t @ cell.weight_ih.t() + cell.bias_ih
    gh = h_prev @ w_hh_eff.t() + cell.bias_hh
    i_r, i_u, i_n = gi.chunk(3, dim=-1)
    h_r, h_u, h_n = gh.chunk(3, dim=-1)
    r_t = torch.sigmoid(i_r + h_r)
    u_pre = i_u + h_u
    if extra_update_bias is not None:
        u_pre = u_pre + extra_update_bias
    u_t = torch.sigmoid(u_pre)
    n_t = torch.tanh(i_n + r_t * h_n)
    h_t = (1 - u_t) * h_prev + u_t * n_t
    pseudo_deriv = torch.cat([r_t * (1 - r_t), u_t * (1 - u_t), 1 - n_t.pow(2)], dim=-1)
    return h_t, u_t, pseudo_deriv


def _run_trial_local(
    front_end, core, heads, reflective_gate, S: int, learners: dict,
    trial_steps_batch: list, image_bank, feature_dim: int, action_dim: int, gate_width: int,
    sigma_p: float, device, dense_reward: bool = False,
) -> tuple[list, list]:
    """Local-learning (L=1) batch of B trials sharing one load: no autograd,
    per-sample eligibility traces accumulated every tick, three-factor
    update applied at trial-batch end. Rungs 1-2: node perturbation injected
    into each traced module's output activity, traced against that noise.
    Rung 3 (e-prop, §6.3): no perturbation -- traced against each unit's own
    local pseudo-derivative instead (`_eprop_gru_step`), reusing the exact
    same trace/update machinery. `dense_reward=True` (warmup phase only,
    see module docstring/A1a): the reward is the fraction of probe-epoch
    ticks whose greedy action matched the ideal target action, rather than
    bare trial-end correct/incorrect."""
    is_eprop = any(lrn.rung == 3 for lrn in learners.values())
    with torch.no_grad():
        B = len(trial_steps_batch)
        T = len(trial_steps_batch[0])
        state = _init_state(core, S, 0, B, device)  # local-learning cells are never plastic (P n/a)
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
                    if is_eprop:
                        h_t, _, pd = _eprop_gru_step(cell, z_t, h_prev, extra_update_bias=gate_bias)
                        learners["flat"].trace_step(pd, h_prev=h_prev, x_t=z_t)
                    else:
                        xi = learners["flat"].sample_perturbation((B, 3 * core.hidden_dim), device=device)
                        h_t, _ = _perturbed_gru_step(cell, z_t, h_prev, xi, extra_update_bias=gate_bias)
                        learners["flat"].trace_step(xi, h_prev=h_prev, x_t=z_t)
                else:
                    xi = torch.zeros(B, 3 * core.hidden_dim, device=device)
                    h_t, _ = _perturbed_gru_step(cell, z_t, h_prev, xi, extra_update_bias=gate_bias)
                state = {"h": h_t}
                h_star = core.readout_state(h_t)
            else:
                h_w_prev, h_m_prev, g_prev = state["h_worker"], state["h_manager"], state["g"]
                worker_in = torch.cat([z_t, g_prev], dim=-1)
                if "worker" in learners:
                    if is_eprop:
                        h_w_t, _, pd_w = _eprop_gru_step(core.worker, worker_in, h_w_prev)
                        learners["worker"].trace_step(pd_w, h_prev=h_w_prev, x_t=worker_in)
                    else:
                        xi_w = learners["worker"].sample_perturbation((B, 3 * core.worker_units), device=device)
                        h_w_t, _ = _perturbed_gru_step(core.worker, worker_in, h_w_prev, xi_w)
                        learners["worker"].trace_step(xi_w, h_prev=h_w_prev, x_t=worker_in)
                else:
                    xi_w = torch.zeros(B, 3 * core.worker_units, device=device)
                    h_w_t, _ = _perturbed_gru_step(core.worker, worker_in, h_w_prev, xi_w)

                s_t = core.pool_worker(h_w_t)
                if is_eprop:
                    if reflective_gate is not None:
                        h_m_t, _, pd_m = _eprop_gru_step(core.manager, s_t, h_m_prev, extra_update_bias=gate_bias)
                    else:
                        is_tick = (i % core.manager_period) == 0
                        if is_tick:
                            h_m_t, _, pd_m = _eprop_gru_step(core.manager, s_t, h_m_prev)
                        else:
                            h_m_t = h_m_prev
                            pd_m = torch.zeros(B, 3 * core.manager_units, device=device)
                    if "manager" in learners:
                        learners["manager"].trace_step(pd_m, h_prev=h_m_prev, x_t=s_t)
                else:
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

            # pi/value are plain Linear layers (no gating nonlinearity), so
            # their own pseudo-derivative is trivially 1 everywhere -- under
            # e-prop this reduces to a plain (un-perturbed) presynaptic
            # eligibility trace; under rungs 1-2, perturb them directly
            # (their output dims differ, n_actions vs. 1, so a single shared
            # perturbation tensor would crash the value weight's trace on a
            # shape mismatch).
            logits = heads.pi(h_star)
            value_pre = heads.value(h_star)  # [batch, 1], pre-squeeze
            if "pi" in learners:
                if is_eprop:
                    learners["pi"].trace_step(torch.ones_like(logits), h_prev=h_star, x_t=h_star)
                else:
                    xi_pi = learners["pi"].sample_perturbation(logits.shape, device=device)
                    logits = logits + xi_pi
                    learners["pi"].trace_step(xi_pi, h_prev=h_star, x_t=h_star)
            if "value" in learners:
                if is_eprop:
                    learners["value"].trace_step(torch.ones_like(value_pre), h_prev=h_star, x_t=h_star)
                else:
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
    # the heads relative to the recurrent core.
    learners["pi"] = NodePerturbationLearner(
        weight_hh=heads.pi.weight, weight_ih=None,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung in (2, 3)), eprop=(rung == 3), seed=seed + 2,
    )
    learners["value"] = NodePerturbationLearner(
        weight_hh=heads.value.weight, weight_ih=None,
        sigma_p=kwargs["sigma_p"], gamma_e=kwargs["gamma_e"], lr_local=kwargs["lr_local"],
        normalize_traces=(rung == 2), adaptive_baseline=(rung in (2, 3)), eprop=(rung == 3), seed=seed + 3,
    )
    return learners


def evaluate_accuracy(front_end, core, heads, S: int, P: int, reflective_gate, task_gen, image_bank, cfg, device) -> dict:
    """Match/non-match accuracy per load, plus `acc["identity_catch"]`
    (§9.4a identity-report accuracy over catch trials, present only when
    `heads.identity_aux is not None`). Catch trials carry no real in/out
    judgment (`_run_trial`'s `correct` entry for them is a meaningless
    always-"incorrect" placeholder), so they are excluded from the
    match/non-match denominator here to keep `load{N}` gates meaningful."""
    m = cfg["model"]
    eval_batch_size = int(cfg["train"].get("eval_batch_size", 8))
    identity_catch_fraction = float(cfg["task"].get("identity_catch_fraction", 0.0))
    categories = cfg["task"]["categories"] if heads.identity_aux is not None else None
    acc = {}
    id_correct_total, id_total_total = 0, 0
    for load in cfg["task"]["loads"]:
        n_trials = cfg["train"]["eval_trials_per_load"]
        n_correct = 0
        n_non_catch = 0
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
                    identity_catch_fraction=identity_catch_fraction,
                )
                batch.append(steps)
            with torch.no_grad():
                _, correct, _, identity_catch = _run_trial(
                    front_end, core, heads, reflective_gate, S, int(reflective_gate is not None), P,
                    batch, image_bank, m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="eval",
                    recurrent_noise_sigma=float(m.get("recurrent_noise_sigma", 0.0)),
                    categories=categories,
                )
            is_catch = [steps[0].is_identity_catch for steps in batch]
            n_correct += sum(c for c, catch in zip(correct, is_catch) if not catch)
            n_non_catch += sum(1 for catch in is_catch if not catch)
            if identity_catch is not None:
                id_correct_total += identity_catch["correct"]
                id_total_total += identity_catch["total"]
            n_done += bsz
        acc[f"load{load}"] = round(n_correct / n_non_catch, 4) if n_non_catch else 0.0
    if categories is not None:
        acc["identity_catch"] = round(id_correct_total / id_total_total, 4) if id_total_total else 0.0
    return acc


def train_one(run: dict, cfg: dict) -> dict:
    from brainalign_wm.utils.seeding import seed_everything
    from brainalign_wm.utils.device import get_device
    from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank
    from brainalign_wm.tasks.generator import TaskGenerator

    full_cfg = _load_full_config()
    # Core cells carry "P" (synaptic plasticity, BPTT throughout, §17
    # decision 6); the four Extended local-learning cells (M**L) carry
    # "L"=1 instead and no "P" key -- the two knobs are mutually exclusive.
    S, M, seed = run["S"], run["M"], run["seed"]
    L = run.get("L", 0)
    P = run.get("P", 0)
    pbwm_gate = bool(run.get("pbwm_gate", False))  # ablation-battery arm M111_pbwm only (§4.4)
    total_steps = int(cfg.get("steps", full_cfg["tiers"]["smoke"]["steps"]))
    run_id = run["run_id"]

    seed_everything(seed)
    device = get_device()

    # Bio-plausible ablation battery (§4.4) and identity-catch (§9.4a) arms
    # are per-run, not global config edits -- folding `run` overrides into
    # `full_cfg` here (before deriving m/mech_cfg/t_cfg or building
    # task_gen/heads/evaluate_accuracy, all of which read from full_cfg)
    # means distinct arms/model_ids can share one config.yaml and every
    # downstream reader (including eval-time accuracy) sees the override.
    if "recurrent_noise_sigma" in run:
        full_cfg = {**full_cfg, "model": {**full_cfg["model"], "recurrent_noise_sigma": float(run["recurrent_noise_sigma"])}}
    if "identity_catch_fraction" in run:
        full_cfg = {**full_cfg, "task": {**full_cfg["task"], "identity_catch_fraction": float(run["identity_catch_fraction"])}}
    # Performance-matched baseline family (flat_gru_2x/l1/dropout, §4.1):
    # `flat_units` changes the S=0 core's actual parameter shape, so (like
    # the other overrides above) it must be folded in before `_build_model`
    # reads it. `l1_weight`/`core_dropout_p` don't change parameter shapes
    # (pure training-time regularizers), so they're read straight from
    # `run` below without touching `full_cfg`.
    if "flat_units" in run:
        full_cfg = {**full_cfg, "model": {**full_cfg["model"], "flat_units": int(run["flat_units"])}}

    m, mech_cfg, t_cfg = full_cfg["model"], full_cfg["mechanisms"], full_cfg["train"]
    value_weight = float(t_cfg["value_loss_weight"])
    entropy_coef = float(t_cfg.get("entropy_coef", 0.0))
    energy_cost_weight = float(run.get("energy_cost_weight", t_cfg.get("energy_cost_weight", 0.0)))
    # T/D (arms of the 5-bit ablation battery, unlike energy/noise/pbwm
    # which are per-run overrides with no config-level "on" state) are
    # architecture-adjacent bits carried on `run["T"]`/`run["D"]` by every
    # battery cell (run_grid.py's CELLS) -- an explicit `topo_loss_weight`/
    # `dale_penalty_weight` key in `run` still wins (for supplementary
    # arms that want a custom magnitude), but otherwise the bit itself
    # selects between the config's "on" and "off" (0.0) magnitudes, so a
    # T=1/D=1 cell actually trains with the penalty active rather than
    # silently defaulting to config.yaml's off value.
    if "topo_loss_weight" in run:
        topo_loss_weight = float(run["topo_loss_weight"])
    else:
        topo_loss_weight = float(t_cfg.get("topo_loss_weight_on", 0.01)) if run.get("T", 0) else float(t_cfg.get("topo_loss_weight", 0.0))
    if "dale_penalty_weight" in run:
        dale_penalty_weight = float(run["dale_penalty_weight"])
    else:
        dale_penalty_weight = float(t_cfg.get("dale_penalty_weight_on", 0.01)) if run.get("D", 0) else float(t_cfg.get("dale_penalty_weight", 0.0))
    dale_ei_split = float(run.get("dale_ei_split", m.get("dale_ei_split", 0.8)))
    flat_grid = tuple(run.get("flat_grid", m.get("flat_grid", [16, 16])))
    recurrent_noise_sigma = float(m.get("recurrent_noise_sigma", 0.0))
    core_dropout_p = float(run.get("core_dropout_p", m.get("core_dropout_p", 0.0)))
    # M7 fix: the per-run override key now matches the config key
    # (`l1_weight_penalty`) instead of the old ad hoc `l1_weight` shortname.
    l1_weight = float(run.get("l1_weight_penalty", t_cfg.get("l1_weight_penalty", 0.0)))
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

    front_end, core, heads = _build_model(full_cfg, S, M, P, device, pbwm_gate=pbwm_gate)
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

    # Audit addition: periodic metrics logging
    eval_every = int(t_cfg.get("eval_every", max(500, total_steps // 30)))
    metrics_logger = _MetricsLogger(run_id)

    rung_check_step = int(t_cfg["rung_check_frac"] * total_steps)
    # Second gate check, rung 2 -> 3 (e-prop): same
    # pattern as rung 1 -> 2, at a later checkpoint so rung 2 gets its own
    # fair share of the budget before e-prop is attempted.
    rung_check_step_2 = int(t_cfg.get("rung_check_frac_2", 0.75) * total_steps)
    checked_rung_1 = rung >= 2
    checked_rung_2 = rung >= 3

    t0 = time.time()
    running_loss = 0.0
    loss_count = 0
    for step in range(start_step, total_steps):
        params = task_gen.curriculum_params(step, total_steps)
        phase = params["phase"]
        trial_batch = task_gen.sample_batch(step, total_steps, batch_size)

        if L == 0:
            optimizer.zero_grad()
            signal = "ce" if phase == "warmup" else "reinforce"
            loss, correct, reward, _ = _run_trial(
                front_end, core, heads, reflective_gate, S, M, P, trial_batch, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), device, mode="bptt",
                signal=signal, value_weight=value_weight, entropy_coef=entropy_coef,
                energy_cost_weight=energy_cost_weight, topo_loss_weight=topo_loss_weight, flat_grid=flat_grid,
                recurrent_noise_sigma=recurrent_noise_sigma,
                core_dropout_p=core_dropout_p, categories=full_cfg["task"]["categories"],
            )
            if l1_weight > 0:
                l1_term = sum(p.abs().mean() for n, p in core.named_parameters() if "weight" in n)
                loss = loss + l1_weight * l1_term
            if dale_penalty_weight > 0:
                loss = loss + dale_penalty_weight * _dale_penalty(core, S, dale_ei_split)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(front_end.parameters()) + list(core.parameters()) + list(heads.parameters()), 5.0
            )
            optimizer.step()
            # Track training metrics
            if loss is not None:
                running_loss += loss.item()
                loss_count += 1
        else:
            grad_norm = 0.0
            dense_reward = phase == "warmup"
            correct, reward = _run_trial_local(
                front_end, core, heads, reflective_gate, S, learners, trial_batch, image_bank,
                m["feature_dim"], m["action_dim"], _gate_width(S, m), mech_cfg["perturb_sigma"], device,
                dense_reward=dense_reward,
            )
            if not checked_rung_1 and step >= rung_check_step:
                interim = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, full_cfg, device)
                if interim.get("load1", 0.0) < full_cfg["gates"]["load1_acc"]:
                    rung = 2
                    learners = _make_local_learners(core, heads, S, mech_cfg, rung, seed + 100)
                checked_rung_1 = True
            elif checked_rung_1 and not checked_rung_2 and rung == 2 and step >= rung_check_step_2:
                interim = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, full_cfg, device)
                if interim.get("load1", 0.0) < full_cfg["gates"]["load1_acc"]:
                    rung = 3
                    learners = _make_local_learners(core, heads, S, mech_cfg, rung, seed + 200)
                checked_rung_2 = True

        # Periodic metrics logging (audit addition)
        if (step + 1) % eval_every == 0 or step == total_steps - 1:
            avg_loss = running_loss / max(loss_count, 1)
            acc = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, full_cfg, device)
            metrics_logger.log({
                "step": step + 1,
                "phase": phase,
                "train_loss": f"{avg_loss:.6f}",
                "train_acc_load1": acc.get("load1", ""),
                "train_acc_load2": acc.get("load2", ""),
                "train_acc_load3": acc.get("load3", ""),
                "grad_norm": f"{grad_norm:.6f}" if isinstance(grad_norm, float) else f"{grad_norm.item():.6f}",
                "wall_s": f"{time.time() - t0:.1f}",
            })
            running_loss = 0.0
            loss_count = 0

            # Early stopping: all 3 loads at 1.0
            if all(acc.get(f"load{i}", 0.0) >= 0.999 for i in (1, 2, 3)):
                print(f"[train] early stop at step {step+1}: all loads >= 0.999", flush=True)
                break

        if (step + 1) % t_cfg["checkpoint_every"] == 0 or step == total_steps - 1:
            torch.save(
                {
                    "step": step + 1, "front_end": front_end.state_dict(), "core": core.state_dict(),
                    "heads": heads.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                    "rung": rung,
                },
                ckpt_path,
            )

    metrics_logger.close()
    accuracy = evaluate_accuracy(front_end, core, heads, S, P, reflective_gate, task_gen, image_bank, full_cfg, device)
    gates = {
        "load1>=0.95": accuracy.get("load1", 0.0) >= full_cfg["gates"]["load1_acc"],
        "load3>=0.80": accuracy.get("load3", 0.0) >= full_cfg["gates"]["load3_acc"],
    }
    return {
        "status": "completed", "gates": gates, "accuracy": accuracy,
        "rung": rung if L == 1 else 0, "wall_clock_train_s": round(time.time() - t0, 1),
    }
