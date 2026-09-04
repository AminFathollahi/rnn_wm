#!/usr/bin/env python3
"""Item 9.2 (comments.txt §5): distills the trained teacher (`M00000_teacher_s0`,
Phase 9c prep -- criterion_met at step 70000, final load1/2/3 =
1.0/0.974/0.958) into a ladder of progressively smaller "tiny RNN" students,
behaviorally cloning its per-tick policy. Reports the smallest student that
(a) clears the §3 criterion itself and (b) has teacher-matched representational
geometry (RSA of student vs teacher RDMs).

DESIGN. The teacher's front_end (frozen, shared across every student) is
reused as-is: item 9.2 is about the CAPACITY of the recurrent core (that is
what "tiny RNN" means here, same axis Phase 9a's capacity curve already
varies), not about re-learning visual features from scratch. Only a fresh,
smaller `_GatedFlatCore` (S=0,M=0,P=0, same family as the teacher) + `Heads`
are trained per student. Given the shared frozen front_end, teacher and
student see the IDENTICAL per-tick input `z_t` -- the only thing that
differs is the recurrent core's capacity and what it learns to do with that
input.

OBJECTIVE. Per-tick soft cross-entropy between the student's policy logits
and the teacher's OWN policy (softmax of `heads.pi`, no_grad) on the same
trials -- pure behavioral cloning, not RL (much cheaper/faster to converge
than the teacher's original REINFORCE training, since supervision is dense
and comes from an already-correct teacher). Comments.txt's "(and optionally
its state through a learned linear map)" is attempted too, cheaply: a small
`nn.Linear(student_h_dim, teacher_h_dim)` trained jointly via an MSE term
against the teacher's own (detached) `h_star`, at a small weight -- this
directly helps criterion (b) (a state-matching auxiliary loss should only
improve RSA alignment, never hurt it, and it costs one small linear layer).

TRAIN-TO-CRITERION. Reuses this session's just-revised hybrid policy
conceptually (not `train_one` itself, which is RL/BPTT-shaped and not a fit
for a supervised distillation loop): periodic `evaluate_accuracy` checks
every `eval_every` steps, a streak of 3 consecutive passing evals stops
training and triggers the official `final_evaluation` (disjoint eval seed,
same anti-leakage property as `train.py`), with `MAX_STEPS` as a ceiling for
students that never confirm the criterion (much smaller than the teacher's
150k, since distillation from a working teacher is a far easier objective
than the teacher's own from-scratch RL training).

RSA. Condition = (load, in_set) -- the coarsest task variable Sternberg
itself defines (3 loads x {in-set, not-in-set} = 6 conditions), matching
`scripts/run_geometry.py`'s own `epoch=='maintain'` (delay-period) reading
convention. `in_set` is only populated at the probe/feedback epoch (see
`sternberg.py::TrialStep`), so it's read off each trial's own probe tick
and paired with that SAME trial's last-maintain-tick `h_star`. A plain
condition-averaged RDM (1 - Pearson correlation between condition-mean
h_star vectors) is used, NOT `rdm.py`'s crossnobis machinery -- crossnobis
cross-validates away trial-to-trial MEASUREMENT noise across repeated
neural recordings of the same condition; comparing two RNNs' own
activations model-to-model has no such repeated-measurement noise problem,
so that machinery doesn't apply here. `analysis/rsa.py::compare_rdms`
(Spearman correlation between the two RDMs' upper triangles) is reused
directly for the actual RSA statistic.

Usage:
  python scripts/run_distillation_students.py --hidden 1 2 4 8 16 32 64
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_grid import ROOT  # noqa: E402

from brainalign_wm.config import get_path  # noqa: E402
from brainalign_wm.analysis.rsa import compare_rdms  # noqa: E402
from brainalign_wm.tasks.image_token_bank import ImageTokenBank  # noqa: E402
from brainalign_wm.tasks.generator import TaskGenerator  # noqa: E402
from brainalign_wm.training.train import (  # noqa: E402
    _build_model, _gate_width, _init_state, _step_core, _load_full_config,
    evaluate_accuracy, final_evaluation, wilson_ci,
)

TEACHER_RUN_ID = "M00000_teacher_s0"
DEFAULT_HIDDEN_LADDER = [1, 2, 4, 8, 16, 32, 64]
MAX_STEPS = 8000            # ceiling; teacher-supervised distillation converges far faster than RL from scratch
EVAL_EVERY = 200
CONSECUTIVE_EVALS_REQUIRED = 3
EVAL_TRIALS_PER_LOAD = 200   # matches gates.criterion's own periodic-eval convention
BATCH_SIZE = 32
LR = 3e-3
STATE_MATCH_WEIGHT = 0.1
RDM_N_TRIALS_PER_CONDITION = 40
CURRICULUM_OFFSET = 200_000  # step_idx offset guaranteeing CurriculumSchedule.params_for returns "target" phase


def _load_teacher(full_cfg: dict, device):
    front_end, core, heads = _build_model(full_cfg, 0, 0, 0, device)
    ckpt = torch.load(get_path("results") / "checkpoints" / TEACHER_RUN_ID / "ckpt.pt", map_location=device, weights_only=False)
    front_end.load_state_dict(ckpt["front_end"])
    core.load_state_dict(ckpt["core"])
    heads.load_state_dict(ckpt["heads"])
    for net in (front_end, core, heads):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    return front_end, core, heads


def _build_student(full_cfg: dict, hidden: int, device):
    cfg2 = copy.deepcopy(full_cfg)
    cfg2["model"]["flat_units"] = hidden
    _, core, heads = _build_model(cfg2, 0, 0, 0, device)  # discard student front_end; teacher's is shared/frozen
    return core, heads


def _soft_ce(teacher_policy: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy: -sum_a teacher_policy[a] * log_softmax(student_logits)[a],
    averaged over the batch. Standard distillation loss (Hinton et al. 2015);
    equal to KL(teacher || student) up to the teacher's own (student-independent)
    entropy term, so it has the same gradient w.r.t. the student."""
    log_probs = F.log_softmax(student_logits, dim=-1)
    return -(teacher_policy * log_probs).sum(dim=-1).mean()


def _forward_tick(front_end, core, heads, z_t, c_t, state):
    """One tick, S=0/M=0/P=0 (no reflective gate, no plasticity, no
    recurrent noise/dropout -- none of the ablation-battery arms apply to
    the teacher or any student here)."""
    h_star, new_state, _u_t = _step_core(core, 0, 0, 0, z_t, state, t=0, gate_bias=None)
    policy, value, logits = heads(h_star)
    return h_star, policy, logits, new_state


def _distill_step(front_end, teacher_core, teacher_heads, student_core, student_heads, state_map,
                   trial_steps_batch, image_bank, feature_dim, device, train_student: bool):
    """Unrolls teacher (no_grad) and student (grad iff train_student) in
    lockstep on the same batch, through the SAME shared frozen front_end.
    Returns (mean per-tick soft-CE loss [scalar tensor or python float],
    mean per-tick state-match MSE [python float])."""
    B = len(trial_steps_batch)
    T = len(trial_steps_batch[0])
    teacher_state = _init_state(teacher_core, 0, 0, B, device)
    student_state = _init_state(student_core, 0, 0, B, device)

    ce_terms = []
    state_terms = []
    for i in range(T):
        ts_list = [trial_steps_batch[b][i] for b in range(B)]
        image_ids = [ts.image_id for ts in ts_list]
        feats = np.zeros((B, feature_dim), dtype=np.float32)
        for b, image_id in enumerate(image_ids):
            if image_id is not None:
                feats[b] = np.asarray(image_bank.feature_of(image_id), dtype=np.float32)
        v_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
        c_t = torch.tensor([ts.c_t for ts in ts_list], dtype=torch.float32, device=device)

        with torch.no_grad():
            z_t = front_end(v_t, c_t)
            t_h_star, t_policy, _t_logits, teacher_state = _forward_tick(
                front_end, teacher_core, teacher_heads, z_t, c_t, teacher_state
            )

        ctx = torch.enable_grad() if train_student else torch.no_grad()
        with ctx:
            s_h_star, _s_policy, s_logits, student_state = _forward_tick(
                front_end, student_core, student_heads, z_t, c_t, student_state
            )
            ce_terms.append(_soft_ce(t_policy, s_logits))
            if state_map is not None:
                mse = F.mse_loss(state_map(s_h_star), t_h_star)
                state_terms.append(mse)

    ce_loss = torch.stack(ce_terms).mean()
    state_loss = torch.stack(state_terms).mean() if state_terms else torch.zeros((), device=device)
    return ce_loss, state_loss


def train_student(full_cfg, front_end, teacher_core, teacher_heads, task_gen, image_bank, hidden: int, seed: int, device,
                   max_steps: int = MAX_STEPS, eval_every: int = EVAL_EVERY):
    torch.manual_seed(seed)
    m = full_cfg["model"]
    student_core, student_heads = _build_student(full_cfg, hidden, device)
    state_map = torch.nn.Linear(hidden, m["flat_units"]).to(device)
    params = list(student_core.parameters()) + list(student_heads.parameters()) + list(state_map.parameters())
    optimizer = torch.optim.Adam(params, lr=LR)

    consecutive = 0
    criterion = full_cfg["gates"]["criterion"]
    task_loads = full_cfg["task"]["loads"]
    criterion_met = False
    steps_to_criterion = None
    eval_call_counter = 0
    t0 = time.time()

    for step in range(max_steps):
        batch = task_gen.sample_batch(CURRICULUM_OFFSET + step, CURRICULUM_OFFSET + max_steps, BATCH_SIZE)
        optimizer.zero_grad()
        ce_loss, state_loss = _distill_step(
            front_end, teacher_core, teacher_heads, student_core, student_heads, state_map,
            batch, image_bank, m["feature_dim"], device, train_student=True,
        )
        loss = ce_loss + STATE_MATCH_WEIGHT * state_loss
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_every == 0:
            eval_call_counter += 1
            acc = evaluate_accuracy(
                front_end, student_core, student_heads, 0, 0, None, task_gen, image_bank, full_cfg, device,
                n_trials=EVAL_TRIALS_PER_LOAD, eval_seed=seed * 1_000_000 + eval_call_counter,
            )
            print(f"[distill H={hidden}] step={step + 1} ce={ce_loss.item():.4f} "
                  f"acc={ {k: acc[k] for k in acc if not k.endswith('_ci_lo') and not k.endswith('_ci_hi')} }", flush=True)
            if not criterion_met:
                if all(acc.get(f"load{i}", 0.0) >= criterion[f"load{i}"] for i in task_loads):
                    consecutive += 1
                else:
                    consecutive = 0
                if consecutive >= CONSECUTIVE_EVALS_REQUIRED:
                    criterion_met = True
                    steps_to_criterion = step + 1

        if criterion_met:
            break

    accuracy = final_evaluation(front_end, student_core, student_heads, 0, 0, None, task_gen, image_bank, full_cfg, device, seed)
    gates = {f"load{i}>={criterion[f'load{i}']}": accuracy.get(f"load{i}", 0.0) >= criterion[f"load{i}"] for i in task_loads}
    return {
        "hidden": hidden, "student_core": student_core, "student_heads": student_heads,
        "accuracy": accuracy, "gates": gates, "criterion_met": criterion_met,
        "steps_to_criterion": steps_to_criterion, "wall_clock_s": round(time.time() - t0, 1),
    }


def _collect_condition_reps(front_end, core, heads, task_gen, full_cfg, device, seed: int, n_per_condition: int) -> dict:
    """{(load, in_set): mean h_star [h_dim]} -- see module docstring for the
    condition scheme and why `in_set` is read from the probe tick."""
    m = full_cfg["model"]
    rng = np.random.RandomState(seed)
    reps: dict = {}
    for load in full_cfg["task"]["loads"]:
        batch = []
        for _ in range(n_per_condition):
            trial_seed = int(rng.randint(0, 2**31 - 1))
            trial_rng = np.random.RandomState(trial_seed)
            steps = task_gen.sternberg.generate_trial(
                rng=trial_rng, loads=[load], lure_fraction=full_cfg["task"]["lure_fraction"],
                maintain_steps=full_cfg["task"]["maintain_steps"], trial_id=-1, split="test",
                identity_catch_fraction=0.0,
            )
            batch.append(steps)
        T = len(batch[0])
        B = len(batch)
        state = _init_state(core, 0, 0, B, device)
        last_maintain_h = [None] * B
        with torch.no_grad():
            for i in range(T):
                ts_list = [batch[b][i] for b in range(B)]
                feats = np.zeros((B, m["feature_dim"]), dtype=np.float32)
                for b, ts in enumerate(ts_list):
                    if ts.image_id is not None:
                        feats[b] = np.asarray(task_gen.bank.feature_of(ts.image_id), dtype=np.float32)
                v_t = torch.as_tensor(feats, dtype=torch.float32, device=device)
                c_t = torch.tensor([ts.c_t for ts in ts_list], dtype=torch.float32, device=device)
                z_t = front_end(v_t, c_t)
                h_star, _policy, _logits, state = _forward_tick(front_end, core, heads, z_t, c_t, state)
                for b, ts in enumerate(ts_list):
                    if ts.epoch == "maintain":
                        last_maintain_h[b] = h_star[b].cpu().numpy()
        probe_i = next(i for i, s in enumerate(batch[0]) if s.epoch == "probe")
        for b in range(B):
            in_set = bool(batch[b][probe_i].in_set)
            if last_maintain_h[b] is None:
                continue
            reps.setdefault((load, in_set), []).append(last_maintain_h[b])
    return {k: np.mean(np.stack(v), axis=0) for k, v in reps.items() if v}


def _rdm_from_reps(reps: dict, condition_order: list) -> np.ndarray:
    vecs = np.stack([reps[c] for c in condition_order])
    n = len(condition_order)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            corr = np.corrcoef(vecs[i], vecs[j])[0, 1]
            rdm[i, j] = 1.0 - corr
    return rdm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, nargs="+", default=DEFAULT_HIDDEN_LADDER)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_cfg = _load_full_config()
    stimuli_root = Path(full_cfg["paths"]["stimuli"])
    image_bank = ImageTokenBank(
        stimuli_root=stimuli_root, categories=full_cfg["task"]["categories"],
        feature_cache_path=Path(full_cfg["paths"]["feature_cache"]) / "image_token_bank.npy", seed=0,
    )
    task_gen = TaskGenerator(full_cfg, image_bank, seed=args.seed)

    print(f"[distill] loading teacher {TEACHER_RUN_ID}", flush=True)
    front_end, teacher_core, teacher_heads = _load_teacher(full_cfg, device)

    teacher_reps = _collect_condition_reps(
        front_end, teacher_core, teacher_heads, task_gen, full_cfg, device,
        seed=args.seed * 1000 + 777, n_per_condition=RDM_N_TRIALS_PER_CONDITION,
    )
    condition_order = sorted(teacher_reps.keys())
    teacher_rdm = _rdm_from_reps(teacher_reps, condition_order)
    print(f"[distill] teacher representations collected over {len(condition_order)} conditions: {condition_order}", flush=True)

    results = []
    for hidden in args.hidden:
        print(f"[distill] === training student H={hidden} ===", flush=True)
        r = train_student(full_cfg, front_end, teacher_core, teacher_heads, task_gen, image_bank, hidden, args.seed, device,
                           max_steps=args.max_steps, eval_every=args.eval_every)
        student_reps = _collect_condition_reps(
            front_end, r["student_core"], r["student_heads"], task_gen, full_cfg, device,
            seed=args.seed * 1000 + 777, n_per_condition=RDM_N_TRIALS_PER_CONDITION,
        )
        student_rdm = _rdm_from_reps(student_reps, condition_order)
        rsa = compare_rdms(teacher_rdm, student_rdm, method="spearman")
        record = {
            "hidden": hidden, "accuracy": r["accuracy"], "gates": r["gates"],
            "criterion_met": r["criterion_met"], "steps_to_criterion": r["steps_to_criterion"],
            "wall_clock_s": r["wall_clock_s"], "rsa_vs_teacher": rsa,
        }
        results.append(record)
        print(f"[distill] H={hidden} criterion_met={r['criterion_met']} "
              f"steps_to_criterion={r['steps_to_criterion']} accuracy={r['accuracy']} rsa={rsa:.4f}", flush=True)

    teacher_self_reps = _collect_condition_reps(
        front_end, teacher_core, teacher_heads, task_gen, full_cfg, device,
        seed=args.seed * 1000 + 778, n_per_condition=RDM_N_TRIALS_PER_CONDITION,
    )
    teacher_rdm_2 = _rdm_from_reps(teacher_self_reps, condition_order)
    teacher_self_consistency = compare_rdms(teacher_rdm, teacher_rdm_2, method="spearman")
    print(f"[distill] teacher self-consistency RSA (independent trial draw): {teacher_self_consistency:.4f}", flush=True)

    out = {
        "teacher_run_id": TEACHER_RUN_ID, "condition_order": [list(c) for c in condition_order],
        "teacher_self_consistency_rsa": teacher_self_consistency, "students": results,
    }
    out_path = get_path("results") / "distillation_students.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[distill] wrote {out_path}", flush=True)

    smallest_ok = None
    for record in sorted(results, key=lambda r: r["hidden"]):
        behaviorally_ok = record["criterion_met"]
        geometry_ok = record["rsa_vs_teacher"] >= 0.8 * teacher_self_consistency
        if behaviorally_ok and geometry_ok:
            smallest_ok = record
            break
    if smallest_ok is not None:
        print(f"\n[distill] SMALLEST student preserving both behaviour and geometry: "
              f"H={smallest_ok['hidden']} (rsa={smallest_ok['rsa_vs_teacher']:.4f}, "
              f"threshold={0.8 * teacher_self_consistency:.4f})", flush=True)
    else:
        print("\n[distill] NO student in the swept ladder cleared both bars -- honest negative result.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
