"""Mechanism tests: the reflective gate opens under sustained surprise, and
the node-perturbation update sign matches the sign of the reward prediction
error. (Whether cell M111 learns load-1 Sternberg is an end-to-end training
gate, exercised separately by the training integration test.)"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from brainalign_wm.mechanisms.reflective_gate import ReflectiveGate, shuffle_reflection
from brainalign_wm.mechanisms.local_learning import NodePerturbationLearner, RUNG_NAMES, RungExhausted, make_learner_for_cell
from brainalign_wm.models.gru_cell import MaskedGRUCell

BATCH = 8


def test_reflection_accumulates_and_decays():
    gate = ReflectiveGate(lambda_R=0.9, beta=1.0)
    R = gate.init_state(BATCH)
    # sustained high surprise -> R should rise monotonically toward a plateau
    Rs = []
    for _ in range(50):
        delta = torch.ones(BATCH, 1) * 2.0
        R = gate.step(delta, R)
        Rs.append(R.mean().item())
    assert Rs[-1] > Rs[0]
    assert Rs[-1] > Rs[len(Rs) // 2]  # still rising/plateauing, not decaying
    # now switch to zero surprise -> R should decay back down
    for _ in range(50):
        delta = torch.zeros(BATCH, 1)
        R = gate.step(delta, R)
    assert R.mean().item() < Rs[-1]


def test_gate_bias_opens_update_gate_under_sustained_surprise():
    """The central causal claim of the reflective gate: a high R_t drives
    u_t toward 1 (the manager overwrites its state), while a low R_t keeps
    the gate closed (state held). Verified directly on a MaskedGRUCell fed a
    constant gate_bias, isolating the mechanism from the rest of HRLCore."""
    torch.manual_seed(0)
    cell = MaskedGRUCell(input_dim=16, hidden_dim=32)
    h = torch.zeros(BATCH, 32)  # gru_cell.MaskedGRUCell has no init_state helper; zeros directly
    x = torch.randn(BATCH, 16)

    _, u_low = cell(x, h, extra_update_bias=torch.zeros(BATCH, 32))
    _, u_high = cell(x, h, extra_update_bias=torch.full((BATCH, 32), 10.0))
    assert u_high.mean().item() > u_low.mean().item()
    assert u_high.mean().item() > 0.95, "large positive bias should saturate the update gate near 1"


def test_reflection_shuffle_preserves_marginal_destroys_order():
    torch.manual_seed(0)
    R_seq = torch.arange(20, dtype=torch.float32).reshape(20, 1, 1).repeat(1, BATCH, 1)
    gen = torch.Generator().manual_seed(0)
    shuffled = shuffle_reflection(R_seq, generator=gen)
    assert not torch.equal(shuffled, R_seq)
    assert torch.equal(shuffled.sort(dim=0).values, R_seq.sort(dim=0).values)  # same multiset


def test_node_perturbation_update_sign_matches_reward():
    """The node-perturbation update sign must match the sign of the reward
    prediction error: a trace that is positive on average should move the
    weight in the reward-consistent direction (reward above baseline
    increases the weight along the traced perturbation-by-presynaptic
    direction; reward below baseline decreases it)."""
    torch.manual_seed(0)
    cell = MaskedGRUCell(input_dim=8, hidden_dim=16)
    learner = NodePerturbationLearner(
        weight_hh=cell.weight_hh, weight_ih=cell.weight_ih,
        sigma_p=0.1, gamma_e=0.9, lr_local=0.1, seed=0,
    )
    h_prev = torch.ones(BATCH, 16) * 0.5
    x_t = torch.ones(BATCH, 8) * 0.5
    xi_t = torch.ones(BATCH, 48)  # positive perturbation on all (3*hidden) gate pre-units
    learner.trace_step(xi_t, h_prev=h_prev, x_t=x_t)

    w_hh_before = cell.weight_hh.data.clone()
    learner.reward_baseline = 0.0
    rpe = learner.apply_update(reward=1.0)  # reward > baseline -> positive rpe
    assert rpe > 0
    delta = (cell.weight_hh.data - w_hh_before)
    assert delta.mean().item() > 0, "positive trace x positive rpe should increase weights on average"

    # now a negative reward relative to (now-updated) baseline should push the other way
    learner2 = NodePerturbationLearner(
        weight_hh=nn.Parameter(w_hh_before.clone()), weight_ih=cell.weight_ih,
        sigma_p=0.1, gamma_e=0.9, lr_local=0.1, seed=0,
    )
    learner2.trace_step(xi_t, h_prev=h_prev, x_t=x_t)
    w_before2 = learner2._traced[0].param.data.clone()
    learner2.reward_baseline = 1.0
    rpe2 = learner2.apply_update(reward=0.0)
    assert rpe2 < 0
    delta2 = learner2._traced[0].param.data - w_before2
    assert delta2.mean().item() < 0, "positive trace x negative rpe should decrease weights on average"


def test_rung_ladder_rejects_unimplemented_rung3():
    cell = MaskedGRUCell(input_dim=8, hidden_dim=16)
    with pytest.raises(RungExhausted):
        make_learner_for_cell(cell, sigma_p=0.05, gamma_e=0.9, lr_local=1e-3, rung=3, seed=0)
    assert RUNG_NAMES[3] == "e_prop"


def test_rung2_normalizes_traces():
    cell = MaskedGRUCell(input_dim=8, hidden_dim=16)
    learner = make_learner_for_cell(cell, sigma_p=0.05, gamma_e=0.9, lr_local=1e-3, rung=2, seed=0)
    assert learner.normalize_traces and learner.adaptive_baseline
    assert learner.rung == 2


def test_rung2_baseline_decay_distinct_from_rung1():
    """Audit fix A1b: rung 2's `adaptive_baseline` must have a real, distinct
    effect from rung 1 -- previously the `if adaptive_baseline` branch was
    byte-identical to the `else` branch (a dead distinction)."""
    cell = MaskedGRUCell(input_dim=8, hidden_dim=16)
    rung1 = make_learner_for_cell(cell, sigma_p=0.05, gamma_e=0.9, lr_local=1e-3, rung=1, seed=0)
    rung2 = make_learner_for_cell(cell, sigma_p=0.05, gamma_e=0.9, lr_local=1e-3, rung=2, seed=0)
    assert rung1.baseline_decay_adaptive == rung2.baseline_decay_adaptive  # same constant available to both
    assert rung1.baseline_decay != rung1.baseline_decay_adaptive  # but only rung 2 actually uses it
    rung1.reward_baseline = rung2.reward_baseline = 0.0
    rung1.apply_update(reward=1.0)
    rung2.apply_update(reward=1.0)
    assert rung1.reward_baseline != rung2.reward_baseline, "rung 1 and rung 2 must track the baseline differently"


def test_batched_node_perturbation_preserves_per_sample_correlation():
    """Audit fix A1c: batching must keep PER-SAMPLE eligibility traces until
    the reward-weighted average at `apply_update` -- averaging the trace
    across the batch BEFORE the reward is known would replace
    mean_b[(r_b-baseline)*xi_b] with mean_b[r_b-baseline]*mean_b[xi_b],
    destroying the reward-perturbation correlation the algorithm exploits.
    Construct a batch where the (wrong) product-of-averages is exactly zero
    but the (correct) average-of-products is not."""
    cell = MaskedGRUCell(input_dim=4, hidden_dim=4)
    learner = NodePerturbationLearner(
        weight_hh=cell.weight_hh, weight_ih=cell.weight_ih, sigma_p=0.1, gamma_e=0.0, lr_local=1.0, seed=0,
    )
    learner.reward_baseline = 0.0
    # batch of 2: perturbation +1 paired with reward 1; perturbation -1
    # paired with reward 0. Batch-mean perturbation is 0 (so the wrong,
    # trace-averaged-first computation would report zero update), but each
    # sample's own (perturbation, reward) pair is genuinely correlated.
    xi_t = torch.ones(2, 12)  # [B=2, 3*hidden=12]
    xi_t[1] = -1.0
    h_prev = torch.ones(2, 4)
    x_t = torch.ones(2, 4)
    learner.trace_step(xi_t, h_prev=h_prev, x_t=x_t)

    w_before = cell.weight_hh.data.clone()
    learner.apply_update(reward=[1.0, 0.0])
    delta = cell.weight_hh.data - w_before
    assert delta.abs().mean().item() > 1e-6, (
        "batched update collapsed to ~0 -- traces were likely averaged across the "
        "batch before the reward was applied (destroys the reward-perturbation correlation)"
    )
    assert delta.mean().item() > 0, "sample with positive perturbation and above-baseline reward should dominate positively"
