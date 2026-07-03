"""M3 gates (protocol §11.5, §14): reflective gate opens under sustained
surprise; node-perturbation update sign matches reward. ("M111 learns
load-1 Sternberg" is an end-to-end training gate -- exercised in M7's
integration test, once `training/train.py::train_one` exists.)"""
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
    """The core §6.1 causal claim: high R_t -> u_t -> 1 (manager overwrites);
    low R_t -> gate closed (state held). Verified directly on a MaskedGRUCell
    fed a constant gate_bias, isolating the mechanism from the rest of HRLCore."""
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
    """§11.5: 'node-perturbation update sign matches reward.' A trace that's
    positive on average should move the weight in the reward-consistent
    direction: reward > baseline -> weight increases along the traced
    (perturbation x presynaptic) direction; reward < baseline -> decreases."""
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
