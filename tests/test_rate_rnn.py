"""Leaky rate substrates: the structural constraints each cell exists to
impose (fixed E/I signs, rank-R recurrence, short-term synaptic state), the
effective-synapse budget they share with the flat GRU baseline cell, and
end-to-end gradient flow."""
import pytest

from brainalign_wm.config import load_config

torch = pytest.importorskip("torch")

from brainalign_wm.models.gru_cell import MaskedGRUCell
from brainalign_wm.models.rate_rnn import (
    DynamicSynapseRateCell, ExcitatoryInhibitoryRateCell, LowRankRateCell, build_rate_cell,
)

FULL_CFG = load_config()
CFG = FULL_CFG["model"]
RATE_CFG = CFG["rate_rnn"]


def _baseline_effective_synapses() -> int:
    """The flat GRU baseline cell's effective synapse count -- the budget
    every substrate in this module is matched to."""
    return MaskedGRUCell(CFG["bottleneck"], CFG["flat_units"], mask=None).effective_param_count()


def _configured_cells():
    return {name: build_rate_cell(name, CFG["bottleneck"], RATE_CFG)
            for name in ("excitatory_inhibitory", "dynamic_synapse", "low_rank")}


def test_excitatory_inhibitory_signs_survive_an_optimizer_step():
    """Hard separation, not a penalty: no gradient step may flip the sign of
    a unit's outgoing weights."""
    cell = ExcitatoryInhibitoryRateCell(4, 10, excitatory_fraction=0.8, recurrent_init_spectral_radius=1.0)
    n_e = cell.n_excitatory
    assert n_e == 8

    before = cell.recurrent_weight().detach().clone()
    target = torch.rand(3, 10) * 3.0
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.05)
    for _ in range(20):
        h = cell.init_state(3)
        for _ in range(4):
            h, _ = cell(torch.randn(3, 4), h)
        optimizer.zero_grad()
        (h - target).pow(2).mean().backward()
        optimizer.step()

    w = cell.recurrent_weight()
    assert not torch.allclose(w, before), "recurrent weights did not move, so the test proves nothing"
    assert torch.all(w[:, :n_e] >= 0), "an excitatory unit's outgoing weights turned negative"
    assert torch.all(w[:, n_e:] <= 0), "an inhibitory unit's outgoing weights turned positive"
    assert (w[:, :n_e] > 0).any() and (w[:, n_e:] < 0).any(), "recurrent weights collapsed to all-zero"


def test_excitatory_inhibitory_rates_are_non_negative():
    cell = ExcitatoryInhibitoryRateCell(4, 10)
    h = cell.init_state(3)
    for _ in range(5):
        h, gate = cell(torch.randn(3, 4), h)
        assert gate is None
        assert torch.all(h >= 0), "softplus rate units must not go negative"


def test_low_rank_recurrent_matrix_has_the_configured_rank():
    for rank in (1, 4, 9):
        cell = LowRankRateCell(4, 32, rank=rank, recurrent_init_spectral_radius=1.0)
        w = cell.recurrent_weight()
        assert w.shape == (32, 32)
        assert int(torch.linalg.matrix_rank(w.detach())) == rank
        assert cell.left.shape == (32, rank) and cell.right.shape == (32, rank)


def test_low_rank_keeps_its_rank_after_an_optimizer_step():
    cell = LowRankRateCell(4, 32, rank=4)
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.1)
    h = cell.init_state(3)
    for _ in range(3):
        h, _ = cell(torch.randn(3, 4), h)
    optimizer.zero_grad()
    h.pow(2).mean().backward()
    optimizer.step()
    assert int(torch.linalg.matrix_rank(cell.recurrent_weight().detach())) == 4


def test_synaptic_state_recovers_toward_rest_at_the_configured_rate():
    """With no presynaptic activity, facilitation relaxes to the resting
    utilization and resources recover to 1, each at tick_ms/tau per tick."""
    cell = DynamicSynapseRateCell(
        4, 8, tick_ms=100.0,
        facilitating={"utilization": 0.15, "tau_facilitation_ms": 1000.0, "tau_depression_ms": 200.0},
        depressing={"utilization": 0.45, "tau_facilitation_ms": 200.0, "tau_depression_ms": 1000.0},
    )
    synaptic = cell.init_synaptic_state(2)
    facilitation, resources = synaptic.unbind(dim=1)
    assert torch.allclose(facilitation[:, :4], torch.full((2, 4), 0.15))
    assert torch.allclose(facilitation[:, 4:], torch.full((2, 4), 0.45))
    assert torch.allclose(resources, torch.ones(2, 8))

    # Perturb away from rest, then run with zero activity and zero input.
    synaptic = torch.stack([torch.zeros(2, 8), torch.zeros(2, 8)], dim=1)
    h = torch.zeros(2, 8)
    previous = synaptic
    for _ in range(3):
        _h, _gate, synaptic = cell(torch.zeros(2, 4), h, previous)
        facilitation, resources = synaptic.unbind(dim=1)
        prev_f, prev_x = previous.unbind(dim=1)
        expected_f = prev_f + cell.decay_facilitation * (cell.utilization - prev_f)
        expected_x = prev_x + cell.decay_depression * (1.0 - prev_x)
        assert torch.allclose(facilitation, expected_f, atol=1e-6)
        assert torch.allclose(resources, expected_x, atol=1e-6)
        previous = synaptic
    # Monotone recovery toward rest, and not yet arrived after three ticks.
    assert torch.all(resources < 1.0) and torch.all(resources > 0.0)


def test_synaptic_state_depresses_under_activity_and_is_carried_across_ticks():
    cell = DynamicSynapseRateCell(4, 8, tick_ms=100.0)
    synaptic = cell.init_synaptic_state(2)
    h = torch.ones(2, 8)
    resources_by_tick = []
    for _ in range(5):
        h_out, _gate, synaptic = cell(torch.zeros(2, 4), h, synaptic)
        facilitation, resources = synaptic.unbind(dim=1)
        resources_by_tick.append(resources.clone())
        assert torch.all(facilitation >= cell.utilization), "activity must facilitate, never de-facilitate"
    assert torch.all(resources_by_tick[0] < 1.0), "sustained activity must deplete resources"
    assert torch.all(resources_by_tick[-1] < resources_by_tick[0])

    # The state is genuinely carried: replaying tick 2 from the rest state
    # instead of from tick 1's output gives a different result.
    fresh = cell.init_synaptic_state(2)
    carried_h, _, _ = cell(torch.zeros(2, 4), h, synaptic)
    fresh_h, _, _ = cell(torch.zeros(2, 4), h, fresh)
    assert not torch.allclose(carried_h, fresh_h)


def test_configured_cells_match_the_baseline_effective_synapse_budget():
    target = _baseline_effective_synapses()
    tolerance = CFG["param_budget_tol"]
    for name, cell in _configured_cells().items():
        achieved = cell.effective_param_count()
        assert abs(achieved - target) / target < tolerance, f"{name}: {achieved} vs baseline {target}"


def test_configured_cells_expose_the_flat_core_interface():
    for name, cell in _configured_cells().items():
        assert cell.n_units() == RATE_CFG["units"]
        state = cell.init_state(2)
        assert state.shape == (2, RATE_CFG["units"])
        assert cell.readout_state(state).shape == state.shape


def test_gradients_reach_every_parameter_through_a_multi_tick_unroll():
    for name, cell in _configured_cells().items():
        synaptic = cell.init_synaptic_state(2) if hasattr(cell, "init_synaptic_state") else None
        h = cell.init_state(2)
        x = torch.randn(2, CFG["bottleneck"])
        for _ in range(6):
            if synaptic is None:
                h, _gate = cell(x, h)
            else:
                h, _gate, synaptic = cell(x, h, synaptic)
        h.pow(2).mean().backward()
        for param_name, p in cell.named_parameters():
            assert p.grad is not None, f"{name}.{param_name} received no gradient"
            assert torch.isfinite(p.grad).all(), f"{name}.{param_name} has a non-finite gradient"
            assert p.grad.abs().sum() > 0, f"{name}.{param_name} received an all-zero gradient"


def test_initial_spectral_radius_is_set_on_the_effective_recurrent_matrix():
    for name in ("excitatory_inhibitory", "dynamic_synapse", "low_rank"):
        cfg = {**RATE_CFG, "units": 64, "recurrent_init_spectral_radius": 1.3}
        cell = build_rate_cell(name, CFG["bottleneck"], cfg)
        radius = torch.linalg.eigvals(cell.recurrent_weight().detach()).abs().max().item()
        assert abs(radius - 1.3) < 1e-3, f"{name}: radius {radius}"


def test_tick_longer_than_a_synaptic_time_constant_is_rejected():
    with pytest.raises(ValueError, match="not below every"):
        DynamicSynapseRateCell(4, 8, tick_ms=400.0)
