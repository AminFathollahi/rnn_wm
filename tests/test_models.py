"""Model tests: mask density matches its target, parameter budget is
matched within tolerance, all S x M x P cell combinations pass a forward
pass, and the hierarchical core's manager-tick semantics are correct."""
import yaml
import pytest

torch = pytest.importorskip("torch")

from brainalign_wm.models.flat_gru import FlatGRUCore
from brainalign_wm.models.front_end import FrontEnd
from brainalign_wm.models.gru_cell import make_locality_mask
from brainalign_wm.models.heads import Heads, N_ACTIONS
from brainalign_wm.models.hrl import HRLCore

ROOT = __file__.rsplit("/tests/", 1)[0]
FULL_CFG = yaml.safe_load(open(f"{ROOT}/configs/config.yaml"))
CFG = FULL_CFG["model"]
BATCH = 4


def test_mask_density_near_target():
    mask = make_locality_mask(tuple(CFG["worker_grid"]), CFG["worker_density"], seed=0)
    achieved = mask.mean().item()
    assert abs(achieved - CFG["worker_density"]) < 0.01, f"achieved density {achieved} far from target {CFG['worker_density']}"
    assert mask.shape == (CFG["worker_units"], CFG["worker_units"])
    assert (mask.diagonal() == 0).all(), "no self-connections expected"


def test_mask_is_symmetric_seedable_not_required_but_deterministic():
    m1 = make_locality_mask(tuple(CFG["worker_grid"]), CFG["worker_density"], seed=42)
    m2 = make_locality_mask(tuple(CFG["worker_grid"]), CFG["worker_density"], seed=42)
    assert torch.equal(m1, m2)


def _build_flat():
    return FlatGRUCore(CFG["bottleneck"], hidden_dim=CFG["flat_units"])


def _build_hrl(reflective: bool):
    return HRLCore(
        input_dim=CFG["bottleneck"],
        worker_units=CFG["worker_units"],
        manager_units=CFG["manager_units"],
        grid=tuple(CFG["worker_grid"]),
        density=CFG["worker_density"],
        manager_period=CFG["manager_period"],
        g_dim=CFG["g_dim"],
        pool_block=CFG["worker_pool_block"],
        reflective=reflective,
    )


def test_param_budget_matched_within_tolerance():
    flat = _build_flat()
    hrl = _build_hrl(reflective=False)
    n_flat = sum(p.numel() for p in flat.parameters())
    n_hrl = sum(p.numel() for p in hrl.parameters())
    rel_diff = abs(n_hrl - n_flat) / n_flat
    tol = CFG["param_budget_tol"]
    assert rel_diff <= tol, f"HRL core {n_hrl} vs flat core {n_flat}: {rel_diff:.3%} > tol {tol:.0%}"


@pytest.mark.parametrize("P", [0, 1])
@pytest.mark.parametrize("M", [0, 1])
@pytest.mark.parametrize("S", [0, 1])
def test_all_cells_forward_and_smoke(S, M, P):
    """All 8 S x M x P cell combinations pass a forward pass with correct
    output shapes (L doesn't change the forward pass shape -- only how
    weights get updated, tested in test_mechanisms.py's node-perturbation
    update-sign test; T/D are training-time loss penalties only, so they
    don't change the forward pass either). Reuses `training/train.py`'s
    own model-construction helpers rather than duplicating them, so this
    test exercises the exact same wiring `train_one` does."""
    from brainalign_wm.training.train import _build_model, _gate_width, _init_state, _step_core

    device = torch.device("cpu")
    front_end, core, heads = _build_model(FULL_CFG, S, M, P, device)
    state = _init_state(core, S, P, BATCH, device)
    gate_width = _gate_width(S, CFG)
    gate_bias = torch.zeros(BATCH, gate_width) if M else None

    v_t = torch.randn(BATCH, CFG["feature_dim"])
    c_t = torch.zeros(BATCH, CFG["task_vec_dim"])
    z_t = front_end(v_t, c_t)
    assert z_t.shape == (BATCH, CFG["bottleneck"])

    h_star, new_state, u_t = _step_core(core, S, M, P, z_t, state, t=0, gate_bias=gate_bias)
    if u_t is not None:
        assert u_t.shape == (BATCH, gate_width)

    policy, value, logits = heads(h_star)
    assert policy.shape == (BATCH, N_ACTIONS)
    assert torch.allclose(policy.sum(-1), torch.ones(BATCH), atol=1e-5)
    assert value.shape == (BATCH,)


def test_hrl_manager_hard_clock_when_not_reflective():
    core = _build_hrl(reflective=False)
    state = core.init_state(BATCH)
    z_t = torch.randn(BATCH, CFG["bottleneck"])
    state1, u1 = core(z_t, state, t=1)  # not a tick (period=5)
    assert torch.equal(state1["h_manager"], state["h_manager"]), "manager should hold state off-tick"
    state5, u5 = core(z_t, state, t=0)  # t=0 IS a tick (0 % period == 0)
    assert not torch.equal(state5["h_manager"], state["h_manager"]), "manager should update on-tick"


def test_hrl_reflective_requires_gate_bias():
    core = _build_hrl(reflective=True)
    state = core.init_state(BATCH)
    z_t = torch.randn(BATCH, CFG["bottleneck"])
    with pytest.raises(ValueError):
        core(z_t, state, t=0, gate_bias=None)


# ---------------- Knob P: Hebbian fast weights (§6.2) ----------------

from brainalign_wm.models.gru_cell import PlasticGRUCell


def test_plastic_gru_cell_hebb_trace_evolves_and_affects_output():
    torch.manual_seed(0)
    hidden = 8
    cell = PlasticGRUCell(4, hidden, eta_decay=0.9, eta_hebb=0.5, hebb_clip=2.0)
    with torch.no_grad():
        cell.alpha.normal_()  # alpha inits to 0 (P=1 == P=0 at init); nonzero to see the trace's effect
    h = torch.randn(2, hidden)  # nonzero presynaptic activity so the Hebb outer product isn't trivially 0
    hebb = cell.init_hebb(2)
    assert hebb.shape == (2, 3 * hidden, hidden)
    x = torch.randn(2, 4)
    h1, u1, hebb1 = cell(x, h, hebb)
    assert not torch.equal(hebb1, hebb), "hebb trace should update from zero after one tick"
    h2, u2, hebb2 = cell(x, h1, hebb1)
    # Same cell, but a nonzero hebb trace now modulates the effective weight
    # -- re-running from a zeroed trace with the same inputs must differ.
    h2_no_trace, _, _ = cell(x, h1, torch.zeros_like(hebb1))
    assert not torch.allclose(h2, h2_no_trace), "nonzero Hebb trace should change the recurrent update"
    assert hebb2.abs().max().item() <= 2.0 + 1e-6, "hebb must stay within hebb_clip"


def test_hrl_core_plastic_worker_only():
    core = _build_hrl(reflective=False)
    core_plastic = HRLCore(
        input_dim=CFG["bottleneck"], worker_units=CFG["worker_units"], manager_units=CFG["manager_units"],
        grid=tuple(CFG["worker_grid"]), density=CFG["worker_density"], manager_period=CFG["manager_period"],
        g_dim=CFG["g_dim"], pool_block=CFG["worker_pool_block"], reflective=False, plastic=True,
    )
    assert isinstance(core_plastic.worker, PlasticGRUCell)
    from brainalign_wm.models.gru_cell import MaskedGRUCell
    assert type(core_plastic.manager) is MaskedGRUCell, "plasticity (§6.2) is worker-only, manager stays plain"

    state = core_plastic.init_state(BATCH)
    assert "hebb_worker" in state
    z_t = torch.randn(BATCH, CFG["bottleneck"])
    state, u_t = core_plastic(z_t, state, t=0)
    assert state["hebb_worker"].shape == (BATCH, 3 * CFG["worker_units"], CFG["worker_units"])


# ---------------- ablation-battery arm M111_pbwm (§4.4/§6.1) ----------------

from brainalign_wm.models.gru_cell import PBWMManagerCell


def test_pbwm_gate_requires_reflective():
    with pytest.raises(ValueError):
        HRLCore(
            input_dim=CFG["bottleneck"], worker_units=CFG["worker_units"], manager_units=CFG["manager_units"],
            grid=tuple(CFG["worker_grid"]), density=CFG["worker_density"], manager_period=CFG["manager_period"],
            g_dim=CFG["g_dim"], pool_block=CFG["worker_pool_block"], reflective=False, pbwm_gate=True,
        )


def test_pbwm_gate_manager_is_lstm_style_with_cell_state():
    core = HRLCore(
        input_dim=CFG["bottleneck"], worker_units=CFG["worker_units"], manager_units=CFG["manager_units"],
        grid=tuple(CFG["worker_grid"]), density=CFG["worker_density"], manager_period=CFG["manager_period"],
        g_dim=CFG["g_dim"], pool_block=CFG["worker_pool_block"], reflective=True, pbwm_gate=True, reflection_beta=1.0,
    )
    assert isinstance(core.manager, PBWMManagerCell)
    state = core.init_state(BATCH)
    assert "c_manager" in state and state["c_manager"].shape == (BATCH, CFG["manager_units"])

    z_t = torch.randn(BATCH, CFG["bottleneck"])
    with pytest.raises(ValueError):
        core(z_t, state, t=0, R_t=None)  # pbwm_gate needs R_t, not gate_bias

    R_t = torch.rand(BATCH, 1)
    new_state, o_t = core(z_t, state, t=0, R_t=R_t)
    assert not torch.equal(new_state["c_manager"], state["c_manager"]), "cell state should evolve"
    assert o_t.shape == (BATCH, CFG["manager_units"])


def test_pbwm_manager_cell_output_gate_not_r_driven():
    """Spec: 'output gate ungated by R_t (reads out normally)' -- only
    input/forget are R-driven."""
    cell = PBWMManagerCell(input_dim=8, hidden_dim=6, beta=5.0)
    x_t = torch.randn(BATCH, 8)
    h_prev = torch.randn(BATCH, 6)
    c_prev = torch.randn(BATCH, 6)
    _, _, o_low = cell(x_t, h_prev, c_prev, R_t=torch.zeros(BATCH, 1))
    _, _, o_high = cell(x_t, h_prev, c_prev, R_t=torch.ones(BATCH, 1) * 10.0)
    assert torch.allclose(o_low, o_high), "output gate must not depend on R_t"
