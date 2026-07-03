"""Model tests: mask density matches its target, parameter budget is
matched within tolerance, all eight cells pass a forward pass, and the
hierarchical core's manager-tick semantics are correct."""
import yaml
import pytest

torch = pytest.importorskip("torch")

from brainalign_wm.models.flat_gru import FlatGRUCore
from brainalign_wm.models.front_end import FrontEnd
from brainalign_wm.models.gru_cell import make_locality_mask
from brainalign_wm.models.heads import Heads, N_ACTIONS
from brainalign_wm.models.hrl import HRLCore

ROOT = __file__.rsplit("/tests/", 1)[0]
CFG = yaml.safe_load(open(f"{ROOT}/configs/config.yaml"))["model"]
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


@pytest.mark.parametrize("S", [0, 1])
@pytest.mark.parametrize("M", [0, 1])
def test_all_cells_forward_and_smoke(S, M):
    """Cheap proxy for 'all 8 cells pass forward + smoke' (L doesn't change
    the forward pass shape -- only how weights get updated, tested in
    test_mechanisms.py's node-perturbation update-sign test)."""
    front = FrontEnd(
        feature_dim=CFG["feature_dim"], task_vec_dim=CFG["task_vec_dim"],
        bottleneck_dim=CFG["bottleneck"], input_noise_sigma=CFG["input_noise_sigma"],
    )
    v_t = torch.randn(BATCH, CFG["feature_dim"])
    c_t = torch.zeros(BATCH, CFG["task_vec_dim"])
    z_t = front(v_t, c_t)
    assert z_t.shape == (BATCH, CFG["bottleneck"])

    if S == 0:
        core = _build_flat()
        h_prev = core.init_state(BATCH)  # FlatGRUCore: state is a plain tensor
        h_t = core(z_t, h_prev)
        h_star = core.readout_state(h_t)
    else:
        core = _build_hrl(reflective=bool(M))
        state = core.init_state(BATCH)
        gate_bias = torch.zeros(BATCH, CFG["manager_units"]) if M else None
        state, u_t = core(z_t, state, t=0, gate_bias=gate_bias)
        h_star = core.readout_state(state)
        assert u_t.shape == (BATCH, CFG["manager_units"])

    heads = Heads(h_star.shape[-1], readout_scale=CFG["readout_scale"])
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
