"""PHASE 1 (comments.txt §5, items 1.1-1.3): effective (mask-aware) synapse
counting on every cell type, the vanilla RNN cell, and the budget solver in
scripts/match_param_budget.py."""
from pathlib import Path

import pytest
import yaml

from brainalign_wm.config import load_config

torch = pytest.importorskip("torch")

from brainalign_wm.models.gru_cell import MaskedGRUCell, PBWMManagerCell, PlasticGRUCell
from brainalign_wm.models.hrl import HRLCore
from brainalign_wm.models.vanilla_rnn import VanillaRNNCell

ROOT = Path(__file__).resolve().parents[1]
FULL_CFG = load_config()
CFG = FULL_CFG["model"]


def test_masked_gru_cell_effective_count_excludes_masked_entries():
    dense = MaskedGRUCell(4, 6, mask=None)
    assert dense.n_units() == 6
    assert dense.effective_param_count() == dense.weight_ih.numel() + dense.weight_hh.numel()

    mask = torch.zeros(6, 6)
    mask[:3, :3] = 1.0  # 9 of 36 entries alive
    sparse = MaskedGRUCell(4, 6, mask=mask)
    assert sparse.effective_param_count() == sparse.weight_ih.numel() + 3 * 9  # x3 gate blocks


def test_plastic_gru_cell_alpha_does_not_inflate_effective_count():
    """alpha modulates an EXISTING synapse (§gru_cell.py docstring), so a
    plastic cell must report the same effective count as a plain masked
    cell with the same shape/mask -- alpha only shows up in the raw
    parameter count (`.parameters()`), not the structural synapse count."""
    mask = torch.zeros(6, 6)
    mask[:3, :3] = 1.0
    plain = MaskedGRUCell(4, 6, mask=mask)
    plastic = PlasticGRUCell(4, 6, mask=mask)
    assert plastic.effective_param_count() == plain.effective_param_count()
    assert sum(p.numel() for p in plastic.parameters()) > sum(p.numel() for p in plain.parameters())


def test_pbwm_manager_cell_effective_count_is_dense():
    cell = PBWMManagerCell(input_dim=5, hidden_dim=7)
    assert cell.n_units() == 7
    assert cell.effective_param_count() == cell.weight_ih.numel() + cell.weight_hh.numel()


def test_hrl_core_effective_count_sums_worker_manager_gproj():
    core = HRLCore(
        input_dim=8, worker_units=9, manager_units=5, grid=(3, 3), density=0.5,
        manager_period=3, g_dim=4, pool_block=1,
    )
    expected = core.worker.effective_param_count() + core.manager.effective_param_count() + core.g_proj.weight.numel()
    assert core.effective_param_count() == expected
    assert core.n_units() == 9 + 5


def test_three_cores_share_one_effective_synapse_budget_and_one_in_degree():
    """The three cores the structure contrast runs over, built the way
    `_build_model` builds them. Matching on effective synapses is what
    licenses the contrast, so the numbers are pinned here rather than being
    recomputed per arm:

      dense flat            128 units, no mask                     73,728
      locality-matched flat 289 units on 17x17, density 0.0681      72,612
      hierarchical          196-unit worker on 14x14 + 24 manager   74,052

    The masked flat core also matches the worker's mean in-degree (19.75
    against 19.69), which is the per-unit quantity that defines the
    connectivity regime; the mask's density target differs from the worker's
    0.10 precisely because the sheet is wider, and at a different width
    in-degree and global density cannot both be held. Unit count is the
    dimension left unmatched: of units, synapses and in-degree only two can
    be held at once, and synapses is the criterion this study matches on."""
    from brainalign_wm.models.gru_cell import make_locality_mask
    from brainalign_wm.training.train import _build_model

    dense_flat = MaskedGRUCell(CFG["bottleneck"], CFG["flat_units"], mask=None)
    assert dense_flat.effective_param_count() == 73_728

    grid, density = (17, 17), 0.0681
    assert grid[0] * grid[1] == 289, "the sheet's cells number the units"
    local_cfg = {**FULL_CFG, "model": {**CFG, "flat_units": 289, "flat_grid": list(grid),
                                       "flat_density": density}}
    _fe, local_flat, _heads = _build_model(local_cfg, S=0, M=0, P=0, device="cpu")
    assert local_flat.n_units() == 289
    assert local_flat.effective_param_count() == 72_612
    # The cell tiles one [H, H] locality mask across the three gate blocks.
    local_mask = make_locality_mask(grid, density, seed=0)
    assert torch.equal(local_flat.cell.mask, local_mask.repeat(3, 1))

    _fe, hierarchical, _heads = _build_model(FULL_CFG, S=1, M=0, P=0, device="cpu")
    assert hierarchical.effective_param_count() == 74_052

    counts = [dense_flat.effective_param_count(), local_flat.effective_param_count(),
              hierarchical.effective_param_count()]
    assert (max(counts) - min(counts)) / min(counts) <= CFG["param_budget_tol"]

    worker_mask = make_locality_mask(tuple(CFG["worker_grid"]), CFG["worker_density"], seed=0)
    assert abs(local_mask.sum(1).mean().item() - worker_mask.sum(1).mean().item()) < 0.5


def test_flat_core_is_densely_recurrent_unless_a_density_is_set():
    """Null density is the setting for every battery cell, so the default
    build must stay bit-identical to the unmasked one."""
    from brainalign_wm.training.train import _build_model

    _fe, core, _heads = _build_model(FULL_CFG, S=0, M=0, P=0, device="cpu")
    assert core.cell.mask is None
    assert core.effective_param_count() == MaskedGRUCell(
        CFG["bottleneck"], CFG["flat_units"], mask=None).effective_param_count()


def test_vanilla_rnn_cell_forward_shape_and_effective_count():
    mask = torch.zeros(6, 6)
    mask[:3, :3] = 1.0  # 9 alive entries
    cell = VanillaRNNCell(4, 6, mask=mask)
    x = torch.randn(2, 4)
    h = torch.randn(2, 6)
    h_t = cell(x, h)
    assert h_t.shape == (2, 6)
    assert torch.all(h_t.abs() <= 1.0), "tanh output must stay in [-1, 1]"
    assert cell.effective_param_count() == cell.weight_ih.numel() + 9
    assert cell.n_units() == 6


def test_flat_grid_product_matches_flat_units():
    """arm T's topo-loss reshape (`train.py::_run_trial`) does
    `.view(-1, gh, gw)` on the S=0 flat core's hidden state -- gh*gw must
    equal `flat_units` or that reshape fails at train time."""
    gh, gw = CFG["flat_grid"]
    assert gh * gw == CFG["flat_units"]


def test_match_param_budget_solve_flat_units_recovers_known_width():
    from scripts.match_param_budget import solve_flat_units

    d_in, gates, h_true = 64, 3, 128
    target = gates * h_true * (h_true + d_in)  # exact synapse count at H=128
    assert solve_flat_units(target, d_in, gates) == h_true


def test_match_param_budget_solve_hierarchical_hits_target_within_5_percent():
    from scripts.match_param_budget import solve_hierarchical

    target = 74000
    result = solve_hierarchical(target, d_in=64, g_dim=32, density=0.10, gates=3)
    assert abs(result["total_synapses"] - target) / target < 0.05
