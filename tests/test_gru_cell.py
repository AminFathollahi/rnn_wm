"""Item 8.9b (comments.txt §5): correctness tests for the bio-statistics
recurrent-weight initializer, `gru_cell.bioinit_weight_hh`."""
import torch

from brainalign_wm.models.gru_cell import bioinit_weight_hh


def test_bioinit_weight_hh_shape_and_positivity():
    H = 12
    w = bioinit_weight_hh(H, n_gates=3, seed=0)
    assert w.shape == (3 * H, H)
    assert torch.all(w > 0.0)  # LogNormal support


def test_bioinit_weight_hh_each_block_hits_target_spectral_radius():
    H = 12
    w = bioinit_weight_hh(H, n_gates=3, seed=1)
    for g in range(3):
        block = w[g * H:(g + 1) * H]
        radius = torch.linalg.eigvals(block).abs().max().item()
        assert abs(radius - 0.95) < 0.05, (g, radius)


def test_bioinit_weight_hh_deterministic_given_same_seed():
    H = 8
    w1 = bioinit_weight_hh(H, n_gates=3, seed=42)
    w2 = bioinit_weight_hh(H, n_gates=3, seed=42)
    assert torch.equal(w1, w2)


def test_bioinit_weight_hh_differs_across_seeds():
    H = 8
    w1 = bioinit_weight_hh(H, n_gates=3, seed=1)
    w2 = bioinit_weight_hh(H, n_gates=3, seed=2)
    assert not torch.equal(w1, w2)
