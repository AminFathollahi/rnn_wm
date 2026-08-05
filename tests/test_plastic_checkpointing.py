"""Gradient-equivalence test for segment-checkpointing the plastic
recurrent loop (`brainalign_wm.training.train._segment_checkpoint_scan`).

`PlasticGRUCell` (brainalign_wm/models/gru_cell.py) retains three
[B, 3H, H] tensors per tick that autograd keeps alive until backward, so
its graph grows linearly with trial length -- large enough at the longest
curriculum load to exceed the whole GPU for the S=1 plastic cells. The
fix is to run the per-tick recurrence through `torch.utils.checkpoint` in
segments, so only segment-boundary state is retained and each segment's
interior is recomputed during backward.

`torch.utils.checkpoint` recomputes every op in backward -- it never skips
one -- so the result must be the EXACT full-BPTT gradient, not an
approximation: this is not truncated BPTT, not a shortened unroll, and
must not become either. This test is the acceptance criterion for that
claim: a checkpointed rollout and a plain (uncheckpointed) rollout of an
identical cell, from identical weights, identical seed, and identical
inputs, must produce identical forward outputs and identical parameter
gradients. If they do not match, the checkpointing change is not exact
and must not ship.

CPU-only by construction (no device is ever requested), so it is safe to
run alongside a live GPU training job.
"""
import copy

import torch

from brainalign_wm.models.gru_cell import PlasticGRUCell
from brainalign_wm.training.train import _segment_checkpoint_scan


def _rollout_plain(cell: PlasticGRUCell, inputs: list, h0: torch.Tensor, hebb0: torch.Tensor) -> torch.Tensor:
    """Direct per-tick calls, standard autograd, no checkpointing at all --
    the reference this test holds checkpointing to."""
    h, hebb = h0, hebb0
    outs = []
    for x_t in inputs:
        h, _u_t, hebb = cell(x_t, h, hebb)
        outs.append(h)
    return torch.stack(outs)


def _rollout_checkpointed(
    cell: PlasticGRUCell, inputs: list, h0: torch.Tensor, hebb0: torch.Tensor, segment_len: int,
) -> torch.Tensor:
    def step_fn(state, t):
        h, hebb = state
        h_t, _u_t, hebb_t = cell(inputs[t], h, hebb)
        return (h_t, hebb_t), h_t

    _final_state, outs = _segment_checkpoint_scan(step_fn, (h0, hebb0), len(inputs), segment_len)
    return torch.stack(outs)


def _matched_cells(seed: int, input_dim: int, hidden_dim: int) -> tuple[PlasticGRUCell, PlasticGRUCell]:
    """Two `PlasticGRUCell`s with identical weights (a deepcopy, not two
    independent seeded inits) so their gradients are directly comparable."""
    torch.manual_seed(seed)
    plain = PlasticGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)
    ckpt = copy.deepcopy(plain)
    return plain, ckpt


def test_checkpointed_rollout_matches_plain_forward_and_gradients():
    B, input_dim, H, T = 3, 5, 6, 12  # small and short, per the brief -- this must run fast
    torch.manual_seed(0)
    inputs = [torch.randn(B, input_dim) for _ in range(T)]

    cell_plain, cell_ckpt = _matched_cells(seed=123, input_dim=input_dim, hidden_dim=H)
    h0 = torch.zeros(B, H)
    hebb0 = cell_plain.init_hebb(B)

    out_plain = _rollout_plain(cell_plain, inputs, h0, hebb0)
    loss_plain = out_plain.pow(2).mean()
    loss_plain.backward()

    # segment_len=4 for T=12 -> 3 segments, i.e. genuine multi-tick
    # segmenting (segment_len=1 would retain every tick's state exactly
    # like no checkpointing at all -- see _segment_checkpoint_scan's
    # docstring -- so this must be > 1 to actually exercise the fix).
    out_ckpt = _rollout_checkpointed(cell_ckpt, inputs, h0.clone(), hebb0.clone(), segment_len=4)
    loss_ckpt = out_ckpt.pow(2).mean()
    loss_ckpt.backward()

    torch.testing.assert_close(out_plain, out_ckpt, rtol=1e-5, atol=1e-6)

    plain_params = dict(cell_plain.named_parameters())
    ckpt_params = dict(cell_ckpt.named_parameters())
    assert plain_params.keys() == ckpt_params.keys()
    for name, p_plain in plain_params.items():
        p_ckpt = ckpt_params[name]
        assert p_plain.grad is not None, f"{name}: plain rollout produced no gradient"
        assert p_ckpt.grad is not None, f"{name}: checkpointed rollout produced no gradient"
        torch.testing.assert_close(p_plain.grad, p_ckpt.grad, rtol=1e-5, atol=1e-6, msg=lambda m, n=name: f"{n}: {m}")


def test_checkpointed_rollout_matches_across_segment_sizes():
    """Not just one arbitrary segment_len -- a single-segment (no real
    checkpointing benefit, segment_len=T) and a maximally-segmented
    (segment_len=1, every tick its own checkpoint) rollout must both agree
    with the plain reference too."""
    B, input_dim, H, T = 2, 4, 5, 9
    torch.manual_seed(1)
    inputs = [torch.randn(B, input_dim) for _ in range(T)]

    cell_plain, cell_ckpt = _matched_cells(seed=7, input_dim=input_dim, hidden_dim=H)
    h0 = torch.zeros(B, H)
    hebb0 = cell_plain.init_hebb(B)

    out_plain = _rollout_plain(cell_plain, inputs, h0, hebb0)
    out_plain.pow(2).mean().backward()

    for segment_len in (1, T):
        cell_ckpt.zero_grad()
        out_ckpt = _rollout_checkpointed(cell_ckpt, inputs, h0.clone(), hebb0.clone(), segment_len=segment_len)
        torch.testing.assert_close(out_plain, out_ckpt, rtol=1e-5, atol=1e-6)
        out_ckpt.pow(2).mean().backward()
        for name, p_plain in cell_plain.named_parameters():
            p_ckpt = dict(cell_ckpt.named_parameters())[name]
            torch.testing.assert_close(
                p_plain.grad, p_ckpt.grad, rtol=1e-5, atol=1e-6, msg=lambda m, n=name, s=segment_len: f"segment_len={s} {n}: {m}"
            )


if __name__ == "__main__":
    test_checkpointed_rollout_matches_plain_forward_and_gradients()
    test_checkpointed_rollout_matches_across_segment_sizes()
    print("OK")
