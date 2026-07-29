"""Item 9.2 (comments.txt Phase 9): planted-answer tests for
`scripts/run_distillation_students.py`'s own non-trivial pure-function
logic -- the soft-CE distillation loss and the condition-averaged RDM
construction -- mirroring `tests/test_tiny_rnn_bandit.py`'s house style
(hand-computed small examples, not integration/"does it run" checks)."""
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from scripts.run_distillation_students import _rdm_from_reps, _soft_ce  # noqa: E402


def test_soft_ce_matches_hand_computation():
    # teacher policy = [0.5, 0.5, 0.0] (one-hot-ish); student logits chosen
    # so log_softmax is easy to hand-verify: logits=[0,0,0] -> uniform
    # log-probs = log(1/3) each -> CE = -(0.5*log(1/3) + 0.5*log(1/3) + 0) = -log(1/3)
    teacher_policy = torch.tensor([[0.5, 0.5, 0.0]])
    student_logits = torch.tensor([[0.0, 0.0, 0.0]])
    ce = _soft_ce(teacher_policy, student_logits)
    expected = -np.log(1.0 / 3.0)
    assert torch.allclose(ce, torch.tensor(expected, dtype=torch.float32), atol=1e-5)


def test_soft_ce_is_zero_when_student_matches_teacher_exactly():
    # student logits that produce EXACTLY the teacher's distribution: a
    # one-hot teacher policy with student logits strongly peaked on that
    # class should give a soft-CE close to 0 (the teacher's own entropy,
    # which is 0 for a one-hot distribution).
    teacher_policy = torch.tensor([[1.0, 0.0, 0.0]])
    student_logits = torch.tensor([[100.0, -100.0, -100.0]])  # ~one-hot softmax
    ce = _soft_ce(teacher_policy, student_logits)
    assert ce.item() < 1e-6


def test_soft_ce_batch_is_mean_over_batch():
    # two independent samples with different (teacher, student) pairs;
    # the batch loss must be the plain mean of the two per-sample CEs.
    teacher_policy = torch.tensor([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    student_logits = torch.tensor([[0.0, 0.0, 0.0], [100.0, -100.0, -100.0]])
    ce = _soft_ce(teacher_policy, student_logits)
    expected = (-np.log(1.0 / 3.0) + 0.0) / 2.0
    assert abs(ce.item() - expected) < 1e-5


def test_rdm_from_reps_planted_example():
    # 3 conditions, 2-dim vectors chosen so pairwise correlations are exact:
    # a=[1,2], b=[2,4] (perfectly correlated, corr=1 -> rdm=0),
    # c=[2,-1] (perfectly anti-correlated with a, since [2,-1] is NOT a
    # scalar multiple of [1,2] -- use an exact anti-correlated pair instead).
    reps = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([2.0, 4.0]),          # b = 2*a -> corr(a,b) = 1 -> rdm[a,b] = 0
        "c": np.array([2.0, -1.0]),         # a . c after centering: corr(a,c) = -1 -> rdm[a,c] = 2
    }
    order = ["a", "b", "c"]
    rdm = _rdm_from_reps(reps, order)
    assert rdm.shape == (3, 3)
    assert np.allclose(np.diag(rdm), 0.0)  # diagonal untouched (self-distance not computed, stays 0)
    assert abs(rdm[0, 1] - 0.0) < 1e-6   # a vs b: perfectly correlated -> distance 0
    assert abs(rdm[1, 0] - 0.0) < 1e-6   # symmetric
    assert abs(rdm[0, 2] - 2.0) < 1e-6   # a vs c: perfectly anti-correlated -> distance 2
    assert abs(rdm[2, 0] - 2.0) < 1e-6


def test_rdm_from_reps_condition_order_controls_matrix_layout():
    reps = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}
    rdm_xy = _rdm_from_reps(reps, ["x", "y"])
    rdm_yx = _rdm_from_reps(reps, ["y", "x"])
    assert np.allclose(rdm_xy, rdm_yx.T)
