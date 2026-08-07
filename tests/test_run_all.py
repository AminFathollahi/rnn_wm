"""Alignment-pipeline unit tests: the minimum-trials-per-condition filter
(guards against crossnobis silently zero-filling undefined RDM entries for
rare conditions, which previously produced a spurious raw_alignment=0.0),
and the (session, trial_id) grouping bug found while validating the
probe-epoch pooled path against real Tier-A data."""
import numpy as np
import pandas as pd

from brainalign_wm.analysis.run_all import (
    _filter_min_trials, MIN_TRIALS_PER_CONDITION, _model_epoch_patterns, _coarse_condition,
    _is_ablation_or_catch_variant,
)


def test_is_ablation_or_catch_variant():
    assert not _is_ablation_or_catch_variant({"model_id": "M11111", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1})
    assert not _is_ablation_or_catch_variant({"model_id": "M00000", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0})
    assert _is_ablation_or_catch_variant({"model_id": "M11111_energy", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1})
    assert _is_ablation_or_catch_variant({"model_id": "M11111_noise", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1})
    assert _is_ablation_or_catch_variant({"model_id": "M11111_pbwm", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1})
    assert _is_ablation_or_catch_variant({"model_id": "M00000_idcatch", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0})
    assert _is_ablation_or_catch_variant({"model_id": "M11111_idcatch", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1})


def test_filter_drops_rare_conditions():
    patterns = np.arange(30).reshape(10, 3).astype(float)
    labels = ["a"] * 8 + ["b"] * 1 + ["c"] * 1
    filtered_patterns, filtered_labels = _filter_min_trials(patterns, labels, min_count=4)
    assert set(filtered_labels) == {"a"}
    assert filtered_patterns.shape == (8, 3)


def test_filter_keeps_all_when_all_conditions_common():
    patterns = np.arange(40).reshape(10, 4).astype(float)
    labels = ["a"] * 5 + ["b"] * 5
    filtered_patterns, filtered_labels = _filter_min_trials(patterns, labels, min_count=5)
    assert len(filtered_labels) == 10
    assert filtered_patterns.shape == (10, 4)


def test_default_threshold_is_reasonable():
    assert MIN_TRIALS_PER_CONDITION >= 4  # crossnobis needs multiple folds with multiple trials each


def _fake_log_row(session, trial_id, epoch, load, in_set, correct, h_val):
    return {
        "session": session, "trial_id": trial_id, "epoch": epoch, "load": load,
        "in_set": in_set, "correct": correct, "h_flat": [h_val, h_val], "h_worker": None, "h_manager": None,
    }


def _fake_hier_log_row(session, trial_id, epoch, load, in_set, correct, worker_val, manager_val):
    return {
        "session": session, "trial_id": trial_id, "epoch": epoch, "load": load,
        "in_set": in_set, "correct": correct, "h_flat": None,
        "h_worker": [worker_val, worker_val], "h_manager": [manager_val],
    }


def test_model_epoch_patterns_does_not_merge_trials_across_sessions():
    """`trial_id` is assigned PER SESSION by `generate_activity_logs.
    replay_session` (reset to 0 for every session) and is NOT globally
    unique across a multi-session activity log. Two different sessions'
    trial_id=0 must produce TWO separate condition-pattern rows, not be
    averaged together into one -- the real bug found while validating the
    probe-epoch pooled path (every pooled score was built from averages of
    unrelated trials that happened to share a trial index)."""
    rows = [
        _fake_log_row("sessA", 0, "probe", load=1, in_set=True, correct=True, h_val=10.0),
        _fake_log_row("sessB", 0, "probe", load=1, in_set=True, correct=True, h_val=1000.0),
    ]
    df = pd.DataFrame(rows)
    patterns, labels = _model_epoch_patterns(df, "probe", _coarse_condition)
    assert patterns.shape[0] == 2, "sessA#trial0 and sessB#trial0 must NOT be merged into one row"
    assert set(patterns[:, 0].tolist()) == {10.0, 1000.0}
    assert labels == [(1, True, True), (1, True, True)]


def test_no_dataframe_transpose_attribute_access_for_the_T_column():
    """`ok.T` is DataFrame.transpose, not the T ablation-bit column, so
    attribute access silently wrote the transpose's first row -- a
    stringified run_id Series -- into every headline row's T field, and
    `results/alignment_results.csv` mislabelled one of the five preregistered
    arms. S/M/P/D have no such collision, which is why only T was wrong.
    Found by reading the dry run's CSV (comments.txt §18.4). Column access
    for T must be `["T"]`."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "brainalign_wm" / "analysis" / "run_all.py").read_text()
    assert not re.search(r"\b\w+\.T\.(iloc|values|mean|unique)\b", src)


def test_model_epoch_patterns_subpop_selects_the_right_hidden_units():
    """comments.txt §20.2 (D43): H1's worker<->MTL / manager<->MFC
    dissociation needs the model side restricted to one subpopulation.
    subpop="worker" must return only h_worker, "manager" only h_manager,
    and "all" their concatenation -- exactly what every existing (pooled)
    caller still gets by not passing subpop at all."""
    rows = [
        _fake_hier_log_row("sessA", 0, "maintain", load=1, in_set=True, correct=True, worker_val=1.0, manager_val=100.0),
        _fake_hier_log_row("sessA", 1, "maintain", load=1, in_set=True, correct=True, worker_val=2.0, manager_val=200.0),
    ]
    df = pd.DataFrame(rows)
    cond_fn = lambda row: (1, int(row.trial_id))  # noqa: E731 -- trivial per-trial label for this test only

    worker_patterns, _ = _model_epoch_patterns(df, "maintain", cond_fn, subpop="worker")
    manager_patterns, _ = _model_epoch_patterns(df, "maintain", cond_fn, subpop="manager")
    all_patterns, _ = _model_epoch_patterns(df, "maintain", cond_fn, subpop="all")

    assert worker_patterns.shape == (2, 2)  # h_worker is 2-dim in the fixture
    assert manager_patterns.shape == (2, 1)  # h_manager is 1-dim in the fixture
    assert all_patterns.shape == (2, 3)  # concatenation of both
    np.testing.assert_array_equal(worker_patterns, [[1.0, 1.0], [2.0, 2.0]])
    np.testing.assert_array_equal(manager_patterns, [[100.0], [200.0]])
    np.testing.assert_array_equal(all_patterns, np.concatenate([worker_patterns, manager_patterns], axis=1))


def test_model_epoch_patterns_subpop_not_applicable_on_a_flat_log():
    """A flat run has no worker/manager subpopulation. Requesting one must
    return the same empty sentinel as no data at this epoch -- callers turn
    that into status="not_applicable", never a fabricated 0.0 -- while
    subpop="all" (the default every pre-existing caller uses) is completely
    unaffected, protecting the frozen pooled DV."""
    rows = [_fake_log_row("sessA", 0, "maintain", load=1, in_set=True, correct=True, h_val=5.0)]
    df = pd.DataFrame(rows)
    cond_fn = lambda row: (1, int(row.trial_id))  # noqa: E731

    worker_patterns, worker_labels = _model_epoch_patterns(df, "maintain", cond_fn, subpop="worker")
    manager_patterns, manager_labels = _model_epoch_patterns(df, "maintain", cond_fn, subpop="manager")
    assert worker_patterns.shape == (0, 0) and worker_labels == []
    assert manager_patterns.shape == (0, 0) and manager_labels == []

    # subpop="all" (default, unchanged) still works exactly as before.
    all_patterns_explicit, all_labels_explicit = _model_epoch_patterns(df, "maintain", cond_fn, subpop="all")
    all_patterns_default, all_labels_default = _model_epoch_patterns(df, "maintain", cond_fn)
    np.testing.assert_array_equal(all_patterns_explicit, all_patterns_default)
    assert all_labels_explicit == all_labels_default == [(1, 0)]
    np.testing.assert_array_equal(all_patterns_default, [[5.0, 5.0]])
