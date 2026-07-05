"""Alignment-pipeline unit tests: the minimum-trials-per-condition filter
(guards against crossnobis silently zero-filling undefined RDM entries for
rare conditions, which previously produced a spurious raw_alignment=0.0),
and the (session, trial_id) grouping bug found while validating the
probe-epoch pooled path against real Tier-A data."""
import numpy as np
import pandas as pd

from brainalign_wm.analysis.run_all import _filter_min_trials, MIN_TRIALS_PER_CONDITION, _model_epoch_patterns, _coarse_condition


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
