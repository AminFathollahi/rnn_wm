"""Alignment-pipeline unit tests: the minimum-trials-per-condition filter
(guards against crossnobis silently zero-filling undefined RDM entries for
rare conditions, which previously produced a spurious raw_alignment=0.0)."""
import numpy as np

from brainalign_wm.analysis.run_all import _filter_min_trials, MIN_TRIALS_PER_CONDITION


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
