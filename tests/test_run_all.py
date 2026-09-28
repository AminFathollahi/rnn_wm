"""Alignment-pipeline unit tests: the minimum-trials-per-condition filter
(guards against crossnobis silently zero-filling undefined RDM entries for
rare conditions, which previously produced a spurious raw_alignment=0.0),
and the (session, trial_id) grouping bug found while validating the
probe-epoch pooled path against real Tier-A data."""
import numpy as np
import pandas as pd

import brainalign_wm.analysis.run_all as run_all
from brainalign_wm.analysis.run_all import (
    _filter_min_trials, MIN_TRIALS_PER_CONDITION, _model_epoch_patterns, _coarse_condition,
    _is_ablation_or_catch_variant, _maintenance_alignment_for_run, _probe_alignment_for_run,
    _valid_model_trials,
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


def test_invalid_dataset_trials_are_removed_from_saved_logs():
    class Dataset:
        def trials(self):
            return pd.DataFrame({"session": ["s", "s"], "source_trial_index": [0, 2]})

    df = pd.DataFrame({"session": ["s", "s", "s"], "trial_id": [0, 1, 2]})
    filtered = _valid_model_trials(df, Dataset())
    assert filtered["trial_id"].tolist() == [0, 2]


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


def test_model_epoch_patterns_unit_idx_subsamples_columns():
    """`unit_idx` unit-count-matches a wide population to a narrower one by
    column subsampling, applied after the subpop is assembled -- used to
    contrast a hierarchical run's worker/manager population against a flat
    run's core population at equal width."""
    rows = [
        _fake_hier_log_row("sessA", 0, "maintain", load=1, in_set=True, correct=True, worker_val=1.0, manager_val=100.0),
    ]
    df = pd.DataFrame(rows)
    cond_fn = lambda row: (1, int(row.trial_id))  # noqa: E731

    full, _ = _model_epoch_patterns(df, "maintain", cond_fn, subpop="all")
    assert full.shape == (1, 3)  # 2 worker dims + 1 manager dim, from the fixture

    subset, labels = _model_epoch_patterns(df, "maintain", cond_fn, subpop="all", unit_idx=np.array([2, 0]))
    assert subset.shape == (1, 2)
    np.testing.assert_array_equal(subset, full[:, [2, 0]])
    assert labels == [(1, 0)]


def test_probe_alignment_not_applicable_on_a_flat_log_before_touching_neural_data():
    """A flat run has no worker/manager population to score against the
    probe epoch either. This must resolve to `status: not_applicable`
    entirely from the model side -- `dandi_data=None` and no crash proves
    the neural side is never touched for a request that cannot be
    satisfied."""
    rows = [_fake_log_row("sessA", t, "probe", load=1, in_set=True, correct=True, h_val=float(t)) for t in range(3)]
    df = pd.DataFrame(rows)

    result = _probe_alignment_for_run("run", df, None, None, subpop="worker")
    assert result == {"run_id": "run", "region": "pooled", "subpop": "worker", "status": "not_applicable"}


def test_probe_alignment_too_few_conditions_carries_subpop():
    """The ordinary (not not-applicable) early-return path must still name
    which subpop it evaluated, so a caller building a per-population table
    doesn't have to special-case status rows."""
    rows = [_fake_log_row("sessA", 0, "probe", load=1, in_set=True, correct=True, h_val=1.0)]
    df = pd.DataFrame(rows)

    result = _probe_alignment_for_run("run", df, None, None)
    assert result == {"run_id": "run", "region": "pooled", "subpop": "all", "status": "too_few_conditions"}


def test_chance_control_resolves_log_after_run_id(monkeypatch, tmp_path):
    expected = tmp_path / "M00000_s2_chance.parquet"
    generated = []
    dataset = object()

    def generate(model_id, seed, dataset):
        generated.append((model_id, seed, dataset))
        return expected

    monkeypatch.setattr(run_all, "get_path", lambda name: tmp_path)
    monkeypatch.setattr(
        "brainalign_wm.training.generate_activity_logs.generate_chance_activity_log", generate,
    )
    monkeypatch.setattr("brainalign_wm.training.logging_schema.read_log", lambda path: pd.DataFrame())
    monkeypatch.setattr(run_all, "_maintenance_alignment_for_run", lambda *args, **kwargs: [])
    monkeypatch.setattr(run_all, "_probe_alignment_for_run", lambda *args, **kwargs: {"status": "empty"})

    result = run_all.chance_control_check("M00000", 2, dataset)

    assert generated == [("M00000", 2, dataset)]
    assert result == {"run_id": "M00000_s2_chance", "model_id": "M00000"}


def test_region_loop_reuses_one_model_rdm_per_session_and_unit_subset(monkeypatch):
    """The model side of the maintenance alignment depends on the session
    and the unit subset but not on the brain region, so the same cache
    handed to successive regions must return identical rows while
    rebuilding no model RDM a second time."""
    from tests.test_measurement_validation import _TwoSessionSternbergFixture

    dataset = _TwoSessionSternbergFixture()
    model_df = dataset.model_log()
    options = dict(match_widths=(2,), match_n_draws=3)

    uncached = [_maintenance_alignment_for_run("run", model_df, dataset, region, **options)
                for region in ("A", "B")]

    calls = []
    original = run_all._model_epoch_patterns
    monkeypatch.setattr(run_all, "_model_epoch_patterns",
                        lambda *args, **kwargs: (calls.append(1), original(*args, **kwargs))[1])

    cache: dict = {}
    first = _maintenance_alignment_for_run("run", model_df, dataset, "A", model_rdm_cache=cache, **options)
    after_first = len(calls)
    second = _maintenance_alignment_for_run("run", model_df, dataset, "B", model_rdm_cache=cache, **options)

    assert after_first > 0
    assert len(calls) == after_first
    assert [first, second] == uncached
    assert any(row.get("match_target_units") == 2 for row in first)
