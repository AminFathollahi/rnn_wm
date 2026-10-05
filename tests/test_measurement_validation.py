"""Checks on the machinery that says what the alignment DVs measure:
the per-session stratified estimator must be reproduced exactly when it
is generalized over epochs, the session split must never cut within a
session, the synthetic geometry mixture must actually vary the true
alignment it claims to, and the patient-clustered interval must resample
whole patients."""
from __future__ import annotations

import numpy as np
import pandas as pd

from brainalign_wm.analysis.measurement_validation import (
    SessionSubsetView,
    geometry_mixture_generator,
    patient_cluster_bootstrap_ci,
    split_sessions,
    stratified_session_alignment,
)
from brainalign_wm.analysis.task_structure import (
    out_of_sample_semipartial,
    semipartial_correlation,
)
from brainalign_wm.analysis.run_all import _maintenance_alignment_for_run

N_UNITS_PER_SESSION = 6
N_BINS = 2


class _TwoSessionSternbergFixture:
    """Two sessions, each with two load strata x four held-item sets x four
    trials. Enough repeat structure that the maintenance path selects the
    real-identity condition schema, and enough conditions per stratum to
    clear the shared-condition and valid-pair floors."""

    bin_ms = 50

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        rows, self._session_unit_data = [], {}
        trial_id = 0
        for session in ("S0", "S1"):
            for load in (1, 2):
                for item_set in range(4):
                    held = tuple(100 * item_set + k for k in range(load))
                    for _ in range(4):
                        rows.append({"session": session, "trial_id": trial_id, "load": load,
                                     "held_items": held, "in_set": True, "correct": True})
                        trial_id += 1
        self._trials = pd.DataFrame(rows)
        signal = np.array([hash(r.held_items) % 7 for r in self._trials.itertuples()], dtype=float)
        load_arr = self._trials.load.values.astype(float)
        self._rates = np.zeros((2 * N_UNITS_PER_SESSION, len(self._trials), N_BINS))
        for si, session in enumerate(("S0", "S1")):
            mask = (self._trials.session.values == session)
            block = (signal[mask][:, None] * 0.6 + load_arr[mask][:, None] * 2.0
                     + rng.randn(mask.sum(), N_UNITS_PER_SESSION) * 0.4)
            for ui in range(N_UNITS_PER_SESSION):
                self._rates[si * N_UNITS_PER_SESSION + ui, mask, :] = block[:, ui][:, None]
        self._signal = signal

    def units(self, region=None):
        return [f"S{s}#u{i}" for s in (0, 1) for i in range(N_UNITS_PER_SESSION)]

    def trials(self):
        return self._trials

    def sessions(self):
        return ["S0", "S1"]

    def patient_of(self, session):
        return {"S0": "P0", "S1": "P1"}[session]

    def rates(self, region, bin_ms, epochs):
        return self._rates

    def model_log(self, seed: int = 1) -> pd.DataFrame:
        """An activity log whose maintenance activity shares the neural
        held-item signal, so the alignment is clearly non-zero."""
        rng = np.random.RandomState(seed)
        rows = []
        for i, trial in enumerate(self._trials.itertuples()):
            vector = (self._signal[i] * 0.5 + rng.randn(4) * 0.3).tolist()
            for epoch in ("maintain", "probe"):
                rows.append({
                    "session": trial.session, "trial_id": trial.trial_id, "epoch": epoch,
                    "load": trial.load, "held_items": trial.held_items, "in_set": trial.in_set,
                    "correct": trial.correct, "h_flat": vector, "h_worker": None, "h_manager": None,
                })
        return pd.DataFrame(rows)


def test_generalized_estimator_reproduces_the_reported_maintenance_values():
    dataset = _TwoSessionSternbergFixture()
    model_df = dataset.model_log()
    reported = [r for r in _maintenance_alignment_for_run("run", model_df, dataset, None)
                if r.get("status") == "ok"]
    generalized = stratified_session_alignment(model_df, dataset, None, epoch="maintain")
    assert len(reported) == len(generalized) > 0
    for a, b in zip(sorted(reported, key=lambda r: r["session"]),
                    sorted(generalized, key=lambda r: r["session"])):
        assert a["session"] == b["session"]
        assert a["raw_alignment"] == b["raw_alignment"]
        assert a["n_shared_conditions"] == b["n_shared_conditions"]


def test_load_selection_reaches_alignment_and_ceiling(monkeypatch):
    from brainalign_wm.analysis import rsa, run_all

    dataset = _TwoSessionSternbergFixture()
    seen_raw = []
    seen_ceiling = []
    shared = run_all._shared_conditions_for_load
    ceiling = rsa.within_session_noise_ceiling

    def shared_spy(model_conditions, neural_conditions, load_value, floor):
        seen_raw.append(load_value)
        return shared(model_conditions, neural_conditions, load_value, floor)

    def ceiling_spy(*args, **kwargs):
        condition_filter = kwargs.get("condition_filter")
        seen_ceiling.append((condition_filter((1, "a")), condition_filter((2, "a"))))
        return ceiling(*args, **kwargs)

    monkeypatch.setattr(run_all, "_shared_conditions_for_load", shared_spy)
    monkeypatch.setattr(rsa, "within_session_noise_ceiling", ceiling_spy)

    rows = _maintenance_alignment_for_run(
        "run", dataset.model_log(), dataset, None, load_value=1,
    )

    assert rows
    assert seen_raw and set(seen_raw) == {1}
    assert seen_ceiling and set(seen_ceiling) == {(True, False)}


def test_the_same_estimator_also_runs_at_the_probe_epoch():
    dataset = _TwoSessionSternbergFixture()
    rows = stratified_session_alignment(dataset.model_log(), dataset, None, epoch="probe")
    assert len(rows) > 0
    assert all(np.isfinite(r["raw_alignment"]) for r in rows)


def test_session_split_never_cuts_within_a_session():
    half_a, half_b = split_sessions(["c", "a", "d", "b"])
    assert set(half_a) | set(half_b) == {"a", "b", "c", "d"}
    assert not set(half_a) & set(half_b)


def test_session_subset_view_restricts_trials_and_rates_together():
    dataset = _TwoSessionSternbergFixture()
    view = SessionSubsetView(dataset, ["S0"])
    assert set(view.trials().session) == {"S0"}
    assert view.rates(None, view.bin_ms, ["maintain"]).shape[1] == len(view.trials())


def test_geometry_mixture_spans_a_range_of_true_alignments():
    dataset = _TwoSessionSternbergFixture()
    from brainalign_wm.analysis import rsa as rsa_module
    from brainalign_wm.analysis.run_all import _maintenance_condition_fn_stratified

    trials = dataset.trials()
    condition_fn = _maintenance_condition_fn_stratified(trials[trials.session == "S0"])
    neural_data, session_trials = rsa_module._session_trial_patterns(dataset, "S0", None, "maintain", dataset.bin_ms)
    labels = [condition_fn(row) for row in session_trials.itertuples()]

    true_values = []
    for weight in (0.0, 0.5, 1.0):
        generator = geometry_mixture_generator(weight)
        patterns, returned = generator(neural_data, labels, "S0")
        assert patterns.shape == neural_data.shape and returned == labels
        true_values.append(generator.true_by_session["S0"])
    assert true_values[2] > 0.99  # weight 1 is the real geometry itself
    assert true_values[0] < true_values[1] < true_values[2]


def test_patient_bootstrap_resamples_whole_patients():
    """Two patients whose sessions disagree in sign: an interval that
    resampled sessions independently would be far narrower than one that
    can draw the same patient twice."""
    values = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    patients = np.array(["A", "A", "A", "B", "B", "B"])
    low, high = patient_cluster_bootstrap_ci(values, patients, n_boot=500, seed=0)
    assert low <= -0.9 and high >= 0.9


def test_patient_bootstrap_ignores_missing_values():
    values = np.array([0.2, np.nan, 0.4])
    patients = np.array(["A", "A", "B"])
    low, high = patient_cluster_bootstrap_ci(values, patients, n_boot=200, seed=0)
    assert np.isfinite(low) and np.isfinite(high) and low <= 0.3 <= high


def _triple(n_pairs: int, model_carries_shared: bool, seed: int):
    """(neural, model, task) RDM vectors where the task vector drives both
    sides, plus a second signal the model carries only when asked to."""
    rng = np.random.RandomState(seed)
    task = rng.rand(n_pairs)
    shared = rng.rand(n_pairs)
    neural = task + 0.6 * shared + 0.05 * rng.randn(n_pairs)
    model = task + (0.6 * shared if model_carries_shared else 0.0) + 0.05 * rng.randn(n_pairs)
    return neural, model, task


def test_out_of_sample_semipartial_matches_the_in_sample_one_on_its_own_data():
    triple = _triple(400, True, seed=0)
    assert out_of_sample_semipartial(triple, triple) == semipartial_correlation(*triple)


def test_task_removal_coefficients_transfer_to_a_shorter_vector():
    """The fit and evaluation halves have different pair counts, which is
    what the session split produces; a task-only model must fall to near
    zero there while a model carrying non-task structure must survive."""
    task_only = out_of_sample_semipartial(_triple(400, False, seed=1), _triple(150, False, seed=2))
    beyond_task = out_of_sample_semipartial(_triple(400, True, seed=1), _triple(150, True, seed=2))
    assert abs(task_only) < 0.2
    assert beyond_task > 0.8


def test_semipartial_ignores_pairs_left_undefined_by_load_stratification():
    neural, model, task = _triple(200, True, seed=3)
    padded = [np.concatenate([v, np.full(50, np.nan)]) for v in (neural, model, task)]
    assert out_of_sample_semipartial(tuple(padded), tuple(padded)) == semipartial_correlation(neural, model, task)
