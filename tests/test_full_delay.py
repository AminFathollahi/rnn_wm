import numpy as np
import pandas as pd
import pytest

from brainalign_wm.analysis.full_delay import FullDelayView, comparison_rows, window_qc_rows, window_rate


class Dataset:
    bin_ms = 50

    def __init__(self):
        self._trials = pd.DataFrame({
            "session": ["s", "s"],
            "load": [1, 2],
            "t_maintain": [0.0, 1.0],
            "t_probe": [0.15, 1.125],
        })
        self.fixed = np.array([[[7.0], [8.0]]])

    def trials(self):
        return self._trials

    def units(self, region=None):
        return ["s#u0"]

    def spike_times(self, unit):
        return np.array([0.0, 0.05, 0.099, 0.1, 0.149, 0.15, 1.0, 1.124, 1.125])

    def rates(self, region, bin_ms, epochs):
        return self.fixed

    def excluded_trials(self):
        return pd.DataFrame({
            "load": [1],
            "t_maintain": [2.0],
            "t_probe": [2.0],
            "exclusion_reason": ["nonpositive_maintenance_duration"],
        })


def test_probe_boundary_is_excluded():
    rate, bins = window_rate(np.array([0.0, 0.05, 0.1, 0.15]), 0.0, 0.15)
    assert rate == pytest.approx(3 / 0.15)
    assert bins == 3


def test_partial_bin_is_duration_weighted():
    rate, bins = window_rate(np.array([1.0, 1.124, 1.125]), 1.0, 1.125)
    assert rate == pytest.approx(2 / 0.125)
    assert bins == 3


def test_malformed_window_has_no_rate_or_bins():
    rate, bins = window_rate(np.array([0.0]), 1.0, 1.0)
    assert np.isnan(rate)
    assert bins == 0


def test_variable_trials_produce_exact_epoch_rates():
    rates = FullDelayView(Dataset()).rates(None, 50, ["maintain"])
    assert rates.shape == (1, 2, 1)
    assert rates[0, 0, 0] == pytest.approx(5 / 0.15)
    assert rates[0, 1, 0] == pytest.approx(2 / 0.125)


def test_fixed_window_path_is_unchanged():
    dataset = Dataset()
    before = dataset.rates(None, 50, ["maintain"]).copy()
    view = FullDelayView(dataset)
    view.rates(None, 50, ["maintain"])
    np.testing.assert_array_equal(dataset.rates(None, 50, ["maintain"]), before)
    np.testing.assert_array_equal(view.rates(None, 50, ["probe"]), before)
    np.testing.assert_array_equal(view.rates(None, 50, ["maintain", "probe"]), before)


def test_unordered_spike_times_give_the_same_rates_as_ordered_ones():
    """A few recorded units store their spike times out of order, and the
    window edges are located by binary search."""

    class Unordered(Dataset):
        def spike_times(self, unit):
            times = super().spike_times(unit)
            return times[[3, 0, 8, 1, 5, 2, 7, 4, 6]]

    ordered = FullDelayView(Dataset()).rates(None, 50, ["maintain"])
    np.testing.assert_allclose(FullDelayView(Unordered()).rates(None, 50, ["maintain"]), ordered)
    assert ordered[0, 0, 0] == pytest.approx(5 / 0.15)


def test_full_delay_rates_are_built_once_then_sliced_by_region():
    class RegionalDataset(Dataset):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def units(self, region=None):
            units = ["s#u0", "s#u1"]
            return units if region is None else [units[0 if region == "A" else 1]]

        def spike_times(self, unit):
            self.calls += 1
            return super().spike_times(unit)

    dataset = RegionalDataset()
    view = FullDelayView(dataset)
    pooled = view.rates(None, 50, ["maintain"])
    region_a = view.rates("A", 50, ["maintain"])
    region_b = view.rates("B", 50, ["maintain"])

    assert dataset.calls == 2
    np.testing.assert_array_equal(region_a, pooled[[0]])
    np.testing.assert_array_equal(region_b, pooled[[1]])


def test_window_qc_reports_duration_bins_and_exclusions():
    rows = window_qc_rows(Dataset(), 50)
    pooled = next(row for row in rows if row["load"] == "all")
    assert pooled["n_trials"] == 2
    assert pooled["n_excluded"] == 1
    assert pooled["excluded_by_reason"] == "nonpositive_maintenance_duration:1"
    assert pooled["duration_s_median"] == pytest.approx(0.1375)
    assert pooled["n_bins_median"] == pytest.approx(3)


def test_comparison_pairs_sessions_and_aggregates_with_existing_estimator(monkeypatch):
    from brainalign_wm.analysis import run_all

    def align(run_id, model_df, dataset, region, subpops):
        full = isinstance(dataset, FullDelayView)
        raw = (0.2, 0.3) if full else (0.1, 0.2)
        ceilings = (0.2, 1.0) if full else (0.5, 1.0)
        return [
            {
                "run_id": run_id,
                "session": session,
                "patient": patient,
                "region": "pooled",
                "subpop": "all",
                "status": "ok",
                "raw_alignment": raw[index],
                "normalized_alignment": raw[index] / ceilings[index],
                "noise_ceiling_lower": 0.4,
                "noise_ceiling_upper": ceilings[index],
                "n_shared_conditions": 4,
            }
            for index, (session, patient) in enumerate((("s0", "p0"), ("s1", "p0")))
        ]

    monkeypatch.setattr(run_all, "_valid_model_trials", lambda model_df, dataset: model_df)
    monkeypatch.setattr(run_all, "_maintenance_alignment_for_run", align)
    model_df = pd.DataFrame({"epoch": ["maintain"], "h_flat": [[0.0]]})
    rows = comparison_rows("run", model_df, Dataset(), None, n_boot=100, seed=0)

    sessions = [row for row in rows if row["analysis_level"] == "session"]
    summary = next(row for row in rows if row["analysis_level"] == "run")
    assert len(sessions) == 2
    assert all(row["raw_difference"] == pytest.approx(0.1) for row in sessions)
    assert summary["fixed_raw_alignment"] == pytest.approx(0.15)
    assert summary["full_delay_raw_alignment"] == pytest.approx(0.25)
    assert summary["raw_difference_ci_lower"] == pytest.approx(0.1)
    assert summary["raw_difference_ci_upper"] == pytest.approx(0.1)
    assert summary["normalized_difference"] == pytest.approx(0.25 / 0.6 - 0.15 / 0.75)
    assert summary["normalized_difference_ci_lower"] == pytest.approx(summary["normalized_difference"])
    assert summary["normalized_difference_ci_upper"] == pytest.approx(summary["normalized_difference"])


def test_hierarchical_comparison_requests_separate_populations(monkeypatch):
    from brainalign_wm.analysis import run_all

    calls = []

    def align(run_id, model_df, dataset, region, subpops):
        calls.append(subpops)
        return []

    monkeypatch.setattr(run_all, "_valid_model_trials", lambda model_df, dataset: model_df)
    monkeypatch.setattr(run_all, "_maintenance_alignment_for_run", align)
    model_df = pd.DataFrame({"epoch": ["maintain"], "h_flat": [None]})
    rows = comparison_rows("run", model_df, Dataset(), None)

    assert calls == [("all", "worker", "manager")] * 2
    assert {row["subpop"] for row in rows} == {"all", "worker", "manager"}
