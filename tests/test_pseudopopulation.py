"""Cross-session pseudopopulation tests (audit fix A2d): a valid pooled
condition-level pseudopopulation must never average a unit over trials it
did not itself record. A minimal synthetic `NeuralDataset`-like fixture with
two non-overlapping sessions makes the structural-zero bug reproducible
without needing the real (large, externally-mounted) Tier A data."""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from brainalign_wm.analysis.pseudopopulation import build_condition_fold_means, cv_euclidean_rdm, pooled_condition_rdm

Row = namedtuple("Row", ["load", "probe_in_set", "correct", "session"])


class _TwoSessionFixture:
    """2 sessions, 2 units each (4 total), 1 condition (load=1, in_set=True,
    correct=True) shared by both sessions. Session A's units fire at rate
    10 Hz for this condition; session B's units fire at rate 100 Hz --
    wildly different scales, chosen so that any cross-session zero-dilution
    bug produces an obviously wrong (much smaller) condition mean than the
    session's own true rate."""

    bin_ms = 50

    def __init__(self):
        self._trials = pd.DataFrame(
            {
                "session": ["A"] * 6 + ["B"] * 6,
                "load": [1] * 12,
                "probe_in_set": [True] * 12,
                "correct": [True] * 12,
                "trial_id": list(range(12)),
            }
        )
        self._rates = {}  # (uid, trial_id) -> scalar rate
        for tid in range(6):
            self._rates[("A#u0", tid)] = 10.0
            self._rates[("A#u1", tid)] = 12.0
        for tid in range(6, 12):
            self._rates[("B#u0", tid)] = 100.0
            self._rates[("B#u1", tid)] = 102.0

    def units(self, region=None):
        return ["A#u0", "A#u1", "B#u0", "B#u1"]

    def trials(self):
        return self._trials

    def rates(self, region, bin_ms, epochs):
        units = self.units(region)
        trials = self.trials()
        out = np.zeros((len(units), len(trials), 1))
        for ui, uid in enumerate(units):
            for ti, tid in enumerate(trials.trial_id):
                if (uid, tid) in self._rates:
                    out[ui, ti, 0] = self._rates[(uid, tid)]
        return out


def _cond_fn(row) -> tuple:
    return (row.load, row.probe_in_set, row.correct)


def test_pooled_pseudopopulation_no_cross_session_dilution():
    ds = _TwoSessionFixture()
    fold_means, conds = build_condition_fold_means(ds, None, ds.bin_ms, "maintain", _cond_fn, n_folds=2, seed=0)
    assert len(conds) == 1
    # session A's units should reflect ONLY session A's own rate (~10-12),
    # never diluted toward 0 by session B's trials (or vice versa).
    unit_means = np.nanmean(fold_means[:, 0, :], axis=0)  # [n_units], averaged over folds
    assert unit_means[0] == pytest.approx(10.0, abs=1e-6)
    assert unit_means[1] == pytest.approx(12.0, abs=1e-6)
    assert unit_means[2] == pytest.approx(100.0, abs=1e-6)
    assert unit_means[3] == pytest.approx(102.0, abs=1e-6)


def test_pooled_condition_rdm_runs_without_crashing_on_single_condition():
    ds = _TwoSessionFixture()
    rdm, conds = pooled_condition_rdm(ds, None, ds.bin_ms, "maintain", _cond_fn, n_folds=2, seed=0)
    assert rdm.shape == (1, 1)
    assert rdm[0, 0] == 0.0  # diagonal always zeroed


def test_cv_euclidean_rdm_handles_nan_gaps():
    """A unit with NO data for one condition (NaN) must not corrupt distances
    between OTHER condition pairs it does have data for."""
    fold_means = np.array(
        [
            [[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]],
            [[1.1, 2.1], [3.1, np.nan], [5.1, 6.1]],
        ]
    )  # [n_folds=2, n_cond=3, n_units=2]
    rdm = cv_euclidean_rdm(fold_means, conds=["a", "b", "c"])
    assert rdm.shape == (3, 3)
    assert np.isfinite(rdm).all()
    assert rdm[0, 2] != 0.0  # a<->c: both units have data, should be nonzero
