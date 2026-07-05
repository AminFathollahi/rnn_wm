"""Per-session RSA machinery tests (audit fixes A2b, A2e, A2f):
- `_session_condition_rdm` drops rare (few-trial) conditions instead of
  letting a single singleton condition zero out fold count for the whole
  session.
- `noise_ceiling_from_dataset` aligns sessions by condition IDENTITY, not
  by incidentally-matching RDM shape -- two sessions with same-size but
  DIFFERENT condition sets must not be spuriously correlated.
- `within_session_noise_ceiling` gives a sane, non-degenerate reliability
  estimate for a single session's own resampled RDMs.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd

from brainalign_wm.analysis.rsa import _session_condition_rdm, noise_ceiling_from_dataset, within_session_noise_ceiling


class _RareConditionFixture:
    """One session; 14 trials over conditions labeled 0..6, where label 6
    has only a single trial (mirrors the real-data pattern that broke
    per-session maintenance alignment: most conditions well-populated, one
    or two singletons)."""

    bin_ms = 50

    def __init__(self):
        labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]  # label 6: 1 trial
        n = len(labels)
        self._trials = pd.DataFrame({
            "session": ["S0"] * n, "cond": labels, "trial_id": list(range(n)),
        })
        rng = np.random.RandomState(0)
        self._data = rng.randn(n, 8) + np.array(labels)[:, None] * 0.3

    def units(self, region=None):
        return [f"S0#u{i}" for i in range(8)]

    def trials(self):
        return self._trials

    def sessions(self):
        return ["S0"]

    def rates(self, region, bin_ms, epochs):
        n_units = len(self.units(region))
        n_trials = len(self._trials)
        return self._data.T.reshape(n_units, n_trials, 1)


def _cond_fn(row) -> tuple:
    return (row.cond,)


def test_session_rdm_survives_a_singleton_condition():
    ds = _RareConditionFixture()
    rdm, conds = _session_condition_rdm(ds, "S0", None, "maintain", ds.bin_ms, n_folds=2, condition_fn=_cond_fn)
    assert rdm is not None, "one rare (1-trial) condition should be dropped, not zero out the whole session"
    assert (6,) not in conds
    assert len(conds) == 6  # labels 0..5 kept, label 6 dropped


def test_within_session_noise_ceiling_sane_bounds():
    ds = _RareConditionFixture()
    lower, upper = within_session_noise_ceiling(ds, "S0", None, "maintain", ds.bin_ms, condition_fn=_cond_fn, n_resamples=10)
    assert 0.0 <= lower <= upper <= 1.0 + 1e-6
    assert upper > 0.0


class _DisjointConditionsFixture:
    """Two sessions, same RDM SHAPE (3x3) but COMPLETELY DIFFERENT condition
    identities -- the shape-matching bug (A2e) would have compared these as
    if aligned; the fix must recognize there is no shared condition and
    decline to report a ceiling rather than silently correlate unaligned
    RDMs."""

    bin_ms = 50

    def __init__(self):
        # session A: conditions "a","b","c"; session B: conditions "x","y","z"
        rows = []
        rng = np.random.RandomState(0)
        for sess, conds in (("A", ["a", "b", "c"]), ("B", ["x", "y", "z"])):
            for c in conds:
                for _ in range(4):
                    rows.append({"session": sess, "cond": c})
        for i, r in enumerate(rows):
            r["trial_id"] = i
        self._trials = pd.DataFrame(rows)
        n = len(self._trials)
        cond_to_offset = {c: i for i, c in enumerate(sorted(self._trials.cond.unique()))}
        offsets = self._trials.cond.map(cond_to_offset).values
        self._data = rng.randn(n, 6) + offsets[:, None] * 0.5

    def units(self, region=None):
        return [f"A#u{i}" for i in range(3)] + [f"B#u{i}" for i in range(3)]

    def trials(self):
        return self._trials

    def sessions(self):
        return ["A", "B"]

    def rates(self, region, bin_ms, epochs):
        n_units = len(self.units(region))
        n_trials = len(self._trials)
        return self._data.T.reshape(n_units, n_trials, 1)


def _cond_fn2(row) -> tuple:
    return (row.cond,)


def test_noise_ceiling_declines_when_sessions_share_no_conditions():
    ds = _DisjointConditionsFixture()
    lower, upper = noise_ceiling_from_dataset(ds, None, "maintain", condition_fn=_cond_fn2)
    assert (lower, upper) == (0.0, 0.0), (
        "sessions with same RDM shape but disjoint condition identities must not be "
        "spuriously correlated as if aligned (A2e)"
    )
