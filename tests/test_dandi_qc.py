"""Unit tests for the accuracy/firing-rate QC in `dandi_nwb.py`'s
`DandiSternbergTierA`, against synthetic `_SessionData` -- unlike
`test_dandi_adapter.py` these do not need the external drive mounted,
since they exercise pure QC logic rather than the NWB-reading path.

Both thresholds (`MIN_SESSION_ACCURACY`=0.55, `MIN_FIRING_HZ`=0.1) come
from Daume et al. 2024's 001187 STAR Methods -- the only one of the three
Tier A/B source papers with a literature-sourced number -- and are applied
UNIFORMLY across all datasets this adapter loads, not just 001187."""
import numpy as np
import pandas as pd

from brainalign_wm.neural.adapters.dandi_nwb import DandiSternbergTierA, _SessionData


def _make_session(dataset, correct_frac, n_trials=10):
    n_correct = round(correct_frac * n_trials)
    trials = pd.DataFrame({
        "correct": [True] * n_correct + [False] * (n_trials - n_correct),
        "t_start": [float(i) for i in range(n_trials)],
        "t_stop": [float(i) + 1.0 for i in range(n_trials)],
    })
    return _SessionData(
        session_id=f"{dataset}-fake", patient_id="fake",
        trials=trials, unit_ids=["u0"], unit_region={"u0": "hippocampus"},
        spikes={"u0": np.array([0.5])},
    )


def _probe():
    p = DandiSternbergTierA.__new__(DandiSternbergTierA)
    p.min_firing_hz = 0.1
    p.min_session_accuracy = 0.55
    return p


def test_session_accuracy_qc_applies_to_every_dataset():
    probe = _probe()
    for ds in ("000469", "000673", "001187"):
        assert not probe._session_passes_accuracy_qc(_make_session(ds, 0.50))
        assert probe._session_passes_accuracy_qc(_make_session(ds, 0.60))


def test_firing_rate_qc_applies_to_every_dataset():
    probe = _probe()
    # one spike over a 10s session == 0.1 Hz: right at the floor, kept
    for ds in ("000469", "000673", "001187"):
        sess = _make_session(ds, 1.0)
        probe._apply_firing_qc(sess)
        assert sess.unit_ids == ["u0"]

    probe.min_firing_hz = 0.2  # a stricter threshold should now drop the same unit, uniformly
    for ds in ("000469", "000673", "001187"):
        sess = _make_session(ds, 1.0)
        probe._apply_firing_qc(sess)
        assert sess.unit_ids == []
