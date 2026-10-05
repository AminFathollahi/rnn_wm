from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from brainalign_wm.analysis import alignment_distribution as dist
from brainalign_wm.analysis.run_all import _shared_conditions_for_load


def _row(session, patient, load, field, raw, status="ok"):
    row = {
        "run_id": "run",
        "session": session,
        "patient": patient,
        "region": "pooled",
        "subpop": "all",
        "status": status,
        "load": load,
        "condition_field": field,
        "condition_schema": f"load_stratified_{field}",
        "n_trials": 20,
        "n_model_trials": 20,
        "n_condition_levels": 6,
    }
    if status == "ok":
        row.update(
            raw_alignment=raw,
            normalized_alignment=max(0.0, raw / 0.5),
            noise_ceiling_upper=0.5,
            n_shared_conditions=5,
        )
    return row


def test_distribution_reports_spread_counts_and_effect_direction():
    rows = [
        _row("s1", "p1", "all", "identity", -0.3),
        _row("s2", "p2", "all", "category", -0.1),
        _row("s1", "p1", 1, "identity", -0.4),
        _row("s2", "p2", 1, "category", -0.2),
        _row("s1", "p1", 2, "identity", 0.1),
        _row("s2", "p2", 2, "category", 0.3),
        _row("s1", "p1", 3, "identity", 0.0, status="insufficient_shared_conditions"),
    ]

    out = pd.DataFrame(dist.summarize_distribution(rows, n_boot=100, seed=3))
    overall = out[out["crossing"] == "overall"].iloc[0]
    by_load = out[out["crossing"] == "load"].set_index("load")

    assert overall.raw_alignment == pytest.approx(-0.2)
    assert overall.raw_std == pytest.approx(np.std([-0.3, -0.1], ddof=1))
    assert overall.session_normalized_mean == 0.0
    assert overall.normalized_alignment == 0.0
    assert overall.n_sessions_ok == 2
    assert overall.n_patients_ok == 2
    assert overall.n_trials == 40
    assert overall.n_condition_levels_sum == 12
    assert by_load.loc[1, "relation_to_pooled"] == "carries"
    assert by_load.loc[2, "relation_to_pooled"] == "opposes"
    assert by_load.loc[3, "status"] == "no_usable_sessions"
    assert by_load.loc[3, "n_sessions_attempted"] == 1
    assert {"load+condition_field+condition_schema+patient", "condition_field+condition_schema"} <= set(out.crossing)


def test_session_rows_reuse_headline_estimator_for_each_load(monkeypatch):
    calls = []

    def align(run_id, model_df, dataset, region, subpops, load_value):
        calls.append(load_value)
        return [{
            "run_id": run_id,
            "session": "s1",
            "region": "pooled",
            "subpop": "all",
            "status": "ok",
            "raw_alignment": 0.2,
            "noise_ceiling_upper": 0.5,
            "n_shared_conditions": 4,
        }]

    class Dataset:
        def trials(self):
            return pd.DataFrame({
                "session": ["s1"] * 8,
                "load": [1] * 4 + [2] * 4,
                "held_items": [[1], [1], [2], [2], [3], [3], [4], [4]],
            })

        def patient_of(self, session):
            return "p1"

    model = pd.DataFrame({
        "session": ["s1"] * 8,
        "epoch": ["maintain"] * 8,
        "load": [1] * 4 + [2] * 4,
        "trial_id": range(8),
        "h_flat": [np.ones(2)] * 8,
    })
    monkeypatch.setattr(dist, "_valid_model_trials", lambda frame, dataset: frame)
    monkeypatch.setattr(dist, "_maintenance_alignment_for_run", align)

    rows = dist.session_distribution_rows("run", model, Dataset(), None)

    assert calls == [None, 1, 2]
    assert [row["load"] for row in rows] == ["all", 1, 2]
    assert rows[0]["condition_field"] == "held_items"
    assert rows[0]["condition_schema"] == "load_stratified_identity"
    assert rows[1]["n_trials"] == 4
    assert rows[1]["n_model_trials"] == 4


def test_shared_condition_load_filter_preserves_headline_default():
    model = [(1, "a"), (1, "b"), (2, "c"), (2, "d")]
    neural = [(1, "a"), (1, "b"), (2, "d"), (2, "e")]

    assert _shared_conditions_for_load(model, neural, None, 3) == [(1, "a"), (1, "b"), (2, "d")]
    assert _shared_conditions_for_load(model, neural, 1, 2) == [(1, "a"), (1, "b")]
    assert _shared_conditions_for_load(model, neural, 2, 2) is None
