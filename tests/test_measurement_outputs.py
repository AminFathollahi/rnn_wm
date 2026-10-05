from __future__ import annotations

import csv
import json

import scripts.run_measurement_outputs as outputs
from scripts.run_measurement_outputs import WorkUnit, enumerate_units, run_units


def test_enumeration_splits_expensive_outputs_into_stable_units():
    records = [{"run_id": "M00000_s0"}, {"run_id": "M11111_s0"}]
    units = enumerate_units(
        ["task_structure", "sensitivity", "epoch_method", "persistence", "full_delay"],
        records,
        [None, "MTL"],
        [0.0, 1.0],
    )

    assert len(units) == 19
    assert len({unit.key for unit in units}) == len(units)
    assert sum(unit.table == "persistence" for unit in units) == 2
    assert sum(unit.table == "sensitivity" for unit in units) == 4
    assert sum(unit.table == "full_delay" and unit.run_id is None for unit in units) == 1
    assert sum(unit.table == "full_delay" and unit.run_id is not None for unit in units) == 4


def test_distribution_units_are_resumable_per_run_and_region():
    records = [{"run_id": "a"}, {"run_id": "b"}]
    units = enumerate_units(["distribution"], records, [None, "MFC"], [])

    assert len(units) == 4
    assert len({unit.key for unit in units}) == 4
    assert {unit.label() for unit in units} == {
        "distribution/a/pooled",
        "distribution/a/MFC",
        "distribution/b/pooled",
        "distribution/b/MFC",
    }


def test_state_version_changes_work_unit_keys(monkeypatch):
    unit = WorkUnit("full_delay", run_id="a", region="MTL")
    key = unit.key
    monkeypatch.setattr(outputs, "STATE_VERSION", outputs.STATE_VERSION + 1)
    assert unit.key != key


def test_completed_units_are_atomic_and_resumable(tmp_path):
    units = [WorkUnit("persistence", run_id="a"), WorkUnit("persistence", run_id="b")]
    calls = []

    def execute(unit):
        calls.append(unit.run_id)
        return [{"run_id": unit.run_id, "status": "ok"}]

    state = tmp_path / "state"
    assert run_units(units, state, tmp_path, "", execute) == (2, 0, 0)
    assert run_units(units, state, tmp_path, "", execute) == (0, 2, 0)
    assert calls == ["a", "b"]
    assert not list(state.rglob("*.partial"))
    assert all(json.loads(path.read_text())["rows"] for path in state.glob("persistence/*.json"))
    with (tmp_path / "dynamics_persistence_corrected.csv").open() as handle:
        assert {row["run_id"] for row in csv.DictReader(handle)} == {"a", "b"}


def test_failed_units_remain_pending(tmp_path):
    unit = WorkUnit("epoch_method", run_id="a", region="MTL")

    result = run_units(
        [unit],
        tmp_path / "state",
        tmp_path,
        "",
        lambda _: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert result == (0, 0, 1)
    assert not list((tmp_path / "state").rglob("*.json"))
