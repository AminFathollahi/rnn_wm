"""Phase 12.3 (comments.txt §12.3): `run_grid.run_grid_loop`'s
`ProcessPoolExecutor`-based execution must be a pure scheduling change --
N workers produce the same set of manifest rows as `--workers 1`, and a
worker raising an exception must not take down the others (isolated,
recorded as a `status: error` row)."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_grid as rg


def _rows(manifest_path: Path) -> dict:
    return {
        json.loads(line)["run_id"]: json.loads(line)
        for line in manifest_path.read_text().splitlines() if line.strip()
    }


def test_n_workers_produce_same_manifest_rows_as_workers_1(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "RESULTS", tmp_path)
    runs = [
        {"model_id": f"M{i}", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": f"M{i}_s0"}
        for i in range(6)
    ]
    cfg = {"steps": 5, "scaffold_sleep_s": 0}

    manifest_1 = tmp_path / "m1.jsonl"
    rg.run_grid_loop(runs, set(), cfg, "hash", "gitrev", "smoke", budget_s=3600, t0=time.time(),
                      manifest=manifest_1, force_scaffold=True, workers=1, log_prefix="[test]")
    manifest_4 = tmp_path / "m4.jsonl"
    rg.run_grid_loop(runs, set(), cfg, "hash", "gitrev", "smoke", budget_s=3600, t0=time.time(),
                      manifest=manifest_4, force_scaffold=True, workers=4, log_prefix="[test]")

    rows1, rows4 = _rows(manifest_1), _rows(manifest_4)
    expected_ids = {r["run_id"] for r in runs}
    assert set(rows1) == set(rows4) == expected_ids
    for rid in expected_ids:
        assert rows1[rid]["status"] == rows4[rid]["status"] == "completed"
        assert rows1[rid]["accuracy"] == rows4[rid]["accuracy"]  # scaffold stub is deterministic per run_id
    assert all(r["workers"] == 1 for r in rows1.values())
    assert all(r["workers"] == 4 for r in rows4.values())


def test_worker_exception_yields_error_row_without_killing_others(tmp_path, monkeypatch):
    monkeypatch.setattr(rg, "RESULTS", tmp_path)

    def _flaky_resolve(force_scaffold):  # noqa: ARG001
        def _fn(run, cfg):
            if run["run_id"] == "BOOM_s0":
                raise RuntimeError("boom")
            return rg._scaffold_train_one(run, cfg)
        return _fn

    monkeypatch.setattr(rg, "resolve_train_fn", _flaky_resolve)
    runs = [
        {"model_id": "OK1", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "OK1_s0"},
        {"model_id": "BOOM", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "BOOM_s0"},
        {"model_id": "OK2", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "OK2_s0"},
    ]
    manifest = tmp_path / "m.jsonl"
    rg.run_grid_loop(runs, set(), {"steps": 5, "scaffold_sleep_s": 0}, "hash", "gitrev", "smoke",
                      budget_s=3600, t0=time.time(), manifest=manifest, force_scaffold=True,
                      workers=3, log_prefix="[test]")

    rows = _rows(manifest)
    assert set(rows) == {"OK1_s0", "BOOM_s0", "OK2_s0"}
    assert rows["BOOM_s0"]["status"] == "error"
    assert "boom" in rows["BOOM_s0"]["error"]
    assert rows["OK1_s0"]["status"] == "completed"
    assert rows["OK2_s0"]["status"] == "completed"
