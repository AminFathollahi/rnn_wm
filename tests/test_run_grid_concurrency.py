"""Phase 12.3 (comments.txt §12.3): `run_grid.run_grid_loop`'s
`ProcessPoolExecutor`-based execution must be a pure scheduling change --
N workers produce the same set of manifest rows as `--workers 1`, and a
worker raising an exception must not take down the others (isolated,
recorded as a `status: error` row)."""
import json
import os
import signal
import sys
import threading
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


def test_gpu_budget_mib_serializes_s1_runs_but_not_s0(tmp_path, monkeypatch):
    """D38: two S=1 (hierarchical) cells alone measured ~11.16 GiB concurrently,
    almost the whole 12227 MiB card. `--gpu-budget-mib` must keep S=1 runs from
    overlapping in wall-clock time even when `workers` would otherwise allow it,
    while S=0 runs (small footprint) still pack in freely."""
    monkeypatch.setattr(rg, "RESULTS", tmp_path)

    def _timed_resolve(force_scaffold):  # noqa: ARG001
        def _fn(run, cfg):
            t_start = time.time()
            time.sleep(cfg.get("scaffold_sleep_s", 0.05))
            return {"status": "completed", "t_start": t_start, "t_end": time.time()}
        return _fn

    monkeypatch.setattr(rg, "resolve_train_fn", _timed_resolve)
    runs = (
        [{"model_id": f"S1_{i}", "S": 1, "seed": 0, "run_id": f"S1_{i}_s0"} for i in range(3)]
        + [{"model_id": f"S0_{i}", "S": 0, "seed": 0, "run_id": f"S0_{i}_s0"} for i in range(3)]
    )
    manifest = tmp_path / "m.jsonl"
    rg.run_grid_loop(runs, set(), {"scaffold_sleep_s": 0.4}, "hash", "gitrev", "smoke",
                      budget_s=3600, t0=time.time(), manifest=manifest, force_scaffold=True,
                      workers=6, log_prefix="[test]", gpu_budget_mib=rg.DEFAULT_GPU_BUDGET_MIB)

    rows = _rows(manifest)
    assert len(rows) == 6
    s1 = [(r["t_start"], r["t_end"]) for r in rows.values() if r["S"] == 1]
    for i, (s_a, e_a) in enumerate(s1):
        for s_b, e_b in s1[i + 1:]:
            assert e_a <= s_b or e_b <= s_a, "two S=1 runs overlapped under the GPU memory budget"


def test_a_run_that_does_not_fit_does_not_block_cheaper_runs_behind_it(tmp_path, monkeypatch):
    """The SUP campaign launched 8 workers and ran 3. Two plastic cells filled
    7076 of 10500 MiB; the next queued cell was a third plastic cell needing
    3538 more, and the scheduler stopped submitting there -- for 15 hours --
    while nine 500 MiB cells sat behind it and five workers idled. Admission
    must skip a run that does not fit and take the first pending one that
    does."""
    monkeypatch.setattr(rg, "RESULTS", tmp_path)

    def _timed_resolve(force_scaffold):  # noqa: ARG001
        def _fn(run, cfg):  # noqa: ARG001
            t_start = time.time()
            time.sleep(1.2 if run["P"] else 0.05)  # plastic cells are the long ones
            return {"status": "completed", "t_start": t_start, "t_end": time.time()}
        return _fn

    monkeypatch.setattr(rg, "resolve_train_fn", _timed_resolve)

    def _big(i):
        return {"model_id": f"BIG{i}", "S": 1, "P": 1, "seed": 0, "run_id": f"BIG{i}_s0"}

    def _cheap(i):
        return {"model_id": f"CHEAP{i}", "S": 0, "P": 0, "seed": 0, "run_id": f"CHEAP{i}_s0"}

    # Queue order mirrors CELLS: plastic cells early, cheap ones interleaved after.
    # Two plastic cells (2*3538) plus one cheap (500) fill 7576 of 10500 MiB; the
    # third plastic needs 3538 more and cannot be admitted until a plastic one ends.
    runs = [_big(0), _big(1), _cheap(0), _big(2), _cheap(1), _cheap(2)]
    manifest = tmp_path / "m.jsonl"
    rg.run_grid_loop(runs, set(), {}, "hash", "gitrev", "smoke",
                      budget_s=3600, t0=time.time(), manifest=manifest, force_scaffold=True,
                      workers=6, log_prefix="[test]", gpu_budget_mib=rg.DEFAULT_GPU_BUDGET_MIB,
                      mem_cfg=_mem_cfg())
    rows = _rows(manifest)
    assert len(rows) == 6

    first_big_end = min(r["t_end"] for r in rows.values() if r["P"] == 1)
    for r in rows.values():
        if r["P"] == 0:
            assert r["t_end"] < first_big_end, (
                f"{r['run_id']} waited behind an unadmittable plastic cell instead of "
                "being skipped ahead to"
            )


def _mem_cfg(max_load: int = 3, checkpointing: bool = True) -> dict:
    """A resolved-config shape carrying only the fields `_run_mib` reads."""
    return {
        "model": {"flat_units": 128, "worker_units": 196},
        "train": {"batch_size": 128},
        "mechanisms": {"plastic_gradient_checkpointing": checkpointing},
        "task": {
            "loads": list(range(1, max_load + 1)), "encode_steps": 15, "maintain_steps": 15,
            "probe_steps": 10, "fixation_steps": 3, "feedback_steps": 1, "iti_steps": 2,
        },
    }


def test_trial_ticks_matches_the_configured_epoch_durations():
    # 31 + 15*load: the arithmetic every memory estimate below rests on.
    assert rg._trial_ticks(_mem_cfg(max_load=1)) == 46
    assert rg._trial_ticks(_mem_cfg(max_load=3)) == 76


def test_estimate_is_monotone_in_max_load_for_plastic_cells():
    """D41: peak memory is linear in trial length, so the estimate MUST rise
    with the curriculum's hardest load. D39's per-cell constants could not,
    which is why they were admitted at load 1 and OOM'd at load 3."""
    plastic = {"model_id": "M11111", "S": 1, "P": 1, "seed": 0, "run_id": "M11111_s0"}
    by_load = [rg._run_mib(plastic, _mem_cfg(max_load=n)) for n in (1, 2, 3)]
    assert by_load == sorted(by_load) and by_load[0] < by_load[-1], by_load


def test_plastic_s1_at_load3_uncheckpointed_exceeds_the_whole_card():
    """The configuration that actually failed. `M11111` OOM'd at --workers 1,
    alone on an 11.5 GiB card, so the scheduler must refuse it outright rather
    than admit it as D39's 6200 MiB constant did."""
    plastic_s1 = {"model_id": "M11111", "S": 1, "P": 1, "seed": 0, "run_id": "M11111_s0"}
    est = rg._run_mib(plastic_s1, _mem_cfg(max_load=3, checkpointing=False))
    assert est > 12227, f"estimated {est} MiB, which a 12227 MiB card would wrongly accept"
    # ... and that checkpointing is what brings it back within budget.
    assert rg._run_mib(plastic_s1, _mem_cfg(max_load=3)) < rg.DEFAULT_GPU_BUDGET_MIB


def test_non_plastic_runs_are_cheap_regardless_of_substrate():
    """Corrects D39's premise. A non-plastic cell never builds the [B, 3H, H]
    tensor at all, so an S=1 non-plastic run is cheap -- measured ~320 MiB --
    and several may overlap. D39 budgeted it 6200 MiB and needlessly serialized
    them."""
    cfg = _mem_cfg()
    for run in ({"S": 1, "P": 0, "run_id": "M11011_s0"}, {"S": 0, "P": 0, "run_id": "M00000_s0"}):
        assert rg._run_mib(run, cfg) == rg._BASE_MIB
    assert 4 * rg._BASE_MIB < rg.DEFAULT_GPU_BUDGET_MIB


def test_estimate_falls_back_to_constants_without_a_resolved_config():
    """`run_grid_loop` is also driven by the scaffold path and by
    `scripts/run_stage1_grid.py`, which pass a tier dict rather than a resolved
    config. Keep the old conservative constants there rather than silently
    estimating from defaults."""
    assert rg._run_mib({"S": 1}, None) == rg._LEGACY_MIB[1]
    assert rg._run_mib({"S": 0}, {"steps": 5}) == rg._LEGACY_MIB[0]


def test_load_completed_ignores_a_row_from_another_tier(tmp_path):
    """A smoke-tier diagnostic must not retire a full-tier campaign cell.

    Regression for the real `M11011_SUP_s0` row: D38's `--tier smoke` OOM
    probe left `status: completed` at 100 steps and chance accuracy, and
    `load_completed` keyed on `run_id` alone would have skipped that core
    cell in every later full-tier pass."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"run_id": "M11011_SUP_s0", "status": "completed", "tier": "smoke"}) + "\n"
        + json.dumps({"run_id": "M00000_SUP_s0", "status": "completed", "tier": "full"}) + "\n"
        + json.dumps({"run_id": "M01111_SUP_s0", "status": "error", "tier": "full"}) + "\n"
    )
    assert rg.load_completed(manifest, "full") == {"M00000_SUP_s0"}
    assert rg.load_completed(manifest, "smoke") == {"M11011_SUP_s0"}
    # No tier given: unfiltered, as before, for callers with no tier concept.
    assert rg.load_completed(manifest) == {"M11011_SUP_s0", "M00000_SUP_s0"}


def test_stop_signal_writes_interrupted_rows_for_in_flight_runs(tmp_path, monkeypatch):
    """D44 (comments.txt §20.4): a killed pass left four runs 8,000-76,000
    steps in with checkpoints on disk and NO manifest row at all, invisible
    to `load_completed` and every manifest-based audit. A caught stop signal
    must record every in-flight run as `status: interrupted` at the moment
    it arrives, not only if the process later reaches its graceful
    shutdown path."""
    monkeypatch.setattr(rg, "RESULTS", tmp_path)
    runs = [{"model_id": f"M{i}", "S": 0, "seed": 0, "run_id": f"M{i}_s0"} for i in range(3)]
    manifest = tmp_path / "m.jsonl"

    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, rg._handle_signal)
    rg._STOP = False
    try:
        thread = threading.Thread(
            target=rg.run_grid_loop,
            args=(runs, set(), {"scaffold_sleep_s": 2.0}, "hash", "gitrev", "smoke"),
            kwargs=dict(budget_s=3600, t0=time.time(), manifest=manifest,
                        force_scaffold=True, workers=3, log_prefix="[test]"),
        )
        thread.start()
        time.sleep(0.3)  # let all three futures start (well inside the 2.0s scaffold sleep)
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.3)  # let the handler run and append rows, without waiting for completion

        rows = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        interrupted = [r for r in rows if r["status"] == "interrupted"]
        assert {r["run_id"] for r in interrupted} == {r["run_id"] for r in runs}
        assert rg.load_completed(manifest) == set()  # interrupted must not count as completed

        thread.join(timeout=10)
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        rg._STOP = False


def test_admission_respects_a_reading_of_the_card_not_only_the_model(tmp_path, monkeypatch):
    """comments.txt §23.1: the scheduler admitted against `_run_mib`'s *model*
    of what the in-flight runs should cost. On 2026-08-08 three runs whose
    modelled cost fit the budget were submitted while five idle pool workers
    held ~9 GiB of an 11.5 GiB card, and all three died of CUDA OOM -- the
    allocator wrote the evidence into their manifest rows. A candidate must fit
    both accounts: what the model says is in flight, AND what the device
    reports as used."""
    monkeypatch.setattr(rg, "RESULTS", tmp_path)

    def _timed_resolve(force_scaffold):  # noqa: ARG001
        def _fn(run, cfg):
            t_start = time.time()
            time.sleep(cfg.get("scaffold_sleep_s", 0.05))
            return {"status": "completed", "t_start": t_start, "t_end": time.time()}
        return _fn

    monkeypatch.setattr(rg, "resolve_train_fn", _timed_resolve)

    # Two S=0 cells, 500 MiB each by the model: 1000 of 10500 MiB, so the model
    # alone would happily run them together.
    def _runs():
        return [{"model_id": f"S0_{i}", "S": 0, "P": 0, "seed": 0, "run_id": f"S0_{i}_s0"}
                for i in range(2)]

    def _overlap(manifest):
        (a_s, a_e), (b_s, b_e) = [(r["t_start"], r["t_end"]) for r in _rows(manifest).values()]
        return not (a_e <= b_s or b_e <= a_s)

    def _go(manifest, used_mib):
        monkeypatch.setattr(rg, "gpu_used_total_mib", lambda: (used_mib, 12227))
        rg.run_grid_loop(_runs(), set(), {"scaffold_sleep_s": 0.4}, "hash", "gitrev", "smoke",
                          budget_s=3600, t0=time.time(), manifest=manifest, force_scaffold=True,
                          workers=2, log_prefix="[test]",
                          gpu_budget_mib=rg.DEFAULT_GPU_BUDGET_MIB, mem_cfg=_mem_cfg())

    # Card already at 10200 of a 10500 MiB budget: 300 MiB of real headroom, so
    # the second 500 MiB run must wait however cheap the model thinks it is.
    full = tmp_path / "full.jsonl"
    _go(full, 10200)
    assert not _overlap(full), "admitted a run the device had no room for"

    # Control: same runs, same model, empty card -- they must still overlap, or
    # the fix would be serializing everything rather than reading the device.
    empty = tmp_path / "empty.jsonl"
    _go(empty, 0)
    assert _overlap(empty), "an empty card should still allow two cheap runs to overlap"


def test_real_runs_get_a_fresh_worker_process_each(tmp_path, monkeypatch):
    """comments.txt §23.1 / user decision 2026-08-09: `empty_cache()` returns the
    caching allocator's blocks but not the CUDA context, which a pooled worker
    holds for the life of the pool -- 370-466 MiB per idle worker during the
    2026-08-08 pass. Only process death frees it, so real (non-scaffold) runs
    must get `max_tasks_per_child=1`."""
    monkeypatch.setattr(rg, "RESULTS", tmp_path)
    seen: dict = {}
    real_pool = rg.ProcessPoolExecutor

    class _Recording(real_pool):
        def __init__(self, *a, **kw):
            seen.update(kw)
            # Drop it here so the pool stays fork-based for this test; the
            # assertion below is on what `run_grid_loop` asked for.
            kw.pop("max_tasks_per_child", None)
            super().__init__(*a, **kw)

    monkeypatch.setattr(rg, "ProcessPoolExecutor", _Recording)
    monkeypatch.setattr(rg, "resolve_train_fn", lambda force_scaffold: rg._scaffold_train_one)

    runs = [{"model_id": "A", "S": 0, "P": 0, "seed": 0, "run_id": "A_s0"}]
    manifest = tmp_path / "m.jsonl"
    rg.run_grid_loop(runs, set(), {"steps": 5, "scaffold_sleep_s": 0}, "hash", "gitrev", "smoke",
                      budget_s=3600, t0=time.time(), manifest=manifest, force_scaffold=False,
                      workers=2, log_prefix="[test]")
    assert seen["max_tasks_per_child"] == 1
    assert _rows(manifest)["A_s0"]["status"] == "completed"


