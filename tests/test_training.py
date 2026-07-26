"""Training entrypoint test: every ablation-battery cell completes a
forward pass and weight update and emits a well-formed result at a
minimal step budget (the smoke tier). Skips if the stimuli pool has not
been built yet."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
STIMULI_READY = (ROOT / "stimuli" / "faces").exists()

pytestmark = pytest.mark.skipif(not STIMULI_READY, reason="stimuli/ pool not built yet")

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("cell", CFG["cells"])
def test_cell_smoke(cell):
    """5-bit ablation-battery cells (S,M,P,T,D), BPTT throughout -- no
    cell here is training-algorithm-gated."""
    import shutil

    from brainalign_wm.training.train import train_one

    S, M, P, T, D = (int(c) for c in cell[1:6])
    run = {"model_id": cell, "S": S, "M": M, "P": P, "T": T, "D": D, "seed": 0, "run_id": f"SMOKETEST_{cell}"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    # Phase 2 (comments.txt §5): global batch_size=128 OOMs arm P
    # (PlasticGRUCell's per-trial Hebbian trace retained across the full
    # BPTT unroll) on this 12GB GPU -- see PHASE_LOG.md. A smoke test's
    # 6-step run doesn't need production batch fidelity, so it overrides
    # down via `cfg["batch_size"]` rather than shrinking the global default.
    cfg = {"steps": 6, "scaffold_sleep_s": 0}
    if P:
        cfg["batch_size"] = 16
    try:
        result = train_one(run, cfg)
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
    # Phase 3 (A3): gate keys/thresholds come from config.yaml's gates.criterion,
    # not hardcoded, so this test doesn't silently drift from config again.
    criterion = CFG["gates"]["criterion"]
    assert set(result["gates"].keys()) == {f"{k}>={v}" for k, v in criterion.items()}
    expected_acc_keys = {f"load{i}" for i in (1, 2, 3)} | {f"load{i}_ci_{b}" for i in (1, 2, 3) for b in ("lo", "hi")}
    assert set(result["accuracy"].keys()) == expected_acc_keys
    assert result["rung"] == 0  # BPTT throughout; rung is local-learning-only
    assert result["criterion_met"] is False  # 6-step smoke run cannot reach 3 consecutive passing evals
    assert result["steps_to_criterion"] is None


@pytest.mark.parametrize("cell", CFG["local_learning_cells"])
def test_local_learning_cell_smoke(cell):
    """The local-learning cells: 3rd char is literally 'L', trained by
    node-perturbation, not P -- a separate S/M-only family, no T/D."""
    import shutil

    from brainalign_wm.training.train import train_one

    S, M = int(cell[1]), int(cell[2])
    assert cell[3] == "L"
    run = {"model_id": cell, "S": S, "M": M, "L": 1, "seed": 0, "run_id": f"SMOKETEST_{cell}"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        result = train_one(run, {"steps": 6, "scaffold_sleep_s": 0})
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
    assert result["rung"] in (1, 2, 3)


@pytest.mark.parametrize(
    "extra",
    [
        {"energy_cost_weight": 0.01},
        {"recurrent_noise_sigma": 0.05},
        {"pbwm_gate": True},
        {"identity_catch_fraction": 0.12},
        {"topo_loss_weight": 0.01},
        {"dale_penalty_weight": 0.01},
    ],
)
def test_run_dict_overrides_bio_plausible_and_identity_catch(extra):
    """Bio-plausible-ablation-battery and identity-catch arms are per-run
    overrides (not global config edits, so distinct arms can share one
    config.yaml) -- each should train without crashing and, for the
    identity-catch arm, report a separate `accuracy["identity_catch"]`."""
    import shutil

    from brainalign_wm.training.train import train_one

    run_id = f"SMOKETEST_override_{list(extra)[0]}"
    run = {"model_id": "M11111", "S": 1, "M": 1, "P": 1, "T": 1, "D": 1, "seed": 0, "run_id": run_id, **extra}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        # M11111 has P=1 -- see the batch_size override note in test_cell_smoke.
        result = train_one(run, {"steps": 6, "scaffold_sleep_s": 0, "batch_size": 16})
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
    if "identity_catch_fraction" in extra:
        assert "identity_catch" in result["accuracy"]
        assert 0.0 <= result["accuracy"]["identity_catch"] <= 1.0
    else:
        assert "identity_catch" not in result["accuracy"]


def test_cfg_batch_size_override_takes_effect():
    """Phase 2 (comments.txt §5 item 2.3/PHASE_LOG.md): `cfg["batch_size"]`
    must override `configs/config.yaml`'s global `train.batch_size` -- added
    so arm-P cells (which OOM at the global default on this GPU) can run at
    a smaller batch without touching every other cell's throughput.
    `flops_per_step` (item 2.4) is exactly linear in batch_size, so its
    logged value is an observable witness that the override actually
    reached the training loop, not just that train_one didn't crash."""
    import csv
    import shutil

    from brainalign_wm.training.train import ROOT as TRAIN_ROOT
    from brainalign_wm.training.train import train_one

    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "SMOKETEST_batch_override"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    metrics_path = TRAIN_ROOT / "results" / "metrics" / f"{run['run_id']}.csv"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        train_one(run, {"steps": 1, "scaffold_sleep_s": 0, "batch_size": 4})
        with open(metrics_path) as fh:
            flops_at_4 = int(next(csv.DictReader(fh))["flops_per_step"])
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        train_one(run, {"steps": 1, "scaffold_sleep_s": 0, "batch_size": 8})
        with open(metrics_path) as fh:
            flops_at_8 = int(next(csv.DictReader(fh))["flops_per_step"])
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if metrics_path.exists():
            metrics_path.unlink()

    assert flops_at_8 == 2 * flops_at_4


@pytest.mark.parametrize(
    "extra",
    [
        {"flat_units": 512},
        {"l1_weight_penalty": 0.001},
        {"core_dropout_p": 0.1},
    ],
)
def test_run_dict_overrides_perf_matched_baselines(extra):
    """Performance-matched baseline overrides (flat_gru_2x/l1/dropout) are
    per-run, same mechanism as the bio-plausible-ablation overrides above
    -- each should train without crashing off the M00000 (flat GRU)
    architecture the baseline family is defined on."""
    import shutil

    from brainalign_wm.training.train import train_one

    run_id = f"SMOKETEST_override_{list(extra)[0]}"
    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0, "seed": 0, "run_id": run_id, **extra}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        result = train_one(run, {"steps": 6, "scaffold_sleep_s": 0})
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
