"""Training entrypoint test: every ablation-battery cell completes a
forward pass and weight update and emits a well-formed result at a
minimal step budget (the smoke tier). Skips if the stimuli pool has not
been built yet."""
from pathlib import Path

import numpy as np
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
    # BPTT unroll) on this 12GB GPU -- see executor.md. A smoke test's
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
    assert result["matched"] is False  # 6-step smoke run cannot reach 3 consecutive passing evals
    assert result["steps_to_load1_0.83"] is None


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
    """Phase 2 (comments.txt §5 item 2.3/executor.md): `cfg["batch_size"]`
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


def test_gate_a_load1_only_does_not_keyerror_against_all_task_loads():
    """Phase 12 (§3.1/12.1b): `gates.criterion` names only `load1`, but
    `task.loads` is [1,2,3] -- the periodic-eval milestone loop and the
    final `gates` dict must iterate `criterion`'s OWN keys, not
    `task_loads`, or every run in the study raises KeyError on load2/load3.
    This is the root-cause regression test for that bug (train.py used to
    index `criterion[f"load{i}"]` while looping `task_loads` at three
    sites)."""
    import shutil

    from brainalign_wm.training.train import train_one

    assert set(CFG["gates"]["criterion"].keys()) == {"load1"}  # the scenario this test exists to cover
    assert CFG["task"]["loads"] == [1, 2, 3]

    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "SMOKETEST_gate_a_subset"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    try:
        result = train_one(run, {"steps": 6, "scaffold_sleep_s": 0})
    finally:
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    assert result["status"] == "completed"
    assert set(result["gates"].keys()) == {"load1>=0.83"}  # not load2/load3 -- no KeyError, no spurious keys either


def test_milestone_confirmed_mid_run_recorded_at_k_times_eval_every_and_training_continues():
    """Phase 12 (§3.3): a milestone's `steps_to_<key>_<threshold>` must be
    recorded at exactly the CONFIRMING evaluation's step
    (k * eval_every, k = gates.consecutive_evals), and reaching it must NOT
    stop training -- §3 is explicit that only `gates.max_steps` (here,
    `cfg["steps"]`, since `gates.max_steps` is still null pre-12.6) ends the
    loop. Forces accuracy to 1.0 on every periodic eval via a stub (real
    accuracy this early in training is nowhere near 0.83/0.80, so a
    milestone would otherwise never fire in a smoke-sized run) and shrinks
    `eval_every` so multiple evals fit inside a short run."""
    import copy
    import shutil

    import brainalign_wm.training.train as train_mod

    fake_cfg = copy.deepcopy(CFG)
    fake_cfg["train"]["eval_every"] = 2
    fake_cfg["gates"]["consecutive_evals"] = 3
    fake_cfg["gates"]["criterion"] = {"load1": 0.83}
    fake_cfg["gates"]["extra_milestones"] = {"load3": 0.80}
    eval_every = fake_cfg["train"]["eval_every"]
    consecutive_evals = fake_cfg["gates"]["consecutive_evals"]
    total_steps = 10

    def fake_evaluate_accuracy(*args, **kwargs):
        return {
            "load1": 1.0, "load1_ci_lo": 1.0, "load1_ci_hi": 1.0,
            "load2": 1.0, "load2_ci_lo": 1.0, "load2_ci_hi": 1.0,
            "load3": 1.0, "load3_ci_lo": 1.0, "load3_ci_hi": 1.0,
        }

    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "SMOKETEST_milestone_continue"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    orig_load_cfg, orig_eval_acc = train_mod._load_full_config, train_mod.evaluate_accuracy
    train_mod._load_full_config = lambda: fake_cfg
    train_mod.evaluate_accuracy = fake_evaluate_accuracy
    try:
        result = train_mod.train_one(run, {"steps": total_steps, "scaffold_sleep_s": 0})
        import torch

        ckpt = torch.load(ckpt_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    finally:
        train_mod._load_full_config = orig_load_cfg
        train_mod.evaluate_accuracy = orig_eval_acc
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    expected_step = consecutive_evals * eval_every  # 3rd confirming eval, at step k*eval_every = 6
    assert result["steps_to_load1_0.83"] == expected_step
    assert result["steps_to_load3_0.8"] == expected_step
    assert result["matched"] is True
    # Training did NOT stop at the milestone: the final checkpoint is at
    # total_steps (10), not at the milestone step (6).
    assert ckpt["step"] == total_steps


def test_resume_preserves_true_milestone_step_and_does_not_reoverwrite_snapshot():
    """comments.txt §16A.2 regression: a milestone already confirmed before
    an interruption must keep its TRUE step after a resume (not the first
    post-resume confirmation), and its `ckpt_at_criterion.pt` snapshot must
    not be silently overwritten with post-resume weights the next time the
    live loop would otherwise re-detect it. This is not hypothetical: it is
    exactly what happened to `FLATGRU_RL_s0`'s 80,000-step resume before
    this fix landed."""
    import copy
    import shutil

    import torch

    import brainalign_wm.training.train as train_mod

    fake_cfg = copy.deepcopy(CFG)
    fake_cfg["train"]["eval_every"] = 2
    fake_cfg["gates"]["consecutive_evals"] = 3
    fake_cfg["gates"]["criterion"] = {"load1": 0.83}
    fake_cfg["gates"]["extra_milestones"] = {}

    def fake_evaluate_accuracy(*args, **kwargs):
        return {
            "load1": 1.0, "load1_ci_lo": 1.0, "load1_ci_hi": 1.0,
            "load2": 1.0, "load2_ci_lo": 1.0, "load2_ci_hi": 1.0,
            "load3": 1.0, "load3_ci_lo": 1.0, "load3_ci_hi": 1.0,
        }

    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "SMOKETEST_resume_milestone"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    metrics_path = ROOT / "results" / "metrics" / f"{run['run_id']}.csv"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    if metrics_path.exists():
        metrics_path.unlink()
    orig_load_cfg, orig_eval_acc = train_mod._load_full_config, train_mod.evaluate_accuracy
    train_mod._load_full_config = lambda: fake_cfg
    train_mod.evaluate_accuracy = fake_evaluate_accuracy
    try:
        first = train_mod.train_one(run, {"steps": 10, "scaffold_sleep_s": 0})
        true_step = first["steps_to_load1_0.83"]  # 3rd confirming eval at eval_every=2
        assert true_step == 6
        snapshot_before = torch.load(ckpt_dir / "ckpt_at_criterion.pt", map_location="cpu", weights_only=False)
        assert snapshot_before["step"] == 6

        second = train_mod.train_one(run, {"steps": 20, "scaffold_sleep_s": 0})  # resumes from step 10
        snapshot_after = torch.load(ckpt_dir / "ckpt_at_criterion.pt", map_location="cpu", weights_only=False)
    finally:
        train_mod._load_full_config = orig_load_cfg
        train_mod.evaluate_accuracy = orig_eval_acc
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        if metrics_path.exists():
            metrics_path.unlink()

    assert second["steps_to_load1_0.83"] == 6
    assert second["first_milestone_step"] == 6
    assert snapshot_after["step"] == 6  # not overwritten with post-resume (step 16) weights


def test_analysis_budget_and_unmet_criterion_cap_stop_at_their_own_limits():
    """Gate-A success stops at the budget; persistent failure reaches the cap."""
    import copy
    import shutil

    import torch

    import brainalign_wm.training.train as train_mod

    fake_cfg = copy.deepcopy(CFG)
    fake_cfg["train"]["eval_every"] = 2
    fake_cfg["gates"]["consecutive_evals"] = 2
    fake_cfg["gates"]["criterion"] = {"load1": 0.83}
    fake_cfg["gates"]["extra_milestones"] = {}
    budget, cap = 6, 10

    def accuracy(value):
        return {
            "load1": value, "load1_ci_lo": value, "load1_ci_hi": value,
            "load2": value, "load2_ci_lo": value, "load2_ci_hi": value,
            "load3": value, "load3_ci_lo": value, "load3_ci_hi": value,
        }

    evaluation_calls = {"met": 0}

    def fake_evaluate_accuracy(*args, **kwargs):
        run_kind = fake_cfg["_test_run_kind"]
        if run_kind == "never":
            return accuracy(0.0)
        evaluation_calls["met"] += 1
        return accuracy(1.0 if evaluation_calls["met"] >= 2 else 0.0)

    run_ids = ["SMOKETEST_budget_met", "SMOKETEST_budget_never_met"]
    ckpt_dirs = [ROOT / "results" / "checkpoints" / run_id for run_id in run_ids]
    metrics_paths = [ROOT / "results" / "metrics" / f"{run_id}.csv" for run_id in run_ids]
    for path in [*ckpt_dirs, *metrics_paths]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    orig_load_cfg = train_mod._load_full_config
    orig_eval_acc = train_mod.evaluate_accuracy
    orig_final_eval = train_mod.final_evaluation
    train_mod._load_full_config = lambda: fake_cfg
    train_mod.evaluate_accuracy = fake_evaluate_accuracy
    train_mod.final_evaluation = lambda *a, **k: accuracy(0.9)
    try:
        fake_cfg["_test_run_kind"] = "met"
        met = train_mod.train_one(
            {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": run_ids[0]},
            {"steps": budget, "max_steps_if_criterion_unmet": cap, "batch_size": 4},
        )
        met_ckpt = torch.load(ckpt_dirs[0] / "ckpt.pt", map_location="cpu", weights_only=False)
        met_gate_ckpt = torch.load(
            ckpt_dirs[0] / "ckpt_at_criterion.pt", map_location="cpu", weights_only=False
        )

        fake_cfg["_test_run_kind"] = "never"
        never = train_mod.train_one(
            {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 1, "run_id": run_ids[1]},
            {"steps": budget, "max_steps_if_criterion_unmet": cap, "batch_size": 4},
        )
        never_ckpt = torch.load(ckpt_dirs[1] / "ckpt.pt", map_location="cpu", weights_only=False)
    finally:
        train_mod._load_full_config = orig_load_cfg
        train_mod.evaluate_accuracy = orig_eval_acc
        train_mod.final_evaluation = orig_final_eval
        for path in ckpt_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in metrics_paths:
            if path.exists():
                path.unlink()

    assert met["training_stop_step"] == budget
    assert met["criterion_met_at_budget"] is True
    assert met_ckpt["step"] == budget
    assert never["training_stop_step"] == cap
    assert never["criterion_met_at_budget"] is False
    assert never["criterion_met_by_stop"] is False
    assert never_ckpt["step"] == budget
    for checkpoint in (met_ckpt, met_gate_ckpt, never_ckpt):
        assert "torch_rng_state" in checkpoint
        assert "numpy_rng_state" in checkpoint

    torch.set_rng_state(met_ckpt["torch_rng_state"])
    expected_torch_draw = torch.rand(4)
    np.random.set_state(met_ckpt["numpy_rng_state"])
    expected_numpy_draw = np.random.rand(4)
    torch.manual_seed(999)
    np.random.seed(999)
    train_mod._restore_checkpoint_rng(met_ckpt)
    assert torch.equal(torch.rand(4), expected_torch_draw)
    assert np.array_equal(np.random.rand(4), expected_numpy_draw)


def test_update_milestone_counters_latches_and_never_refires():
    """comments.txt §16A.2: the shared per-eval update rule used by both the
    live loop and history replay. A milestone latches after
    `consecutive_evals_required` consecutive hits and never re-fires."""
    import brainalign_wm.training.train as train_mod

    criterion = {"load1": 0.83}
    thresholds = {"load1": 0.83, "load3": 0.80}
    mc = {k: 0 for k in thresholds}
    mr = {k: False for k in thresholds}
    cc = {k: 0 for k in criterion}
    mst = {k: None for k in thresholds}
    mtt = {k: None for k in thresholds}
    mwt = {k: None for k in thresholds}
    mjt = {k: None for k in thresholds}

    accs = [
        {"load1": 0.5, "load3": 0.5}, {"load1": 0.5, "load3": 0.5},
        {"load1": 0.9, "load3": 0.5}, {"load1": 0.9, "load3": 0.5}, {"load1": 0.9, "load3": 0.5},
        {"load1": 0.9, "load3": 0.5},  # a fourth consecutive pass must not re-fire
    ]
    steps = [2000, 4000, 6000, 8000, 10000, 12000]
    newly = [
        train_mod._update_milestone_counters(
            acc, step, float(step), None, 128, criterion, thresholds, 3,
            mc, mr, cc, mst, mtt, mwt, mjt,
        )
        for acc, step in zip(accs, steps)
    ]
    assert newly == [[], [], [], [], ["load1"], []]
    assert mst["load1"] == 10000
    assert mr["load1"] is True
    assert mst["load3"] is None  # load3 never crossed 0.80


def test_seed_milestone_state_from_history_reconstructs_true_step(tmp_path):
    """comments.txt §16A.2: replaying a resumed run's pre-resume CSV rows
    must recover the true confirmation step, not the first post-resume one."""
    import csv as csv_mod

    import brainalign_wm.training.train as train_mod

    csv_path = tmp_path / "RUN.csv"
    fieldnames = train_mod._MetricsLogger.FIELDNAMES
    with open(csv_path, "w", newline="") as fh:
        w = csv_mod.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for step, load1 in [(10000, 0.9), (12000, 0.9), (14000, 0.9), (40000, 0.95)]:
            w.writerow({
                **{k: "" for k in fieldnames},
                "step": step, "train_acc_load1": load1, "train_acc_load3": 0.5,
                "wall_s": step / 10,
            })

    criterion = {"load1": 0.83}
    thresholds = {"load1": 0.83, "load3": 0.80}
    seeded = train_mod._seed_milestone_state_from_history(csv_path, 40000, criterion, thresholds, 3, 128)

    assert seeded["milestone_reached"]["load1"] is True
    assert seeded["milestone_steps_to"]["load1"] == 14000  # 3rd consecutive eval, not the 4th row (40000)
    assert seeded["first_milestone_step"] == 14000
    assert seeded["milestone_reached"]["load3"] is False


def test_matched_requires_criterion_to_hold_at_final_checkpoint_not_just_ever():
    """`matched` reports inclusion at the checkpoint accuracy/geometry
    actually use (the final, max_steps evaluation), not merely that the
    criterion was reached at some earlier point and then lost. A run whose
    accuracy clears the threshold for exactly the required streak and then
    drops back to chance for the rest of training must report `matched`
    False, even though the streak-based milestone (an efficiency DV, not an
    inclusion verdict) still records the step where it first confirmed."""
    import copy
    import shutil

    import brainalign_wm.training.train as train_mod

    fake_cfg = copy.deepcopy(CFG)
    fake_cfg["train"]["eval_every"] = 2
    fake_cfg["gates"]["consecutive_evals"] = 3
    fake_cfg["gates"]["criterion"] = {"load1": 0.83}
    fake_cfg["gates"]["extra_milestones"] = {}
    consecutive_evals = fake_cfg["gates"]["consecutive_evals"]
    eval_every = fake_cfg["train"]["eval_every"]
    total_steps = 10  # periodic evals at steps 2, 4, 6, 8, 10

    calls = {"n": 0}

    def fake_evaluate_accuracy(*args, **kwargs):
        calls["n"] += 1
        load1 = 1.0 if calls["n"] <= consecutive_evals else 0.5  # clears for evals 1-3, then chance
        return {
            "load1": load1, "load1_ci_lo": load1, "load1_ci_hi": load1,
            "load2": load1, "load2_ci_lo": load1, "load2_ci_hi": load1,
            "load3": load1, "load3_ci_lo": load1, "load3_ci_hi": load1,
        }

    run = {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "seed": 0, "run_id": "SMOKETEST_matched_at_checkpoint"}
    ckpt_dir = ROOT / "results" / "checkpoints" / run["run_id"]
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    orig_load_cfg, orig_eval_acc = train_mod._load_full_config, train_mod.evaluate_accuracy
    train_mod._load_full_config = lambda: fake_cfg
    train_mod.evaluate_accuracy = fake_evaluate_accuracy
    try:
        result = train_mod.train_one(run, {"steps": total_steps, "scaffold_sleep_s": 0})
    finally:
        train_mod._load_full_config = orig_load_cfg
        train_mod.evaluate_accuracy = orig_eval_acc
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)

    # The milestone (efficiency DV) still records the step of the 3rd
    # confirming eval, unaffected by the later collapse.
    assert result["steps_to_load1_0.83"] == consecutive_evals * eval_every
    # But inclusion at the final checkpoint must be False: the last
    # `consecutive_evals` evaluations (steps 6, 8, 10) do not all clear.
    assert result["matched"] is False


def test_metrics_logger_resume_appends_without_truncating(tmp_path, monkeypatch):
    """comments.txt §16 item 16.1: resuming a run must not destroy its
    earlier accuracy trace, which is exactly what Gate B (§3.2/12.6) is
    derived from. A second logger for the same run_id must append, keep
    the original rows, and write the header exactly once."""
    import brainalign_wm.training.train as train_mod

    monkeypatch.setattr(train_mod, "ROOT", tmp_path)

    logger = train_mod._MetricsLogger("RESUME_TEST")
    logger.log({"step": 2000, "train_loss": "0.5"})
    logger.log({"step": 4000, "train_loss": "0.4"})
    logger.close()

    resumed = train_mod._MetricsLogger("RESUME_TEST", resume=True)
    resumed.log({"step": 6000, "train_loss": "0.3"})
    resumed.close()

    text = (tmp_path / "results" / "metrics" / "RESUME_TEST.csv").read_text()
    lines = [l for l in text.splitlines() if l]
    assert lines.count(",".join(train_mod._MetricsLogger.FIELDNAMES)) == 1
    steps = [l.split(",")[0] for l in lines[1:]]
    assert steps == ["2000", "4000", "6000"]


def test_metrics_logger_skips_duplicate_step_at_resume_boundary(tmp_path, monkeypatch):
    """If a resumed run re-executes a step already logged before the
    interruption (possible if a future checkpoint cadence outpaces
    eval_every), it must not double-write that row."""
    import brainalign_wm.training.train as train_mod

    monkeypatch.setattr(train_mod, "ROOT", tmp_path)

    logger = train_mod._MetricsLogger("RESUME_DUP_TEST")
    logger.log({"step": 4000, "train_loss": "0.4"})
    logger.close()

    resumed = train_mod._MetricsLogger("RESUME_DUP_TEST", resume=True)
    resumed.log({"step": 4000, "train_loss": "0.4"})  # re-logged, must be dropped
    resumed.log({"step": 6000, "train_loss": "0.3"})
    resumed.close()

    text = (tmp_path / "results" / "metrics" / "RESUME_DUP_TEST.csv").read_text()
    rows = [l for l in text.splitlines() if l][1:]
    assert [r.split(",")[0] for r in rows] == ["4000", "6000"]


def test_metrics_logger_non_resume_overwrites_stale_file(tmp_path, monkeypatch):
    """A genuinely fresh run (resume=False, the default -- no checkpoint was
    restored) must still start a clean trace, even if a stale CSV from an
    earlier run of the same run_id is sitting on disk. This is the regression
    the append fix must not introduce: two independent `train_one` calls that
    reuse a run_id without a checkpoint (e.g. re-running a smoke test) are not
    a resume and must not accumulate rows."""
    import csv

    import brainalign_wm.training.train as train_mod

    monkeypatch.setattr(train_mod, "ROOT", tmp_path)

    first = train_mod._MetricsLogger("FRESH_TEST")
    first.log({"step": 1, "train_loss": "0.9"})
    first.close()

    second = train_mod._MetricsLogger("FRESH_TEST")  # resume=False (default)
    second.log({"step": 1, "train_loss": "0.1"})
    second.close()

    with open(tmp_path / "results" / "metrics" / "FRESH_TEST.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["train_loss"] == "0.1"
