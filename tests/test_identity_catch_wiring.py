"""The identity-report sub-experiment's run ids must reach the replay,
variant-guard and alignment paths intact for any architecture the launcher
is pointed at, not only the two it used to hardcode."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from brainalign_wm.analysis.run_all import _is_ablation_or_catch_variant
from brainalign_wm.training.generate_activity_logs import _parse_run_id, _run_id_extras

import run_identity_catch
from align_extended_variants import _base_run_id, demand_contrast


def test_cells_are_selectable_and_validated():
    cells = run_identity_catch.parse_cells("M00000,M10010")
    assert cells == [
        {"model_id": "M00000", "S": 0, "M": 0, "P": 0, "T": 0, "D": 0},
        {"model_id": "M10010", "S": 1, "M": 0, "P": 0, "T": 1, "D": 0},
    ]
    with pytest.raises(ValueError):
        run_identity_catch.parse_cells("M1001")


def test_enumerated_runs_are_seed_major_and_name_their_signal():
    runs = run_identity_catch.enumerate_runs([0, 1], run_identity_catch.parse_cells("M00000,M10010"), "SUP")
    assert [r["run_id"] for r in runs] == [
        "M00000_idcatch_SUP_s0", "M10010_idcatch_SUP_s0", "M00000_idcatch_SUP_s1", "M10010_idcatch_SUP_s1",
    ]
    assert all(r["model_id"] in ("M00000_idcatch", "M10010_idcatch") for r in runs)  # unsuffixed by signal
    assert all(r["supervision"] == "SUP" for r in runs)
    assert all(r["identity_catch_fraction"] == run_identity_catch.IDENTITY_CATCH_FRACTION for r in runs)
    assert runs[1]["S"] == 1 and runs[1]["T"] == 1


def test_enumerated_run_ids_never_collide_across_signal_passes():
    cells = run_identity_catch.parse_cells("M00000,M10010")
    sup_ids = {r["run_id"] for r in run_identity_catch.enumerate_runs([0, 1], cells, "SUP")}
    rl_ids = {r["run_id"] for r in run_identity_catch.enumerate_runs([0, 1], cells, "RL")}
    assert sup_ids.isdisjoint(rl_ids)


def test_run_id_reaches_replay_and_variant_guard():
    """Pre-namespacing ids already on disk (never renamed) still parse."""
    assert _parse_run_id("M10010_idcatch_s5") == ("M10010_idcatch", 1, 0, 0, 1, 0, 5)
    assert _run_id_extras("M10010_idcatch") == (False, run_identity_catch.IDENTITY_CATCH_FRACTION, 1, {})
    assert _is_ablation_or_catch_variant(
        {"model_id": "M10010_idcatch", "S": 1, "M": 0, "P": 0, "T": 1, "D": 0}
    )
    assert not _is_ablation_or_catch_variant(
        {"model_id": "M10010", "S": 1, "M": 0, "P": 0, "T": 1, "D": 0}
    )


def test_signal_namespaced_run_id_reaches_replay_and_variant_guard():
    assert _parse_run_id("M10010_idcatch_RL_s5") == ("M10010_idcatch_RL", 1, 0, 0, 1, 0, 5)
    assert _run_id_extras("M10010_idcatch_RL") == (False, run_identity_catch.IDENTITY_CATCH_FRACTION, 1, {})
    assert _is_ablation_or_catch_variant(
        {"model_id": "M10010_idcatch", "S": 1, "M": 0, "P": 0, "T": 1, "D": 0}
    )


def test_base_run_resolves_to_the_signal_qualified_campaign_run():
    known = {"M10010_SUP_s3": {}, "M10010_s3": {}}
    assert _base_run_id("M10010", {"seed": 3, "supervision": "SUP"}, known) == "M10010_SUP_s3"
    assert _base_run_id("M10010", {"seed": 3}, known) == "M10010_s3"
    assert _base_run_id("M10010", {"seed": 9, "supervision": "SUP"}, {}) == "M10010_s9"


def _pair(model_id, seed, gap):
    return {"run_id": f"{model_id}_idcatch_s{seed}", "base_run_id": f"{model_id}_SUP_s{seed}",
            "base_model_id": model_id, "seed": seed, "accuracy_gap_load3": gap}


def test_demand_contrast_recovers_a_planted_paired_difference():
    sessions = {f"sess{i}": f"patient{i // 2}" for i in range(8)}
    alignment, pairs = {}, []
    rng = np.random.default_rng(0)
    for seed in range(4):
        pair = _pair("M10010", seed, 0.2 if seed == 3 else 0.01)
        pairs.append(pair)
        noise = rng.normal(0, 0.01, len(sessions))
        alignment[pair["base_run_id"]] = {s: (0.10 + n, p) for (s, p), n in zip(sessions.items(), noise)}
        alignment[pair["run_id"]] = {s: (0.15 + n, p) for (s, p), n in zip(sessions.items(), noise)}
    rows = demand_contrast(pairs, alignment, n_boot=200)
    all_seeds = next(r for r in rows if not r["matched_on_accuracy"])
    matched = next(r for r in rows if r["matched_on_accuracy"])
    assert all_seeds["status"] == "ok" and all_seeds["n_seeds"] == 4
    assert all_seeds["effect"] == pytest.approx(0.05, abs=1e-6)
    assert all_seeds["ci_lo"] > 0.0 and all_seeds["ci_hi"] > all_seeds["ci_lo"]
    assert all_seeds["n_patients"] == 4
    assert matched["n_seeds"] == 3  # the seed whose accuracy gap exceeds the tolerance is dropped
