"""Pure run_id-parsing tests for generate_activity_logs.py -- no stimuli
pool, GPU, or neural data required. Model IDs are the 5-arm ablation
battery's "M" + 5 binary digits (S,M,P,T,D fixed order; comments.txt
2026-07-08 v6.0 pivot), or the separate, unchanged local-learning family
(M**L)."""
import pytest

from brainalign_wm.training.generate_activity_logs import _parse_model_id, _parse_run_id, _run_id_extras


def test_parse_model_id_5bit_full_and_knockout():
    assert _parse_model_id("M11111") == (1, 1, 1, 1, 1)
    assert _parse_model_id("M10111") == (1, 0, 1, 1, 1)
    assert _parse_model_id("M00000") == (0, 0, 0, 0, 0)


def test_parse_model_id_local_learning_still_routes_separately():
    # T/D never apply to the local-learning study -- both 0.
    assert _parse_model_id("M10L") == (1, 0, 0, 0, 0)
    assert _parse_model_id("M11L") == (1, 1, 0, 0, 0)


def test_parse_model_id_rejects_unrecognized_string():
    with pytest.raises(ValueError):
        _parse_model_id("M1")


def test_parse_run_id_plain_core_cell():
    model_id, S, M, P, T, D, seed = _parse_run_id("M11111_s3")
    assert (model_id, S, M, P, T, D, seed) == ("M11111", 1, 1, 1, 1, 1, 3)


def test_parse_run_id_ablation_and_identity_catch_suffixes():
    """§4.4/§9.4a run_id convention (re-anchored to M11111/M00000 in the
    v6.0 pivot): the suffix rides along in `model_id` (keeps parquet
    filenames unambiguous) while S/M/P/T/D still come from the leading 5
    characters."""
    for run_id, expected_model_id in [
        ("M11111_pbwm_s0", "M11111_pbwm"),
        ("M11111_energy_s2", "M11111_energy"),
        ("M11111_noise_s4", "M11111_noise"),
        ("M00000_idcatch_s1", "M00000_idcatch"),
        ("M11111_idcatch_s3", "M11111_idcatch"),
    ]:
        model_id, S, M, P, T, D, seed = _parse_run_id(run_id)
        assert model_id == expected_model_id
        assert (S, M, P, T, D) == tuple(int(c) for c in expected_model_id[1:6])


def test_run_id_extras_detects_pbwm_and_identity_catch():
    assert _run_id_extras("M11111") == (False, 0.0, 1)
    assert _run_id_extras("M11111_pbwm") == (True, 0.0, 1)
    assert _run_id_extras("M11111_energy") == (False, 0.0, 1)
    assert _run_id_extras("M11111_noise") == (False, 0.0, 1)
    assert _run_id_extras("M00000_idcatch") == (False, 0.12, 1)
    assert _run_id_extras("M11111_idcatch") == (False, 0.12, 1)


def test_run_id_extras_detects_perf_matched_baselines():
    assert _run_id_extras("M00000_2x") == (False, 0.0, 2)
    assert _run_id_extras("M00000_l1") == (False, 0.0, 1)
    assert _run_id_extras("M00000_dropout") == (False, 0.0, 1)


def test_parse_run_id_handles_supervision_namespaced_ids():
    """advisor.md D32: campaign run_ids carry the supervision level
    (`M00000_SUP_s0`). The 5-bit prefix still drives the replay
    architecture, and `model_id` keeps the full string so the parquet
    filename stays unambiguous -- same convention as the `_pbwm` suffix."""
    assert _parse_run_id("M00000_SUP_s0") == ("M00000_SUP", 0, 0, 0, 0, 0, 0)
    assert _parse_run_id("M11111_RL_s7") == ("M11111_RL", 1, 1, 1, 1, 1, 7)
    assert _parse_run_id("M01L_RL_s2") == ("M01L_RL", 0, 1, 0, 0, 0, 2)


def test_activity_log_path_namespaces_non_default_checkpoints():
    """advisor.md D33: the Gate A log must not overwrite the Gate B log.
    `ckpt.pt` keeps the original filename so nothing that already reads
    these logs changes."""
    from pathlib import Path

    from brainalign_wm.training.generate_activity_logs import activity_log_path

    out = Path("/tmp/logs")
    assert activity_log_path("M00000_SUP_s0", "ckpt.pt", out) == out / "M00000_SUP_s0.parquet"
    assert activity_log_path("M00000_SUP_s0", "ckpt_at_criterion.pt", out) == out / "M00000_SUP_s0_at_criterion.parquet"
    assert activity_log_path("X", "ckpt.pt", out) != activity_log_path("X", "ckpt_at_criterion.pt", out)
