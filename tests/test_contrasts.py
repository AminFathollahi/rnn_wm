import json

import numpy as np
import pandas as pd
import pytest

from brainalign_wm.analysis.contrasts import (
    CORE_CELL,
    add_one_cell,
    assign_patient_folds,
    build_panels,
    enabled_arm_count,
    knock_out_cell,
    paired_seed_contrast,
    training_signal,
    write_patient_folds,
)

PATIENTS = [f"P{i:02d}CS" for i in range(57)]


def test_patient_folds_are_deterministic_disjoint_and_exhaustive():
    first = assign_patient_folds(PATIENTS)
    second = assign_patient_folds(list(reversed(PATIENTS)))
    assert first == second
    discovery = {p for p, f in first.items() if f == "discovery"}
    confirmation = {p for p, f in first.items() if f == "confirmation"}
    assert not discovery & confirmation
    assert discovery | confirmation == set(PATIENTS)
    assert len(confirmation) == round(len(PATIENTS) / 3)


def test_written_patient_folds_round_trip(tmp_path):
    path = tmp_path / "folds.json"
    payload = write_patient_folds(PATIENTS, path)
    on_disk = json.loads(path.read_text())
    assert on_disk["folds"] == payload["folds"] == assign_patient_folds(PATIENTS)
    assert on_disk["n_discovery"] + on_disk["n_confirmation"] == len(PATIENTS)


def _synthetic_tables(cell_offsets: dict, n_seeds: int = 8, n_patients: int = 6, noise: float = 0.01):
    """One run per (cell, signal, seed); two sessions for the first patient so
    that sessions are genuinely nested within patients."""
    rng = np.random.default_rng(0)
    sessions = [(f"s{p}", f"P{p}") for p in range(n_patients)] + [("s0b", "P0")]
    runs, per_session = [], []
    for (cell, signal), offset in cell_offsets.items():
        for seed in range(n_seeds):
            run_id = f"{cell}_{signal}_s{seed}"
            values = offset + rng.normal(0.0, noise, len(sessions))
            runs.append(
                dict(
                    run_id=run_id,
                    model_id=cell,
                    S=int(cell[1]), M=int(cell[2]), P=int(cell[3]),
                    T=int(cell[4]), D=int(cell[5]),
                    seed=seed,
                    maintenance_signed_raw_alignment=float(values.mean()),
                    probe_raw_alignment=float(values.mean()) + 0.5,
                    accuracy_load1=0.95,
                    accuracy_load3=0.90,
                )
            )
            for (session, patient), value in zip(sessions, values):
                per_session.append(
                    dict(
                        run_id=run_id, session=session, region="pooled", subpop="all",
                        patient=patient, raw_alignment=float(value),
                    )
                )
    return pd.DataFrame(runs), pd.DataFrame(per_session)


def test_contrast_recovers_a_planted_difference():
    planted = 0.10
    runs, sessions = _synthetic_tables(
        {("M00000", "SUP"): 0.20, (add_one_cell("S"), "SUP"): 0.20 + planted}
    )
    panel = build_panels(runs, sessions)["maintenance_signed_alignment"]
    assert panel.cluster_unit == "patient"
    rng = np.random.default_rng(0)
    boot = panel.bootstrap_values(400, rng)
    result = paired_seed_contrast(
        panel, boot, {add_one_cell("S"): 1.0, "M00000": -1.0}, "SUP", rng
    )
    assert result["n_seeds"] == 8
    assert result["effect"] == pytest.approx(planted, abs=0.01)
    assert result["ci_lo"] < planted < result["ci_hi"]
    assert result["mde"] > 0.0


def test_supervised_and_reinforcement_runs_are_never_pooled():
    """Opposite planted effects under the two signals: a pooled estimate would
    cancel to zero, and each per-signal estimate must recover its own sign."""
    runs, sessions = _synthetic_tables(
        {
            ("M00000", "SUP"): 0.20,
            (add_one_cell("S"), "SUP"): 0.30,
            ("M00000", "RL"): 0.20,
            (add_one_cell("S"), "RL"): 0.10,
        }
    )
    panel = build_panels(runs, sessions)["maintenance_signed_alignment"]
    rng = np.random.default_rng(0)
    boot = panel.bootstrap_values(400, rng)
    weights = {add_one_cell("S"): 1.0, "M00000": -1.0}
    supervised = paired_seed_contrast(panel, boot, weights, "SUP", rng)
    reinforcement = paired_seed_contrast(panel, boot, weights, "RL", rng)
    assert supervised["effect"] == pytest.approx(0.10, abs=0.01)
    assert reinforcement["effect"] == pytest.approx(-0.10, abs=0.01)
    assert supervised["n_seeds"] == reinforcement["n_seeds"] == 8
    assert training_signal("M10010_SUP_s3") == "SUP"


def test_fold_restriction_uses_only_the_requested_patients():
    runs, sessions = _synthetic_tables({("M00000", "SUP"): 0.20}, noise=0.0)
    sessions.loc[sessions["patient"] == "P3", "raw_alignment"] = 1.0
    folds = {p: ("confirmation" if p == "P3" else "discovery") for p in sessions["patient"].unique()}
    discovery = {p for p, f in folds.items() if f == "discovery"}
    panel = build_panels(runs, sessions, discovery)["maintenance_signed_alignment"]
    assert np.allclose(panel.point_values(), 0.20)
    assert "P3" not in set(panel.session_patient)


def test_knock_out_cell_complements_add_one_cell():
    for arm in "SMPTD":
        assert add_one_cell(arm).count("1") == 1
        assert knock_out_cell(arm).count("0") == 1
        assert add_one_cell(arm)[1:].index("1") == knock_out_cell(arm)[1:].index("0")


def test_arm_count_reads_the_bits_of_a_tagged_control_variant():
    assert enabled_arm_count("M10010") == 2
    assert enabled_arm_count("M10010_w128") == 2
    assert enabled_arm_count("M00000_idcatch") == 0
    with pytest.raises(ValueError):
        enabled_arm_count("M10L")


def test_only_untagged_battery_cells_match_the_core_pattern():
    assert CORE_CELL.match("M11111")
    assert not CORE_CELL.match("M11111_energy")
    assert not CORE_CELL.match("M10L")
