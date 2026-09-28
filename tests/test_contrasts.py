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


def test_equal_patient_and_equal_session_estimands_remain_distinct():
    runs, sessions = _synthetic_tables({("M00000", "SUP"): 0.0}, n_seeds=1, n_patients=2, noise=0.0)
    sessions.loc[sessions["patient"].eq("P0"), "raw_alignment"] = 0.0
    sessions.loc[sessions["patient"].eq("P1"), "raw_alignment"] = 1.0
    panels = build_panels(runs, sessions)
    equal_session = panels["maintenance_signed_alignment"]
    equal_patient = panels["maintenance_signed_alignment_equal_patient"]
    assert equal_session.point_values()[0] == pytest.approx(1 / 3)
    assert equal_patient.point_values()[0] == pytest.approx(1 / 2)
    assert equal_patient.runs.iloc[0]["n_sessions"] == 3
    assert equal_patient.runs.iloc[0]["n_patients"] == 2


def test_equal_patient_bootstrap_matches_its_point_weights():
    runs, sessions = _synthetic_tables({("M00000", "SUP"): 0.0}, n_seeds=1, n_patients=2, noise=0.0)
    sessions.loc[sessions["patient"].eq("P0"), "raw_alignment"] = 0.0
    sessions.loc[sessions["patient"].eq("P1"), "raw_alignment"] = 1.0
    panel = build_panels(runs, sessions)["maintenance_signed_alignment_equal_patient"]
    boot = panel.bootstrap_values(20000, np.random.default_rng(0))
    assert boot.mean() == pytest.approx(panel.point_values()[0], abs=0.01)


def test_knock_out_cell_complements_add_one_cell():
    for arm in "SMPTD":
        assert add_one_cell(arm).count("1") == 1
        assert knock_out_cell(arm).count("0") == 1
        assert add_one_cell(arm)[1:].index("1") == knock_out_cell(arm)[1:].index("0")


def test_arm_count_reads_the_bits_of_a_tagged_control_variant():
    assert enabled_arm_count("M10010") == 2
    assert enabled_arm_count("M10010_lowrank") == 2
    assert enabled_arm_count("M00000_idcatch") == 0
    with pytest.raises(ValueError):
        enabled_arm_count("M10L")


def test_only_untagged_battery_cells_match_the_core_pattern():
    assert CORE_CELL.match("M11111")
    assert not CORE_CELL.match("M11111_energy")
    assert not CORE_CELL.match("M10L")


def _population_tables(worker=0.30, manager=-0.10, flat=0.10, n_seeds=4):
    """A flat cell and a hierarchical cell, the latter scored once per
    population, with an extra row for each population wide enough to be
    subsampled down to 24 units."""
    sessions = [("s0", "P0"), ("s1", "P1"), ("s2", "P2")]
    runs, per_session = [], []
    for cell, values in (
        ("M00000", {"all": flat}),
        ("M00010", {"all": flat + 0.02}),
        ("M10000", {"worker": worker, "manager": manager, "all": 0.5}),
    ):
        for seed in range(n_seeds):
            run_id = f"{cell}_SUP_s{seed}"
            runs.append(dict(
                run_id=run_id, model_id=cell, S=int(cell[1]), M=0, P=0, T=0, D=0, seed=seed,
                maintenance_signed_raw_alignment=0.0, probe_raw_alignment=0.0,
                accuracy_load1=0.95, accuracy_load3=0.90,
            ))
            for subpop, value in values.items():
                for session, patient in sessions:
                    per_session.append(dict(
                        run_id=run_id, session=session, region="MTL", subpop=subpop,
                        patient=patient, raw_alignment=value, S=int(cell[1]),
                        match_target_units=float("nan"),
                    ))
                    if subpop != "manager":
                        per_session.append(dict(
                            run_id=run_id, session=session, region="MTL", subpop=subpop,
                            patient=patient, raw_alignment=value - 0.05, S=int(cell[1]),
                            match_target_units=24.0,
                        ))
    return pd.DataFrame(runs), pd.DataFrame(per_session)


def test_each_population_is_paired_against_the_flat_run_separately():
    runs, sessions = _population_tables()
    rng = np.random.default_rng(0)
    effects = {}
    for population in ("worker", "manager"):
        panel = build_panels(runs, sessions, region="MTL", population=population)[
            f"maintenance_signed_alignment__MTL__{population}"
        ]
        boot = panel.bootstrap_values(200, rng)
        effects[population] = paired_seed_contrast(
            panel, boot, {"M10000": 1.0, "M00000": -1.0}, "SUP", rng
        )
    assert effects["worker"]["effect"] == pytest.approx(0.20)
    assert effects["manager"]["effect"] == pytest.approx(-0.20)
    assert effects["worker"]["hierarchical_cells"] == "M10000"


def test_a_contrast_between_two_flat_cells_names_no_hierarchical_cell():
    runs, sessions = _population_tables()
    panel = build_panels(runs, sessions, region="MTL", population="worker")[
        "maintenance_signed_alignment__MTL__worker"
    ]
    rng = np.random.default_rng(0)
    result = paired_seed_contrast(
        panel, panel.bootstrap_values(50, rng), {"M00010": 1.0, "M00000": -1.0}, "SUP", rng
    )
    assert result["hierarchical_cells"] == ""
    assert result["effect"] == pytest.approx(0.02)


def test_common_width_slice_prefers_the_subsampled_row_where_one_exists():
    runs, sessions = _population_tables()
    native = build_panels(runs, sessions, region="MTL", population="worker")[
        "maintenance_signed_alignment__MTL__worker"
    ]
    matched = build_panels(runs, sessions, region="MTL", population="worker", match_units="24")[
        "maintenance_signed_alignment__MTL__worker"
    ]
    assert native.runs.loc["M10000_SUP_s0", "value"] == pytest.approx(0.30)
    assert matched.runs.loc["M10000_SUP_s0", "value"] == pytest.approx(0.25)
    # The manager is already at the target width, so its own row stands in.
    manager = build_panels(runs, sessions, region="MTL", population="manager", match_units="24")[
        "maintenance_signed_alignment__MTL__manager"
    ]
    assert manager.runs.loc["M10000_SUP_s0", "value"] == pytest.approx(-0.10)


def test_a_population_absent_from_a_table_yields_no_rows_rather_than_flat_ones():
    runs, sessions = _population_tables()
    sessions = sessions[sessions["subpop"] != "worker"]
    panel = build_panels(runs, sessions, region="MTL", population="worker")[
        "maintenance_signed_alignment__MTL__worker"
    ]
    assert "M10000_SUP_s0" not in panel.runs.index
    assert "M00000_SUP_s0" in panel.runs.index
