import numpy as np
import pandas as pd
import torch

import run_grid
from brainalign_wm.models.gru_cell import make_locality_mask, rewire_mask
from brainalign_wm.training.generate_activity_logs import _run_id_extras
from brainalign_wm.training.model_variants import FLAT_CONTROLS
from brainalign_wm.training.train import _build_model
from brainalign_wm.config import load_config
from scripts.analyze_flat_controls import add_design, paired_contrasts


def test_random_masks_preserve_local_mask_degrees():
    for tag in ("random128", "random289"):
        spec = FLAT_CONTROLS[tag]
        local = make_locality_mask(tuple(spec["flat_grid"]), spec["flat_density"], spec["flat_mask_seed"])
        random = rewire_mask(local, spec["flat_mask_seed"])
        assert torch.equal(local.sum(0), random.sum(0))
        assert torch.equal(local.sum(1), random.sum(1))
        assert int(torch.diagonal(random).sum()) == 0
        assert not torch.equal(local, random)


def test_flat_control_enumeration_is_complete_and_replayable():
    run_ids = set()
    for tag, spec in FLAT_CONTROLS.items():
        runs = run_grid.enumerate_runs(
            range(8), supervision="SUP", cells=["M00000", "M00010"], flat_control=tag
        )
        assert len(runs) == 16
        assert all(run["run_id"] not in run_ids for run in runs)
        run_ids.update(run["run_id"] for run in runs)
        assert all(_run_id_extras(run["model_id"])[3] == spec for run in runs)
        if spec["flat_density"]:
            assert all(run["flat_mask_edges"] > 0 and len(run["flat_mask_sha256"]) == 64 for run in runs)
    assert len(run_ids) == 80


def test_initialization_policy_is_constant_within_width():
    for width in (128, 289):
        units = {
            spec["flat_recurrent_init_units"]
            for spec in FLAT_CONTROLS.values()
            if spec["flat_units"] == width
        }
        if width == 128:
            units.add(128)
        assert units == {width}


def test_random_control_builds_the_recorded_mask():
    spec = FLAT_CONTROLS["random128"]
    cfg = load_config()
    cfg = {**cfg, "model": {**cfg["model"], **spec}}
    _, core, _ = _build_model(cfg, S=0, M=0, P=0, device="cpu")
    run = run_grid.enumerate_runs(
        [0], supervision="SUP", cells=["M00000"], flat_control="random128"
    )[0]
    from brainalign_wm.models.gru_cell import mask_digest

    assert int(core.cell.mask[:128].sum()) == run["flat_mask_edges"]
    assert mask_digest(core.cell.mask[:128]) == run["flat_mask_sha256"]


def test_flat_control_contrasts_include_interactions():
    rows = []
    for seed in (0, 1):
        for width in (128, 289):
            for connectivity, offset in (("dense", 0.0), ("random", 1.0), ("local", 2.0)):
                for topography in (0, 1):
                    tag = None if width == 128 and connectivity == "dense" else {
                        (128, "random"): "random128",
                        (128, "local"): "local128",
                        (289, "dense"): "dense289",
                        (289, "random"): "random289",
                        (289, "local"): "local289native",
                    }[(width, connectivity)]
                    model = f"M000{topography}0"
                    run_id = f"{model}{'_' + tag if tag else ''}_SUP_s{seed}"
                    rows.append({"run_id": run_id, "status": "ok", "epoch": "maintenance", "region": "MFC",
                                 "subpop": "all", "raw_alignment": width / 100 + offset + topography * width / 128})
    design = add_design(pd.DataFrame(rows))
    out = paired_contrasts(design)
    assert set(out["contrast"]) >= {
        "width_289_minus_128", "dense_minus_random", "random_minus_local",
        "topography_on_minus_off", "width_by_topography", "connectivity_by_topography",
        "width_by_connectivity",
    }
    interaction = out[(out["contrast"] == "width_by_topography") & (out["connectivity"] == "dense")]
    assert np.isclose(interaction["effect"].iloc[0], 289 / 128 - 1)
