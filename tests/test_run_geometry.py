"""Phase 12.4 (comments.txt §12.4): `--checkpoint NAME` must let
`scripts/run_geometry.py` analyse `ckpt_at_criterion.pt` as well as the
default `ckpt.pt`, writing each to its OWN output file so one doesn't
silently overwrite the other."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_geometry import _build_single_trial_ensemble, out_csv_for


def test_default_checkpoint_keeps_original_filename():
    assert out_csv_for("ckpt.pt") == ROOT / "results" / "geometry_results.csv"


def test_at_criterion_checkpoint_gets_a_distinct_filename():
    at_criterion = out_csv_for("ckpt_at_criterion.pt")
    assert at_criterion != out_csv_for("ckpt.pt")
    assert at_criterion == ROOT / "results" / "geometry_results_ckpt_at_criterion.csv"


def test_single_trial_ensemble_supports_hierarchical_joint_state():
    rows = []
    for trial_id in (0, 1):
        for t in (0, 1):
            rows.append({
                "trial_id": trial_id, "t": t, "epoch": "maintain", "h_flat": None,
                "h_worker": np.array([trial_id, t], dtype=float),
                "h_manager": np.array([trial_id + t], dtype=float),
            })
    ensemble = _build_single_trial_ensemble(pd.DataFrame(rows))
    assert ensemble.shape == (2, 2, 3)
    np.testing.assert_array_equal(ensemble[1, 1], [1.0, 1.0, 2.0])
