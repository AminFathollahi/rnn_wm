"""Ranking/MDD logic in scripts/rsa_encoder_partial.py (pure, no data needed)."""
import math

import pandas as pd

from scripts.rsa_encoder_partial import Z_ALPHA_POWER, add_ranks_and_mdd


def _rows():
    return [
        {"run_id": "r1", "region": "a", "partial_status": "ok", "partial_correlation": 0.5, "partial_boot_se": 0.1},
        {"run_id": "r1", "region": "b", "partial_status": "ok", "partial_correlation": 0.2, "partial_boot_se": 0.05},
        {"run_id": "r1", "region": "c", "partial_status": "unstable_covariate_collinear", "partial_correlation": float("nan"), "partial_boot_se": float("nan")},
        {"run_id": "r2", "region": "a", "partial_status": "ok", "partial_correlation": 0.1, "partial_boot_se": 0.08},
    ]


def test_ranks_ok_regions_by_partial_descending_within_run():
    df = add_ranks_and_mdd(pd.DataFrame(_rows()))
    r1 = df[df.run_id == "r1"].set_index("region")
    assert r1.loc["a", "rank"] == 1
    assert r1.loc["b", "rank"] == 2
    assert math.isnan(r1.loc["c", "rank"])  # excluded, not ranked


def test_mdd_only_defined_between_consecutive_ok_ranks():
    df = add_ranks_and_mdd(pd.DataFrame(_rows()))
    r1 = df[df.run_id == "r1"].set_index("region")
    expected = Z_ALPHA_POWER * math.hypot(0.1, 0.05)
    assert math.isclose(r1.loc["a", "mdd_to_next_rank"], expected)
    assert math.isnan(r1.loc["b", "mdd_to_next_rank"])  # last ranked region: no next rank


def test_runs_are_ranked_independently():
    df = add_ranks_and_mdd(pd.DataFrame(_rows()))
    assert df[df.run_id == "r2"].iloc[0]["rank"] == 1
