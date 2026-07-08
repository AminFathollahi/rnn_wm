import numpy as np
import pandas as pd

from brainalign_wm.analysis.dv_relationship import compute_correlations, mixed_effects_fit


def _planted_correlation_table(rng: np.random.RandomState, n_cells: int = 12, n_seeds: int = 5) -> pd.DataFrame:
    rows = []
    for cell_idx in range(n_cells):
        base_accuracy = cell_idx / n_cells  # spreads accuracy across cells
        for seed in range(n_seeds):
            accuracy = base_accuracy + rng.normal(0, 0.02)
            rsa_alignment = 0.8 * accuracy + rng.normal(0, 0.02)  # planted positive correlation
            modularity_q = -0.6 * accuracy + rng.normal(0, 0.02)  # planted negative correlation
            rows.append({
                "cell": f"cell{cell_idx}", "seed": seed, "S": cell_idx % 2, "M": 0, "P": 0, "T": 0, "D": 0,
                "accuracy": accuracy, "rsa_alignment": rsa_alignment, "modularity_q": modularity_q,
                "small_worldness": np.nan, "mixed_selectivity": np.nan, "persistence_index": np.nan,
            })
    return pd.DataFrame(rows)


def test_compute_correlations_recovers_planted_sign_and_direction():
    rng = np.random.RandomState(0)
    df = _planted_correlation_table(rng)
    correlations = compute_correlations(df)

    pos = correlations["accuracy<->rsa_alignment"]
    assert pos is not None
    assert pos["pearson_r"] > 0.5
    assert pos["pearson_ci"][0] > 0, "CI for a strong planted positive correlation should exclude 0"

    neg = correlations["accuracy<->modularity_q"]
    assert neg is not None
    assert neg["pearson_r"] < -0.5
    assert neg["pearson_ci"][1] < 0, "CI for a strong planted negative correlation should exclude 0"


def test_compute_correlations_returns_none_for_all_nan_pair():
    rng = np.random.RandomState(0)
    df = _planted_correlation_table(rng)
    result = compute_correlations(df)["accuracy<->small_worldness"]
    assert result is None


def test_mixed_effects_fit_recovers_positive_accuracy_coefficient():
    rng = np.random.RandomState(1)
    df = _planted_correlation_table(rng)
    fit = mixed_effects_fit(df)
    assert fit is not None
    assert fit["params"]["accuracy"] > 0
