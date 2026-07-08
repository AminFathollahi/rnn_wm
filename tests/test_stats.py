import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from brainalign_wm.analysis.stats import accuracy_vif, bayes_factor_bic, mixed_effects_alignment


def test_bayes_factor_bic_favors_alternative_for_clear_difference():
    rng = np.random.RandomState(0)
    a = rng.normal(0.0, 0.05, size=20)
    b = rng.normal(2.0, 0.05, size=20)
    bf10 = bayes_factor_bic(a, b)
    assert bf10 > 10  # a huge, low-noise mean separation should give strong evidence for a difference


def test_bayes_factor_bic_favors_null_for_identical_groups():
    rng = np.random.RandomState(1)
    a = rng.normal(0.5, 0.2, size=40)
    b = rng.normal(0.5, 0.2, size=40)
    bf10 = bayes_factor_bic(a, b)
    assert bf10 < 1  # same generating distribution -> evidence should lean toward "no difference"


def test_bayes_factor_bic_is_symmetric_in_argument_order():
    rng = np.random.RandomState(1)
    a = rng.normal(0.0, 1.0, size=10)
    b = rng.normal(1.0, 1.0, size=10)
    assert np.isclose(bayes_factor_bic(a, b), bayes_factor_bic(b, a))


def _synthetic_alignment_frame(rng: np.random.RandomState, n_seeds: int = 6) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        for S in (0, 1):
            rows.append({
                "align_score": 0.3 * S + rng.normal(0.5, 0.05),
                "S": S, "M": 0, "P": 0,
                "accuracy": rng.normal(0.7, 0.05), "seed": seed,
            })
    return pd.DataFrame(rows)


def test_mixed_effects_alignment_recovers_known_positive_effect():
    rng = np.random.RandomState(0)
    df = _synthetic_alignment_frame(rng)
    res = mixed_effects_alignment(df, formula="align_score ~ S + accuracy")
    ci = res["conf_int"]["S"]  # {0: lower, 1: upper}, per DataFrame.to_dict(orient="index")
    assert ci[0] > 0, f"a strong planted positive S effect should give a CI excluding 0, got {ci}"


def test_mixed_effects_alignment_ols_fallback_gives_finite_estimates(monkeypatch):
    """Forces the statsmodels.MixedLM path to fail (rather than engineering
    a data pathology that may or may not trip it) so the fallback branch
    is exercised deterministically."""
    def _raise(*args, **kwargs):
        raise RuntimeError("forced MixedLM failure for test")

    monkeypatch.setattr(smf, "mixedlm", _raise)
    rng = np.random.RandomState(1)
    df = _synthetic_alignment_frame(rng)
    res = mixed_effects_alignment(df, formula="align_score ~ S + accuracy")
    assert res["method"].startswith("OLS + cluster-bootstrap")
    assert all(np.isfinite(v) for v in res["params"].values())
    for lo, hi in res["conf_int"].values():
        assert np.isfinite(lo) and np.isfinite(hi)


def test_accuracy_vif_high_for_collinear_low_for_orthogonal():
    rng = np.random.RandomState(0)
    n = 40
    S = rng.randint(0, 2, n)
    M = rng.randint(0, 2, n)
    P = rng.randint(0, 2, n)

    collinear = pd.DataFrame({
        "S": S, "M": M, "P": P,
        "accuracy": 0.5 * S + 0.3 * M + 0.2 * P + rng.normal(0, 0.001, n),
    })
    orthogonal = pd.DataFrame({"S": S, "M": M, "P": P, "accuracy": rng.normal(0.7, 0.1, n)})

    assert accuracy_vif(collinear) > 20
    assert accuracy_vif(orthogonal) < 2.0
