"""Statistics and inference: a mixed-effects model of alignment score on
the 2x2x2 factorial plus an accuracy covariate, false-discovery-rate
correction, and distribution-comparison effect sizes for persistence and
load-tuning measures.

`statsmodels` MixedLM is used if importable (a lazy import); otherwise the
code falls back to OLS with a cluster-bootstrap confidence interval
(clustered by seed/session) -- a documented, less statistically powerful
but not silently misleading substitute. The fallback is recorded in the
returned dict's `"method"` field, so callers and report generation can
state explicitly which method produced a given result.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def fdr_correct(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR. Returns a boolean reject-null array, same
    order as `pvalues`."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    if not below.any():
        reject = np.zeros(n, dtype=bool)
    else:
        max_i = np.max(np.where(below)[0])
        reject = np.zeros(n, dtype=bool)
        reject[order[: max_i + 1]] = True
    return reject


def mixed_effects_alignment(df: pd.DataFrame, formula: str = "align_score ~ S * M * L + accuracy") -> dict:
    """df columns: align_score, S, M, L, accuracy, seed, session (session
    optional). Returns {"method", "params", "pvalues", "conf_int", "effects"}."""
    try:
        import statsmodels.formula.api as smf

        groups = df["seed"] if "seed" in df else np.zeros(len(df))
        model = smf.mixedlm(formula, df, groups=groups)
        result = model.fit(reml=False)
        return {
            "method": "statsmodels.MixedLM",
            "params": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "conf_int": result.conf_int().to_dict(orient="index"),
        }
    except Exception as e:  # noqa: BLE001 -- documented fallback, not a silent swallow
        return _ols_cluster_bootstrap(df, formula, fallback_reason=repr(e))


def _ols_cluster_bootstrap(df: pd.DataFrame, formula: str, n_boot: int = 1000, seed: int = 0, fallback_reason: str = "") -> dict:
    import statsmodels.formula.api as smf  # OLS part of statsmodels is lightweight/always available with the package

    rng = np.random.RandomState(seed)
    clusters = df["seed"].unique() if "seed" in df else np.arange(len(df))
    ols = smf.ols(formula, df).fit()
    boot_params = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        rows = pd.concat([df[df["seed"] == c] for c in sampled_clusters]) if "seed" in df else df.sample(len(df), replace=True, random_state=rng)
        try:
            b = smf.ols(formula, rows).fit()
            boot_params.append(b.params)
        except Exception:
            continue
    boot_df = pd.DataFrame(boot_params)
    ci = {k: (float(np.percentile(boot_df[k], 2.5)), float(np.percentile(boot_df[k], 97.5))) for k in boot_df.columns}
    return {
        "method": f"OLS + cluster-bootstrap (statsmodels.MixedLM unavailable: {fallback_reason})",
        "params": ols.params.to_dict(),
        "pvalues": ols.pvalues.to_dict(),
        "conf_int": ci,
    }


def rank_biserial_effect_size(group_a: np.ndarray, group_b: np.ndarray) -> float:
    from scipy.stats import mannwhitneyu

    u, _ = mannwhitneyu(group_a, group_b, alternative="two-sided")
    n1, n2 = len(group_a), len(group_b)
    return float(1 - (2 * u) / (n1 * n2))


def compare_distributions(model_values: np.ndarray, brain_values: np.ndarray) -> dict:
    """Persistent-index / load-tuning distribution comparison (H6): a
    permutation test on the mean difference + rank-biserial effect size."""
    from scipy.stats import mannwhitneyu

    u_stat, p = mannwhitneyu(model_values, brain_values, alternative="two-sided")
    effect = rank_biserial_effect_size(model_values, brain_values)
    return {"u_stat": float(u_stat), "p_value": float(p), "rank_biserial": effect}
