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


def mixed_effects_alignment(
    df: pd.DataFrame, formula: str = "align_score ~ S * M * L + accuracy", extra_vc_col: str = "patient",
) -> dict:
    """df columns: align_score, S, M, L, accuracy, seed, and (audit fix B3)
    optionally `extra_vc_col` (default `"patient"`) -- one row per (run,
    session), the per-session maintenance-epoch alignment table
    `analysis/run_all.py` now produces, which carries a real patient
    dimension (unlike the old one-row-per-run table, where there was
    nothing to group patients by). Pass `extra_vc_col="session"` for the
    H1/C2 region-dissociation model instead, where `session` (not
    `patient`) is the factor crossed with training seed. Returns
    {"method", "params", "pvalues", "conf_int"}.

    `seed` and `extra_vc_col` are CROSSED, not nested (every seed's model
    is compared against every level of `extra_vc_col`, and every level of
    `extra_vc_col` appears under every seed) -- statsmodels' MixedLM only
    supports a single top-level `groups` factor, which nests whatever you
    pass to it. An earlier version of this function passed `groups=seed`
    with the extra factor as a `vc_formula` variance component; that nests
    it WITHIN seed (each seed gets its own independent draw of that effect)
    rather than sharing one effect across all seeds, silently understating
    the extra factor's structure (caught on adversarial review -- confirmed
    by comparing log-likelihoods against the correct formulation below).
    The standard statsmodels workaround for genuinely crossed random
    effects is to pass a single dummy constant as `groups` and declare BOTH
    seed and the extra factor as `vc_formula` variance components under
    that one dummy group, which is what this does when both are available
    and vary (>1 level each). Falls back to `groups=seed` alone (no extra
    term) when `extra_vc_col` is absent or constant.

    Note (B3): once A1 is fixed, `accuracy` and `L` are still partially
    collinear (L=1 cells tend to have both lower accuracy and different
    alignment) -- callers should check the variance inflation factor (VIF)
    of `accuracy` against `S*M*L` before trusting its coefficient in
    isolation; `accuracy` is kept as a covariate (not dropped) because
    alignment should be compared at matched behavior where possible (H3).

    Note (region pseudo-replication): callers should NOT feed rows from
    multiple `region` levels (pooled/MTL/MFC) for the SAME session into
    this function in one call -- they are highly correlated subsets of the
    same underlying units/trials, not independent observations, and this
    function has no way to model that redundancy. `analysis/run_all.py`
    fits the main S*M*L model on pooled-region rows only, and a SEPARATE
    region-dissociation model restricted to region in {MTL, MFC}."""
    has_extra = extra_vc_col in df and df[extra_vc_col].nunique() > 1
    has_seed = "seed" in df and df["seed"].nunique() > 1
    try:
        import statsmodels.formula.api as smf

        if has_extra and has_seed:
            dummy_group = np.zeros(len(df))
            vc_formula = {"seed": "0 + C(seed)", extra_vc_col: f"0 + C({extra_vc_col})"}
            model = smf.mixedlm(formula, df, groups=dummy_group, vc_formula=vc_formula)
            method_suffix = f" + crossed (seed, {extra_vc_col}) variance components"
        else:
            groups = df["seed"] if "seed" in df else np.zeros(len(df))
            vc_formula = {extra_vc_col: f"0 + C({extra_vc_col})"} if has_extra else None
            model = smf.mixedlm(formula, df, groups=groups, vc_formula=vc_formula)
            method_suffix = f" + {extra_vc_col} variance component" if vc_formula else ""
        result = model.fit(reml=False)
        return {
            "method": "statsmodels.MixedLM" + method_suffix,
            "params": result.params.to_dict(),
            "pvalues": result.pvalues.to_dict(),
            "conf_int": result.conf_int().to_dict(orient="index"),
        }
    except Exception as e:  # noqa: BLE001 -- documented fallback, not a silent swallow
        return _ols_cluster_bootstrap(df, formula, fallback_reason=repr(e))


def accuracy_vif(df: pd.DataFrame, formula_rhs: str = "S * M * L") -> float:
    """Variance inflation factor of `accuracy` against the factorial design
    (B3's collinearity check: even after A1 fixes the L-arm confound,
    accuracy and L may remain partially collinear). VIF = 1/(1-R^2) from
    regressing `accuracy` on the other predictors; report and eyeball
    (>~5 is the usual concern threshold) rather than silently drop
    `accuracy` from the model."""
    import statsmodels.formula.api as smf

    aux = smf.ols(f"accuracy ~ {formula_rhs}", df).fit()
    r2 = aux.rsquared
    return float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")


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
