"""Regression tests for the 2026-07-06 PI review pass (comments.txt N1/N2):

- N1: a load-only synthetic RDM (item identity carries NO true signal,
  only load structures the data) must NOT drive maintenance alignment once
  the crossnobis RDM is load-stratified (`rdm.stratified_crossnobis_rdm`) --
  unlike the pre-fix, non-stratified, single pooled-RDM approach, which is
  dominated by cross-load condition pairs and reports a large spurious
  correlation whenever model and neural data merely share load structure.
- N2: a null (label-shuffled / symmetric-noise) maintenance DV must NOT
  report a positive floor once per-session raw alignment is aggregated as
  a signed mean BEFORE normalizing, instead of averaging each session's
  already-[0,1]-clipped `normalized_alignment`.
"""
from __future__ import annotations

import numpy as np
import pytest

from brainalign_wm.analysis.rdm import crossnobis_rdm, stratified_crossnobis_rdm
from brainalign_wm.analysis.rsa import compare_rdms
from brainalign_wm.analysis.run_all import _aggregate_maintenance


def _load_only_conditions(loads, items_per_load, reps):
    """Conditions where LOAD is the only structured factor and item
    identity is assigned arbitrarily (uninformative for the RDM's true
    generative process below) -- `conds_full` is the OLD `(item, load)`
    schema; `strata`/`conds_only` split that into load-stratum + bare item
    label for the N1-fixed stratified path."""
    conds_full, strata, conds_only = [], [], []
    for load in loads:
        for item in range(items_per_load):
            for _ in range(reps):
                conds_full.append((item, load))
                strata.append(load)
                conds_only.append(item)
    return conds_full, strata, conds_only


def test_load_only_signal_inflates_nonstratified_but_not_stratified_alignment():
    """N1: build two INDEPENDENT synthetic datasets ("model" and "neural")
    whose only true structure is a large, shared LOAD offset -- item
    identity is pure independent noise on both sides, so there is NO real
    identity-driven signal for either side to detect. The pre-fix
    (non-stratified, pooled-load) crossnobis RDM should still show a large
    SPURIOUS correlation (both RDMs' off-diagonals are dominated by the
    same cross-load distances), while the post-fix (load-stratified) RDM,
    which never computes a cross-load pair at all, should show
    approximately zero correlation -- there is no real per-load identity
    signal for it to find."""
    loads = [1, 2, 3]
    conds_full, strata, conds_only = _load_only_conditions(loads, items_per_load=5, reps=4)
    n = len(conds_full)
    load_arr = np.array([c[1] for c in conds_full], dtype=float)

    nonstrat_vals, strat_vals = [], []
    for trial_seed in range(8):
        rng = np.random.RandomState(trial_seed)
        # Both sides share the SAME load-driven macrostructure (offset
        # 5.0x load) but independent noise and independent (uninformative)
        # item-to-trial assignment order -- item identity carries zero
        # true cross-dataset signal.
        model_data = rng.randn(n, 10) * 0.5 + load_arr[:, None] * 5.0
        neural_data = rng.randn(n, 10) * 0.5 + load_arr[:, None] * 5.0

        model_rdm, model_conds = crossnobis_rdm(model_data, conds_full, n_folds=2, seed=0)
        neural_rdm, neural_conds = crossnobis_rdm(neural_data, conds_full, n_folds=2, seed=1)
        assert model_conds == neural_conds
        nonstrat_vals.append(compare_rdms(model_rdm, neural_rdm))

        model_rdm_s, model_conds_s = stratified_crossnobis_rdm(model_data, conds_only, strata, n_folds=2, seed=0)
        neural_rdm_s, neural_conds_s = stratified_crossnobis_rdm(neural_data, conds_only, strata, n_folds=2, seed=1)
        assert model_conds_s == neural_conds_s
        strat_vals.append(compare_rdms(model_rdm_s, neural_rdm_s))

    mean_nonstrat = float(np.mean(nonstrat_vals))
    mean_strat = float(np.mean(strat_vals))
    assert mean_nonstrat > 0.5, (
        f"sanity check on the fixture itself: pooled-load RDMs sharing load structure should show a large "
        f"spurious correlation pre-fix (got {mean_nonstrat:.3f})"
    )
    assert abs(mean_strat) < 0.25, (
        f"load-stratified RDMs must NOT be driven by shared load structure alone when item identity carries "
        f"no true signal (got mean stratified corr {mean_strat:.3f}, vs. non-stratified {mean_nonstrat:.3f})"
    )
    assert mean_strat < mean_nonstrat - 0.4, "stratification must materially reduce the load-driven inflation"


def test_stratified_rdm_has_nan_cross_stratum_blocks():
    """N1: `stratified_crossnobis_rdm` must never compute a cross-stratum
    (cross-load) condition pair -- those entries stay NaN, structurally,
    not merely small."""
    loads = [1, 2]
    conds_full, strata, conds_only = _load_only_conditions(loads, items_per_load=3, reps=3)
    rng = np.random.RandomState(0)
    data = rng.randn(len(conds_full), 6) + np.array([c[1] for c in conds_full])[:, None] * 3.0
    rdm, conds = stratified_crossnobis_rdm(data, conds_only, strata, n_folds=2, seed=0)
    assert rdm is not None
    strata_of_conds = [c[0] for c in conds]
    cross_stratum_is_nan = all(
        np.isnan(rdm[i, j])
        for i in range(len(conds))
        for j in range(len(conds))
        if strata_of_conds[i] != strata_of_conds[j]
    )
    assert cross_stratum_is_nan, "every cross-stratum (cross-load) entry must be NaN, never a computed distance"
    within_stratum_has_values = any(
        not np.isnan(rdm[i, j]) and i != j
        for i in range(len(conds))
        for j in range(len(conds))
        if strata_of_conds[i] == strata_of_conds[j]
    )
    assert within_stratum_has_values, "within-stratum entries should still be real computed distances"


def test_signed_aggregation_does_not_rectify_symmetric_noise_into_a_positive_floor():
    """N2: per-session raw alignment that is pure, SIGNED (mean ~0) noise
    must aggregate to ~0 normalized alignment under the fixed pipeline --
    not the strictly positive floor the old (clip-per-session-then-average)
    aggregation produced. Session raw values are symmetric around zero by
    construction (a null/label-shuffled DV should look like this)."""
    ceiling = 0.5
    raws = [-0.10, 0.08, -0.05, 0.12, -0.09, 0.03, -0.02, 0.06, -0.11, 0.10]
    assert abs(np.mean(raws)) < 0.02, "fixture sanity: raws should be ~symmetric around zero"

    rows_ok = [{"raw_alignment": r, "noise_ceiling_upper": ceiling} for r in raws]
    agg = _aggregate_maintenance(rows_ok)

    # The OLD (buggy) aggregation: clip each session's raw/ceiling ratio to
    # [0,1] first, THEN average -- rectifies the symmetric noise into a
    # strictly positive number.
    old_style_normalized = float(np.mean([np.clip(r / ceiling, 0.0, 1.0) for r in raws]))

    assert agg["maintenance_signed_raw_alignment"] == pytest.approx(np.mean(raws))
    assert agg["maintenance_normalized_alignment"] < 0.03, (
        f"signed-then-normalize aggregation of symmetric noise should sit near zero, "
        f"got {agg['maintenance_normalized_alignment']:.4f}"
    )
    assert old_style_normalized > 0.05, (
        "fixture sanity: the OLD clip-then-average aggregation should show the spurious positive floor "
        f"this test guards against (got {old_style_normalized:.4f})"
    )
    assert agg["maintenance_normalized_alignment"] < old_style_normalized, (
        "fixed aggregation must report a lower (less rectified) normalized alignment than the old, buggy one"
    )
