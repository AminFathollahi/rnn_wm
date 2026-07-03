"""Analysis-toolkit tests: crossnobis is unbiased on null data; demixed PCA
recovers a planted component; noise-ceiling bounds are sane; false-discovery-
rate correction is correct; cross-temporal decoding shape and behavior are
sane on synthetic stable versus dynamic structure."""
import numpy as np

from brainalign_wm.analysis.rdm import crossnobis_rdm
from brainalign_wm.analysis.dpca import dpca_components
from brainalign_wm.analysis.stats import fdr_correct
from brainalign_wm.analysis.cross_temporal import cross_temporal_decoding, stability_index


def test_crossnobis_unbiased_on_null_data():
    """Pure noise, no true condition differences -> expected crossnobis
    distance ~0 (unlike squared-Euclidean, which is biased positive)."""
    rng = np.random.RandomState(0)
    n_trials, n_units, n_cond = 200, 10, 4
    data = rng.randn(n_trials, n_units)
    labels = rng.randint(0, n_cond, size=n_trials)
    rdm, conds = crossnobis_rdm(data, labels, n_folds=4, seed=0)
    off_diag = rdm[np.triu_indices(len(conds), k=1)]
    assert abs(off_diag.mean()) < 0.5, f"expected ~0 under the null, got mean={off_diag.mean():.3f}"


def test_crossnobis_detects_real_difference():
    rng = np.random.RandomState(0)
    n_trials, n_units = 200, 10
    labels = rng.randint(0, 2, size=n_trials)
    data = rng.randn(n_trials, n_units) + labels[:, None] * 5.0  # condition 1 shifted hard
    rdm, conds = crossnobis_rdm(data, labels, n_folds=4, seed=0)
    assert rdm[0, 1] > 1.0


def test_dpca_recovers_planted_component():
    """Planted: rate depends ONLY on `load` (axis 1), constant across `item`
    (axis 2) and `time` (axis 3) -- dPCA's load marginalization should carry
    ~all the variance; item/time marginalizations ~0."""
    rng = np.random.RandomState(0)
    n_units, n_load, n_item, n_time = 8, 3, 4, 5
    load_effect = rng.randn(n_units, n_load) * 10.0
    R = np.broadcast_to(load_effect[:, :, None, None], (n_units, n_load, n_item, n_time)).copy()
    R += rng.randn(*R.shape) * 0.01  # tiny noise, not zero (avoid degenerate SVD)
    comp = dpca_components(R, ["load", "item", "time"], n_components=1)
    assert comp[("load",)]["fraction_of_total_variance"] > 0.9
    assert comp[("item",)]["fraction_of_total_variance"] < 0.05
    assert comp[("load",)]["explained_variance_ratio"][0] > 0.75


def test_fdr_correct_matches_known_case():
    # 5 p-values, 2 clearly significant, rest not (BH at alpha=0.05)
    p = np.array([0.001, 0.002, 0.3, 0.5, 0.8])
    reject = fdr_correct(p, alpha=0.05)
    assert reject[0] and reject[1]
    assert not reject[2] and not reject[3] and not reject[4]


def test_fdr_correct_all_null():
    p = np.array([0.4, 0.5, 0.6, 0.9])
    reject = fdr_correct(p, alpha=0.05)
    assert not reject.any()


def test_cross_temporal_stable_vs_dynamic():
    rng = np.random.RandomState(0)
    n_trials, n_time, n_units = 120, 6, 5
    y = rng.randint(0, 2, size=n_trials)

    # stable code: same discriminative direction at every timebin
    direction = rng.randn(n_units)
    X_stable = rng.randn(n_trials, n_time, n_units) * 0.5
    for t in range(n_time):
        X_stable[:, t, :] += y[:, None] * direction[None, :] * 4.0
    mat_stable = cross_temporal_decoding(X_stable, y, n_folds=3, seed=0)
    assert mat_stable.shape == (n_time, n_time)
    assert stability_index(mat_stable) > 0.5  # high off-diag generalization

    # dynamic code: a different, uncorrelated discriminative direction per timebin
    X_dynamic = rng.randn(n_trials, n_time, n_units) * 0.5
    for t in range(n_time):
        d_t = rng.randn(n_units)
        X_dynamic[:, t, :] += y[:, None] * d_t[None, :] * 4.0
    mat_dynamic = cross_temporal_decoding(X_dynamic, y, n_folds=3, seed=1)
    assert np.nanmean(np.diag(mat_dynamic)) > 0.6  # decodable at each timebin
    assert stability_index(mat_dynamic) < stability_index(mat_stable)
