"""Ridge-regression encoding models: model units predicting each neuron's
rate, and the reverse direction, with nested cross-validation and
noise-ceiling-normalized R^2. Complements representational similarity
analysis by providing a predictive-mapping view alongside the geometric
one."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def encoding_r2(
    X: np.ndarray, Y: np.ndarray, n_folds: int = 5, alphas: tuple = (0.1, 1.0, 10.0, 100.0), seed: int = 0
) -> np.ndarray:
    """X: [n_samples, n_source_units] (predictors); Y: [n_samples, n_target_units].
    Returns per-target-unit cross-validated R^2 (outer CV; inner CV picks
    ridge alpha via `RidgeCV`)."""
    n = X.shape[0]
    kf = KFold(n_splits=min(n_folds, n), shuffle=True, random_state=seed)
    n_targets = Y.shape[1]
    preds = np.zeros_like(Y, dtype=float)
    for train_idx, test_idx in kf.split(X):
        model = RidgeCV(alphas=alphas)
        model.fit(X[train_idx], Y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    ss_res = ((Y - preds) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / np.maximum(ss_tot, 1e-12), 0.0)
    return r2


def noise_ceiling_normalized_r2(r2: np.ndarray, ceiling_upper: float) -> np.ndarray:
    if ceiling_upper <= 0:
        return np.zeros_like(r2)
    return np.clip(r2 / ceiling_upper, 0.0, 1.0)
