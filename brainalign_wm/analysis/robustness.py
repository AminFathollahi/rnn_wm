"""Robustness analyses for conditions excluded from model selection."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd


def accuracy_matched(df, tol: float = 0.02, acc_col: str = "mean_acc"):
    median = df[acc_col].median()
    return df[(df[acc_col] >= median - tol) & (df[acc_col] <= median + tol)].copy(), float(median)


def longer_delay_cfg(cfg: dict, factor: float = 3.0) -> dict:
    out = copy.deepcopy(cfg)
    out["task"]["maintain_steps"] = int(round(cfg["task"]["maintain_steps"] * factor))
    return out


def novel_load_cfg(cfg: dict, load: int) -> dict:
    if load in cfg["task"]["loads"]:
        raise ValueError(f"load {load} is in the trained set {cfg['task']['loads']}, not novel")
    out = copy.deepcopy(cfg)
    out["task"]["loads"] = [load]
    return out


def bootstrap_correlation(x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    from scipy.stats import pearsonr, spearmanr

    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    r, p_r = pearsonr(x, y)
    rho, p_rho = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    boot_r, boot_rho = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        if np.std(x[idx]) == 0 or np.std(y[idx]) == 0:
            continue
        boot_r.append(pearsonr(x[idx], y[idx])[0])
        boot_rho.append(spearmanr(x[idx], y[idx])[0])
    return {
        "n": n, "pearson_r": float(r), "pearson_p": float(p_r),
        "pearson_ci": (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))) if boot_r else (np.nan, np.nan),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "spearman_ci": (float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5))) if boot_rho else (np.nan, np.nan),
    }


def min_detectable_r(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    if n <= 3:
        return float("nan")
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return float(np.tanh((z_alpha + z_beta) / np.sqrt(n - 3)))


def correlation_table(
    df: pd.DataFrame, predictors: list[str], outcomes: list[str], n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            values = df[[predictor, outcome]].dropna()
            result = bootstrap_correlation(
                values[predictor].to_numpy(), values[outcome].to_numpy(), n_boot=n_boot, seed=seed,
            )
            rows.append({
                "predictor": predictor,
                "outcome": outcome,
                "n": result["n"],
                "pearson_r": result["pearson_r"],
                "pearson_p": result["pearson_p"],
                "pearson_ci_lo": result["pearson_ci"][0],
                "pearson_ci_hi": result["pearson_ci"][1],
                "spearman_rho": result["spearman_rho"],
                "spearman_p": result["spearman_p"],
                "spearman_ci_lo": result["spearman_ci"][0],
                "spearman_ci_hi": result["spearman_ci"][1],
                "min_detectable_r": min_detectable_r(result["n"]),
            })
    return pd.DataFrame(rows)
