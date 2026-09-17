import numpy as np
import pandas as pd
import pytest

from brainalign_wm.analysis.robustness import (
    accuracy_matched, bootstrap_correlation, correlation_table, longer_delay_cfg, min_detectable_r, novel_load_cfg,
)


def test_accuracy_matched_keeps_only_the_band():
    df = pd.DataFrame({"mean_acc": [0.80, 0.94, 0.95, 0.96, 0.99]})
    matched, median = accuracy_matched(df, tol=0.02)
    assert median == 0.95
    assert sorted(matched["mean_acc"]) == [0.94, 0.95, 0.96]


def test_longer_delay_cfg_scales_maintain_steps_only():
    cfg = {"task": {"maintain_steps": 15, "loads": [1, 2, 3]}}
    out = longer_delay_cfg(cfg, factor=3.0)
    assert out["task"]["maintain_steps"] == 45
    assert out["task"]["loads"] == [1, 2, 3]
    assert cfg["task"]["maintain_steps"] == 15  # original untouched


def test_novel_load_cfg_rejects_a_trained_load():
    cfg = {"task": {"loads": [1, 2, 3]}}
    with pytest.raises(ValueError):
        novel_load_cfg(cfg, 3)
    assert novel_load_cfg(cfg, 4)["task"]["loads"] == [4]


def test_bootstrap_correlation_recovers_a_perfect_relationship():
    x = np.linspace(0, 1, 30)
    res = bootstrap_correlation(x, x)
    assert res["pearson_r"] == pytest.approx(1.0)
    assert res["pearson_ci"][0] > 0.9


def test_min_detectable_r_shrinks_with_n():
    assert min_detectable_r(200) < min_detectable_r(20)
    assert np.isnan(min_detectable_r(3))


def test_correlation_table_preserves_signed_predictor():
    df = pd.DataFrame({
        "signed": [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
        "clipped": [0.0, 0.0, 0.0, 0.1, 0.2, 0.3],
        "outcome": [-3, -2, -1, 1, 2, 3],
    })
    out = correlation_table(df, ["signed", "clipped"], ["outcome"], n_boot=100)
    signed = out.loc[out["predictor"].eq("signed")].iloc[0]
    assert signed["n"] == 6
    assert signed["pearson_r"] == pytest.approx(1.0)


def test_correlation_table_flags_a_relationship_it_cannot_detect_at_this_n():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({"predictor": rng.randn(20), "outcome": rng.randn(20)})
    out = correlation_table(df, ["predictor"], ["outcome"], n_boot=100).iloc[0]
    assert abs(out["pearson_r"]) < out["min_detectable_r"]
