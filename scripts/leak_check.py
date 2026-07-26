#!/usr/bin/env python3
"""Probe-label leak check (Phase 0 of the 2026-07-26 audit, comments.txt).

Generates trials and asks: can `in_set` be predicted at the probe tick from
information available WITHOUT using working memory -- the probe category
alone, any single `c_t` dimension, or a combination of the two? The old
`lure_flag` broadcast (c_t[8], removed under F1) scored 1.000 on this check;
this script is the permanent regression guard for that whole class of bug.
`tests/test_tasks.py::test_probe_context_vector_does_not_leak_the_answer`
imports `generate_probe_dataset` and `best_single_dim_accuracy` from here so
the leak logic lives in one place.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

from brainalign_wm.tasks.sternberg import SternbergGenerator

ROOT = Path(__file__).resolve().parent.parent


def generate_probe_dataset(
    cfg: dict, bank, n_trials: int, seed_start: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ctx [n, C_DIM], category_match [n] bool, truth [n] bool) at
    the probe tick of `n_trials` freshly generated trials."""
    gen = SternbergGenerator(cfg, bank)
    task = cfg["task"]
    ctx, category_match, truth = [], [], []
    for i in range(n_trials):
        rng = np.random.RandomState(seed_start + i)
        steps = gen.generate_trial(
            rng,
            loads=task["loads"],
            lure_fraction=task["lure_fraction"],
            maintain_steps=task["maintain_steps"],
            trial_id=seed_start + i,
        )
        probe = next(s for s in steps if s.epoch == "probe")
        ctx.append(probe.c_t)
        category_match.append(probe.probe_category in probe.held_categories)
        truth.append(bool(probe.in_set))
    return np.asarray(ctx), np.asarray(category_match, dtype=bool), np.asarray(truth, dtype=bool)


def best_single_dim_accuracy(col: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the better of the two threshold rules on `col`, or 0.0 if
    `col` is constant (carries no information)."""
    if len(np.unique(col)) < 2:
        return 0.0
    return float(max(((col > 0.5) == y).mean(), ((col <= 0.5) == y).mean()))


def all_single_dim_accuracies(ctx: np.ndarray, y: np.ndarray) -> dict[int, float]:
    return {dim: best_single_dim_accuracy(ctx[:, dim], y) for dim in range(ctx.shape[1])}


def conjunction_accuracy(category_match: np.ndarray, ctx: np.ndarray, y: np.ndarray) -> float:
    """Accuracy of the best LINEAR combination of category-match and every
    c_t dimension -- generalizes the original exploit ("NO if c[8] else YES
    iff category matches"), which was exactly this kind of conjunction."""
    features = np.column_stack([category_match.astype(float), ctx])
    varying = features[:, np.ptp(features, axis=0) > 0]
    if varying.shape[1] == 0:
        return float((category_match == y).mean())
    clf = LogisticRegression().fit(varying, y)
    return float(clf.score(varying, y))


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    bank = ImageTokenBank(
        stimuli_root=ROOT / "stimuli",
        categories=cfg["task"]["categories"],
        feature_cache_path=ROOT / "results" / "feat_cache" / "leak_check.npy",
        seed=0,
    )
    ctx, category_match, truth = generate_probe_dataset(cfg, bank, n_trials=3000)

    cat_acc = float((category_match == truth).mean())
    dim_accs = all_single_dim_accuracies(ctx, truth)
    conj_acc = conjunction_accuracy(category_match, ctx, truth)

    print(f"category-match accuracy:        {cat_acc:.4f}")
    print("single c_t dimension accuracy:")
    for dim, acc in dim_accs.items():
        print(f"  c_t[{dim}]: {acc:.4f}")
    print(f"conjunction (category+c_t) accuracy: {conj_acc:.4f}")


if __name__ == "__main__":
    main()
