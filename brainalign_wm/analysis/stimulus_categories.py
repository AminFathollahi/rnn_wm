"""Imputed category labels for the real datasets' own stimulus images
(audit fix A2a, revised scope).

`comments.txt`'s A2a asked for delay-period RDM conditions built on held
item/category IDENTITY at fixed load, instead of load alone. Implementing
that literally with the real Tier-A datasets' own PicIDs turned out to be
empirically infeasible, discovered while building the per-session
maintenance-epoch alignment path: real sessions draw held items from a
large, effectively non-repeating pool -- e.g. every one of 70 load-1 trials
in a representative `000673` session held a distinct single item (70/70
unique). Exact item-identity conditions therefore have no repeated trials
to average or cross-validate a condition mean from, and the real datasets
carry no semantic category metadata (no `category` column in the NWB
schema -- see `dandi_nwb.py`'s module docstring) that could substitute a
coarser, recurring label directly.

This module imputes a category for each dataset image via nearest-neighbor
cosine similarity, in the SAME frozen ResNet-18 feature space the rest of
the pipeline already uses, against the four category centroids of the
model's own broad training pool (`ImageTokenBank`). This is not a ground-
truth label -- it is a content-based proxy, built from the same encoder the
model's own front end uses, so "does the delay period discriminate held-
category" is at least testing category-level content rather than pure
load count. Held-CATEGORY multisets recur far more often across trials
than exact item identity (only `n_categories` possible values per slot),
making a category+load condition schema for the maintenance epoch
tractable where raw identity was not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def category_centroids(image_bank) -> dict[str, np.ndarray]:
    """Mean frozen-encoder feature per training-pool category."""
    feats = image_bank.features()
    cats = [image_bank.category_of(i) for i in range(len(image_bank))]
    out: dict[str, np.ndarray] = {}
    for c in set(cats):
        idx = [i for i, cc in enumerate(cats) if cc == c]
        out[c] = feats[idx].mean(axis=0)
    return out


def impute_categories_for_session(session_id: str, centroids: dict[str, np.ndarray]) -> dict[str, str]:
    """{PicID (str) -> imputed category} for one session's cached stimulus
    features (`results/feat_cache/dataset_stimuli/<session>.npz`, built by
    `dandi_nwb.cache_stimulus_features`). Returns {} if the cache is
    missing for this session."""
    cache_path = ROOT / "results" / "feat_cache" / "dataset_stimuli" / f"{session_id}.npz"
    if not cache_path.exists():
        return {}
    data = np.load(cache_path)
    pic_ids, feats = data["pic_ids"], data["features"]
    cat_names = list(centroids.keys())
    cat_mat = np.stack([centroids[c] for c in cat_names])  # [n_categories, D]
    feats_n = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    cat_n = cat_mat / (np.linalg.norm(cat_mat, axis=1, keepdims=True) + 1e-8)
    sims = feats_n @ cat_n.T  # [n_images, n_categories]
    best = sims.argmax(axis=1)
    return {str(pid): cat_names[best[i]] for i, pid in enumerate(pic_ids)}
