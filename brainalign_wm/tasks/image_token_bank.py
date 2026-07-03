"""ImageTokenBank (protocol §7.3): the broad naturalistic training pool.

Deterministic (seeded) sampling, a fixed train/test image split (for
generalization tests -- a true WM system should maintain novel images it
never trained on, §7.4), and on-disk-cached ResNet-18 features keyed by
image id (frozen encoder => compute once).

Populate `stimuli/<category>/*.png` first via `scripts/build_stimuli_pool.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ImageTokenBank:
    stimuli_root: Path
    categories: list[str]
    feature_cache_path: Path
    test_fraction: float = 0.2
    seed: int = 0
    _index: list[dict] = field(init=False, default_factory=list)
    _features: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.stimuli_root = Path(self.stimuli_root)
        self.feature_cache_path = Path(self.feature_cache_path)
        if self.feature_cache_path.suffix != ".npy":  # np.save auto-appends .npy otherwise
            self.feature_cache_path = self.feature_cache_path.with_suffix(
                self.feature_cache_path.suffix + ".npy"
            )
        self._index = self._build_index()
        rng = np.random.RandomState(self.seed)
        # Stratify the train/test split PER CATEGORY (not one global shuffle):
        # a global split can leave a category with zero test images by chance
        # (real failure mode with a small pool -- category `sample(..., split="test",
        # category=c)` then raises "not enough images" for that category).
        self._test_ids: set[int] = set()
        self._train_ids: set[int] = set()
        by_category: dict[str, list[int]] = {}
        for d in self._index:
            by_category.setdefault(d["category"], []).append(d["image_id"])
        for cat, ids in by_category.items():
            ids = np.array(ids)
            perm = rng.permutation(len(ids))
            n_test = max(1, int(round(len(ids) * self.test_fraction))) if len(ids) > 1 else 0
            self._test_ids.update(ids[perm[:n_test]].tolist())
            self._train_ids.update(ids[perm[n_test:]].tolist())

    def _build_index(self) -> list[dict]:
        index = []
        image_id = 0
        for cat in sorted(self.categories):
            cat_dir = self.stimuli_root / cat
            if not cat_dir.exists():
                continue
            paths = sorted(cat_dir.glob("*.png")) + sorted(cat_dir.glob("*.jpg"))
            for p in paths:
                index.append({"image_id": image_id, "category": cat, "path": str(p)})
                image_id += 1
        if not index:
            raise FileNotFoundError(
                f"No images found under {self.stimuli_root}/<category>/ for categories "
                f"{self.categories}. Run `python scripts/build_stimuli_pool.py` first."
            )
        return index

    def __len__(self) -> int:
        return len(self._index)

    def category_of(self, image_id: int) -> str:
        return self._index[image_id]["category"]

    def sample(
        self,
        n: int,
        rng: np.random.RandomState,
        split: str = "train",
        category: str | None = None,
        exclude: set[int] | None = None,
    ) -> list[int]:
        pool_ids = self._train_ids if split == "train" else self._test_ids
        candidates = [
            d["image_id"]
            for d in self._index
            if d["image_id"] in pool_ids
            and (category is None or d["category"] == category)
            and (exclude is None or d["image_id"] not in exclude)
        ]
        if len(candidates) < n:
            raise ValueError(
                f"not enough images: have {len(candidates)}, need {n} "
                f"(split={split}, category={category})"
            )
        return rng.choice(candidates, size=n, replace=False).tolist()

    def load_image(self, image_id: int) -> np.ndarray:
        from PIL import Image

        return np.array(Image.open(self._index[image_id]["path"]).convert("RGB"))

    def features(self) -> np.ndarray:
        """[N, 512] cached ResNet features; computed once, then memmapped from disk."""
        if self._features is not None:
            return self._features
        if self.feature_cache_path.exists():
            cached = np.load(self.feature_cache_path)
            if cached.shape[0] == len(self._index):
                self._features = cached
                return self._features
        from brainalign_wm.encoders.resnet18_encoder import encode_images

        images = [self.load_image(d["image_id"]) for d in self._index]
        feats = encode_images(images)
        self.feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.feature_cache_path, feats)
        self._features = feats
        return self._features

    def feature_of(self, image_id: int) -> np.ndarray:
        return self.features()[image_id]
