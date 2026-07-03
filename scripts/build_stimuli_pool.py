#!/usr/bin/env python3
"""Materialize the ImageTokenBank's broad naturalistic pool (protocol §7.3)
from CIFAR-100 superclasses, grouped into 4 categories comparable to the
human Sternberg datasets' picture categories (protocol §7.1: "use
superordinate categories comparable to the dataset's -- faces, animals,
objects, places").

Why CIFAR-100: disk-light (~170MB), license-clean, auto-downloads via
torchvision, and ships fine + superclass labels -- a tractable way to build a
*broad* training pool without hand-curating thousands of images (protocol §0
rule 7: falsifiable / precedented / tractable default; logged in
DECISIONS.md). Images are low-res (32x32, upsampled to 224 by the encoder
transform) but this pool is only used to train a *general* WM operation --
the brain-alignment comparison uses the datasets' own embedded stimulus
images (§7.4), not this pool, so photorealism here is not load-bearing.

Caveat logged: CIFAR-100 has no genuine face closeups; the 'people' fine
classes (baby/boy/girl/man/woman) are the closest available proxy for
'faces' and are treated as such throughout -- this is weaker than the
primate/human face-patch literature would want and is noted as a limitation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

CATEGORY_MAP = {
    "faces": ["baby", "boy", "girl", "man", "woman"],
    "animals": [
        "bear", "leopard", "lion", "tiger", "wolf",
        "elephant", "cattle", "camel", "chimpanzee", "kangaroo",
        "fox", "porcupine", "possum", "raccoon", "skunk",
        "hamster", "mouse", "rabbit", "shrew", "squirrel",
    ],
    "objects": [
        "bottle", "bowl", "can", "cup", "plate",
        "chair", "couch", "table", "wardrobe", "bed",
        "clock", "keyboard", "lamp", "telephone", "television",
    ],
    "places": [
        "cloud", "forest", "mountain", "plain", "sea",
        "bridge", "castle", "house", "road", "skyscraper",
    ],
}
FINE_TO_CATEGORY = {fine: cat for cat, fines in CATEGORY_MAP.items() for fine in fines}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="stimuli")
    ap.add_argument("--per-class", type=int, default=80, help="images per fine class per CIFAR split")
    ap.add_argument("--cache-dir", default="./results/cifar100_raw")
    args = ap.parse_args()

    from torchvision.datasets import CIFAR100

    out_root = Path(args.out)
    counts = {cat: 0 for cat in CATEGORY_MAP}
    for split_train in (True, False):
        ds = CIFAR100(root=args.cache_dir, train=split_train, download=True)
        fine_names = ds.classes
        per_fine_seen: dict[str, int] = {}
        for img, label in zip(ds.data, ds.targets):
            fine = fine_names[label]
            cat = FINE_TO_CATEGORY.get(fine)
            if cat is None:
                continue
            seen = per_fine_seen.get(fine, 0)
            if seen >= args.per_class:
                continue
            per_fine_seen[fine] = seen + 1
            cat_dir = out_root / cat
            cat_dir.mkdir(parents=True, exist_ok=True)
            idx = counts[cat]
            Image.fromarray(img).save(cat_dir / f"{fine}_{idx:05d}.png")
            counts[cat] += 1
    for cat, n in counts.items():
        print(f"[build_stimuli_pool] {cat}: {n} images -> {out_root / cat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
