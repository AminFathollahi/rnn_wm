"""Precompute and cache frozen ResNet features for the ImageTokenBank and
for the neural datasets' embedded stimulus images (via
`neural/adapters/dandi_nwb.py`). The encoder is frozen, so each feature is
computed once and reused.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = cfg["paths"]

    from brainalign_wm.tasks.image_token_bank import ImageTokenBank

    bank = ImageTokenBank(
        stimuli_root=paths["stimuli"],
        categories=[d.name for d in Path(paths["stimuli"]).iterdir() if d.is_dir()]
        if Path(paths["stimuli"]).exists()
        else ["faces", "animals", "objects", "places"],
        feature_cache_path=Path(paths["feature_cache"]) / "image_token_bank.npy",
    )
    feats = bank.features()
    print(f"[cache_features] ImageTokenBank: {feats.shape[0]} images -> {feats.shape[1]}-d "
          f"features cached at {bank.feature_cache_path}")

    try:
        from brainalign_wm.neural.adapters.dandi_nwb import cache_stimulus_features

        cache_stimulus_features(cfg)
        print("[cache_features] dataset embedded stimulus features cached.")
    except (ImportError, NotImplementedError, ModuleNotFoundError, AttributeError) as e:
        print(f"[cache_features] skipping dataset stimulus-image caching: {e!r}. "
              f"Confirm the external data drive is mounted and retry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
