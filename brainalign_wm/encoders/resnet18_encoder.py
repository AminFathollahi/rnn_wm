"""Frozen ResNet-18 (ImageNet) visual encoder (protocol §0.1, §5.1).

v_t = f_enc(I_t) in R^512: penultimate global-average-pool output. Frozen and
eval()-only -- no learning rule (BPTT or local) ever touches it, so the
recurrent-learning question stays isolated from the visual front end.
Features are meant to be cached to disk by callers (ImageTokenBank,
NWB-stimulus caching); this module holds only the encoder itself.
"""
from __future__ import annotations

import numpy as np

_ENCODER = None
_TRANSFORM = None


def _load():
    global _ENCODER, _TRANSFORM
    if _ENCODER is not None:
        return _ENCODER, _TRANSFORM
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    from torchvision.models.feature_extraction import create_feature_extractor

    weights = ResNet18_Weights.IMAGENET1K_V1
    enc = resnet18(weights=weights)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    _ENCODER = create_feature_extractor(enc, {"avgpool": "feat"})
    _TRANSFORM = weights.transforms()
    return _ENCODER, _TRANSFORM


def encode_images(images: list[np.ndarray], device: str = "cpu", batch_size: int = 64) -> np.ndarray:
    """images: list of HxWx3 uint8 arrays -> [N, 512] float32 features."""
    import torch
    from PIL import Image

    extractor, transform = _load()
    extractor = extractor.to(device)
    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tens = torch.stack([transform(Image.fromarray(im)) for im in batch]).to(device)
            out = extractor(tens)["feat"]
            out = out.flatten(1)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)
