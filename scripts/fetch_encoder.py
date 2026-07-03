#!/usr/bin/env python3
"""Downloads the ResNet-18 ImageNet weights into the PyTorch hub cache,
using only the standard library so it can run before torch is installed.

torchvision names weight files as `<arch>-<first8_of_sha256>.pth`, so
integrity is verified by checking that the file's SHA-256 hash starts with
the prefix embedded in its filename.
"""
from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
FILENAME = URL.rsplit("/", 1)[-1]
SHA_PREFIX = FILENAME.split("-")[1].split(".")[0]  # "f37072fd"


def cache_dir() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    base = Path(torch_home) if torch_home else Path.home() / ".cache" / "torch"
    d = base / "hub" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_prefix(path: Path, n: int = 8) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main() -> int:
    dest = cache_dir() / FILENAME
    if dest.exists() and sha256_prefix(dest) == SHA_PREFIX:
        print(f"[fetch-encoder] already present & verified: {dest}")
        return 0
    print(f"[fetch-encoder] downloading {URL}\n              -> {dest}")
    try:
        urllib.request.urlretrieve(URL, dest)
    except Exception as e:  # noqa: BLE001
        print(f"[fetch-encoder] download failed: {e}\n"
              f"  Offline: manually place {FILENAME} into {dest.parent}", file=sys.stderr)
        return 1
    got = sha256_prefix(dest)
    if got != SHA_PREFIX:
        print(f"[fetch-encoder] CHECKSUM MISMATCH: got {got}, expected {SHA_PREFIX}", file=sys.stderr)
        return 2
    print(f"[fetch-encoder] OK ({dest.stat().st_size/1e6:.1f} MB, sha256 {got}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
