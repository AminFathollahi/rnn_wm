"""Capability-aware device selection.

`torch.cuda.is_available()` is not enough on this machine: torch 2.5.1 reports a CUDA
device but cannot run on the RTX 5070 Ti (sm_120). Check the compiled arch list.
"""
from __future__ import annotations


def get_device(prefer_cuda: bool = True):
    import torch

    if prefer_cuda and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            want = f"sm_{major}{minor}"
            archs = torch.cuda.get_arch_list()
            if any(a.startswith(want) for a in archs):
                return torch.device("cuda")
            print(f"[device] GPU is {want} but torch was built for {archs}; "
                  f"using CPU. Run `make setup` to upgrade PyTorch.")
        except Exception as e:  # noqa: BLE001
            print(f"[device] GPU capability check failed ({e}); using CPU.")
    return torch.device("cpu")
