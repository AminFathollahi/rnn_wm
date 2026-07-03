# brainalign_wm

Bio-plausible inductive biases and human single-neuron working-memory geometry.

**Master spec:** [`../protocol_v4_master_prompt.md`](../protocol_v4_master_prompt.md) — read it fully, especially **§0.2** (bootstrap + overnight run) before building. This repo is a **scaffold**: the orchestration, entrypoints, config contract, and tests are real; the science modules are TODO stubs mapped to protocol sections.

## Quickstart (single machine, RTX 5070 Ti)

```bash
cd brainalign_wm
make help                 # list targets
make setup                # upgrade PyTorch (CUDA 12.8 / sm_120) + install deps  [needs internet]
make verify-gpu           # assert the GPU is usable by torch
make fetch-encoder        # download ResNet-18 weights (stdlib; works pre-torch)
make test                 # scaffold tests pass today
make overnight-demo       # prove the orchestration end-to-end (synthetic stub, no torch)
```

Then implement the science (protocol §5–§10) in milestone order (§14), and:

```bash
make features             # cache frozen ResNet features
make recovery             # sim-spike geometry-recovery gate (§8.3) — must pass first
make overnight            # real training grid: 8 cells x seeds, resumable, breadth-first
```

## What works now vs. TODO

| Component | State |
|---|---|
| `run_grid.py` orchestrator (resume, breadth-first, budget, isolation, manifest, `MORNING_REPORT.md`) | **working** (stdlib) |
| `Makefile`, `scripts/fetch_encoder.py`, `pyproject.toml`, `configs/config.yaml` | **working** |
| `utils/device.py` (capability-aware CPU/GPU) | **working** |
| `training/train.py::train_one` | **stub** — implement (§5–§7) |
| models / mechanisms / tasks / neural adapters / analysis / figures | **stubs** — implement per protocol |

## Key facts baked in

- Env: conda `wm_dynamics`; PyTorch upgraded to CUDA-12.8 for sm_120 (`make setup`).
- Data: human single-neuron Sternberg NWB on the external USB (`configs/config.yaml: paths.data_root`); read via `h5py` (no pynwb needed). Tier A = `000469` + `000673` pooled.
- Visual WM only for Core; train on a broad image pool, **align on the datasets' exact images**.
- Never fabricate a gate pass; a cell that can't train is a logged result.
