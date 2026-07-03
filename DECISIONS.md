# Decisions log

Append-only. Record any non-obvious choice (protocol §0 rule 7), especially deviations.

## Resolved before build (from protocol §17)
- **Neural target:** human single-neuron ephys (Sternberg), not fMRI.
- **Datasets:** combine all — Tier A pooled (`000469` + `000673`, ~1,800 WM units, MTL+MFC), Tier B `001187` (MTL), Tier C `000574` (verbal, coarse conditions only). Read via h5py.
- **Modality:** visual WM only for Core; verbal = coarse-condition replication (no encoder path).
- **Stimuli:** train on a broad image pool; align using the datasets' exact embedded images.
- **Compute:** single machine, one GPU (RTX 5070 Ti). Grid runs sequentially, breadth-before-depth, resumable overnight.
- **PyTorch:** upgrade to the CUDA-12.8 / sm_120 build (`make setup`); build device-agnostic with CPU fallback.

## During build
- **2026-07-03: broad ImageTokenBank pool = CIFAR-100, grouped into 4 superordinate categories.** §7.3 needs a *broad naturalistic* training pool distinct from the exact dataset stimuli used only for alignment (§7.4). No such pool existed on disk. Chose CIFAR-100 (auto-downloads via torchvision, ~170MB, license-clean) grouped by fine-label into `faces` (people fine classes; no genuine face closeups available -- logged as a limitation), `animals`, `objects`, `places` -- matching the picture-Sternberg category structure (§7.1). Images are 32x32 (upsampled by the ResNet transform); acceptable because this pool trains a *general* WM operation, not the alignment comparison itself (which uses the datasets' embedded `StimulusTemplates`). See `scripts/build_stimuli_pool.py`.
- **2026-07-03: skip env clone, upgrade `wm_dynamics` in place.** §0.2 Step 0.1 offers cloning to `wm_dynamics_gpu` as an "optional safety" since other projects share `wm_dynamics`. Disk check: root filesystem has only 12GB free vs. the env's 8.2GB size — cloning would leave ~3.8GB free, too tight for the torch/torchvision cu128 wheel download (~1-2GB) plus install. Upgrading in place instead (this was always the fallback per §0.2). Risk noted: other conda envs/projects depending on `wm_dynamics`'s torch==2.5.1 will now see torch upgraded; device-agnostic code + CPU fallback (`utils/device.py`) is unaffected either way.
