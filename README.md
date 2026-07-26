# RNN models of working memory with varying degrees of bioplausibility

A factorial study of biologically-motivated inductive biases (hierarchical
structure with spatial sparsity, neuromodulatory gating, and local
reward-modulated learning) and their effect on the alignment between a
recurrent neural network's working-memory representations and human
single-neuron electrophysiology recorded during a Sternberg working-memory
task. The full experimental design, hypotheses, and specification are
recorded in `../protocol_v4_master_prompt.md`; this document covers only
setup and day-to-day usage.

## Environment

```bash
cd brainalign_wm
make help                 # list available targets
make setup                # install PyTorch (CUDA 12.8 / sm_120 build) and analysis dependencies
make verify-gpu           # confirm the GPU is usable by PyTorch
make fetch-encoder        # download the ResNet-18 ImageNet weights
make test                 # run the test suite
```

## Execution

```bash
make features             # precompute and cache frozen visual-encoder features
make recovery             # simulated-spike geometry-recovery gate (must pass before real-data alignment)
make run-grid             # execute the training grid: 8 cells x N seeds, resumable
```

`make run-grid` is safe to interrupt and resume: completed runs are recorded
in `results/manifest.jsonl` and skipped on the next invocation, and each
individual run resumes from its own last checkpoint under
`results/checkpoints/<run_id>/`. Progress is summarized in `RUN_REPORT.md`
after each invocation.

## Repository layout

| Component | Description |
|---|---|
| `run_grid.py` | Grid orchestrator: resumable, breadth-before-depth over cells x seeds, per-run isolation, wall-clock budget |
| `brainalign_wm/models/` | Recurrent architectures: flat GRU, hierarchical manager-worker core with a spatially-masked worker |
| `brainalign_wm/mechanisms/` | Reflective gating and node-perturbation local learning |
| `brainalign_wm/tasks/` | Image-Sternberg trial generator, annealed training curriculum |
| `brainalign_wm/neural/` | Neural-data interface, simulated-spike generator, NWB adapter for the human single-neuron datasets |
| `brainalign_wm/analysis/` | Representational similarity analysis, demixed PCA, cross-temporal decoding, encoding models, statistics |
| `brainalign_wm/training/` | Logging schema and the single-run training entrypoint (`train.py::train_one`) |
| `configs/config.yaml` | The project's configuration contract |
| `RESPONSES.md` | Project status, design decisions and their rationale, and the response to `comments.txt` |

## Notes

- Configuration: `configs/config.yaml`, in particular `paths.data_root` for
  the external-drive location of the human single-neuron recordings, read
  directly via `h5py` (no `pynwb` dependency required).
- Scope: visual working memory only in the core experimental design; the
  training image pool is broad and general-purpose, while alignment against
  the neural recordings uses each session's exact stimulus images.
- No behavioral gate is ever reported as passed without being met: a
  training run that fails its accuracy threshold is a recorded result, not
  a condition to be silently corrected.
