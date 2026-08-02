# RNN models of working memory with varying degrees of bioplausibility

A study of biologically-motivated inductive biases (hierarchical structure
with spatial sparsity, neuromodulatory gating, synaptic plasticity,
topographic smoothness, and Dale's law) and their effect on the alignment
between a recurrent neural network's working-memory representations and
human single-neuron electrophysiology recorded during a Sternberg
working-memory task. The model is also benchmarked as a digital twin of the
maintenance-code findings (C1-C4) from the companion project at
`../wm_dynamics/PAPER_REPORT.tex`.

The study was designed around three sequential stages:

- **Stage 1** — a vanilla tanh-RNN training-regime map: supervision
  {SUP, RL} x structure {S=0 flat, S=1 hierarchical} x diet {WM-only,
  multi-task} = 8 cells x 3 seeds, run by `scripts/run_stage1_grid.py`.
  **Deferred.** The vanilla tanh substrate does not learn this task under
  any of the three training signals tested (`legacy`, `SUP`, `RL`) at any
  recurrent-init spectral radius tested — a NO-GO that survived every
  prespecified recovery lever. Stage 1's own rationale (map the regime on a
  *cheap* substrate before committing to the GRU battery) died with that
  substrate, so it is deferred rather than rebuilt on GRU; see `advisor.md`
  D17/D21/D23b. `scripts/run_stage1_grid.py` remains vanilla-hardcoded.
- **Stage 2** — the 15-cell, 5-arm (S, M, P, T, D) bio-plausibility ablation
  battery on the **GRU** substrate (the study's substrate, per `advisor.md`
  D17/D21), run by `run_grid.py`. This is the study's main event and the
  one `make run-grid` drives.
- **Stage 3** — n-back task inference / meta-RL, model-internal only.

The full experimental design, hypotheses, gates, and phase-by-phase
specification are consolidated in `advisor.md` (the current, authoritative
scientific record and decision ledger) and `implementation.md` (how the
model and pipeline are actually built); `../protocol_v4_master_prompt.md`
is the superseded original design doc. This README covers only setup and
day-to-day usage.

## Environment

```bash
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
make run-grid             # Stage 2: execute the 15-cell training grid, N seeds, resumable
python scripts/run_stage1_grid.py --max-steps <N> --budget <B>   # Stage 1: 8 cells x 3 seeds
```

Both grid runners are safe to interrupt and resume: completed runs are
recorded in `results/manifest.jsonl` and skipped on the next invocation, and
each individual run resumes from its own last checkpoint under
`results/checkpoints/<run_id>/`. Progress is summarized in
`results/RUN_REPORT.md` after each invocation. Stage 1's `--max-steps` has no default: it is set
from the Phase 11.1 pilot's `steps_to_criterion` (1.5x the slower of S=0/S=1,
rounded up to the nearest 10k) — a deliberate value chosen once per pilot
result, not a reused constant.

## Repository layout

| Component | Description |
|---|---|
| `run_grid.py` | Stage 2 grid orchestrator: resumable, breadth-before-depth over cells x seeds, per-run isolation, wall-clock budget |
| `scripts/run_stage1_grid.py` | Stage 1 grid orchestrator: same resumability discipline, over the 8-cell supervision x structure x diet factorial |
| `brainalign_wm/models/` | Recurrent architectures: flat GRU, hierarchical manager-worker core with a spatially-masked worker |
| `brainalign_wm/mechanisms/` | Reflective gating and node-perturbation local learning |
| `brainalign_wm/tasks/` | Image-Sternberg trial generator, annealed training curriculum |
| `brainalign_wm/neural/` | Neural-data interface, simulated-spike generator, NWB adapter for the human single-neuron datasets |
| `brainalign_wm/analysis/` | Representational similarity analysis, demixed PCA, cross-temporal decoding, encoding models, statistics |
| `brainalign_wm/training/` | Logging schema and the single-run training entrypoint (`train.py::train_one`) |
| `configs/config.yaml` | The project's configuration contract |
| `advisor.md` | The scientific record: current state, decision ledger, guardrails, literature-to-decision traceability, and the frozen preregistration (hypotheses, analysis lock, amendments) as an appendix |
| `implementation.md` | How the model and pipeline are actually built: architectures, mechanisms, task/curriculum generation, training loop, metrics schema, config contract |
| `executor.md` | The execution record: current status, next steps, and the full implementation chronology with acceptance output |

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
