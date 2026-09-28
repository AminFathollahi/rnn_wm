# Brain-aligned recurrent networks for working memory

This repository tests which biologically motivated constraints on recurrent
neural networks produce working-memory representations that match human
single-neuron geometry during a visual Sternberg task.

Models are trained on the same visual Sternberg paradigm recorded in humans
and compared against human medial temporal lobe and medial frontal cortex
single units at both the behavioral and representational level.

## Task

Delayed match / non-match Sternberg with set sizes 1-3: fixation, sequential
encoding of one item per step, a delay, a probe, and feedback. A fraction of
non-match probes are same-category lures (same category, different identity).
Training follows a curriculum from load 1 with no lures and a short delay to
the full load and lure distribution at the full delay. Evaluation is held out
by stimulus identity and reported per load with Wilson confidence intervals
and percentiles of the human accuracy distribution.

Model input on every tick is a frozen ImageNet-pretrained ResNet-18 image
feature (512 dimensions) plus a small task-context vector (task family, load
during encoding only, epoch). The encoder is never fine-tuned, so all memory
and decision computation lives in the recurrent core.

## Models

The recurrent core is gated (GRU). A fixed front end and fixed policy/value
heads are shared across all conditions so only the core varies.

Five binary mechanisms are ablated in a 15-condition, reference-anchored
design (baseline, full model, five add-one-to-baseline, five
remove-one-from-full, three pairwise interaction probes):

- **Structure:** flat recurrence versus a hierarchical manager/worker core in
  which a locality-masked worker runs every tick and a dense manager updates
  on a slow clock and projects top-down context back to the worker.
- **Modulation:** a reflective gate that accumulates rectified surprise
  (reward-prediction error at feedback, self-generated uncertainty otherwise)
  and biases the manager update gate.
- **Plasticity:** differentiable Hebbian fast weights added to the slow
  trained recurrent weights and reset each trial.
- **Topography:** a smoothness loss on the recurrent layout.
- **Dale's law:** a soft excitatory/inhibitory sign penalty (80/20 split).

A separate extended tier replaces backpropagation with reward-modulated local
learning (node perturbation with eligibility traces, escalating to e-prop) on
matched architectures to isolate the effect of the credit-assignment rule.

## Neural comparison

Human reference data are single-neuron recordings from hippocampus and
amygdala (medial temporal lobe) and dorsal anterior cingulate, pre-supplementary
motor area, and ventromedial prefrontal cortex (medial frontal cortex) during
visual picture Sternberg. Representational similarity analysis is the primary
comparison because it compares population geometry without assuming a
unit-to-unit map between model and neurons. Supporting analyses include
cross-temporal decoding, demixed PCA, fixed-point and topology measures, and
single-trial dynamics, with mixed-effects statistics across sessions. A
simulated-geometry recovery check validates the pipeline on planted geometry
before real-data alignment.

## Installation

Python 3.11 or newer is required. Install a PyTorch build appropriate for
your hardware first, then the project and its analysis dependencies:

```bash
python -m pip install torch torchvision
python -m pip install -e ".[analysis]"
```

For development, also install the test dependencies:

```bash
python -m pip install -e ".[dev]"
pytest
```

The `Makefile` provides shortcuts for the local CUDA setup and common
commands. Run `make help` to list them.

## Configuration

Scientific and training settings live in `configs/config.yaml`. Machine-specific
paths for datasets, stimuli, cached features, checkpoints, and results live in
`configs/paths.yaml`.

To inspect the resolved configuration:

```bash
python -m brainalign_wm.config
```

An alternate paths file can be supplied without modifying the repository:

```bash
export BRAINALIGN_WM_PATHS_CONFIG=/path/to/paths.yaml
```

Individual locations can also be overridden with `BRAINALIGN_WM_DATA_ROOT`,
`BRAINALIGN_WM_STIMULI_ROOT`, `BRAINALIGN_WM_RESULTS_ROOT`,
`BRAINALIGN_WM_FEATURE_CACHE_ROOT`, and `BRAINALIGN_WM_ACTIVITY_LOGS_ROOT`.

## Running the project

Precompute visual features and run the simulated-data recovery check:

```bash
make fetch-encoder
make features
make recovery
```

Run the main supervised ablation grid:

```bash
make run-grid
```

For a lightweight demonstration of the orchestration pipeline without
PyTorch training:

```bash
make run-grid-demo
```

Generated data, checkpoints, logs, figures, and results are excluded from
version control.

## Repository structure

| Path | Purpose |
| --- | --- |
| `brainalign_wm/models/` | Flat and hierarchical recurrent architectures |
| `brainalign_wm/mechanisms/` | Reflective gating and local-learning mechanisms |
| `brainalign_wm/tasks/` | Sternberg, n-back, curriculum, and multi-task generators |
| `brainalign_wm/training/` | Training entry points and activity logging |
| `brainalign_wm/neural/` | Neural-data adapters and simulated-neural-data tools |
| `brainalign_wm/analysis/` | Representation, geometry, dynamics, and statistical analyses |
| `brainalign_wm/figures/` | Figure-generation utilities |
| `configs/` | Experiment and runtime configuration |
| `scripts/` | Experiment, audit, analysis, and benchmarking commands |
| `tests/` | Automated tests |
| `run_grid.py` | Main ablation-grid orchestrator |

## Data and outputs

Raw neural recordings, stimulus images, feature caches, checkpoints, and
generated results are not distributed with this repository. Configure their
locations in `configs/paths.yaml` or through the environment variables above.

A mechanical self-consistency check over run records, resolved
configurations, and metrics is available:

```bash
make audit-campaign
```
