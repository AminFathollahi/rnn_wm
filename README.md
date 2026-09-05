# Brain-aligned recurrent networks for working memory

This repository studies how biologically motivated constraints shape working-memory representations in recurrent neural networks. It compares flat and hierarchical recurrent architectures while varying spatial sparsity, reflective gating, local synaptic plasticity, topographic organization, and Dale's law.

Models are trained on visual working-memory tasks and evaluated at both behavioral and representational levels. The analysis pipeline includes representational similarity analysis, cross-temporal decoding, demixed PCA, encoding models, fixed-point analysis, and comparisons with human single-neuron recordings from a Sternberg task.

## Experimental design

The main experiment is a 15-condition ablation study over five model properties:

- hierarchical structure;
- multi-task training;
- local synaptic plasticity;
- topographic organization; and
- Dale's law.

The repository also contains a vanilla RNN baseline, n-back and multi-task experiments, simulated-neural-data recovery checks, and utilities for comparing model activity with electrophysiology.

## Installation

Python 3.11 or newer is required. Install a PyTorch build appropriate for your hardware first, then install the project and its analysis dependencies:

```bash
python -m pip install torch torchvision
python -m pip install -e ".[analysis]"
```

For development, install the test dependency as well:

```bash
python -m pip install -e ".[dev]"
pytest
```

The `Makefile` provides shortcuts for the local CUDA setup and common project commands. Run `make help` to list them.

## Configuration

Scientific and training settings are defined in `configs/config.yaml`. Runtime paths for datasets, stimuli, cached features, checkpoints, and results are defined in `configs/paths.yaml`.

To inspect the resolved configuration:

```bash
python -m brainalign_wm.config
```

You can use another paths file without modifying the repository:

```bash
export BRAINALIGN_WM_PATHS_CONFIG=/path/to/paths.yaml
```

Individual locations can also be overridden with `BRAINALIGN_WM_DATA_ROOT`, `BRAINALIGN_WM_STIMULI_ROOT`, `BRAINALIGN_WM_RESULTS_ROOT`, `BRAINALIGN_WM_FEATURE_CACHE_ROOT`, and `BRAINALIGN_WM_ACTIVITY_LOGS_ROOT`.

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

For a lightweight demonstration of the orchestration pipeline that does not require PyTorch training:

```bash
make run-grid-demo
```

Runs are resumable. Completed conditions are recorded in the results manifest, and interrupted runs resume from their latest checkpoint. Generated data, checkpoints, logs, figures, and results are intentionally excluded from version control.

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

Raw neural recordings, stimulus images, feature caches, checkpoints, and generated results are not distributed in this repository. Configure their locations in `configs/paths.yaml` or through the environment variables above.

Before using results from a campaign, run the consistency audit:

```bash
make audit-campaign
```
