# Build and execution targets for the working-memory RNN brain-alignment project.
# Override the interpreter if needed:  make PY=/path/to/python <target>
PY ?= /home/amin/miniconda3/envs/wm_dynamics/bin/python
CU_INDEX := https://download.pytorch.org/whl/cu128

.DEFAULT_GOAL := help
.PHONY: help setup verify-gpu fetch-encoder features test smoke recovery run-grid run-grid-demo reproduce clean

help: ## show targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## upgrade PyTorch (CUDA 12.8 / sm_120) + install package & analysis deps
	$(PY) -m pip install --upgrade --index-url $(CU_INDEX) torch torchvision
	$(PY) -m pip install -e ".[analysis]"

verify-gpu: ## confirm the RTX 5070 Ti (sm_120) is usable by torch
	$(PY) -c "import torch; al=torch.cuda.get_arch_list(); print('torch',torch.__version__,'cuda',torch.version.cuda,'archs',al); \
x=torch.zeros(1,device='cuda'); print('GPU OK:',torch.cuda.get_device_name(0)); \
assert any(a.startswith('sm_120') for a in al), 'sm_120 not in arch_list -> upgrade torch (make setup)'"

fetch-encoder: ## download + checksum ResNet-18 ImageNet weights into torch hub cache
	$(PY) scripts/fetch_encoder.py

features: ## precompute + cache frozen ResNet features (ImageTokenBank + dataset stimuli)
	$(PY) -m brainalign_wm.encoders.cache_features --config configs/config.yaml

test: ## run unit tests (scaffold + package)
	$(PY) -m pytest -q

smoke: ## end-to-end check of the training entrypoint (8 cells x 1 seed, minimal step budget)
	$(PY) run_grid.py --seeds 1 --budget 20m --tier smoke

recovery: ## simulated-spike geometry-recovery gate -- must pass before real-data alignment
	$(PY) -m brainalign_wm.neural.sim_brain.recovery_gate --config configs/config.yaml

run-grid: ## execute the full training grid (resumable, breadth-before-depth over cells x seeds)
	$(PY) run_grid.py --seeds 8 --budget 48h --tier full

run-grid-demo: ## demonstrate the orchestrator with the synthetic stub (no torch required)
	$(PY) run_grid.py --scaffold --seeds 3 --budget 30m --tier dev

reproduce: fetch-encoder features test recovery run-grid ## full pipeline from scratch
	$(PY) -m brainalign_wm.analysis.run_all --config configs/config.yaml
	$(PY) -m brainalign_wm.figures.make_all --config configs/config.yaml

clean: ## remove caches and __pycache__ (keeps results/manifest.jsonl)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
