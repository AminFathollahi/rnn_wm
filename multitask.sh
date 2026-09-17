#!/usr/bin/env bash
# Multi-task training from random initialization. Resumable; safe to re-run.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./_start.sh multitask "${RNNWM_PY:-/home/amin/miniconda3/envs/wm_dynamics/bin/python}" \
    scripts/run_multitask_from_init.py --supervision SUP --execute
