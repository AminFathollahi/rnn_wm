#!/usr/bin/env bash
# Continuation training, one stream over every cell. Resumable; safe to re-run.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./_start.sh train "${RNNWM_PY:-/home/amin/miniconda3/envs/wm_dynamics/bin/python}" \
    scripts/run_continuations.py --execute
