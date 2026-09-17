#!/usr/bin/env bash
# Activity logs and post-training analysis. Resumable; safe to re-run.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./_start.sh replay ./analysis.sh
