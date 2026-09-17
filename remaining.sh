#!/usr/bin/env bash
# What GPU work is left, and whether any is running.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY="${RNNWM_PY:-/home/amin/miniconda3/envs/wm_dynamics/bin/python}"

PYTHONPATH=$PWD "$PY" - <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.run_continuations import enumerate_continuations

completed = set()
for line in Path("results/manifest.jsonl").read_text().splitlines():
    if line.strip():
        record = json.loads(line)
        if record.get("status") == "completed":
            completed.add(record["run_id"])

runs = {r["run_id"] for r in enumerate_continuations()}
left = len(runs - completed)
print(f"1. continuation training   {len(runs) - left}/{len(runs)} done"
      + ("" if left else "   [done]"))
PY

listed=$(PYTHONPATH=$PWD "$PY" scripts/list_active_analysis_runs.py | wc -l)
shards=$(ls results/analysis_checkpoints/*/.complete 2>/dev/null | wc -l)
printf '2. replay and analysis     %s/%s done' "$shards" "$listed"
[ "$shards" -ge "$listed" ] && printf '   [done]'
printf '\n\n'

streams=$(pgrep -fc "run_continuations\.py --models" || true)
if [ "${streams:-0}" -gt 0 ]; then
    echo "running: continuation training ($streams streams)"
elif pgrep -f analysis.sh > /dev/null; then
    echo "running: replay and analysis"
else
    echo "running: nothing"
fi
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu: /'
