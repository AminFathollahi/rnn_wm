#!/usr/bin/env bash
# Resumable post-training analysis runner.
set -Eeuo pipefail
exec </dev/null

ANALYSIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${ANALYSIS_PY:-/home/amin/miniconda3/envs/wm_dynamics/bin/python}"
ANALYSIS_CHECKPOINT="${ANALYSIS_CHECKPOINT:-ckpt.pt}"
if [[ "$ANALYSIS_CHECKPOINT" == */* || "$ANALYSIS_CHECKPOINT" != *.pt ]]; then
    echo "[analysis] checkpoint must be a .pt filename: $ANALYSIS_CHECKPOINT" >&2
    exit 2
fi
if [[ "$ANALYSIS_CHECKPOINT" == "ckpt.pt" ]]; then
    CHECKPOINT_TAG=""
else
    CHECKPOINT_TAG="_$(basename "$ANALYSIS_CHECKPOINT" .pt | sed 's/^ckpt_//')"
fi
CHECKPOINT_ROOT="$ANALYSIS_ROOT/results/analysis_checkpoints${CHECKPOINT_TAG}"
RUN_LIST="$CHECKPOINT_ROOT/active_core_runs.txt"
CONTROL_SHARD="$CHECKPOINT_ROOT/controls"
LOCK_FILE="$ANALYSIS_ROOT/results/analysis_checkpoints/analysis.lock"

cd "$ANALYSIS_ROOT"
mkdir -p "$(dirname "$LOCK_FILE")" "$CHECKPOINT_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "[analysis] another analysis.sh is already running (lock: $LOCK_FILE); exiting" >&2
    exit 1
fi

active_tmp="$RUN_LIST.active.$$"
run_list_tmp="$RUN_LIST.tmp.$$"
trap 'rm -f "$active_tmp" "$run_list_tmp"' EXIT
"$ANALYSIS_PY" scripts/list_active_analysis_runs.py > "$active_tmp"
active_count=0
eligible_count=0
while IFS= read -r run_id; do
    active_count=$((active_count + 1))
    if [[ -f "results/checkpoints/$run_id/$ANALYSIS_CHECKPOINT" ]]; then
        printf '%s\n' "$run_id"
        eligible_count=$((eligible_count + 1))
    else
        printf '[analysis] %s: no %s; skipping from this pass\n' "$run_id" "$ANALYSIS_CHECKPOINT" >&2
    fi
done < "$active_tmp" > "$run_list_tmp"
mv "$run_list_tmp" "$RUN_LIST"
rm -f "$active_tmp"
printf '[analysis] checkpoint=%s eligible=%d active=%d omitted=%d\n' \
    "$ANALYSIS_CHECKPOINT" "$eligible_count" "$active_count" "$((active_count - eligible_count))"

while IFS= read -r run_id; do
    [[ -n "$run_id" ]] || continue
    shard="$CHECKPOINT_ROOT/$run_id"
    if [[ -f "$shard/.complete" && -s "$shard/alignment_results${CHECKPOINT_TAG}.csv" && -s "$shard/alignment_by_session${CHECKPOINT_TAG}.csv" && -s "$shard/alignment_probe_by_region${CHECKPOINT_TAG}.csv" && -s "$shard/alignment_by_population${CHECKPOINT_TAG}.csv" ]]; then
        printf '[analysis] %s: checkpoint found; skipping\n' "$run_id"
        continue
    fi

    mkdir -p "$shard"
    printf '[analysis] %s: running representation shard (%s)\n' "$run_id" "$ANALYSIS_CHECKPOINT"
    "$ANALYSIS_PY" -u -m brainalign_wm.analysis.run_all \
        --config configs/config.yaml \
        --runs "$run_id" \
        --checkpoint "$ANALYSIS_CHECKPOINT" \
        --out-dir "$shard" \
        --skip-baselines \
        --skip-chance-control \
        --skip-dv-relationship
    touch "$shard/.complete"
done < "$RUN_LIST"

"$ANALYSIS_PY" scripts/merge_analysis_shards.py \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --run-list "$RUN_LIST" \
    --suffix "$CHECKPOINT_TAG"

if [[ -n "$CHECKPOINT_TAG" ]]; then
    exit 0
fi

if [[ ! -f "$CONTROL_SHARD/.complete" || ! -s "$CONTROL_SHARD/baselines.csv" || ! -s "$CONTROL_SHARD/chance_control.csv" ]]; then
    mkdir -p "$CONTROL_SHARD"
    "$ANALYSIS_PY" -u -m brainalign_wm.analysis.run_all \
        --config configs/config.yaml \
        --runs M11110_SUP_s0 \
        --out-dir "$CONTROL_SHARD" \
        --skip-reflection-shuffle \
        --skip-dynamics \
        --skip-encoding \
        --skip-dpca \
        --skip-dv-relationship \
        --skip-permutation-null
    touch "$CONTROL_SHARD/.complete"
fi
cp "$CONTROL_SHARD/baselines.csv" "$ANALYSIS_ROOT/results/baselines.csv"
cp "$CONTROL_SHARD/chance_control.csv" "$ANALYSIS_ROOT/results/chance_control.csv"

"$ANALYSIS_PY" scripts/analyze_campaign_performance.py
"$ANALYSIS_PY" scripts/run_geometry.py
"$ANALYSIS_PY" scripts/analyze_network_properties.py --campaign-only
"$ANALYSIS_PY" scripts/analyze_attractors.py --campaign-only
"$ANALYSIS_PY" -m brainalign_wm.analysis.dv_relationship
"$ANALYSIS_PY" -m brainalign_wm.figures.make_all --config configs/config.yaml
"$ANALYSIS_PY" scripts/audit_campaign.py --tier full --seeds 8

printf 'done: all %s runs analysed\n' "$eligible_count"
