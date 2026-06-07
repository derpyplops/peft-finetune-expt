#!/usr/bin/env bash
# args: each is "dataset|METHOD:LR:SEED:EP:REG". Runs them in order (worker once per job, right DS).
set -uo pipefail
RESULTS="${RESULTS:-/workspace/sweep_results.jsonl}"; touch "$RESULTS"
n=0; total=$#
for item in "$@"; do
  n=$((n+1)); ds="${item%%@*}"; spec="${item#*@}"
  echo "=== [$(date -u +%H:%M:%S)] job $n/$total  dataset=$ds  spec=$spec ==="
  DS="$ds" RESULTS="$RESULTS" WORK=/workspace bash /workspace/wsc_grid_worker.sh "$spec" || echo "JOBERR $ds $spec"
done
echo "=== SWEEP_BOX DONE $(hostname) $(date -u +%H:%M:%S) ($total jobs) ==="
