#!/usr/bin/env bash
# GPU-blocked fixes from the diagnostic workflow — run AFTER the main seed-42 sweep frees the GPU.
#   wsc  : undertrained at 5ep (f1 31.7 < base 43.6); 10ep gave 54.0 -> rerun at 10 epochs.
#   gsm8k: full-FT forgets (51.1 < base 79.2); svamp was better at 3ep than 5ep, so TEST 3ep (may reduce forgetting).
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"; export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=peftbench-fullft-seed42
SEED=42; LR=5e-6; GRAD_ACCUM=1
CFG=examples/peftbench/full/llama-3-8b-instruct
mkdir -p results_fixes

run() {  # dataset epochs out_suffix
  local d=$1 ep=$2 sfx=$3 tag="${d}${sfx}"
  OUT="saves/fix/train_${tag}"; mkdir -p "$OUT"
  echo "=== TRAIN $tag (ep=$ep) $(date -u +%H:%M:%S) ==="
  DATASET="$d" SEED="$SEED" EPOCHS="$ep" LEARNING_RATE="$LR" GRAD_ACCUM="$GRAD_ACCUM" MODEL="$MODEL" \
    OUTPUT_DIR="$OUT" WANDB_NAME="fix_${tag}" envsubst < "$CFG/train.yaml" > "$OUT/train.yaml"
  llamafactory-cli train "$OUT/train.yaml" > "$OUT/t.log" 2>&1 || { echo "TRAIN FAIL $tag"; return; }
  EV="saves/fix/eval_${tag}"; mkdir -p "$EV"
  DATASET="$d" SEED="$SEED" TRAINED_MODEL="$OUT" OUTPUT_DIR="$EV" WANDB_NAME="fix_eval_${tag}" \
    envsubst < "$CFG/eval.yaml" > "$EV/eval.yaml"
  llamafactory-cli train "$EV/eval.yaml" > "$EV/e.log" 2>&1 || { echo "EVAL FAIL $tag"; return; }
  python scripts/peftbench/compute_metrics.py "$EV" "$d" && cp "$EV/results.jsonl" "results_fixes/${tag}.jsonl"
  echo "=== $tag -> $(cat results_fixes/${tag}.jsonl | tr '\n' ' ')"; rm -rf "$OUT"
}

run wsc   10 _ep10     # wsc at 10 epochs -> expect ~54 (> base 43.6)
run gsm8k 3  _ep3      # gsm8k at 3 epochs -> test if less forgetting beats 51.1
echo "=== POST-SWEEP FIXES DONE ==="
