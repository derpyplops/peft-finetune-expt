#!/usr/bin/env bash
# Phase 3a: one seed (42), all 27 datasets, ASCENDING size order. Run only after smoke locks LR.
# Idempotent (skip-if-done) + auto-resumable (stable OUTPUT_DIR + overwrite_output_dir:false) +
# weights deleted after eval (denominator only needs metrics).  Pass LR=<chosen> from the smoke test.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"
export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export WANDB_PROJECT=peftbench-fullft-seed42
: "${LR:?pass LR=<chosen-from-smoke> }"
SEED=42; EPOCHS=10
CFG=examples/peftbench/full/llama-3-8b-instruct
mkdir -p results

# ascending train-set size (cheap/fast first → bugs surface early; cost ramps last)
datasets=(cb copa wsc svamp conala rte mrpc openbookqa apps wic stsb gsm8k cola boolq \
          piqa codealpacapy multirc math_qa siqa hellaswag winogrande sst2 mmlu record qnli qqp mnli)

for d in "${datasets[@]}"; do
  tag="${d}_s${SEED}"
  [ -f "results/full_${tag}.jsonl" ] && { echo "skip done: $tag"; continue; }
  OUT="saves/full/seed42/train_${tag}"; mkdir -p "$OUT"   # STABLE path → auto-resume on restart
  echo "=== TRAIN $tag $(date -u +%H:%M:%S) ==="
  DATASET="$d" SEED="$SEED" EPOCHS="$EPOCHS" LEARNING_RATE="$LR" MODEL="$MODEL" \
    OUTPUT_DIR="$OUT" WANDB_NAME="full_${tag}" \
    envsubst < "$CFG/train.yaml" > "$OUT/train.yaml"
  llamafactory-cli train "$OUT/train.yaml" || { echo "TRAIN FAIL $tag"; continue; }
  EV="saves/full/seed42/eval_${tag}"; mkdir -p "$EV"
  echo "=== EVAL $tag ==="
  DATASET="$d" SEED="$SEED" TRAINED_MODEL="$OUT" \
    OUTPUT_DIR="$EV" WANDB_NAME="full_eval_${tag}" \
    envsubst < "$CFG/eval.yaml" > "$EV/eval.yaml"
  llamafactory-cli train "$EV/eval.yaml" || { echo "EVAL FAIL $tag"; continue; }
  python scripts/peftbench/compute_metrics.py "$EV" "$d"
  cp "$EV/results.jsonl" "results/full_${tag}.jsonl"
  rm -rf "$OUT"   # free ~100GB; keep eval preds + metrics
  echo "=== DONE $tag → $(cat results/full_${tag}.jsonl) ==="
done
echo "=== SEED42 SWEEP COMPLETE ==="
