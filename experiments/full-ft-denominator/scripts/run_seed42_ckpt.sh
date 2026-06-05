#!/usr/bin/env bash
# Phase 3a (checkpoint-saving variant): seed 42, ascending order, idempotent skip-if-done.
# Same as run_seed42.sh EXCEPT: after eval, push the trained checkpoint to Wasabi instead of
# just deleting it. Big/expensive datasets keep optimizer state (resumable); small ones weights-only.
#   - keeps the denominator internally consistent with the in-flight sweep: EPOCHS default 5, LR passed in.
#   - lets us resume/extend the expensive datasets later without recomputing from scratch.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"
export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export WANDB_PROJECT=peftbench-fullft-seed42
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f /workspace/.wasabi.env ] && source /workspace/.wasabi.env   # WASABI_ACCESS_KEY/SECRET_KEY/BUCKET
: "${LR:?pass LR=<chosen-from-smoke> }"
: "${WASABI_ACCESS_KEY:?wasabi keys missing — source /workspace/.wasabi.env}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
SEED=42; EPOCHS="${EPOCHS:-5}"   # match the in-flight sweep (5ep + early stop) for a consistent denominator
CFG=examples/peftbench/full/llama-3-8b-instruct
CKPT_TMPL="$CFG/train_ckpt.yaml"   # parametric save_only_model — separate file so the running sweep is untouched
mkdir -p results

# Datasets that keep FULL optimizer state (resumable) — the expensive ones where retraining costs hours.
# Everything else: weights-only (~16GB), cheap to retrain if ever needed.
BIG=" sst2 mmlu qnli apps qqp mnli "

datasets=(cb copa wsc svamp rte mrpc openbookqa wic stsb gsm8k codealpacapy cola boolq piqa \
          multirc math_qa siqa hellaswag winogrande sst2 mmlu qnli apps qqp mnli)

for d in "${datasets[@]}"; do
  tag="${d}_s${SEED}"
  OUT="saves/full/seed42/train_${tag}"
  # Skip if already scored AND its weights are gone (old sweep deleted them — nothing to push).
  # If result exists but OUT still present, it was trained but not yet pushed → fall through to push.
  [ -f "results/full_${tag}.jsonl" ] && [ ! -d "$OUT" ] && { echo "skip done: $tag"; continue; }
  if [[ "$BIG" == *" $d "* ]]; then SAVE_ONLY_MODEL=false; mode="full+optim"; else SAVE_ONLY_MODEL=true; mode="weights"; fi
  export SAVE_ONLY_MODEL
  mkdir -p "$OUT"
  if [ ! -f "results/full_${tag}.jsonl" ]; then
    echo "=== TRAIN $tag ($mode) $(date -u +%H:%M:%S) ==="
    DATASET="$d" SEED="$SEED" EPOCHS="$EPOCHS" LEARNING_RATE="$LR" MODEL="$MODEL" \
      OUTPUT_DIR="$OUT" WANDB_NAME="full_${tag}" SAVE_ONLY_MODEL="$SAVE_ONLY_MODEL" \
      envsubst < "$CKPT_TMPL" > "$OUT/train.yaml"
    llamafactory-cli train "$OUT/train.yaml" || { echo "TRAIN FAIL $tag"; continue; }
    EV="saves/full/seed42/eval_${tag}"; mkdir -p "$EV"
    echo "=== EVAL $tag ==="
    DATASET="$d" SEED="$SEED" TRAINED_MODEL="$OUT" \
      OUTPUT_DIR="$EV" WANDB_NAME="full_eval_${tag}" \
      envsubst < "$CFG/eval.yaml" > "$EV/eval.yaml"
    llamafactory-cli train "$EV/eval.yaml" || { echo "EVAL FAIL $tag"; continue; }
    python scripts/peftbench/compute_metrics.py "$EV" "$d"
    cp "$EV/results.jsonl" "results/full_${tag}.jsonl"
    echo "=== DONE $tag → $(cat results/full_${tag}.jsonl) ==="
  fi
  # push checkpoint to Wasabi, then free local disk
  echo "=== PUSH $tag ($mode) → Wasabi $(date -u +%H:%M:%S) ==="
  if python3 /workspace/push_ckpt.py "$OUT" "$d" "$SEED"; then
    mkdir -p pushed; touch "pushed/${tag}.done"
    rm -rf "$OUT"
  else
    echo "PUSH FAIL $tag — keeping local copy"
  fi
done
echo "=== SEED42 CKPT SWEEP COMPLETE ==="
