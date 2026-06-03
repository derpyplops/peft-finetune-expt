#!/usr/bin/env bash
# Smoke test / LR sweep — the correctness gate before the $400 sweep (premortem Tier 1).
# Sweeps full-FT learning rate on 3 small datasets (NLU / math / code) and validates the
# whole train->eval->metrics pipeline + that generations parse to labels.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"
export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export WANDB_PROJECT=peftbench-fullft-smoke
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation for full-FT 8B on 1 GPU

datasets=(copa svamp)             # NLU(400) / math(700) — clean kinit/peft-factory sources, small/fast
                                  # (cb/conala deferred: super_glue script-load + missing conala config — see data audit)
lrs=(5e-6 1e-5 2e-5)
SEED=42; EPOCHS=10
CFG=examples/peftbench/full/llama-3-8b-instruct

mkdir -p results
for d in "${datasets[@]}"; do
  for lr in "${lrs[@]}"; do
    tag="${d}_lr${lr}"
    [ -f "results/smoke_${tag}.jsonl" ] && { echo "skip $tag"; continue; }
    OUT="saves/full/smoke/train_${tag}"; mkdir -p "$OUT"
    echo "=== TRAIN $tag (lr=$lr) ==="
    DATASET="$d" SEED="$SEED" EPOCHS="$EPOCHS" LEARNING_RATE="$lr" MODEL="$MODEL" \
      OUTPUT_DIR="$OUT" WANDB_NAME="smoke_${tag}" \
      envsubst < "$CFG/train.yaml" > "$OUT/train.yaml"
    llamafactory-cli train "$OUT/train.yaml" 2> "$OUT/train.err" || { echo "TRAIN FAIL $tag (see $OUT/train.err)"; tail -5 "$OUT/train.err"; continue; }
    EV="saves/full/smoke/eval_${tag}"; mkdir -p "$EV"
    echo "=== EVAL $tag ==="
    DATASET="$d" SEED="$SEED" TRAINED_MODEL="$OUT" \
      OUTPUT_DIR="$EV" WANDB_NAME="smoke_eval_${tag}" \
      envsubst < "$CFG/eval.yaml" > "$EV/eval.yaml"
    llamafactory-cli train "$EV/eval.yaml" || { echo "EVAL FAIL $tag"; continue; }
    python scripts/peftbench/compute_metrics.py "$EV" "$d" || { echo "METRIC FAIL $tag"; continue; }
    cp "$EV/results.jsonl" "results/smoke_${tag}.jsonl"
    # a few raw generations for the label-parsing sanity check (premortem 1.2)
    echo "--- sample preds vs labels ($tag) ---"; head -3 "$EV/generated_predictions.jsonl"
    rm -rf "$OUT"   # free disk (full-FT checkpoint ~100GB); keep eval preds+metrics
  done
done
echo "=== SMOKE DONE ==="
echo "scores:"; for f in results/smoke_*.jsonl; do echo "$f: $(cat $f)"; done
