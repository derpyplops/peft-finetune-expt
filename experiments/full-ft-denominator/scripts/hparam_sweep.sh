#!/usr/bin/env bash
# Hparam search on the 5 smallest datasets to find full-FT settings that behave like a proper
# ceiling (full-FT >= base, ideally >= best PEFT). 2x2x2 grid: LR x epochs x effective-batch.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"
export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=peftbench-fullft-hparam
SEED=42
CFG=examples/peftbench/full/llama-3-8b-instruct
mkdir -p results_hp

datasets=(cb copa wsc svamp rte)   # 5 smallest
lrs=(1e-5 5e-6)                    # current vs lower (extend to 2e-6 if both too hot)
epochs_list=(3 10)                # 3 (cheap) vs 10 (paper PEFT used 10 -> fair compare)
accums=(1 8)                      # effective batch 4 (paper) vs 32 (standard for 8B FT)

for ga in "${accums[@]}"; do
 for ep in "${epochs_list[@]}"; do
  for lr in "${lrs[@]}"; do
   for d in "${datasets[@]}"; do
     tag="${d}_lr${lr}_ep${ep}_ga${ga}"
     [ -f "results_hp/hp_${tag}.jsonl" ] && { echo "skip $tag"; continue; }
     OUT="saves/hp/train_${tag}"; mkdir -p "$OUT"
     echo "=== TRAIN $tag $(date -u +%H:%M:%S) ==="
     DATASET="$d" SEED="$SEED" EPOCHS="$ep" LEARNING_RATE="$lr" GRAD_ACCUM="$ga" MODEL="$MODEL" \
       OUTPUT_DIR="$OUT" WANDB_NAME="hp_${tag}" \
       envsubst < "$CFG/train.yaml" > "$OUT/train.yaml"
     llamafactory-cli train "$OUT/train.yaml" > "$OUT/t.log" 2>&1 || { echo "TRAIN FAIL $tag"; tail -3 "$OUT/t.log"; continue; }
     EV="saves/hp/eval_${tag}"; mkdir -p "$EV"
     DATASET="$d" SEED="$SEED" TRAINED_MODEL="$OUT" OUTPUT_DIR="$EV" WANDB_NAME="hp_eval_${tag}" \
       envsubst < "$CFG/eval.yaml" > "$EV/eval.yaml"
     llamafactory-cli train "$EV/eval.yaml" > "$EV/e.log" 2>&1 || { echo "EVAL FAIL $tag"; continue; }
     python scripts/peftbench/compute_metrics.py "$EV" "$d" > /dev/null 2>&1 && \
       cp "$EV/results.jsonl" "results_hp/hp_${tag}.jsonl"
     echo "=== $tag -> $(cat results_hp/hp_${tag}.jsonl 2>/dev/null | tr '\n' ' ')"
     rm -rf "$OUT"
   done
  done
 done
done
echo "=== HPARAM SWEEP DONE ==="
