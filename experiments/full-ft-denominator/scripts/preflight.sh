#!/usr/bin/env bash
# Phase 2a preflight: validate the FULL chain (load->template->train->eval->metrics->label-parse)
# for ALL 27 datasets on a tiny 1B model with max_samples=16. Fast, model-agnostic, ~free.
# Finds every data-source / config / metric bug at once, before any 8B GPU time.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"
export MODEL="${MODEL:-unsloth/Llama-3.2-1B-Instruct}"   # ungated 1B, llama3 template
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
MS=16
datasets=(cb copa wsc svamp conala rte mrpc openbookqa apps wic stsb gsm8k cola boolq \
          piqa codealpacapy multirc math_qa siqa hellaswag winogrande sst2 mmlu record qnli qqp mnli)
mkdir -p /workspace/preflight
PASS=(); FAIL=()
for d in "${datasets[@]}"; do
  OUT=/workspace/preflight/$d; rm -rf "$OUT"; mkdir -p "$OUT"
  cat > "$OUT/train.yaml" <<YAML
model_name_or_path: $MODEL
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: full
dataset: $d
template: llama3
cutoff_len: 1024
max_samples: $MS
overwrite_cache: true
output_dir: $OUT/m
overwrite_output_dir: true
save_strategy: "no"
per_device_train_batch_size: 4
learning_rate: 1.0e-5
num_train_epochs: 1
bf16: true
report_to: none
YAML
  cat > "$OUT/eval.yaml" <<YAML
model_name_or_path: $OUT/m
trust_remote_code: true
stage: sft
do_train: false
do_predict: true
predict_with_generate: true
finetuning_type: full
eval_dataset: ${d}_eval
template: llama3
cutoff_len: 1024
max_samples: $MS
overwrite_cache: true
output_dir: $OUT/e
per_device_eval_batch_size: 4
report_to: none
YAML
  if llamafactory-cli train "$OUT/train.yaml" > "$OUT/train.log" 2>&1 \
     && llamafactory-cli train "$OUT/eval.yaml" > "$OUT/eval.log" 2>&1 \
     && python scripts/peftbench/compute_metrics.py "$OUT/e" "$d" > "$OUT/metric.log" 2>&1; then
    PASS+=("$d"); echo "PASS $d : $(tr '\n' ' ' < $OUT/e/results.jsonl 2>/dev/null)"
  else
    FAIL+=("$d"); echo "FAIL $d :"; \
    grep -hiE "error|not found|no such|BuilderConfig|no longer supported|KeyError|ValueError" \
      "$OUT"/train.log "$OUT"/eval.log "$OUT"/metric.log 2>/dev/null | tail -2 | sed 's/^/    /'
  fi
  rm -rf "$OUT/m"
done
echo "=== PREFLIGHT: ${#PASS[@]}/27 PASS, ${#FAIL[@]}/27 FAIL ==="
echo "FAIL: ${FAIL[*]}"
