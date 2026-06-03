#!/usr/bin/env bash
# One-time environment setup on the vast H200 box (image: hiyouga/llamafactory:latest).
# Secrets (HF_TOKEN optional, WANDB_API_KEY) are passed via env — never baked into the repo.
set -euo pipefail

: "${WANDB_API_KEY:?set WANDB_API_KEY in env}"
export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"   # ungated mirror; weights == meta-llama

# work on the big rented disk
export WORK=/workspace
mkdir -p "$WORK" 2>/dev/null || export WORK=/root
cd "$WORK"
export HF_HOME="$WORK/hf"; mkdir -p "$HF_HOME"

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# PEFT-Bench: dataset registry (data/dataset_info.json), compute_metrics.py, configs
[ -d PEFT-Bench ] || git clone --depth 1 https://github.com/kinit-sk/PEFT-Bench.git

# extra deps the image may lack (CodeBLEU for code datasets; evaluate metrics)
pip install -q codebleu evaluate sacrebleu scipy scikit-learn 2>&1 | tail -3 || true

# auth
wandb login "$WANDB_API_KEY"
[ -n "${HF_TOKEN:-}" ] && huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || echo "(no HF_TOKEN; using ungated mirror)"

# place full-FT configs (scp'd to $WORK by the launcher)
DST=PEFT-Bench/examples/peftbench/full/llama-3-8b-instruct
mkdir -p "$DST"
cp "$WORK/full_train.yaml.tmpl" "$DST/train.yaml"
cp "$WORK/full_eval.yaml.tmpl"  "$DST/eval.yaml"

# pre-fetch the model once (so runs don't race on first download)
python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', ignore_patterns=['original/*','*.pth','*.gguf'])"

echo "=== SETUP DONE === WORK=$WORK MODEL=$MODEL"
