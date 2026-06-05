#!/usr/bin/env bash
# Detached swap watcher (runs on the box). When the in-flight old sweep finishes multirc,
# kill it and start the checkpoint-saving sweep so every remaining dataset (incl. the big ones)
# gets pushed to Wasabi. multirc itself completes naturally under the old script (no recompute waste).
set -uo pipefail
LOG=/workspace/swap_watch.log
cd /workspace/PEFT-Bench
echo "$(date -u) watcher up; waiting for results/full_multirc_s42.jsonl" >> "$LOG"
while [ ! -f /workspace/PEFT-Bench/results/full_multirc_s42.jsonl ]; do sleep 30; done
echo "$(date -u) multirc scored -> swapping" >> "$LOG"
pkill -f '[r]un_seed42.sh' 2>/dev/null || true        # old loop (won't match run_seed42_ckpt.sh)
sleep 3
pkill -9 -f '[l]lamafactory-cli train saves/full/seed42' 2>/dev/null || true
for i in $(seq 1 120); do                              # wait up to 10min for VRAM to free
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${m:-99999}" -lt 5000 ] && { echo "$(date -u) gpu free (${m}MiB)" >> "$LOG"; break; }
  sleep 5
done
WORK=/workspace LR=5e-6 EPOCHS=5 GRAD_ACCUM=1 MODEL=NousResearch/Meta-Llama-3-8B-Instruct \
  setsid bash /workspace/run_seed42_ckpt.sh >> /workspace/seed42_ckpt.log 2>&1 < /dev/null &
disown
sleep 8
echo "$(date -u) ckpt sweep launched PID $(pgrep -f '[r]un_seed42_ckpt'|head -1)" >> "$LOG"
