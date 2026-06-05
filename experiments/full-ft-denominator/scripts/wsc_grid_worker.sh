#!/usr/bin/env bash
# Runs a batch of WSC grid jobs on one box. Each arg is a job: METHOD:LR:SEED:EPOCHS
#   full:5e-6:42:10   lora:5e-5:42:10
# Appends one result row per job to $RESULTS (jsonl). Idempotent (skips tags already present).
#
# Correctness guards (from plan audit + self-review):
#  - LoRA eval scores the HELD-OUT split: vendored lora/eval.yaml uses `eval_dataset: ${DATASET}`
#    (no _eval), so we pass DATASET=${D}_eval. full/eval.yaml bakes `${DATASET}_eval`, so it gets DATASET=${D}.
#    Both evaluate wsc_eval (validation), never the train split.
#  - After every eval we ASSERT prediction count == |wsc_eval|.
#  - H3 (selection criterion): for full+seed42 keep all checkpoints, eval each, record per-checkpoint metrics.
#  - All result-row JSON is built via argv + temp files (never interpolated into python -c), since metric
#    JSON contains quotes/spaces.
set -uo pipefail
export WORK="${WORK:-/workspace}"; cd "$WORK/PEFT-Bench"
export HF_HOME="$WORK/hf"; export MODEL="${MODEL:-NousResearch/Meta-Llama-3-8B-Instruct}"
export WANDB_PROJECT=peftbench-wsc-grid
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GRAD_ACCUM="${GRAD_ACCUM:-1}"   # full_train.yaml.tmpl renders ${GRAD_ACCUM}; unset -> empty -> int*None crash
RESULTS="${RESULTS:-$WORK/grid_results.jsonl}"; touch "$RESULTS"
D="${DS:-wsc}"   # dataset; set DS=svamp etc. Configs use ${DATASET}; compute_metrics keys off $D.
CFGF=examples/peftbench/full/llama-3-8b-instruct
CFGL=examples/peftbench/lora/llama-3-8b-instruct
TMP="$WORK/saves/grid"; mkdir -p "$TMP"
NEVAL=$(python3 -c "import json;print(len(json.load(open('data/local/${D}_eval.json'))))")
echo "expected eval examples: $NEVAL"

# eval a full model dir or lora adapter on wsc_eval; writes metric json to $4; asserts pred count.
do_eval() {  # kind(full|lora)  model_dir  out_dir  metric_out_file  wandb_name
  local kind=$1 mdl=$2 ev=$3 mout=$4 wn=$5; mkdir -p "$ev"
  if [ "$kind" = full ]; then
    DATASET=$D SEED=$SEEDv TRAINED_MODEL=$mdl OUTPUT_DIR=$ev WANDB_NAME=$wn envsubst < $CFGF/eval.yaml > $ev/eval.yaml
  else
    DATASET=${D}_eval SEED=$SEEDv ADAPTER=$mdl OUTPUT_DIR=$ev WANDB_NAME=$wn envsubst < $CFGL/eval.yaml > $ev/eval.yaml
    sed -i "s#meta-llama/Meta-Llama-3-8B-Instruct#$MODEL#" $ev/eval.yaml
  fi
  llamafactory-cli train $ev/eval.yaml > $ev/log 2>&1 || { echo "EVALFAIL $wn" >&2; return 1; }
  local pf=$(ls $ev/generated_predictions.jsonl 2>/dev/null | head -1)
  if [ -n "$pf" ]; then local n=$(wc -l < "$pf"); [ "$n" = "$NEVAL" ] || { echo "ASSERT FAIL $wn: $n != $NEVAL" >&2; return 2; }; fi
  python scripts/peftbench/compute_metrics.py "$ev" "$D" > /dev/null 2>&1 || { echo "METRICFAIL $wn" >&2; return 3; }
  head -1 "$ev/results.jsonl" > "$mout"
}

run_one() {
  local job="$1"; IFS=: read -r method lr SEEDv ep reg <<< "$job"
  reg="${reg:-none}"   # optional 5th field: none|neftune|labelsmooth|bigbatch|wd|all (full-FT regularizers)
  local tag="${method}_lr${lr}_s${SEEDv}_ep${ep}_${reg}"
  grep -q "\"tag\": \"$tag\"" "$RESULTS" && { echo "skip done: $tag"; return; }
  local OUT="$TMP/train_$tag" EV="$TMP/eval_$tag" MFILE="$TMP/m_$tag.json" H3FILE="$TMP/h3_$tag.tsv"
  rm -rf "$OUT" "$EV" "$MFILE" "$H3FILE"; mkdir -p "$OUT"
  local H3="${H3_ENABLE:-0}"   # selection-criterion probe; OFF by default (needs big disk for multi-ckpt). Set H3_ENABLE=1 on the 400GB box.
  [ "$H3" = 1 ] && { [ "$method" = full ] && [ "$SEEDv" = 42 ] || H3=0; }
  echo "=== TRAIN $tag (H3=$H3) $(date -u +%H:%M:%S) ==="
  if [ "$method" = full ]; then
    DATASET=$D SEED=$SEEDv EPOCHS=$ep LEARNING_RATE=$lr MODEL=$MODEL OUTPUT_DIR=$OUT \
      SAVE_ONLY_MODEL=true WANDB_NAME=$tag envsubst < $CFGF/train.yaml > $OUT/train.yaml
    # H3: keep 2 checkpoints (ep~5, ep~10) for per-checkpoint metric scoring — 2x16GB fits the 100GB boxes
    [ "$H3" = 1 ] && sed -i 's/save_steps: 0.2/save_steps: 0.5/; s/eval_steps: 0.2/eval_steps: 0.5/; s/save_total_limit: 1/save_total_limit: 2/' $OUT/train.yaml
    case "$reg" in   # full-FT regularizer profiles (Tier-1, config-only)
      neftune)    echo "neftune_noise_alpha: 5.0" >> $OUT/train.yaml;;
      labelsmooth) echo "label_smoothing_factor: 0.1" >> $OUT/train.yaml;;
      bigbatch)   sed -i 's/^gradient_accumulation_steps:.*/gradient_accumulation_steps: 16/' $OUT/train.yaml;;
      wd)         sed -i 's/^weight_decay:.*/weight_decay: 0.1/' $OUT/train.yaml;;
      all)        echo "neftune_noise_alpha: 5.0" >> $OUT/train.yaml; echo "label_smoothing_factor: 0.1" >> $OUT/train.yaml
                  sed -i 's/^gradient_accumulation_steps:.*/gradient_accumulation_steps: 16/; s/^weight_decay:.*/weight_decay: 0.1/' $OUT/train.yaml;;
    esac
    if grep -nE '^(gradient_accumulation_steps|learning_rate|num_train_epochs|save_only_model|model_name_or_path|output_dir|seed|run_name|dataset):[[:space:]]*$' $OUT/train.yaml; then echo "RENDERFAIL $tag empty value(s) above"; rm -rf "$OUT" "$EV"; return; fi
    llamafactory-cli train $OUT/train.yaml > $OUT/log 2>&1 || { echo "TRAINFAIL $tag"; tail -8 $OUT/log; rm -rf "$OUT" "$EV"; return; }
  else
    DATASET=$D SEED=$SEEDv EPOCHS=$ep MODEL=$MODEL OUTPUT_DIR=$OUT WANDB_NAME=$tag envsubst < $CFGL/train.yaml > $OUT/train.yaml
    sed -i 's/push_to_hub: true/push_to_hub: false/; /push_to_hub_organization/d' $OUT/train.yaml
    sed -i "s#meta-llama/Meta-Llama-3-8B-Instruct#$MODEL#" $OUT/train.yaml
    if grep -nE '^(gradient_accumulation_steps|learning_rate|num_train_epochs|save_only_model|model_name_or_path|output_dir|seed|run_name|dataset):[[:space:]]*$' $OUT/train.yaml; then echo "RENDERFAIL $tag empty value(s) above"; rm -rf "$OUT" "$EV"; return; fi
    llamafactory-cli train $OUT/train.yaml > $OUT/log 2>&1 || { echo "TRAINFAIL $tag"; tail -8 $OUT/log; rm -rf "$OUT" "$EV"; return; }
  fi

  # primary eval = best-by-loss model (load_best_model_at_end leaves it at OUT root)
  local kind=full; [ "$method" = lora ] && kind=lora
  do_eval "$kind" "$OUT" "$EV" "$MFILE" eval_$tag || { echo "EVALFAIL $tag"; rm -rf "$OUT" "$EV"; return; }

  # H3: eval every kept checkpoint by the metric (results -> H3FILE as step<TAB>json)
  : > "$H3FILE"
  if [ "$H3" = 1 ]; then
    for ck in $(ls -d $OUT/checkpoint-* 2>/dev/null); do
      local step=$(basename $ck | sed 's/checkpoint-//') cmf="$TMP/ckm_${tag}_${step}.json"
      if do_eval full "$ck" "${EV}_ck${step}" "$cmf" eval_${tag}_ck${step} 2>/dev/null; then
        printf '%s\t%s\n' "$step" "$(cat "$cmf")" >> "$H3FILE"
      fi
      rm -rf "${EV}_ck${step}" "$cmf"
    done
  fi

  # build the result row via argv + files (no json interpolation into -c)
  python3 - "$tag" "$method" "$lr" "$SEEDv" "$ep" "$(hostname)" "$MFILE" "$H3FILE" >> "$RESULTS" <<'PYEOF'
import json,sys,os
tag,method,lr,seed,ep,host,mfile,h3file=sys.argv[1:9]
m=json.loads(open(mfile).read().strip().split('\n')[0])
row={'tag':tag,'method':method,'lr':lr,'seed':int(seed),'epochs':int(ep),'metrics':m,'host':host}
if os.path.exists(h3file) and os.path.getsize(h3file)>0:
    cd={}
    for line in open(h3file):
        line=line.rstrip('\n')
        if not line: continue
        s,j=line.split('\t',1)
        try: cd[s]=json.loads(j)
        except Exception: pass
    if cd: row['checkpoint_metrics']=cd
print(json.dumps(row))
PYEOF
  echo "=== DONE $tag -> $(tail -1 $RESULTS)"
  [ "${KEEP:-0}" = 1 ] && echo "KEEP=1: preserving $EV (generated_predictions for inspection)" || rm -rf "$OUT" "$EV" "$MFILE" "$H3FILE"
}

for job in "$@"; do run_one "$job"; done
echo "=== BATCH DONE on $(hostname) $(date -u +%H:%M:%S) ($# jobs) ==="
