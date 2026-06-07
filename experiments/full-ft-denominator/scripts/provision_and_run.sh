#!/usr/bin/env bash
# $1 = vast contract id ; $2 = file with this box's jobs ("dataset|spec" per line)
# Resolves address, waits (dud-detects), sets up, materializes all datasets, launches the box's detached sweep queue.
CID=$1; JOBSFILE=$2
WK=$(grep WANDB_API_KEY /home/jon/projects/peft-finetune-expt/.env.local | sed 's/.*=//')
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i /home/jon/.ssh/id_ed25519"
addr() { vastai show instances --raw 2>/dev/null | python3 -c "import sys,json
for x in json.load(sys.stdin):
 if x['id']==$CID and x.get('ssh_host') and x.get('ssh_port'): print(x['ssh_host'],x['ssh_port']);break"; }
# wait for SSH-ready (dud detection ~6 min)
H=""; P=""; ready=0
for i in $(seq 1 36); do
  read -r H P < <(addr)
  if [ -n "$H" ] && [ -n "$P" ]; then
    ssh $SSHO -p "$P" root@"$H" 'echo READY' 2>/dev/null | grep -q READY && { ready=1; break; }
  fi
  sleep 10
done
[ "$ready" = 0 ] && { echo "DUD_OR_UNREACHABLE $CID (addr=$H:$P)"; exit 1; }
echo "box $CID ready at $H:$P"
# deploy bundle + setup (clone, deps, model prefetch, configs)
scp $SSHO -P "$P" -r /tmp/grid_bundle root@"$H":/workspace/bundle >/dev/null 2>&1
ssh $SSHO -p "$P" root@"$H" "bash /workspace/bundle/setup_box.sh NEW $WK" 2>&1 | grep -E "SETUP_BOX_DONE|Error|error" | tail -2
ssh $SSHO -p "$P" root@"$H" "cp /workspace/bundle/materialize.py /workspace/; cp /workspace/bundle/recipes_all.json /workspace/; cp /workspace/bundle/sweep_box.sh /workspace/; chmod +x /workspace/sweep_box.sh /workspace/wsc_grid_worker.sh"
# materialize all datasets (detached + poll up to ~30 min)
ssh $SSHO -p "$P" root@"$H" "rm -f /workspace/MAT_DONE; cd /workspace && setsid bash -c 'python3 /workspace/materialize.py > /workspace/materialize.log 2>&1; touch /workspace/MAT_DONE' >/dev/null 2>&1 & echo MAT_LAUNCHED"
for i in $(seq 1 60); do
  st=$(ssh $SSHO -p "$P" root@"$H" '[ -f /workspace/MAT_DONE ] && echo DONE; ls /workspace/PEFT-Bench/data/local/*_eval.json 2>/dev/null | wc -l' 2>/dev/null)
  echo "$CID materialize: $(echo $st | tr '\n' ' ')"
  echo "$st" | grep -q DONE && break
  sleep 30
done
nds=$(ssh $SSHO -p "$P" root@"$H" 'ls /workspace/PEFT-Bench/data/local/*_eval.json 2>/dev/null | wc -l' 2>/dev/null)
echo "$CID materialized $nds datasets"
[ "${nds:-0}" -lt 5 ] && { echo "MATERIALIZE_FAIL $CID ($nds datasets) — not launching"; exit 2; }
# launch the box's ordered sweep queue, detached
JOBS=$(tr '\n' ' ' < "$JOBSFILE")
ssh $SSHO -p "$P" root@"$H" "rm -f /workspace/SWEEP_DONE; cd /workspace/PEFT-Bench && setsid bash -c 'RESULTS=/workspace/sweep_results.jsonl WORK=/workspace bash /workspace/sweep_box.sh $JOBS > /workspace/sweep.log 2>&1; touch /workspace/SWEEP_DONE' >/dev/null 2>&1 & echo SWEEP_LAUNCHED_$CID"
sleep 5
echo "PROVISION_RUN_DONE $CID host=$H port=$P datasets=$nds jobs=$(wc -l < $JOBSFILE)"
