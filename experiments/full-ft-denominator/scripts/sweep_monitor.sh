#!/usr/bin/env bash
# Self-contained overnight monitor for the denominator sweep. Polls 3 boxes, ntfy progress,
# commits results to git hourly, and TEARS DOWN all boxes at completion or an 11h hard cap.
CIDS="39922608 39923256 39922612"
REPO=/home/jon/projects/peft-finetune-expt
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -i /home/jon/.ssh/id_ed25519"
NTFY=https://ntfy.sh/claude-jon-alerts
addr() { vastai show instances --raw 2>/dev/null | python3 -c "import sys,json
for x in json.load(sys.stdin):
 if x['id']==$1 and x.get('ssh_host') and x.get('ssh_port'): print(x['ssh_host'],x['ssh_port']);break"; }
pull_commit() {
  mkdir -p $REPO/experiments/full-ft-denominator/data/sweep
  for c in $CIDS; do
    read -r H P < <(addr $c)
    [ -n "$H" ] && scp $SSHO -P "$P" root@"$H":/workspace/sweep_results.jsonl $REPO/experiments/full-ft-denominator/data/sweep/box_$c.jsonl >/dev/null 2>&1
  done
  cat $REPO/experiments/full-ft-denominator/data/sweep/box_*.jsonl > $REPO/experiments/full-ft-denominator/data/sweep/all.jsonl 2>/dev/null
  ( cd $REPO && git add experiments/full-ft-denominator/data/sweep/ >/dev/null 2>&1 && git commit -q -m "denom sweep results checkpoint $(date -u +%H%M)" >/dev/null 2>&1 && git push >/dev/null 2>&1 )
}
curl -s -H "Title: Denominator sweep started" -d "3 nodes. reg=all (LR2e-6/5ep). cheap x3 seeds + mid x1 (~43 jobs; giants deferred). Hourly progress; auto-teardown by morning." $NTFY >/dev/null
START=$(date +%s); CAP=$((11*3600))
for i in $(seq 1 200); do
  rows=0; done=0; detail=""
  for c in $CIDS; do
    read -r H P < <(addr $c); r=""
    [ -n "$H" ] && r=$(timeout 25 ssh $SSHO -p "$P" root@"$H" '[ -f /workspace/SWEEP_DONE ] && echo D; wc -l</workspace/sweep_results.jsonl 2>/dev/null' 2>/dev/null)
    n=$(echo "$r"|grep -oE '^[0-9]+'|tail -1); rows=$((rows+${n:-0}))
    echo "$r"|grep -q D && done=$((done+1)); detail="$detail $c:${n:-0}$(echo "$r"|grep -q D && echo D)"
  done
  el=$(( $(date +%s)-START ))
  echo "$(date -u -d '+7 hours' +%H:%M 2>/dev/null) ICT: rows=$rows done=$done/3 elapsed=$((el/60))min [$detail ]"
  if [ $((i % 6)) = 0 ]; then pull_commit; curl -s -H "Title: Denom sweep: $rows rows, $done/3 boxes done" -d "[$detail ] elapsed $((el/60))min" $NTFY >/dev/null; fi
  { [ "$done" = 3 ] || [ "$el" -ge "$CAP" ]; } && break
  sleep 600
done
pull_commit
final=$(wc -l < $REPO/experiments/full-ft-denominator/data/sweep/all.jsonl 2>/dev/null)
curl -s -H "Title: Denom sweep FINISHED — tearing down" -H "Priority: high" -d "Collected ${final:-0} rows, committed to git. Destroying 3 boxes. Run a summary in the morning session." $NTFY >/dev/null
for c in $CIDS; do printf 'y\n' | vastai destroy instance $c >/dev/null 2>&1; done
echo "TEARDOWN_DONE destroyed $CIDS ; final_rows=${final:-0}"
