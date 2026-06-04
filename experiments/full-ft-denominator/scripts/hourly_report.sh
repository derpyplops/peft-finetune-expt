#!/usr/bin/env bash
# Hourly status of the hparam sweep -> ntfy channel. Usage: hourly_report.sh <topic> [server]
# Runs as a local background loop; SSHes to the box each hour, posts progress + latest results.
TOPIC="${1:?usage: hourly_report.sh <ntfy-topic> [server]}"
SERVER="${2:-https://ntfy.sh}"
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p 18288 root@ssh8.vast.ai"

for i in $(seq 1 16); do
  n=$(ssh $SSHO 'ls /workspace/PEFT-Bench/results/ 2>/dev/null | wc -l' 2>/dev/null)
  done=$(ssh $SSHO 'grep -c "SEED42 SWEEP COMPLETE" /workspace/seed42.log 2>/dev/null' 2>/dev/null)
  cur=$(ssh $SSHO 'grep "TRAIN " /workspace/seed42.log | tail -1 | sed "s/=== TRAIN //;s/ ===.*//"' 2>/dev/null)
  latest=$(ssh $SSHO 'for f in $(ls -t /workspace/PEFT-Bench/results/full_*.jsonl 2>/dev/null | head -4); do echo "$(basename $f .jsonl|sed s/hp_//): $(head -1 $f | python3 -c "import sys,json;d=json.load(sys.stdin);print(next(iter(d.values())))" 2>/dev/null)"; done' 2>/dev/null)
  ict=$(date -u -d "+7 hours" +"%H:%M" 2>/dev/null || date -u +"%H:%M")
  if [ "${done:-0}" = "1" ]; then
    curl -s -H "Title: ✅ Full-FT sweep DONE (25/25) — ${ict} ICT" -H "Priority: high" \
      -d "All 25 datasets done. Recommended-config analysis pending." "$SERVER/$TOPIC" >/dev/null
    exit 0
  fi
  curl -s -H "Title: Full-FT sweep ${n:-?}/25 — ${ict} ICT" \
    -d "running: ${cur:-?}"$'\n'"recent:"$'\n'"${latest:-...}" "$SERVER/$TOPIC" >/dev/null
  sleep 3600
done
curl -s -H "Title: Hparam reporter stopped (16h cap)" -d "still ${n}/25; check manually" "$SERVER/$TOPIC" >/dev/null
