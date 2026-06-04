#!/usr/bin/env python3
"""Analyze the 40-run hparam sweep: full-FT score per config vs base & best-PEFT,
to pick settings where full-FT is a proper ceiling (>= base, ideally >= best PEFT)."""
import json, glob, os, re
from collections import defaultdict

LOCAL = "/home/jon/projects/peft-finetune-expt/experiments/full-ft-denominator/data/results_hp"
BASE = {"cb":47.0,"copa":73.8,"wsc":43.6,"svamp":56.7,"rte":69.7}
BESTPEFT = {"cb":83.9,"copa":99.7,"wsc":53.6,"svamp":88.0,"rte":87.0}  # max over PEFT methods (paper)
DSETS = ["cb","copa","wsc","svamp","rte"]

def headline(d):
    for k in ["macro_f1","pearsonr","accuracy","codebleu","f1"]:
        if k in d: return d[k]*100
    return None

# parse: hp_<d>_lr<lr>_ep<ep>_ga<ga>.jsonl
scores = {}  # (config) -> {dataset: score}
for f in glob.glob(f"{LOCAL}/hp_*.jsonl"):
    m = re.match(r"hp_(\w+?)_lr([\d.e-]+)_ep(\d+)_ga(\d+)", os.path.basename(f))
    if not m: continue
    d, lr, ep, ga = m.group(1), m.group(2), m.group(3), m.group(4)
    merged = {}
    for line in open(f):
        line=line.strip()
        if line: merged.update(json.loads(line))
    cfg = f"lr{lr}/ep{ep}/eff{4*int(ga)}"
    scores.setdefault(cfg, {})[d] = headline(merged)

print(f"{'config':22s} " + " ".join(f"{d:>6s}" for d in DSETS) + "  | mean Δbase  meanΔbestPEFT  #>=base #>=PEFT")
print("-"*100)
rows=[]
for cfg in sorted(scores):
    sc = scores[cfg]
    vals = [sc.get(d) for d in DSETS]
    dbase = [vals[i]-BASE[d] for i,d in enumerate(DSETS) if vals[i] is not None]
    dpeft = [vals[i]-BESTPEFT[d] for i,d in enumerate(DSETS) if vals[i] is not None]
    nbase = sum(1 for i,d in enumerate(DSETS) if vals[i] is not None and vals[i]>=BASE[d])
    npeft = sum(1 for i,d in enumerate(DSETS) if vals[i] is not None and vals[i]>=BESTPEFT[d])
    mb = sum(dbase)/len(dbase); mp = sum(dpeft)/len(dpeft)
    rows.append((cfg, vals, mb, mp, nbase, npeft))
# sort by: beats base everywhere, then mean vs PEFT
for cfg, vals, mb, mp, nbase, npeft in sorted(rows, key=lambda r:(-r[4], -r[3])):
    vs = " ".join(f"{v:6.1f}" if v is not None else "   n/a" for v in vals)
    print(f"{cfg:22s} {vs}  | {mb:+7.1f}    {mp:+7.1f}      {nbase}/5    {npeft}/5")

print(f"\nbase:     " + " ".join(f"{BASE[d]:6.1f}" for d in DSETS))
print(f"bestPEFT: " + " ".join(f"{BESTPEFT[d]:6.1f}" for d in DSETS))
