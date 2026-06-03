#!/usr/bin/env python3
"""Materialize all 27 PEFT-Bench datasets locally from verified build() recipes.
Self-contained: bakes each task instruction into the prompt, verbalizes labels, writes
data/local/<name>_{train,eval}.json, and rewrites dataset_info.json to point at them.
Run on the box: python3 materialize.py"""
import json, os, sys, traceback, warnings
warnings.filterwarnings("ignore")
from datasets import load_dataset

ROOT = "/workspace/PEFT-Bench"
DATA = f"{ROOT}/data"
LOCAL = f"{DATA}/local"
os.makedirs(LOCAL, exist_ok=True)
recipes = json.load(open("/workspace/recipes_all.json"))
info = json.load(open(f"{DATA}/dataset_info.json"))

def bake(instruction, prompt, query):
    parts = []
    if instruction: parts.append(instruction.strip())
    if prompt: parts.append(str(prompt).strip())
    if query: parts.append(str(query).strip())
    return "\n\n".join(parts)

def materialize_split(build, src, sub, split, instruction):
    ds = load_dataset(src, sub if sub else None, split=split)
    rows = []
    for ex in ds:
        try:
            r = build(ex)
        except Exception:
            r = None
        if not r: continue
        p = bake(instruction, r.get("prompt"), r.get("query"))
        resp = r.get("response")
        if not p or resp is None or str(resp).strip() == "": continue
        rows.append({"prompt": p, "response": str(resp)})
    return rows

summary = []
for rec in recipes:
    name = rec["name"]
    try:
        ns = {}
        exec(rec["builder_code"], ns)
        build = ns["build"]
        instr = (info.get(name) or {}).get("instruction", "")
        tr = materialize_split(build, rec["hf_hub_url"], rec["subset"], rec["train_split"], instr)
        ev = materialize_split(build, rec["hf_hub_url"], rec["subset"], rec["eval_split"], instr)
        if not tr or not ev:
            summary.append((name, f"EMPTY train={len(tr)} eval={len(ev)}")); continue
        json.dump(tr, open(f"{LOCAL}/{name}_train.json", "w"))
        json.dump(ev, open(f"{LOCAL}/{name}_eval.json", "w"))
        cols = {"prompt": "prompt", "response": "response"}
        info[name] = {"file_name": f"local/{name}_train.json", "columns": cols}
        info[f"{name}_eval"] = {"file_name": f"local/{name}_eval.json", "columns": cols}
        summary.append((name, f"OK train={len(tr)} eval={len(ev)}"))
        print(f"  {name}: train={len(tr)} eval={len(ev)}", flush=True)
    except Exception as e:
        summary.append((name, f"FAIL {type(e).__name__}: {str(e)[:80]}"))
        print(f"  {name}: FAIL {e}", flush=True)

json.dump(info, open(f"{DATA}/dataset_info.json", "w"), indent=2)
ok = [s for s in summary if s[1].startswith("OK")]
print(f"\n=== MATERIALIZED {len(ok)}/27 ===")
for n, s in summary:
    if not s.startswith("OK"): print(f"  NOT OK: {n}: {s}")
