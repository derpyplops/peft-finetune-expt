#!/usr/bin/env python3
"""Patch PEFT-Bench data/dataset_info.json for datasets==4.0 (no script-based loading).
- cb / boolq (+ _eval): super_glue -> aps/super_glue (parquet, identical columns/data).
- conala: probe parquet mirrors; patch if one has snippet + an intent-like column.
Idempotent. Run from anywhere; path is fixed."""
import json, sys, warnings
warnings.filterwarnings("ignore")

PATH = "/workspace/PEFT-Bench/data/dataset_info.json"
info = json.load(open(PATH))

# 1) cb/boolq -> aps/super_glue (script-based super_glue is dead on datasets 4.0)
for k in ["cb", "cb_eval", "boolq", "boolq_eval"]:
    if k in info and info[k].get("hf_hub_url") == "super_glue":
        info[k]["hf_hub_url"] = "aps/super_glue"
        print(f"patched {k} -> aps/super_glue")

# 2) conala: find a working parquet mirror with snippet + (rewritten_intent|intent)
from datasets import load_dataset
cands = ["AhmedSSoliman/CoNaLa", "codeparrot/conala-mined-curated", "HydraLM/conala_standardized"]
chosen = None
for c in cands:
    try:
        ex = next(iter(load_dataset(c, split="train", streaming=True)))
        cols = set(ex.keys())
        if "snippet" in cols and (cols & {"rewritten_intent", "intent"}):
            chosen = (c, "rewritten_intent" if "rewritten_intent" in cols else "intent")
            print(f"conala mirror OK: {c} cols={sorted(cols)}")
            break
        print(f"conala mirror {c} missing cols: {sorted(cols)}")
    except Exception as e:
        print(f"conala mirror {c} BAD: {type(e).__name__}: {str(e)[:70]}")
if chosen:
    src, intent_col = chosen
    for k in ["conala", "conala_eval"]:
        if k in info:
            info[k]["hf_hub_url"] = src
            info[k]["subset"] = None
            info[k]["columns"] = {"prompt": intent_col, "response": "snippet"}
    print(f"patched conala/_eval -> {src} (prompt={intent_col})")
else:
    print("conala: NO clean parquet mirror found — leave as-is (will fail preflight; candidate to drop to 26/27)")

json.dump(info, open(PATH, "w"), indent=2)
print("dataset_info.json written")
