#!/usr/bin/env python3
"""Compare our full-FT (SFT) results vs the paper's best PEFT result per dataset.
PEFT numbers parsed verbatim from arxiv 2511.21285v1 Tables 1 (NLU) & 2 (math/QA/code).
NOTE: the paper reports 6 methods (IA3, Prompt, Prefix, P-Tuning, LoRA, LNTuning) — NO BitFit.
Writes reports/sft_vs_best_peft.md."""
import json, glob, os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "data", "results")

# paper Tables 1+2 — per (dataset): base, and {method: score}. Verbatim from HTML parse.
PEFT = {
    # NLU (Table 1, F1; STS-B Pearson)
    "cb":   (47.0, {"IA3":68.9,"Prompt":41.3,"Prefix":49.8,"P-Tuning":6.0,"LoRA":83.9,"LNTuning":67.6}),
    "copa": (73.8, {"IA3":96.6,"Prompt":79.5,"Prefix":7.5,"P-Tuning":92.3,"LoRA":97.8,"LNTuning":97.7}),
    "wsc":  (43.6, {"IA3":42.0,"Prompt":5.6,"Prefix":0.0,"P-Tuning":0.0,"LoRA":53.6,"LNTuning":48.4}),
    "rte":  (69.7, {"IA3":84.5,"Prompt":68.7,"Prefix":72.5,"P-Tuning":0.0,"LoRA":85.7,"LNTuning":84.4}),
    "mrpc": (78.8, {"IA3":86.7,"Prompt":81.4,"Prefix":38.8,"P-Tuning":89.4,"LoRA":91.0,"LNTuning":86.8}),
    "wic":  (66.6, {"IA3":72.1,"Prompt":65.1,"Prefix":60.1,"P-Tuning":70.8,"LoRA":75.2,"LNTuning":74.2}),
    "stsb": (67.4, {"IA3":88.3,"Prompt":59.3,"Prefix":82.8,"P-Tuning":3.1,"LoRA":90.7,"LNTuning":89.8}),
    "cola": (78.3, {"IA3":88.9,"Prompt":76.8,"Prefix":45.5,"P-Tuning":85.8,"LoRA":89.7,"LNTuning":89.3}),
    "boolq":(80.8, {"IA3":89.7,"Prompt":77.0,"Prefix":72.2,"P-Tuning":86.4,"LoRA":91.0,"LNTuning":89.7}),
    # math/QA/code (Table 2; accuracy for math, CodeBLEU for code, F1 others)
    "openbookqa":(78.3,{"IA3":83.7,"Prompt":65.8,"Prefix":50.1,"P-Tuning":79.7,"LoRA":87.7,"LNTuning":85.3}),
    "piqa": (46.0, {"IA3":86.0,"Prompt":57.4,"Prefix":45.3,"P-Tuning":80.3,"LoRA":88.9,"LNTuning":86.4}),
    "gsm8k":(79.2, {"IA3":68.3,"Prompt":73.6,"Prefix":37.8,"P-Tuning":49.2,"LoRA":69.1,"LNTuning":68.3}),
    "svamp":(56.7, {"IA3":85.3,"Prompt":55.7,"Prefix":85.7,"P-Tuning":30.7,"LoRA":88.0,"LNTuning":85.7}),
    "codealpacapy":(31.2,{"IA3":32.0,"Prompt":32.1,"Prefix":9.1,"P-Tuning":32.3,"LoRA":35.0,"LNTuning":32.6}),
}

def headline(d):  # same priority as analyze_hparam.py
    for k in ("macro_f1","pearsonr","accuracy","codebleu","f1"):
        if k in d: return d[k]*100
    return None

def our_score(ds):
    merged = {}
    for f in glob.glob(os.path.join(RESULTS, f"full_{ds}_s42.jsonl")):
        for line in open(f):
            line=line.strip()
            if line: merged.update(json.loads(line))
    return headline(merged)

rows=[]
for ds,(base,methods) in PEFT.items():
    sft = our_score(ds)
    bestm = max(methods, key=methods.get); bestv = methods[bestm]
    rows.append((ds, base, sft, bestv, bestm))

lines = ["# Full-FT (SFT) vs best PEFT — per dataset\n",
         "SFT = our full-parameter fine-tune (seed 42, 5 epochs, best-val checkpoint).",
         "Best PEFT = max over the paper's 6 methods (Tables 1–2 of arXiv 2511.21285). No BitFit in paper.\n",
         "| Dataset | Base | SFT (ours) | Best PEFT | (method) | Δ SFT−PEFT | Winner |",
         "|---|---|---|---|---|---|---|"]
sft_wins=0; n=0
for ds,base,sft,bestv,bestm in rows:
    if sft is None:
        lines.append(f"| {ds} | {base:.1f} | _pending_ | {bestv:.1f} | {bestm} | — | — |"); continue
    n+=1; d=sft-bestv; win="**SFT**" if d>=0 else "PEFT"
    if d>=0: sft_wins+=1
    lines.append(f"| {ds} | {base:.1f} | {sft:.1f} | {bestv:.1f} | {bestm} | {d:+.1f} | {win} |")
lines.append(f"\n**SFT beats best PEFT on {sft_wins}/{n} datasets.**")
out = os.path.join(HERE, "reports", "sft_vs_best_peft.md")
os.makedirs(os.path.dirname(out), exist_ok=True)
open(out,"w").write("\n".join(lines)+"\n")
print("\n".join(lines))
print(f"\nwrote {out}")
