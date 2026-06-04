#!/usr/bin/env python3
"""Preliminary recovery plot from the seed-42 denominators we have so far (10 datasets).
Reveals: full-FT (our denominator) frequently underperforms PEFT and even base on several
datasets -> the "% recovered" framing inverts (>100%). Flags datasets where full-FT < base."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# our full-FT denominators (seed 42), headline metric x100
FULLFT = {"cb":95.0,"copa":77.5,"mrpc":88.4,"rte":79.4,"stsb":85.0,"wic":71.4,
          "wsc":0.0,"openbookqa":10.1,"svamp":62.0,"gsm8k":38.5}
BASE = {"cb":47.0,"copa":73.8,"mrpc":78.8,"rte":69.7,"stsb":67.4,"wic":66.6,
        "wsc":43.6,"openbookqa":78.3,"svamp":56.7,"gsm8k":79.2}
# paper PEFT scores (Tables 1-2), per method
METHODS = ["BitFit","IA3","LNTuning","PromptTuning","LoRA","PrefixTuning","PTuning"]
PARAMS = {"BitFit":163840,"IA3":196608,"LNTuning":266240,"PromptTuning":409600,
          "LoRA":14680064,"PrefixTuning":34177536,"PTuning":53130752}
PEFT = {  # dataset -> {method: score}
 "cb":{"BitFit":67.7,"IA3":68.9,"LNTuning":67.6,"PromptTuning":41.3,"LoRA":83.9,"PrefixTuning":49.8,"PTuning":6.0},
 "copa":{"BitFit":96.6,"IA3":96.6,"LNTuning":99.7,"PromptTuning":79.5,"LoRA":97.8,"PrefixTuning":7.5,"PTuning":92.3},
 "mrpc":{"BitFit":88.6,"IA3":86.7,"LNTuning":86.8,"PromptTuning":81.4,"LoRA":91.0,"PrefixTuning":38.8,"PTuning":89.4},
 "rte":{"BitFit":87.0,"IA3":84.5,"LNTuning":84.4,"PromptTuning":68.7,"LoRA":85.7,"PrefixTuning":72.5,"PTuning":0.0},
 "stsb":{"BitFit":89.8,"IA3":88.3,"LNTuning":89.8,"PromptTuning":59.3,"LoRA":90.7,"PrefixTuning":82.8,"PTuning":3.1},
 "wic":{"BitFit":66.5,"IA3":72.1,"LNTuning":74.2,"PromptTuning":65.1,"LoRA":75.2,"PrefixTuning":60.1,"PTuning":70.8},
 "wsc":{"BitFit":36.1,"IA3":42.0,"LNTuning":48.4,"PromptTuning":5.6,"LoRA":53.6,"PrefixTuning":0.0,"PTuning":0.0},
 "openbookqa":{"BitFit":84.1,"IA3":83.7,"LNTuning":85.3,"PromptTuning":65.8,"LoRA":87.7,"PrefixTuning":50.1,"PTuning":79.7},
 "svamp":{"BitFit":86.7,"IA3":85.3,"LNTuning":85.7,"PromptTuning":55.7,"LoRA":88.0,"PrefixTuning":85.7,"PTuning":30.7},
 "gsm8k":{"BitFit":68.3,"IA3":68.3,"LNTuning":68.3,"PromptTuning":73.6,"LoRA":69.1,"PrefixTuning":37.8,"PTuning":49.2},
}
DSETS = list(FULLFT)
# valid ceiling = full-FT >= base (an actual improvement from fine-tuning)
valid = [d for d in DSETS if FULLFT[d] >= BASE[d]]
broken = [d for d in DSETS if FULLFT[d] < BASE[d]]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

# --- LEFT: per-dataset Base vs best-PEFT(LoRA) vs Full-FT ---
x = np.arange(len(DSETS)); w = 0.27
lora = [PEFT[d]["LoRA"] for d in DSETS]
axL.bar(x-w, [BASE[d] for d in DSETS], w, label="Base (frozen)", color="#bbb")
axL.bar(x,   lora, w, label="LoRA (best PEFT)", color="#4C72B0")
axL.bar(x+w, [FULLFT[d] for d in DSETS], w, label="Full-FT (our denom.)", color="#C44E52")
axL.set_xticks(x); axL.set_xticklabels(DSETS, rotation=45, ha="right")
axL.set_ylabel("score (%)"); axL.set_title("Full-FT vs LoRA vs Base (seed 42, preliminary)")
axL.legend(); axL.grid(axis="y", alpha=0.3)
for d in broken:
    i = DSETS.index(d); axL.annotate("full-FT<base", (i+w, FULLFT[d]+2), fontsize=7, color="#C44E52", ha="center")

# --- RIGHT: recovery vs adapter size, mean over valid-ceiling datasets ---
xs = [PARAMS[m] for m in METHODS]
ys = [np.mean([PEFT[d][m]/FULLFT[d]*100 for d in valid]) for m in METHODS]
axR.scatter(xs, ys, s=80, color="#4C72B0", zorder=3)
for m, xv, yv in zip(METHODS, xs, ys):
    axR.annotate(m, (xv, yv), textcoords="offset points", xytext=(6,4), fontsize=8)
axR.axhline(100, ls="--", color="green", label="100% = full-FT ceiling")
axR.scatter([8.03e9], [100], marker="*", s=200, color="green", zorder=3, label="Full-FT (8.03B params)")
axR.set_xscale("log"); axR.set_xlabel("trainable params (adapter size, log)")
axR.set_ylabel("% of full-FT recovered (mean over valid datasets)")
axR.set_title(f"Recovery vs adapter size  (n={len(valid)} valid-ceiling datasets)")
axR.legend(loc="lower left", fontsize=8); axR.grid(alpha=0.3)

fig.suptitle(f"PRELIMINARY (seed 42, {len(DSETS)}/25 datasets, mixed 3-10 epochs) — "
             f"full-FT < base on: {', '.join(broken)}", fontsize=11)
fig.tight_layout()
out = "/home/jon/projects/peft-finetune-expt/experiments/full-ft-denominator/figures/preliminary_recovery.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print("valid-ceiling datasets:", valid)
print("BROKEN (full-FT < base):", broken)
print("\nmean % recovered per method (over valid datasets):")
for m, yv in zip(METHODS, ys): print(f"  {m:14s} {yv:5.1f}%  (adapter {PARAMS[m]:,})")
