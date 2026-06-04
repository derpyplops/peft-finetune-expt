#!/usr/bin/env python3
"""For each completed seed-42 TRAIN run, find the epoch of min eval/loss (= the best-val
checkpoint load_best_model_at_end picks). Tells us whether 3 epochs captures the best."""
import wandb
api = wandb.Api()
seen = {}
for run in api.runs("caais/peftbench-fullft-seed42"):
    if run.name.startswith("full_eval_"):
        continue
    evs = []
    for row in run.scan_history(keys=["eval/loss", "train/epoch"]):
        if row.get("eval/loss") is not None and row.get("train/epoch") is not None:
            evs.append((row["train/epoch"], row["eval/loss"]))
    if not evs:
        continue
    best_ep, best_loss = min(evs, key=lambda x: x[1])
    last_ep = max(e for e, _ in evs)
    # keep the run with the most evals per name (the full 10-epoch one)
    prev = seen.get(run.name)
    if prev is None or len(evs) > prev[3]:
        seen[run.name] = (best_ep, best_loss, last_ep, len(evs))
print(f"=== best-val epoch per dataset ({len(seen)} runs) ===")
late = 0
for name in sorted(seen):
    bep, bl, last, n = seen[name]
    flag = "  <-- best is LATE (>3)" if bep > 3 else ""
    if bep > 3: late += 1
    print(f"  {name:24s} best eval/loss @ epoch {bep:5.2f} of {last:4.1f}  ({n} evals){flag}")
print(f"\n{late}/{len(seen)} runs have their best-val checkpoint AFTER epoch 3.")
