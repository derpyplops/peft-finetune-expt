# WSC investigation: why does full-FT (SFT) underperform PEFT?

Started 2026-06-05. Sub-investigation of [the full-FT denominator experiment](plan.md).

## Motivation

The denominator sweep shows **full-parameter fine-tuning (SFT) loses to the best PEFT method on 12/14 scored datasets** (see [`reports/sft_vs_best_peft.md`](reports/sft_vs_best_peft.md)). Before treating that as a real result, we need to know *why*. WSC is the sharpest case to probe:

| WSC (F1) | value |
|---|---|
| Base (un-tuned Llama-3-8B-Instruct) | 43.6 |
| Our SFT (seed 42, 5 epochs, best-val) | **31.7** ← below base |
| Best PEFT (LoRA, paper) | 53.6 |

SFT scoring *below the base model* on a tiny (554-example) 2-class task is exactly what high-variance training + an unlucky seed, a bad LR, or a flawed selection/eval looks like. WSC is also the smallest dataset, so we can run a full grid in minutes per run.

## Hypotheses

1. **Seed variance.** 554 examples → full-FT is unstable; a single seed (42) may be an outlier. Test with 3 seeds.
2. **Learning rate.** We fixed LR=5e-6 from a small sweep. Full-FT in the low-data regime is LR-sensitive. Sweep 4 LRs.
3. **Selection criterion.** `load_best_model_at_end` picks the **min eval-loss** checkpoint, which for generation tasks can be a worse *metric* checkpoint than another. (Probed in a follow-up; not in this grid.)
4. **Eval fairness.** Our SFT and the paper's PEFT may not run through an identical eval. **Decisive control: reproduce LoRA through our own pipeline** and check we recover ~53.6.

## Method

Focus: **WSC only.** Parallelize 4× across H200s to keep wall-clock ~30–40 min.

### Grid (15 runs)
- **SFT:** LR ∈ {2e-6, 5e-6, 1e-5, 2e-5} × seed ∈ {42, 1, 2} @ **10 epochs** → 12 runs.
  (10 epochs both matches the paper's PEFT budget — removing the 5-vs-10 asymmetry — and gives WSC the room it wants; its best-val landed at epoch 6 earlier.)
- **LoRA reproduction:** paper config (rank 16, α16, dropout 0.05, target k/v/down_proj, LR 5e-5, 10 epochs, greedy decode) × seed ∈ {42, 1, 2} → 3 runs.

### Compute
- 4 workers: the existing box (39288289, H200, paused mid-sweep) + **3 new H200** (contracts 39560474, 39560479, 39560481), image `hiyouga/llamafactory:latest`, ~$3.8/hr each.
- Jobs partitioned 4 ways (~4 each). Full-FT 8B needs ~126 GB so one run per GPU (no intra-box parallelism).
- **Cost:** 3 new × ~$3.8/hr × ~1 hr ≈ **$11–12** + the existing box.

### Orchestration — a Workflow manages it end to end
- **Setup** (parallel, one agent per new box): `setup_remote.sh` (clone PEFT-Bench, install envsubst + deps, prefetch model, wandb login) + scp WSC data (`wsc_{train,eval}.json`), patched `dataset_info.json`, full + LoRA config templates, and the worker script; verify GPU.
- **Run** (parallel, one agent per box): run its job batch via `wsc_grid_worker.sh`, each job = train → eval → score WSC F1, append a result row.
- **Synthesize** (one agent): collect all rows → WSC F1 table with **mean ± std per LR** and the **LoRA-repro vs paper-LoRA(53.6) vs base(43.6)** comparison; write `reports/wsc_grid.md`.

## Decision criteria (what each outcome means)

- **LoRA repro ≈ 53.6** → eval is fair; the SFT gap is real → keep investigating SFT (selection criterion, regularization, partial-FT).
- **LoRA repro ≠ 53.6** → our pipeline differs from the paper → the entire SFT-vs-PEFT comparison needs recalibration before any conclusion.
- **SFT mean ± std spans 31.7 → ~50** across seeds → variance was the culprit; report mean, not the seed-42 point.
- **A better LR lifts SFT above base (43.6) / toward 53.6** → we were mistuned; re-run the denominator at that LR.

## Status
- 2026-06-05: 3× H200 launched (39560474/79/81). WSC data pulled. LoRA config located (`vendor/PEFT-Factory/examples/peftbench/lora/llama-3-8b-instruct/`). Main denominator sweep **paused** at 15/25 (resumes after this). Building worker + workflow.
