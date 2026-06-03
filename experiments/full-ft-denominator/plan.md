# Plan: Full-parameter fine-tune of LLaMA-3-8B as the PEFT-Bench denominator

**Goal.** Produce per-dataset *full fine-tune* scores for LLaMA-3-8B-Instruct on the PEFT-Bench
datasets, to serve as the 100% reference (denominator) in the plot:

- **x-axis:** adapter size (trainable params per PEFT method, log scale)
- **y-axis:** % of full-FT performance recovered by each PEFT method

The PEFT-Bench paper only reports a frozen `Base` model row; it never ran a full fine-tune. This
experiment fills that gap.

---

## Open questions to confirm with collaborators (don't block setup, but resolve before the full sweep)

1. **Recovery metric definition** — `PEFT / FullFT` (raw ratio) vs `(PEFT − Base) / (FullFT − Base)`
   (gain-recovered). Changes every y-value. "Recovered" leans toward the gain-normalized form.
   → We only need full-FT scores either way, so this does **not** block fine-tuning. It does decide
   whether we must also re-confirm the `Base` numbers (we can lift those from the paper / repo).
2. **Dataset scope** — all 27, or just the subset that appears on the plot? Recommend all 27 to match
   the paper; the generation/reasoning tasks (GSM8K, Conala, CodeAlpacaPy, APPS) are the most
   interesting denominators since PEFT *hurt* there.
3. **Seeds** — paper uses 5. **Decision (2026-06-03): do seed 42 only first** (Phase 3a) for a full cheap
   pass across all 27 datasets, then decide whether to add the other 4 seeds (Phase 3b) based on results +
   budget. 5 seeds makes error bars directly comparable to the paper; 3 is a reasonable middle.

---

## Working principle: shorten feedback loops wherever possible

Default to the **fastest test that can surface the next failure**, not a faithful full run. The first launch
violated this — we discovered ~5 bugs (GNU `time` missing, datasets-4.0 source breakage, optimizer OOM,
120GB-checkpoint I/O, save_only_model) *serially*, each costing a ~25-min full-run cycle, when a 2-minute
tiny run would have caught most at once. Rules:
- **Tiny before full:** validate plumbing with `max_samples: 16` + `max_steps: 2` (whole train→save→eval→
  metrics chain in ~2 min). OOM hits at step 1, not epoch 10; save cost shows on save 1.
- **Small model before big:** iterate model-agnostic logic (configs, envsubst, dataset sources, metrics,
  label parsing) on **Llama-3.2-1B** (seconds, no OOM, ~2GB) — flip `MODEL` to 8B only for real runs.
- **Cheap before GPU:** data-source/config/metric bugs are laptop-discoverable; don't burn $4/hr H200 time on them.
- **Front-load breadth:** preflight ALL 27 datasets at once (find every data bug up front) rather than one-per-sweep.
- **Measure, don't infer:** a `max_steps: 20` run at real seq len gives true tok/s directly.

## Training regime: SINGLE-TASK, per-dataset (do NOT mix datasets)

Confirmed from configs: the benchmark trains **27 independent single-task fine-tunes** per (method, seed)
— each `train.yaml` has `dataset: ${DATASET}` (one dataset), and `run_exp.sh` starts a fresh
`llamafactory-cli train` per dataset. No cross-dataset mixing/curriculum (multi-task is the paper's
*future work*). Within a dataset, examples are shuffled each epoch (`disable_shuffling` default `False`),
seeded by the run seed → same example order as the PEFT runs when seeds match. **Our denominator must be
the same: 27 separate full fine-tunes, never one model trained on the union** — mixing would compute the
denominator under a different paradigm than the PEFT numerators (a Tier-1 apples-to-oranges failure).

## Key technical decision: framework = PEFT-Factory (LLaMA-Factory fork)

Use the paper's own tooling so the denominator is comparable by construction:
- Same data loading, label verbalization (`0 → not_duplicate`, etc.), prompt templates, train/val/test splits.
- Same metric code (F1 / accuracy / Pearson for STS-B / CodeBLEU for code).
- Switch `finetuning_type: peft` → `finetuning_type: full` in the train config. Everything else identical.

Repos: `github.com/kinit-sk/PEFT-Factory` (training) + `github.com/kinit-sk/PEFT-Bench` (configs/datasets/metrics).

---

## ⚠️ Hyperparameter caveat (most important risk)

The paper's hyperparameters (lr **5e-5**, cosine, 10 epochs, batch 4, weight decay 1e-5, 0.1 warmup)
were tuned for **PEFT**, where the frozen backbone regularizes training. For **full** fine-tuning of an
8B model those settings will likely **overfit or diverge** — full FT typically wants lr ≈ 1e-5–2e-5 and
fewer epochs (1–3) with early stopping on val loss.

**Mitigation:** run a small LR sweep ({1e-5, 2e-5, 5e-6}) on 2–3 representative datasets during the smoke
phase, pick best-on-val, then lock it for the sweep. Keep epochs at 10 but rely on PEFT-Factory's
"save best val-loss checkpoint" (already its default per the paper) so overfitting is bounded by early
stopping rather than the epoch count. Document the chosen LR in the log — it's a deviation from the paper
and reviewers will ask.

---

## Compute: memory math for full FT of 8B

Mixed-precision AdamW, 8.03B params:
- bf16 params 16 GB + bf16 grads 16 GB
- fp32 optimizer master + m + v ≈ 3 × 32 GB = 96 GB
- activations (batch 4, grad checkpointing on) ≈ 5–15 GB

→ **~100–130 GB total. Does not fit on one 80 GB H100, but fits on one 141 GB H200.** Live vast prices
(2026-06-03, rentable, >300 GB disk, fast net):

| Option | Config | VRAM | $/hr | Fits? | Notes |
|---|---|---|---|---|---|
| **A (CHOSEN)** | **1× H200** | 141 GB | **$3.82** | ✅ single-card | no ZeRO-3 / no offload — simplest + fast. ~13 GB headroom; smoke test must confirm 2048-ctx peak. |
| B (fallback) | 2× H100 SXM, ZeRO-3 | 2×80 GB | $4.00 | ✅ sharded | if a single H200 is too slow on apps/record. More moving parts (DeepSpeed). |
| C | 1× H100 80GB, ZeRO-3 + CPU offload | 80 GB | $2.64 | ✅ offload | cheapest/hr but offload tax → likely slower *per job*. Skip unless cost-critical. |
| — | 1× B200 | 180 GB | $4.38 | ✅ | overkill + newer-driver/torch-compat risk. |

**Recommendation: Option A — single H200.** Same choice for smoke test and full sweep (peak memory is set
by model+optimizer, not dataset size). Fall back to B (2× H100 ZeRO-3) only if the smoke test's long-context
run OOMs or is too slow. Note: provision **~300 GB disk** for the ~80–110 GB full-FT checkpoints.

---

## ⚠️ vast.ai divergence from CLAUDE.md notes

The vast.ai recipe in `~/.claude/CLAUDE.md` is for the **`deterministic-serving` Nix image** — that image
is SM_90-only and has a custom entrypoint, hence "H100 only" and the `--args` / entrypoint-hash dance.
**None of that applies here.** For fine-tuning use a **standard CUDA/PyTorch image** so:
- any modern 80 GB GPU works (H100, A100 80GB, even though we prefer H100),
- normal `--ssh` launch mode works (no entrypoint hash extraction, no CUDA symlink repair),
- the CUDA-symlink / `LD_LIBRARY_PATH` fixes from CLAUDE.md are **not** needed.

Image: `hiyouga/llamafactory:latest` (ships LLaMA-Factory + deps) **or** `pytorch/pytorch:2.x-cuda12.x` +
`pip install -e .` of PEFT-Factory. Prefer the LLaMA-Factory image and overlay the PEFT-Factory fork on top.

---

## Checkpointing, resume & weight storage

**Three layers of interruption recovery:**
1. **Sweep level** — driver skips any (dataset,seed) whose `results.jsonl` already exists. A fresh box
   resumes the sweep where it stopped.
2. **Run level** — LLaMA-Factory auto-resumes a partially-trained run from its last checkpoint
   (`overwrite_output_dir: false` + stable `output_dir`; restores weights+optimizer+scheduler+RNG+step).
   Max work lost on a kill = one `save_steps` interval (5% of the run).
3. **Durability** — vast local disk is **ephemeral and wiped on reclaim**. For resume to mean anything,
   checkpoints must survive the box: `rsync` the `output_dir` to durable storage after each save, OR run the
   giant datasets (mnli/qqp/qnli/record/mmlu) on **on-demand (non-interruptible)** instances. Results
   (`results.jsonl`, tiny) are always synced off-box immediately.

**Weight storage — the full-FT-specific problem (doesn't exist for PEFT):**
- A full-FT checkpoint keeps optimizer state too → **~80–110 GB on disk each** (16 GB bf16 weights +
  ~64–96 GB fp32 AdamW state, ZeRO-3-sharded). `save_total_limit: 1` bounds it to ~1 (+best) at a time.
  → provision **~300 GB disk** per box (the plan's earlier 200 GB is too tight for full FT), not 50 GB.
- A *final* full model is ~16 GB. Keeping all **135 of them = ~2 TB** — pointless for a denominator.
- **We only need the metrics, not the weights.** Driver does train → eval → `compute_metrics.py` → persist
  the small `results.jsonl`, then **delete the checkpoint/model**. Optionally keep ONE final model
  (e.g. best NLU dataset) if collaborators want an artifact. `push_to_hub: false` (135×16 GB to the Hub is absurd).

## Phases

### Phase 0 — Local prep (no GPU) ✅ DONE 2026-06-03
- [x] Clone `PEFT-Factory` + `PEFT-Bench` → `vendor/`.
- [x] Extract exact data setup, splits, hyperparams, metric mapping → see "Phase 0 findings" above.
      Key result: **no subsampling, full HF train splits, 5 seeds, 10 epochs**.
- [x] Build `configs/full_train.yaml.tmpl` (`finetuning_type: full` + ZeRO-3, `push_to_hub: false`,
      parameterized `DATASET/SEED/LEARNING_RATE/EPOCHS/OUTPUT_DIR/DEEPSPEED_CONFIG`).
- [x] Build `configs/full_eval.yaml.tmpl` (`model_name_or_path = trained checkpoint`, `predict_with_generate`).
- [ ] **TODO (still needs doing before Phase 1):** confirm HF access to gated `meta-llama/Meta-Llama-3-8B-Instruct`
      (token + accepted Meta license); decide seeds (3 vs 5) and dataset scope with collaborators.

### Phase 1 — Provision vast.ai (1 instance)
- [ ] `vastai search offers 'gpu_name=H100_SXM num_gpus>=2 disk_space>200'` (or `num_gpus=1` for Option A).
- [ ] `vastai create instance <id> --image hiyouga/llamafactory:latest --disk 200 --ssh` (standard mode).
- [ ] Setup script (`scripts/setup_remote.sh`): clone PEFT-Factory, `pip install -e .`, `huggingface-cli
      login`, pre-download the base model + all 27 datasets to a persistent disk path so re-launches are fast.

### Phase 2 — Tiered validation gates (fast → slow; only 2d does real training)

The original single heavy "smoke test" (3 datasets × 3 LRs × 10 epochs × full data) was the wrong first
test — it took ~25 min/run to surface 2-minute bugs. Replace with tiered gates, each the cheapest test
that can fail:

- [ ] **2a Preflight — 1B model, all 27 datasets, `max_samples:16`, 1 epoch** (`scripts/preflight.sh`).
      Validates every dataset's load + column mapping + llama3 template + train→eval→`compute_metrics`→
      label-parsing. ~min/dataset, ~free. Catches data-source bugs (cb/boolq/conala), config errors, metric
      breakage — all at once, before any 8B time. **Also where the dataset_info.json source fixes get verified.**
- [ ] **2b Canary — 8B, `max_steps:2`, 1 dataset** (`scripts/canary.sh`). ~3 min. Catches OOM (hits at the
      step-1 optimizer step), save mechanics, real 8B memory headroom.
- [ ] **2c Throughput — 8B, `max_steps:20`, seq 2048, batch 4** (`scripts/canary.sh --throughput`). ~2 min.
      Real tok/s on cost-driving long sequences → firm cost estimate (don't infer from slow tiny datasets).
- [ ] **2d LR sweep — 8B, real small datasets (copa/svamp), 10 epochs, {5e-6,1e-5,2e-5}**. The only gate that
      needs real training. Output = locked LR + full-FT ≥ Base sanity + label-parsing eyeball. Append to log.

### Phase 3 — Sweep (staged: one seed first, ascending dataset size)

**Run order = ascending dataset size, ALWAYS.** Cheap/fast datasets first so bugs surface in minutes on a
250-row run, not after hours on MNLI; cost ramps up only once the pipeline is proven. The driver iterates
datasets in this fixed order (train-set example count):

```
cb(250) copa(400) wsc(554) svamp(700) conala(2.4k) rte(2.5k) mrpc(3.7k) openbookqa(5.0k) apps(5.0k)
wic(5.4k) stsb(5.7k) gsm8k(7.5k) cola(8.6k) boolq(9.4k) piqa(16k) codealpacapy(18k) multirc(27k)
math_qa(30k) siqa(33k) hellaswag(40k) winogrande(40k) sst2(67k) mmlu(100k) record(101k) qnli(105k)
qqp(364k) mnli(393k)
```
(Note: by *token/compute* cost — not row count — apps, record, and multirc rank higher than their position
above because of long-context truncation at 2048. Row-count order is still the right default; just expect
those three to take longer than their neighbors.)

**Phase 3a — one seed (seed 42), all 27 datasets, ascending order.** Get a complete denominator pass cheaply
(~1/5 the cost) before committing to more seeds. This is also the real end-to-end validation of the whole
sweep on the giant datasets. Inspect the 27 scores for sanity (full-FT ≥ base on NLU; nothing collapsed to 0)
**before** Phase 3b.
- [ ] Generate the seed-42 run list: `data/run_matrix.csv`, one row per dataset, sorted ascending by size.
- [ ] Driver script (`scripts/run_sweep.sh`): iterate in order, skip runs whose result JSON already exists
      (idempotent/resumable), `envsubst` the template, `llamafactory-cli train` → eval → `compute_metrics.py`,
      write `data/results/<dataset>_<seed>.json`.
- [ ] **Resilience:** vast instances can be reclaimed. After every run, `rsync`/`scp` `data/results/` back to
      local. Skip-if-exists makes a fresh instance resume cleanly mid-sweep.
- [ ] Gate: review the 27 seed-42 scores. Decide go/no-go on additional seeds with collaborators.

**Phase 3b — remaining seeds (123, 456, 789, 101112), only if needed.** Same ascending order per seed.
Optionally shard across 2–3 instances by dataset range (giant datasets on on-demand, not interruptible, boxes).

### Phase 4 — Aggregate & deliver
- [ ] `scripts/aggregate.py`: collapse seeds → per-dataset mean ± std full-FT score → `reports/full_ft_scores.csv`.
- [ ] Join with PEFT method scores (from the paper's Tables 1–2 / repo) + each method's trainable-param count.
- [ ] Compute % recovered under **both** definitions (so the metric decision can be made from data).
- [ ] Plot: x = adapter size (log), y = % recovered, one point per (method, dataset) or aggregated per method
      → `figures/recovery_vs_size.png`. Expect a rising curve asymptoting to ~100%.
- [ ] Tear down all vast instances. Verify `vastai show instances` is empty (don't leak spend).

---

## Phase 0 findings (CONFIRMED from the repos — supersedes earlier assumptions)

- **NO subsampling.** Configs load the full HF `train` split per dataset (`data/dataset_info.json`,
  `split: train`, no `max_samples` anywhere). My earlier "~$150–600" estimate assumed subsampling that
  does not exist — it is wrong. See cost section below for the corrected figure.
- **Canonical run** (`scripts/peftbench/run_exp.sh`): 27 datasets, 5 seeds `(42,123,456,789,101112)`,
  `EPOCHS=10`, per-device batch 4, lr 5e-5, cosine, warmup 0.1, weight decay 1e-5, `adamw_torch`, bf16,
  `cutoff_len 2048`, `val_size 0.1`, `load_best_model_at_end` (no EarlyStoppingCallback → runs all 10 epochs).
- **Full FT is a one-line change:** the `base` config already uses `finetuning_type: full`; we add
  `do_train: true` + a DeepSpeed ZeRO-3 config. Templates written to `configs/full_train.yaml.tmpl` and
  `configs/full_eval.yaml.tmpl`. (Also disabled `push_to_hub`, which the paper sets to org `rbelanec`.)
- **Eval is generation-based** (`predict_with_generate`) on a separate `<name>_eval` split, scored by
  `scripts/peftbench/compute_metrics.py`. Metric per dataset: macro-F1 (mnli, cb, mmlu, siqa, hellaswag,
  openbookqa, math_qa), binary-F1 (qqp, qnli, sst2, mrpc, rte, cola, multirc, boolq, wic, wsc, copa,
  piqa, winogrande), Pearson+Spearman (stsb), ReCoRD-EM (record), exact/accuracy (gsm8k, svamp), CodeBLEU
  (conala, codealpacapy, apps).
- **Cost is dominated by 5 datasets.** mnli (393k) + qqp (364k) + qnli (105k) + record (~101k) +
  mmlu (~99k) ≈ **76% of all training examples**. The 22 others (incl. wsc 554, cb 250, copa 400,
  svamp 700) are rounding error.

## Cost estimate (corrected after Phase 0; refine after smoke test)

Full benchmark ≈ **1.39M train examples/epoch** (one seed, all 27). ×0.9 val split ×10 epochs ×5 seeds
≈ **62.6M example-passes**. At a blended ~200 tokens/example and ~6N FLOPs/token for full FT 8B:

| Scenario | Instance-hrs (2×H100) | $ @ ~$4.8/hr | Wall-clock |
|---|---|---|---|
| **Full fidelity** (27 ds, 5 seeds, 10 ep, full data) | ~350 | **~$1,700–1,900** | ~15 days on 1 instance / ~4 days on 4 |
| 3 seeds instead of 5 | ~210 | ~$1,000–1,150 | ~9 days / ~2 days |
| + early-stopping callback (full FT converges in 2–4 ep) | ~100–150 | ~$500–700 | — |
| + cap the 5 giant datasets at 20k samples ⚠️ | ~90 | ~$430 | — but breaks fairness, see note |

Sensitivity: blended seq length 150→300 scales ×0.75–1.5; MFU uncertainty ±40%. Honest range for full
fidelity: **$1,200–3,500** and **250–550 instance-hours**. Smoke test (3 datasets, LR sweep) ≈ **$20–40**.

⚠️ **Fairness constraint on cost-cutting.** To reuse the paper's existing PEFT scores as the numerator,
the full-FT denominator must train on the *same* data the PEFT methods saw — i.e. full data, 10 epochs.
Capping `max_samples` or epochs only stays fair if we *also* re-run the affected PEFT cells on the same
cap (more work, not less). Dropping seeds (5→3) and adding early stopping are the safe levers; data/epoch
capping requires collaborator sign-off.

---

## Deliverables

- `reports/full_ft_scores.csv` — per-dataset full-FT denominator (mean ± std).
- `figures/recovery_vs_size.png` — the recovery-vs-adapter-size plot.
- `EXPERIMENT_LOG.md` — append-only log of each milestone, the chosen LR, deviations from the paper, costs.
