# Premortem — full-FT denominator experiment

**Framing.** It is ~3 weeks from now. We spent ~$1.8k and two weeks of wall-clock, and the result is
unusable — either the plot is silently *wrong*, or we burned the money and have nothing. This document
is the autopsy written *in advance*. Failure modes are ranked by (likelihood × blast radius), with the
mitigation we should commit to **now** or bake into the smoke test.

The defining feature of this experiment's risk profile: **the worst failures don't crash — they produce
a believable denominator that's quietly wrong.** A crash we'd notice in an hour. A denominator that's
20% too low because the model's generations don't parse to label strings, we might ship.

---

## Tier 1 — silent-wrong-answer failures (the dangerous ones)

### 1.1 Learning rate: full FT at the paper's 5e-5 diverges or overfits → invalid ceiling
**Autopsy:** we reused the paper's PEFT hyperparameters (lr 5e-5, 10 epochs). For full-parameter 8B that
LR is ~3–5× too hot. Either loss diverges (denominator = garbage), or the model overfits the small
datasets and the best-val checkpoint is still worse than some PEFT methods → **"% recovered" > 100%**,
which is nonsensical and makes the whole plot look broken.
**Likelihood:** high. This is the single most likely way the numbers come out wrong.
**Mitigation (committed):** LR sweep {5e-6, 1e-5, 2e-5} on 2–3 datasets in the smoke phase, pick best-on-val,
lock it. Treat any dataset where full-FT < best-PEFT as a red flag to investigate, not a data point to ship.

### 1.2 Generation eval doesn't parse to label strings → denominator artificially ~0
**Autopsy:** eval is `predict_with_generate` — the model emits free text, and `compute_metrics.py` does
exact string matching against label vocab (`"entailment"`, `"not_duplicate"`, …). A full-FT instruct
model that's more verbose than the tiny PEFT adapters ("The answer is entailment.") fails the exact match;
`macro_f1` counts it as wrong. The denominator collapses toward zero on classification tasks even though
the model knows the answers → every PEFT method shows >100% recovery, or the ceiling looks absurdly low.
**Likelihood:** medium-high, and **easy to miss** because it looks like "full FT is just bad here."
**Mitigation:** in the smoke test, **read the raw `generated_predictions.jsonl`** for 1 classification task and
eyeball pred-vs-label formatting before trusting any metric. Confirm full-FT predictions land in the label
vocab at a sane rate. If not, the training target format / template is off — fix before the sweep.

### 1.3 Numerator/denominator computed in different worlds → apples-to-oranges plot
**Autopsy:** PEFT scores (numerator) come from the paper / the authors' W&B; our full-FT scores
(denominator) come from our vast.ai stack. Different transformers/peft/llamafactory/flash-attn versions,
different HF dataset snapshots (the `rbelanec/*` uploads, `glue`, mmlu `auxiliary_train`, winogrande config),
different metric *choice* (each dataset maps to **two** metrics, e.g. `[macro_f1, em]` — which one is P_t on
the plot?). The ratio mixes environments and nobody notices until the curve looks wrong.
**Likelihood:** high that *some* drift exists; medium that it's large enough to matter.
**Mitigation (important):** **re-run 1–2 PEFT cells ourselves** (e.g. LoRA on RTE + MMLU) under our exact
stack and confirm we reproduce the paper's Table 1/2 numbers within noise. That single check validates the
whole environment for the cross-comparison. Also: pin package versions; freeze the exact metric-per-dataset
mapping (and which of the two metrics is "the" score) in a config before plotting.

### 1.4 Full FT < base on generation tasks → "% recovered" undefined/negative
**Autopsy:** the paper showed PEFT *hurt* vs the base model on gsm8k/conala/codealpacapy/apps. Narrow SFT
of an instruct model can degrade it (catastrophic forgetting of its chat/reasoning ability). If full FT also
lands below base, then for the gain-normalized metric `(PEFT−Base)/(FullFT−Base)` the denominator goes ≤0
and the formula explodes; for the raw ratio it's just misleading.
**Likelihood:** medium-high on exactly those 4–5 generation datasets.
**Mitigation:** decide the metric definition (raw vs gain) **with collaborators before the run**, and
pre-register how we handle "full FT ≤ base" cells (drop them / floor at base / report separately). Don't
discover this at plot time.

---

## Tier 2 — execution failures (expensive, but loud)

### 2.1 Spot reclaim kills a long run with no resume → wasted hours, schedule slip
**Autopsy:** mnli/qqp/qnli/record/mmlu are single multi-hour runs. vast reclaimed the box mid-run; we
didn't checkpoint/resume, so we restarted from epoch 0 and paid twice. These 5 datasets are 76% of the
compute and the *most* exposed because each is one long job.
**Mitigation:** set `save_steps` to wall-clock-bounded checkpoints + `resume_from_checkpoint`; rsync the
output dir off-box after every checkpoint; make the driver skip-if-result-exists. Put the big-5 on
on-demand (not interruptible) instances even if pricier — cheaper than re-running.

### 2.2 Smoke test passes, the sweep fails on the paths the smoke test never touched
**Autopsy:** smoke used cb/svamp/conala — all *small, short-context*. The sweep then died on the giant /
long-context datasets: OOM from 2048-token record/apps batches, ZeRO-3 fragmentation on long runs,
generation-eval timeouts on 40k-example val sets. None of those were exercised.
**Mitigation:** smoke test must include **one large + one long-context run** (e.g. a `max_samples`-capped
mnli for throughput, and an *uncapped-length* record/apps mini-run for the 2048 memory path). Measure peak
memory at full `cutoff_len`, not just on toy data.

### 2.3 Cost/time blow-up from token & MFU underestimate
**Autopsy:** estimate was ~9B tokens / ~350 hrs. record/apps truncation pushed tokens to ~13B, ZeRO-3
offload + comms gave half the assumed MFU, and generation eval (autoregressive, under-budgeted) added 30%.
Real bill ~$4k, double the quote; sponsor unhappy.
**Mitigation:** re-derive cost from *measured* smoke-test throughput before committing the full sweep
(go/no-go gate). Budget generation-eval explicitly. Tokenize a sample to firm up the 9B estimate. Cap a
hard $ ceiling and alert.

### 2.4 HF gating / env not actually verified on the remote
**Autopsy:** `meta-llama/Meta-Llama-3-8B-Instruct` is gated; the token wasn't on the box (or license not
accepted), so every run failed at model download — after we'd paid for provisioning and setup.
**Mitigation:** Phase-1 setup script asserts model+dataset download succeed *before* launching any training.

### 2.5 Instances left running → idle-spend leak
**Mitigation:** teardown step + `vastai show instances` assert-empty in the driver's `finally`; a watchdog
that kills idle boxes.

---

## Tier 3 — annoyances / process

- **Scope not locked:** spend 2 weeks, then "we only needed the 8 GLUE datasets" or "3 seeds was fine" or
  "we meant LoRA-only." → Confirm dataset scope + seed count + metric definition in writing *before* the sweep.
- **Determinism:** bf16 + flash-attn + multi-GPU is nondeterministic; per-seed numbers won't bit-match across
  reruns. Fine for mean±std, but don't promise exact reproducibility.
- **The x-axis point for full FT:** full FT has 8.03B trainable params — is it plotted as the rightmost point
  or only used as the normalizer (=100% line)? Decide so the plot is unambiguous.
- **Bus factor:** one person drives it on a remote box; capture commands in the driver script + log so it's
  re-runnable.

---

## What this premortem changes in the plan

1. **Redesign the smoke test.** Current plan (cb/svamp/conala) is necessary but insufficient. Add: (a) read
   raw generations for label-parsing [1.2], (b) one large + one long-context run [2.2], (c) reproduce 1–2
   paper PEFT cells under our stack [1.3]. Smoke test is now the gate for *correctness*, not just "does it run."
2. **Lock definitions before spend** [1.4, 3]: raw vs gain ratio, dataset scope, seed count, which metric is P_t,
   and the "full FT ≤ base" handling rule.
3. **Harden the big-5 runs** [2.1]: on-demand instances, checkpoint+resume, rsync-off-box, skip-if-exists.
4. **Go/no-go gate after smoke** [2.3]: re-derive cost from measured throughput; abort/renegotiate if >1.5× quote.

## Go / no-go gates before the full sweep
- [ ] LR locked from sweep; no divergence; full-FT ≥ base on at least the NLU smoke tasks.
- [ ] Raw generations inspected; predictions parse to label vocab at sane rate.
- [ ] Reproduced ≥1 paper PEFT cell within noise (environment validated).
- [ ] Measured throughput → cost re-estimate within 1.5× of quote.
- [ ] Metric definition, dataset scope, seed count, "≤base" rule signed off by collaborators.
- [ ] Model + all datasets verified downloadable on the remote; checkpoint/resume tested by a forced kill.
