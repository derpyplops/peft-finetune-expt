# Executive summary — time & money for the full-FT denominator experiment

**Bottom line.** Producing the full-parameter fine-tune denominator at the same fidelity as the
PEFT-Bench paper (27 datasets × 5 seeds × 10 epochs × **full, un-subsampled** training data) costs
roughly **$1,700–1,900** (honest range **$1.2k–3.5k**) and **~350 GPU-instance-hours** — about
**2 weeks** of wall-clock on one 2×H100 instance, or **~4 days** if sharded across four. The smoke
test to de-risk it is ~**$20–40** and a few hours.

> ⚠️ This corrects an earlier "$150–600" figure that assumed the benchmark subsamples its big datasets.
> Phase 0 confirmed it does **not** — configs load full HF `train` splits. That's the whole story below.

## Why it costs this much: 5 datasets do 76% of the work

| Driver | Value | Effect |
|---|---|---|
| Datasets | 27 | mnli (393k) + qqp (364k) + qnli (105k) + record (~101k) + mmlu (~99k) = **76%** of all examples |
| Seeds | 5 (42,123,456,789,101112) | linear ×5 multiplier on everything |
| Epochs | 10, no early-stop callback | runs all 10 even after convergence |
| Subsampling | **none** | the cost driver — full GLUE train sets in full |
| Total work | ~62.6M example-passes | 1.39M ex/epoch × 0.9 val × 10 ep × 5 seeds |

## Compute model

- Full FT 8B ≈ 6N FLOPs/token = 4.8e10 FLOPs/token. 2×H100 ZeRO-3 + activation checkpointing ≈
  ~10k tokens/sec effective (≈35% MFU, comm overhead included).
- Blended ~200 tokens/example (short GLUE classification dominates) → 62.6M × 200 = 1.25e10 tokens →
  ~350 instance-hours → ~$1,700 at ~$4.8/instance-hr (2×H100 SXM on vast).
- Eval (generation over `<name>_eval` splits) adds ~10–20% → budget ~$250 on top.

## Phase 3a — ONE seed (seed 42), the immediate ask

1 seed = ~1.8B tokens. On **1× H200 @ $3.82/hr** at ~5,500 tok/s (full-FT 8B, ~37% MFU): training ~91
GPU-hrs + eval ~14 = **~105 GPU-hrs ≈ $400** (range $300–600). ~1/5 of full fidelity by construction.
- Cost concentration: the 5 giant datasets ≈ $290 (record ~$140, mnli ~$60, qqp ~$45, mmlu ~$30, qnli ~$17);
  the other 22 datasets ≈ $100 combined.
- Wall-clock: ~4.5 days serial on one H200, or ~1.5 days if the 5 giants are sharded onto their own boxes.
- All-in first pass incl. one-time smoke test (~$30): **~$430**.

## Scenarios & levers

| Scenario | Instance-hrs | Cost | Notes |
|---|---|---|---|
| **Full fidelity (recommended baseline)** | ~350 | **~$1,700–1,900** | exactly matches the paper; safest comparison |
| 3 seeds instead of 5 | ~210 | ~$1,000–1,150 | error bars slightly wider; fine for a denominator |
| + early-stopping callback | ~100–150 | ~$500–700 | full FT converges in 2–4 epochs; safe-ish, mild deviation |
| + cap 5 giant datasets @20k ⚠️ | ~90 | ~$430 | **breaks fairness** unless PEFT cells re-run on same cap |
| Smoke test (3 ds + LR sweep) | ~5–10 | ~$20–40 | do first, then re-estimate from measured throughput |

**Safe cost-cutting:** drop seeds 5→3, add early stopping. **Unsafe without sign-off:** capping data or
epochs — to reuse the paper's PEFT scores as the numerator, the denominator must see the same data the
PEFT methods saw.

## Time

- One 2×H100 instance, serial: **~15 days** wall-clock (mostly the 5 big datasets × 5 seeds).
- Sharded across 4 instances by dataset range: **~4 days**, same total $ (parallelism buys time, not money).
- Recommended: shard the 5 giant datasets onto their own instances; sweep the 22 small ones on one box in ~1 day.

## Recommendation

1. Run the **smoke test** (~$30) to measure real throughput + lock the full-FT learning rate.
2. Re-derive this table from measured numbers.
3. Default to **3 seeds + early stopping** unless collaborators want exact 5-seed/10-epoch parity —
   that lands around **$700–1,150** and ~1 week, vs ~$1,800 / ~2 weeks for full parity.
