# svamp/WSC probe — is full-FT's underperformance mistuning or real?

2026-06-05. Follow-up to [wsc_variance_investigation.md](../wsc_variance_investigation.md). The WSC grid turned out to be degenerate (constant-predictor collapse), so we re-ran the LR×seed probe on **svamp** (math accuracy, no trivial baseline) and kept a WSC prediction dump to confirm the degeneracy.

## Eval pipeline is VALID
**svamp LoRA control = 90.3** (90.7 / 89.0 / 91.3) vs paper's 88.0 — reproduced (slightly above). So our eval/metric pipeline is trustworthy on non-degenerate data; the WSC `exact_match=0` / `f1=53.5` was **WSC-specific degeneracy**, not a pipeline bug.

## svamp full-FT (accuracy ×100)
| LR | seed42 | seed1 | seed2 | mean ± sd |
|---|---|---|---|---|
| **2e-6** | 84.3 | 74.3 | 84.0 | **80.9 ± 4.6** ← best |
| 5e-6 (orig) | 75.3 | 72.0 | 64.7 | 70.7 ± 4.5 |
| 1e-5 | 65.7 | 68.7 | 67.7 | 67.3 ± 1.2 |

Reference: base = 56.7, prior denominator (5e-6, 5ep, seed42) = **75.3** (reproduced exactly), **LoRA = 90.3**.

## Findings
1. **Full-FT is partly mistuned.** Lower LR helps a lot: 2e-6 (mean 80.9, best 84.3) > 5e-6 (70.7) > 1e-5 (67.3). The original fixed **5e-6 understates full-FT by ~6–9 pts** on svamp.
2. **But the gap is real.** Even best-tuned full-FT (~84) stays **below LoRA (90.3)**. So PEFT genuinely beats full-FT on svamp — just by a smaller margin than the raw denominator (75.3 vs 90) implied.
3. **Moderate seed variance** (sd ≈ 4.5; 2e-6 seed1 dropped to 74.3 vs ~84 for the others). Single-seed is unreliable.
4. **Different datasets want different LRs.** WSC was "stable" at 5e-6; svamp prefers 2e-6. No single global LR is optimal → the fixed-LR denominator is systematically off.
5. **WSC confirmed degenerate.** The kept 5e-6/seed42 model predicts 62 `True` / 42 `False` vs gold 38/66 — a biased, barely-discriminating predictor. WSC's `f1` clusters near the predict-all-True baseline (0.535). WSC is a bad probe; don't lean on WSC-style datasets for strong claims.

## Implication for the denominator sweep
The current denominator (fixed 5e-6, 5 epochs, single seed) **understates full-FT** → it would **inflate every PEFT method's "% recovery"**. The qualitative story (PEFT ≥ full-FT) survives, but the denominator needs to be re-tuned to be defensible.

### Recommended: middle path
- **Re-run the denominator at a lower LR (~2e-6) × 3 seeds**, report **mean ± std**.
- **Keep 5 epochs** — 10 epochs caused WSC collapse; svamp at 5 is fine. Do NOT "fix the 5-vs-10 asymmetry" by going to 10.
- **Per-dataset LR check only where full-FT is close to PEFT** (avoid the full 27×3LR×3seed = 243-run blowout unless this goes in a paper).
- Estimated ~81 runs (~3× the current sweep).

### Epochs caveat
Matching PEFT's 10 epochs is *not* obviously fairer — on tiny datasets it pushes full-FT into overfit-collapse. 5 epochs + best-val is the safer regime.
