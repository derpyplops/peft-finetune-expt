# Regularizer ablation on full-FT (svamp, LR 2e-6, 5 epochs, 3 seeds)

Tests whether Tier-1 config-only regularizers close full-FT's gap to LoRA and/or fix its seed instability.

| Profile | s42 / s1 / s2 | mean ± sd | Δ vanilla |
|---|---|---|---|
| vanilla | 84.3 / 74.3 / 84.0 | 80.9 ± 4.6 | — |
| neftune (α=5) | 85.0 / 71.7 / 85.0 | 80.6 ± 6.3 | −0.3 |
| weight_decay 0.1 | 82.0 / 77.7 / 81.3 | 80.3 ± 1.9 | −0.6 |
| label_smoothing 0.1 | 84.7 / 81.3 / 86.3 | 84.1 ± 2.1 | +3.2 |
| **bigbatch (grad_accum 16, eff batch 64)** | 84.7 / 86.0 / 85.7 | **85.4 ± 0.6** | **+4.5** |
| **all combined** | 87.0 / 85.3 / 88.0 | **86.8 ± 1.1** | **+5.9** |
| LoRA (target) | — | 90.3 | — |

## Findings
1. **Effective batch size was the main instability driver.** bigbatch collapsed seed sd 4.6→0.6 and added +4.5 acc. The collapses/variance we saw were largely a batch-size-4 artifact.
2. **Label smoothing** adds +3.2 and halves variance.
3. **all** stacks to 86.8 ± 1.1 — closes ~60% of the gap to LoRA (9.4→3.5) and makes full-FT stable.
4. **neftune** (math task, not instruction) and **weight_decay** don't help accuracy; WD only damps variance.
5. A small real gap remains: regularized full-FT (86.8) still < LoRA (90.3) by ~3.5.

## Recommendation for the denominator
Use **"vanilla full-FT done right"**: LR ~2e-6, 5 epochs, **grad_accum 16 (eff batch 64)**, **label_smoothing 0.1**, 3 seeds, report mean±std. These are standard, defensible training choices (not exotic methods), so the denominator stays an honest full-FT reference — just a *stable, well-tuned* one (~87 on svamp vs the old 75). This raises the denominator and shrinks every PEFT method's "% recovery" margin accordingly.
