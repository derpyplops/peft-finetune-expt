# LoRA reproduction — 5 smallest PEFT-Bench datasets

Reproduces the paper's best-PEFT result (LoRA is best on all 5) using the paper's methodology:
LoRA rank16 / alpha16 / dropout0.05, target k_proj,v_proj,down_proj, lr 5e-5, 10 epochs, best-val checkpoint, eval on validation split. 3 seeds (42/1/2). Paper: arXiv 2511.21285.

| Dataset | train | our LoRA (mean±sd) | paper best PEFT | Δ (paper−ours) | reproduced? |
|---|---|---|---|---|---|
| cb | 250 | 75.5±14.3 (n=3) | 83.9 | +8.4 | reproduced |
| copa | 400 | 93.2±0.4 (n=3) | 97.8 | +4.6 | off |
| wsc | 554 | 53.2±1.0 (n=3) | 53.6 | +0.4 | reproduced |
| svamp | 700 | 88.9±1.4 (n=3) | 88.0 | -0.9 | reproduced |
| rte | 2490 | 88.7±1.6 (n=3) | 85.7 | -3.0 | off |

**Reproduced 3/5 within ±max(sd, 2pt) of the paper's value.** wsc is degenerate (best-PEFT ≈ predict-majority baseline) — treat with caution.
