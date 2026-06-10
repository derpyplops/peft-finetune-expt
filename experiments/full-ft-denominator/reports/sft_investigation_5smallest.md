# SFT investigation — why full-FT ≤ PEFT on the small datasets

Hypothesis under test (H1): the denominator's full-FT config (effective batch 64, 5 epochs) gives the tiny
datasets ~30x fewer optimizer steps than the paper's PEFT runs (effective batch 4, 10 epochs), under-training them.

Arms (5 smallest datasets, 3 seeds each):
- **A** full-FT baseline — reg=all, eff-batch 64, 5ep, LR 2e-6 (the current denominator config)
- **B** full-FT *paper-budget* — eff-batch 4, 10ep, LR 2e-6, label-smoothing (matches paper PEFT training budget)
- **D** freeze-top16 — train only the top 16 of 32 transformer layers, eff-batch 4, 10ep, LR 2e-6
- **LoRA** — our reproduction of the paper's LoRA (rank16/α16/k,v,down/5e-5/10ep)
- **paper** — paper's reported best-PEFT score

| Dataset | size | base | A: full baseline | B: full paper-budget | D: freeze-top16 | LoRA (ours) | paper bestPEFT |
|---|---|---|---|---|---|---|---|
| cb | 250 | 47.0 | 53.9±5.1 | 96.1±1.8 | 85.7±1.2 | 75.5±14.3 | 83.9 |
| copa | 400 | 73.8 | 91.8±1.7 | 92.9±0.9 | 88.5±1.7 | 93.2±0.4 | 97.8 |
| wsc | 554 | 43.6 | 0.0±0.0 | 0.0±0.0 | 23.4±17.2 | 53.2±1.0 | 53.6 |
| svamp | 700 | 56.7 | 89.4±0.4 | 83.7±3.3 | 86.6±3.7 | 88.9±1.4 | 88.0 |
| rte | 2490 | 69.7 | 90.1±0.9 | 88.3±2.4 | 81.0±1.8 | 88.7±1.6 | 85.7 |

## Read

**B−A (paper-budget vs baseline full-FT):** cb +42.2, copa +1.1, wsc +0.0, svamp -5.8, rte -1.9
**D−A (freeze-top16 vs baseline full-FT):** cb +31.9, copa -3.3, wsc +23.4, svamp -2.9, rte -9.1

If B closes most of the gap to LoRA/paper, full-FT was under-trained, not inherently worse — and the denominator config must be re-tuned (smaller batch / more epochs) on small datasets. wsc is degenerate (metric artifact), discount it.
