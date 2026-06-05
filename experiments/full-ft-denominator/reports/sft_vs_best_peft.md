# Full-FT (SFT) vs best PEFT — per dataset

SFT = our full-parameter fine-tune (seed 42, 5 epochs, best-val checkpoint).
Best PEFT = max over the paper's 6 methods (Tables 1–2 of arXiv 2511.21285). No BitFit in paper.

| Dataset | Base | SFT (ours) | Best PEFT | (method) | Δ SFT−PEFT | Winner |
|---|---|---|---|---|---|---|
| cb | 47.0 | 98.7 | 83.9 | LoRA | +14.8 | **SFT** |
| copa | 73.8 | 91.8 | 97.8 | LoRA | -6.0 | PEFT |
| wsc | 43.6 | 31.7 | 53.6 | LoRA | -21.9 | PEFT |
| rte | 69.7 | 81.6 | 85.7 | LoRA | -4.1 | PEFT |
| mrpc | 78.8 | 88.7 | 91.0 | LoRA | -2.3 | PEFT |
| wic | 66.6 | 76.4 | 75.2 | LoRA | +1.2 | **SFT** |
| stsb | 67.4 | 89.0 | 90.7 | LoRA | -1.7 | PEFT |
| cola | 78.3 | 84.3 | 89.7 | LoRA | -5.4 | PEFT |
| boolq | 80.8 | 88.3 | 91.0 | LoRA | -2.7 | PEFT |
| openbookqa | 78.3 | 79.6 | 87.7 | LoRA | -8.1 | PEFT |
| piqa | 46.0 | 77.8 | 88.9 | LoRA | -11.1 | PEFT |
| gsm8k | 79.2 | 51.1 | 73.6 | Prompt | -22.5 | PEFT |
| svamp | 56.7 | 75.3 | 88.0 | LoRA | -12.7 | PEFT |
| codealpacapy | 31.2 | 30.4 | 35.0 | LoRA | -4.6 | PEFT |

**SFT beats best PEFT on 2/14 datasets.**
