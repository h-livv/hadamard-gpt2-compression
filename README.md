# Hadamard decomposition against truncated SVD, at equal storage

At equal storage, is the Hadamard decomposition of Panchal, Sahoo and Behera ([*Neurocomputing* 699 (2026) 134372](https://doi.org/10.1016/j.neucom.2026.134372)) a better way to compress a weight matrix than truncated SVD, and what decides the answer? An independent study, not affiliated with the authors. The full write-up is [`docs/REPORT.md`](docs/REPORT.md).

Hadamard at rank r stores 2r(m + n) numbers, as SVD does at rank 2r, so every comparison here pits HadNet(r) against SVD(2r). GPT-2 small scores a WikiText-2 perplexity of 25.17 uncompressed.

| Stage | What was done | What it found | Why it leads on |
|---|---|---|---|
| 1. Plain fits on GPT-2 | All 24 MLP matrices, SVD(2r) vs HadNet(r), weight-error fits, r = 8–64 | HadNet has the lower weight error on 85 of 96 fits, but every model is broken (perplexity 3,904–25,286), and HadNet loses at r = 32 and 64 | Why doesn't the better weight approximation give a better model? |
| 2. Diagnostics | Bug check, r up to 256, one matrix at a time, error statistics, output error | No bug. From r = 128 up HadNet's models are far better (45.2 vs 1,399 at r = 256). The six worst single-matrix runs are all in blocks 0–1. Neither error predicts one matrix's damage | So fit the error on real inputs, and protect blocks 0–1 |
| 3. Activation-aware fits | Blocks 2–11, SVD, HadNet, whitened SVD, weighted HadNet, r = 64–256 | The largest improvement (2–12×). Weighted HadNet wins at r = 64 (90.0 vs 252.2); whitened SVD at r = 128 and 256 (59.1, 29.3), where weighted HadNet's `c_proj` fits barely move. Output error predicts perplexity (Spearman +0.97) | Where does the damage come from, block by block? |
| 4. Cumulative damage | Blocks 2–11 added one at a time | Each block multiplies perplexity (×1.009–1.818). Blocks 3–8 cause 62–85% of the log increase, following `c_proj` output error (Spearman +0.78) | Is the `c_proj` failure the method or the optimiser? Isolate single matrices |
| 5. Single matrices | An image at n = 64–512 and four storage shares under three weightings, a random control, GPT-2 block 5 `c_proj` | The spectrum decides SVD vs HadNet (Prop 3.5): SVD wins on the image everywhere, HadNet on the flat control everywhere. On block 5 `c_proj`, the failure was the learning rate: at 0.03 or 0.01 weighted HadNet beats whitened SVD by 24–26% | Open questions (report) |

*Stage 2 originally read "output error, not weight error, tracks the damage when many matrices are compressed". Across stage 2's eight all-matrix runs both errors track it (Spearman +0.95 for weight error, +0.90 for output error); output error separates from weight error only in stage 3. The table follows the data.*

![Perplexity against model size, the four fits of stage 3](figures/stage3/perplexity_vs_model_size_blocks2-11.png)

## Install and run

Python 3.12. Stages 1–4 and the GPT-2 part of stage 5 need a CUDA GPU with about 6 GB, and download GPT-2 small and WikiText-2 from the Hugging Face Hub on first use.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/checks.py                          # 16 checks, ~6 s; must end with "All checks passed."
python experiments/stage1_plain_fits.py         # ~11 min
python experiments/stage2_diagnostics.py        # ~75 min
python experiments/stage3_activation_aware.py   # ~35 min
python experiments/stage4_cumulative.py         # ~50 min
python experiments/stage5_single_image.py       # ~27 min (CPU)
python experiments/stage5_gpt2_single_matrix.py # ~3.5 min
python experiments/plot.py                      # every figure
```

Each script writes to `results/stageN/`, and `plot.py` writes to `figures/stageN/`. The shared code is in `src/`, and every check is in `tests/checks.py`.
