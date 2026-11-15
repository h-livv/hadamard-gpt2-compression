# Hadamard decomposition vs. truncated SVD on GPT-2 small

A small, fully traceable experiment on the method of Panchal, Sahoo and Behera, [*Tensor decomposition and neural architectures through generalized Hadamard products*](https://doi.org/10.1016/j.neucom.2026.134372), Neurocomputing 699 (2026) 134372.

It compresses the MLP weight matrices of GPT-2 small with the paper's Hadamard decomposition (HadNet) and with truncated SVD **at the same number of stored parameters**, and measures WikiText-2 perplexity.

This is an independent study project. It is not affiliated with the paper's authors. Every function is traced to the paper or to public code in [`VERIFICATION.md`](VERIFICATION.md).

## What the experiment tests

GPT-2 small has 12 transformer blocks. Each block's MLP has two weight matrices, `c_fc` (3072×768) and `c_proj` (768×3072), 24 matrices in all. We replace every one of them by a compressed version and measure how much the language model degrades, using WikiText-2 test perplexity (lower is better).

Two compression methods are compared. Write (m, n) for the matrix shape.

1. **Truncated SVD at rank 2r.** W ≈ U Σ Vᵀ with the top 2r singular values kept. By the Eckart–Young theorem this is the best possible approximation of rank 2r.
2. **Hadamard decomposition at rank r.** W ≈ (W1 H1) ∘ (W2 H2), where ∘ is the entrywise product, W1 and W2 are (m, r), H1 and H2 are (r, n). The factors are fitted with HadNet (paper §4.1, Algorithm 7), started from the SVD initialisation (paper §3.3).

### Why the fair baseline is SVD at rank 2r

The Hadamard factors hold 2·r·m + 2·r·n = 2r(m+n) numbers. A rank-k SVD, stored as U_kΣ_k and V_kᵀ, holds k(m+n). The two are equal when k = 2r. The paper's tables compare Hadamard at rank r with SVD at rank r, which gives the Hadamard side twice the storage. Here both methods get exactly the same number of stored numbers (`checks.py`, check 5).

The interest of the Hadamard form is that its result can have rank up to r² (paper Prop. 3.3), far above 2r, for the same storage. Whether that helps depends on the matrix: it can only help if the singular values between positions 2r+1 and r² carry real weight (paper Prop. 3.5).

### What is and is not measured

After fitting, the dense matrix is rebuilt from the factors and copied back into the model. This measures **storage compression only**. The model runs exactly as fast as before, and nothing is fine-tuned afterwards.

## Maths behind the checks

`checks.py` tests these four facts numerically. They are short enough to derive here.

**M1. Khatri–Rao identity (why the rank can reach r²).** Expand one entry of the Hadamard product:

```
[(W1 H1) ∘ (W2 H2)]_ij = (Σ_k W1_ik H1_kj) (Σ_l W2_il H2_lj)
                       = Σ_{k,l} (W1_ik W2_il) (H1_kj H2_lj)
                       = Σ_{k,l} E_{i,(k,l)} F_{(k,l),j}
```

So (W1 H1) ∘ (W2 H2) = E F, where E (m, r²) has rows `W1[i] ⊗ W2[i]` (row-wise Khatri–Rao product) and F (r², n) has columns `H1[:, j] ⊗ H2[:, j]` (column-wise Khatri–Rao product). A product with inner dimension r² has rank at most r². This is the paper's Prop. 3.3.

**M2. Eckart–Young error formula.** If X has singular values σ₁ ≥ σ₂ ≥ …, its rank-k truncated SVD X_k is the best rank-k approximation in Frobenius norm, and ‖X − X_k‖_F² = Σ_{i>k} σᵢ². The paper uses this as its Theorem 3.1 (summed over tensor slices).

**M3. Parameter count.** Hadamard at rank r stores four factors: 2·(m·r) + 2·(r·n) = 2r(m+n). SVD at rank k stores U_kΣ_k (m, k) and V_kᵀ (k, n): k(m+n). Equal at k = 2r.

**M4. How much Hadamard can gain over SVD (paper Prop. 3.5).** Let Â be the best Hadamard approximation at rank r. By M1 it has rank ≤ r², so by M2 it cannot beat the rank-r² SVD: ‖A − Â‖² ≥ Σ_{i>r²} σᵢ². Subtracting from the rank-2r SVD error Σ_{i>2r} σᵢ² gives

```
‖A − A_2r‖² − ‖A − Â‖²  ≤  Σ_{i=2r+1}^{r²} σᵢ²
```

The possible gain is the energy in singular values 2r+1 … r². It is large for a flat spectrum and small for a fast-decaying one. It is only an upper bound: the actual gain can be smaller, or negative. Check 6 shows both cases.

## How to run it

Needs Python 3.12 and, for reasonable speed, a CUDA GPU (about 6 GB is enough).

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python checks.py          # 8 correctness checks, a few seconds, must end with "All checks passed."
python compress_gpt2.py   # main experiment, r = 8..64     -> results.csv, spectra.npz
python diagnostics.py     # diagnostics, r = 64..256       -> diagnostics.csv
python plot.py            # figures                        -> *.png
```

GPT-2 small and WikiText-2 are downloaded from the Hugging Face Hub on first use. They are not part of this repository.

On the laptop GPU used for the results below (RTX 4050), `compress_gpt2.py` takes about 11 minutes: about 3 s per HadNet fit (96 fits) and about 35 s per perplexity evaluation (9 evaluations). `diagnostics.py` takes about 75 minutes; it makes 104 perplexity evaluations.

To change ranks, methods or layers, edit the constants at the top of `compress_gpt2.py` or `diagnostics.py`. There are no command-line options.

## What each file does

| File | Purpose |
|---|---|
| `hadamard.py` | The maths on one matrix: `svd_init`, `hadnet_fit`, `reconstruct`, `svd_truncate`. |
| `checks.py` | Eight correctness checks (Khatri–Rao identity, rank bound, exact SVD init, Eckart–Young, parameter count, spectrum dependence, output-error identity, Spearman). |
| `compress_gpt2.py` | Loads GPT-2 small, compresses the 24 MLP matrices for each method and rank, measures perplexity, writes `results.csv` and `spectra.npz`. |
| `diagnostics.py` | Bug check, gentler budgets, output-error and entrywise-error measurements; writes `diagnostics.csv`. See "Diagnostics" in `RESULTS.md`. |
| `plot.py` | Draws `perplexity.png` and `spectra.png`, and from `diagnostics.csv` also `perplexity_gentle.png` and `error_vs_perplexity.png`. |
| `RESULTS.md` | Results of both scripts, the diagnostics, and the conclusions. |
| `VERIFICATION.md` | One row per function: where it comes from (paper section or URL) and how it differs from the source. |
| `requirements.txt` | torch, transformers, datasets, numpy, matplotlib, with the versions used. |
| `results.csv` | Output of `compress_gpt2.py`. One row per compressed matrix (method, r, layer, matrix, stored numbers, relative Frobenius error, fit time) with the run's perplexity repeated on each row; plus one baseline row. |
| `diagnostics.csv` | Output of `diagnostics.py`. One row per compressed matrix: weight error, output error, entrywise statistics, perplexity with that matrix alone compressed (r = 64, 128), perplexity with all 24 compressed. |
| `spectra.npz` | Output of `compress_gpt2.py`. Singular values of the 24 **original** matrices, keys `h0_c_fc` … `h11_c_proj`. |

## Results

The results, the diagnostics and the conclusions drawn from them are in [`RESULTS.md`](RESULTS.md).

## Limits

- The loss fitted is plain weight error ‖W − Ŵ‖_F. Nothing here is activation-aware, and nothing is fine-tuned after compression.
- Only the MLP matrices are compressed; attention and embedding weights are untouched.
- HadNet used the paper's settings unchanged (Adam, learning rate 0.1, 2000 epochs, ×0.1 after 100 epochs without improvement). They were not tuned for these matrices.
- `hadnet_fit` returns the best factors seen, whereas the paper's Algorithm 7 and code return the last ones (see `VERIFICATION.md`, note 1).
- One model, one dataset, one run.

## Sources and acknowledgements

- **The method.** A. Panchal, J. K. Sahoo, R. Behera, *Tensor decomposition and neural architectures through generalized Hadamard products*, Neurocomputing 699 (2026) 134372, <https://doi.org/10.1016/j.neucom.2026.134372>. Their code: <https://github.com/PanchalAneesh/HadNet-HadDNN>. `hadamard.py` is a re-implementation of the single-matrix case, written from the paper and checked against that code; no file is copied from it.
- **Perplexity evaluation.** Hugging Face guide "Perplexity of fixed-length models", <https://huggingface.co/docs/transformers/perplexity>.
- **Calibration activations and the output-error measure.** SVD-LLM (X. Wang, Y. Zheng, Z. Wan, M. Zhang, ICLR 2025, <https://openreview.net/forum?id=LNYIUouhdt>, code <https://github.com/AIoT-MLSys-Lab/SVD-LLM>) and ASVD (Z. Yuan et al., <https://arxiv.org/abs/2312.05821>).
- **Model and data.** GPT-2 small (`gpt2`) and WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`), both loaded from the Hugging Face Hub.

If you use this repository, please cite the paper above.
