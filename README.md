# Hadamard decomposition vs. truncated SVD on GPT-2 small

An independent study of the method of Panchal, Sahoo and Behera, [*Tensor decomposition and neural architectures through generalized Hadamard products*](https://doi.org/10.1016/j.neucom.2026.134372), Neurocomputing 699 (2026) 134372. Not affiliated with the authors.

It compresses the 24 MLP weight matrices of GPT-2 small with the paper's Hadamard decomposition and with truncated SVD **at the same storage**, and measures WikiText-2 test perplexity (uncompressed: 25.17).

**Result:** at equal storage, the Hadamard form approximates GPT-2's weights better than SVD and, with plain weight-error fits, gives far better models from about 40% storage up (45 against 1,399 at 83%). The larger gain comes from fitting the error on real activations instead: whitened SVD then reaches perplexity 29.3 with blocks 2–11 compressed. Weighted Hadamard wins at the smallest budget; at larger ones its optimiser, run with the paper's settings, does not converge on some matrices under this new loss. Details in [`docs/RESULTS.md`](docs/RESULTS.md).

## Method

- **What is compressed.** Each of GPT-2 small's 12 blocks has an MLP with two weight matrices, `c_fc` (768 → 3072) and `c_proj` (3072 → 768). Each is replaced by a low-storage approximation Ŵ, rebuilt as a dense matrix and copied back. Only storage is reduced; the model runs as before, and nothing is fine-tuned.
- **Two methods at equal storage.** Hadamard at rank r, Ŵ = (W₁H₁) ∘ (W₂H₂) with ∘ the entrywise product, fitted with HadNet (paper Algorithm 7) from the paper's SVD initialisation. Against truncated SVD at rank **2r**. Both store 2r(m+n) numbers. (The paper's tables compare the two at equal rank r, where the Hadamard factors take twice the storage; this study fixes storage instead.)
- **Two losses.** Plain weight error ‖W − Ŵ‖_F. Or the activation-weighted error ‖X(W − Ŵ)‖_F on the layer's real inputs X (calibration text from the WikiText-2 training split), as in SVD-LLM and ASVD. For SVD the optimum of the latter is SVD-LLM's whitened SVD; for Hadamard, HadNet is run with that loss, an extension made here that is not part of the paper.

### Maths behind the checks

`tests/checks.py` verifies these numerically.

- **M1. Rank up to r².** Expanding an entry, (W₁H₁) ∘ (W₂H₂) = E·F, where E (m × r²) has rows W₁[i] ⊗ W₂[i] and F (r² × n) has columns H₁[:, j] ⊗ H₂[:, j] (Khatri–Rao products). So its rank is at most r² (paper Prop. 3.3), against 2r for SVD at the same storage.
- **M2. Eckart–Young.** Truncated SVD at rank k is the best rank-k approximation, with ‖X − X_k‖_F² = Σ_{i>k} σᵢ² (the matrix case of paper Theorem 3.1).
- **M3. Equal storage.** Hadamard at rank r stores 2·(m·r) + 2·(r·n) = 2r(m+n); SVD at rank k stores k(m+n). Equal at k = 2r.
- **M4. Possible gain.** By M1 and M2, ‖A − A₂ᵣ‖² − ‖A − Â‖² ≤ Σ_{i=2r+1}^{r²} σᵢ² (our reading of paper Prop. 3.5). The gain can be large only for a flat spectrum, which GPT-2's MLP matrices have. It is an upper bound; the real gain can be smaller or negative.

## How to run it

Python 3.12 and, for reasonable speed, a CUDA GPU with about 6 GB. GPT-2 small and WikiText-2 are downloaded from the Hugging Face Hub on first use. Times are for an RTX 4050 laptop GPU.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/checks.py                  # 9 checks, seconds; must end with "All checks passed."
python experiments/compress_gpt2.py     # benchmark, all 24 matrices, r = 8..64     (~11 min)
python experiments/diagnostics.py       # bug check, r = 64..256, one matrix at a time (~75 min)
python experiments/activation_aware.py  # four fits, blocks 2-11, r = 64..256          (~35 min)
python experiments/cumulative.py        # blocks 2..11 compressed one by one           (~50 min)
python experiments/plot.py              # every figure
```

Settings are constants at the top of each script. `activation_aware.py` and `cumulative.py` resume if interrupted.

## Layout

```
src/
  hadamard.py   the method: SVD initialisation, HadNet, truncated and whitened SVD
  metrics.py    weight error, output error, entrywise statistics, Spearman
  gpt2.py       GPT-2 weights, test perplexity, calibration Gram matrices
  methods.py    the methods at equal storage, damping, model size
  paths.py      locations of results/ and figures/
experiments/    one script per experiment, plus plot.py
tests/          checks.py
results/        results.csv, spectra.npz (compress_gpt2); diagnostics.csv; activation_aware.csv; cumulative.csv
figures/        weight_error_fits/, activation_aware_fits/, and the singular-value spectra
docs/           RESULTS.md (results), VERIFICATION.md (where every function comes from)
```

## Limits

- One model, one dataset; only the MLP weight matrices (45.5% of the model) are compressed.
- HadNet uses the paper's settings unchanged (Adam, learning rate 0.1, 2000 steps); `hadnet_fit` returns the best factors seen, the paper's code the last ones.
- Calibration inputs come from the original model, not from the partly compressed one. No fine-tuning.

## Sources

- **Method:** Panchal, Sahoo, Behera, Neurocomputing 699 (2026) 134372, <https://doi.org/10.1016/j.neucom.2026.134372>; code <https://github.com/PanchalAneesh/HadNet-HadDNN>. `src/hadamard.py` re-implements the single-matrix case; no file is copied.
- **Activation-aware fitting:** SVD-LLM (Wang, Zheng, Wan, Zhang, ICLR 2025, <https://openreview.net/forum?id=LNYIUouhdt>; code <https://github.com/AIoT-MLSys-Lab/SVD-LLM>) and ASVD (Yuan et al., <https://arxiv.org/abs/2312.05821>).
- **Perplexity:** Hugging Face, "Perplexity of fixed-length models", <https://huggingface.co/docs/transformers/perplexity>.
- **Model and data:** `gpt2` and `Salesforce/wikitext` (`wikitext-2-raw-v1`) from the Hugging Face Hub.
