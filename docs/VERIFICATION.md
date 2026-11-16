# Verification: where every function comes from

**Paper** = Panchal, Sahoo, Behera, *Tensor decomposition and neural architectures through generalized Hadamard products*, Neurocomputing 699 (2026) 134372.

**Repo** = <https://github.com/PanchalAneesh/HadNet-HadDNN>, read at commit `4aabe3d` (9 July 2026), file `random-tensor/main.py`. The HadNet code in `application-images/main_image.py` and `application-videos/main_video.py` uses the same settings.

**PPL guide** = <https://huggingface.co/docs/transformers/perplexity>.

**README §M1–M4** = the short derivations in the README section "Maths behind the checks".

**SVD-LLM** = Wang, Zheng, Wan, Zhang, ICLR 2025, <https://openreview.net/forum?id=LNYIUouhdt>; code <https://github.com/AIoT-MLSys-Lab/SVD-LLM>, read at commit `7538cca`. **ASVD** = Yuan et al., <https://arxiv.org/abs/2312.05821>.

Layout: `src/` is the library (`hadamard.py`, the method; `metrics.py`, error measures; `gpt2.py`, model and data plumbing; `methods.py`, the compression dispatchers; `paths.py`), `experiments/` holds one script per experiment plus `plot.py`, and `tests/checks.py` holds the correctness checks. The headings below give the path of each file.

The paper is written for tensors and treats each matrix slice independently (its Remark 2.4). Every function here is the one-slice case, so the paper's `⊡` is an ordinary `@`.

## `src/hadamard.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `svd_init(X, r)` | Builds P = √\|X\| and Q = √\|X\|·sign(X), takes a rank-r SVD of each, returns W1 = U_r√S_r, H1 = √S_r V_rᵀ (from P) and W2, H2 (from Q). | Paper §3.3. Repo `svd_initialize`. `torch.linalg.svd` docs: <https://pytorch.org/docs/stable/generated/torch.linalg.svd.html>. | No loop over slices. Keeps the dtype of X (the repo allocates float32 with `torch.zeros`); this lets `checks.py` use float64. GPT-2 weights are float32, so the experiment matches the repo. |
| `reconstruct(W1, H1, W2, H2)` | Returns X_pred = (W1 @ H1) * (W2 @ H2). | Paper §4.1 forward pass (M1, M2, X_pred) and Eq. (5). Repo `HadamardNet.forward`, `h_prod`. | `@` replaces the repo's `h_prod` slice loop. |
| `hadnet_fit(X, r, max_epochs=2000, lr=0.1, patience=100)` | Adam on the four factors, loss ‖X − X_pred‖_F (not squared), learning rate ×0.1 after 100 epochs without a new best loss, returns the best factors and the loss history. | Paper Algorithm 7 and §4.1; settings from §5 ("η = 0.1, wait = 100 and max_epochs = 2000"). Repo `HadamardNet_train`. | **Returns the best factors, not the last ones.** See note 1. Uses `torch.linalg.norm(·, "fro")` where the repo uses the older `torch.norm(·, 'fro')`; same value. Returns the loss history (the repo also records `hist` but does not return it). |
| `hadnet_fit(..., G=None)` | With `G` given, the loss is `weighted_error(X − X_pred, G)` instead of ‖X − X_pred‖_F. | Paper Algorithm 7 with SVD-LLM's loss. | **Not in the paper.** The paper only fits weight error. Start (`svd_init` of W), optimiser and schedule are unchanged. |
| `svd_truncate(X, k)` | Rank-k truncated SVD; returns A = U_k S_k (m, k) and B = V_kᵀ (k, n). | `torch.linalg.svd` docs (URL above). Repo `svd_transform`. Paper Theorem 3.1 (one slice). | The repo returns the dense product `(U_r * S_r) @ Vh_r`; we return the two halves so the 2r(m+n) count is visible. The caller forms `A @ B`. |
| `weighted_error(E, G)` | √sum(E ∘ (E G)) = ‖E Xᵀ‖_F for E (out, in), G = XᵀX. | Loss ‖WX − W′X‖_F of SVD-LLM and ASVD (URLs above), written with the Gram matrix. | Same identity as `metrics.output_error`, in the (out, in) layout and unnormalised. Tested by `check_weighted`. |
| `whitened_svd(X, k, G)` | G = SSᵀ; truncated SVD of X S at rank k; returns A and B S⁻¹. | SVD-LLM §3.1, "truncation-aware data whitening"; code `SVDLLM.py`, `whitening`. | SVD-LLM takes S from a Cholesky factorisation (adding to the diagonal if it fails); we use `torch.linalg.eigh`, S = Q diag(√λ). Any S with SSᵀ = G gives the same result. Tested by `check_weighted`. |

## `src/metrics.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `relative_error(W, W_hat)` | ‖W − Ŵ‖_F / ‖W‖_F. | The paper's "Relative error" (repo `random-tensor/main.py`, `err / norm_tensor`). `torch.linalg.norm` docs: <https://pytorch.org/docs/stable/generated/torch.linalg.norm.html>. | None. |
| `output_error(W, W_hat, G)` | √trace(EᵀGE) / √trace(WᵀGW) = ‖XE‖_F / ‖XW‖_F with E = W − Ŵ, in GPT-2's (in, out) layout. | The calibration objective ‖WX − W′X‖_F of SVD-LLM and ASVD. Identity ‖XE‖²_F = trace(EᵀXᵀXE) derived in the docstring and tested by `check_output_error`. | Those papers minimise this quantity; we only measure it and normalise by ‖XW‖_F. Their W is (out, in) with X (in, tokens); ours is the transpose of both. |
| `entry_stats(E)` | max \|E\|, the 99.9th percentile of \|E\|, and largest column norm / median column norm. | Descriptive statistics. `torch.quantile`, <https://pytorch.org/docs/stable/generated/torch.quantile.html>; `torch.linalg.vector_norm`, <https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html>. | Columns are those of the (in, out) layout: one column per output unit. |
| `spearman(a, b)` | Spearman rank correlation = Pearson correlation of the ranks. | Definition in the `scipy.stats.spearmanr` docs (URL in the `checks.py` table); `numpy.corrcoef`. | No tie handling (scipy averages tied ranks); our inputs have no ties. Written in numpy to avoid adding scipy as a dependency. |

## `src/gpt2.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `get_weight(model, layer, name)` | Reads `transformer.h[layer].mlp.c_fc` or `.c_proj` and returns the weight transposed to (out, in). | `Conv1D` in `transformers/pytorch_utils.py`: `self.weight = nn.Parameter(torch.empty(nx, nf))`, forward `torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)`, docstring "works like a linear layer but the weights are transposed". <https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py>. `GPT2MLP` in `models/gpt2/modeling_gpt2.py` defines `c_fc = Conv1D(intermediate_size, embed_dim)` and `c_proj = Conv1D(embed_dim, intermediate_size)`. Read in the installed version 5.23.0. | None. Checked numerically in the smoke test: `conv(x)` equals `x @ X.T + bias` for the returned X. |
| `set_weight(model, layer, name, X)` | Copies a dense (out, in) matrix back, transposed to (in, out). | As `get_weight`. | None. |
| `load_test_tokens(tokenizer)` | Loads the WikiText-2 raw test split and tokenises it as one sequence. | PPL guide, the `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")` block. | None. |
| `perplexity(model, input_ids)` | Sliding-window perplexity, window 1024, stride 512. | PPL guide, the loop under "Example: Calculating perplexity with GPT-2 in 🤗 Transformers". | No `tqdm`, no `accelerate` (neither is in `requirements.txt`). The guide's example loads `gpt2-large`; we load `gpt2`. `.item()` is applied to the loss before accumulating; same arithmetic. |
| `load_calibration_tokens(tokenizer)` | 64 random windows of 512 tokens from the WikiText-2 **train** split, seed 42. | SVD-LLM `utils/data_utils.py`, `get_calib_train_data` (`random.seed(seed)`, train split joined with `"\n\n"`, `random.randint(0, length - seqlen - 1)`). | SVD-LLM draws a random character offset and tokenises from there; we tokenise once and draw a random token offset, so every window is exactly 512 tokens. Seed 42 and 64 × 512 are this project's settings. |
| `add_to_gram(module, args, output)` | Forward hook: adds XᵀX of the layer's input to `module.G`. | `torch.nn.Module.register_forward_hook`, <https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook>. SVD-LLM `SVDLLM.py`, `profle_svdllm`, inner `hook` (`adds = torch.matmul(inp.transpose(1,2), inp)` summed into `module.raw_scaling_diag_matrix`). | Batch and sequence are flattened first (same sum). Accumulated in float64. |
| `collect_gram_matrices(model, calibration_ids)` | Registers the hook on all 24 `c_fc` / `c_proj` modules, runs the calibration sequences through the original model, removes the hooks, returns the 24 Gram matrices. | Hook points from `GPT2MLP.forward` in `transformers/models/gpt2/modeling_gpt2.py`: `hidden_states = self.c_fc(hidden_states)`; `hidden_states = self.act(hidden_states)`; `hidden_states = self.c_proj(hidden_states)`. Hook removal via the handle returned by `register_forward_hook`. | SVD-LLM hooks every `nn.Linear`; GPT-2's MLP layers are `Conv1D`, so we hook those by name. |

## `src/methods.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `compress_matrix(X, method, r)` | Calls `svd_truncate(X, 2r)` or `hadnet_fit(X, r)` and returns the dense reconstruction. | Functions above. The 2r pairing is the equal-budget rule of README §M3; the paper itself compares at equal rank (its tables). | A dispatcher, not a method. |
| `damped(G)` | G + 0.01·mean(diag G)·I, divided by mean(diag G), float32. | Relative damping as GPTQ's `percdamp = .01` (Frantar et al., <https://arxiv.org/abs/2210.17323>, code <https://github.com/IST-DASLab/gptq>). | GPTQ damps the Hessian 2XXᵀ before inverting it; we damp G before whitening and use the same damped G as the HadNet loss. The division only rescales the loss. |
| `compress(X, method, r, G)` | Dispatcher for `svd`, `hadamard`, `svd_w` (`whitened_svd`, rank 2r, float64) and `hadamard_w` (`hadnet_fit` with `G`). | `hadamard.py`. | A dispatcher, not a method. All four store 2r(m+n) numbers. |
| `model_params(r, n_compressed)`, `model_megabytes(r, n_compressed)` | Numbers stored for the whole model, and megabytes at float32, when `n_compressed` MLP matrices are kept as 2r(m+n) factors. | `GPT2_TOTAL_PARAMS` = `sum(p.numel() for p in GPT2LMHeadModel.from_pretrained("gpt2").parameters())`, measured with transformers 5.23.0. | Counts stored numbers only; the experiments run the model with rebuilt dense matrices. |

## `src/paths.py`

| Name | What it is | Source | Difference |
|---|---|---|---|
| `ROOT`, `RESULTS_DIR`, `FIGURES_DIR` | Repository root, `results/` and `figures/`, found from the location of `src/paths.py`. | `pathlib`, <https://docs.python.org/3/library/pathlib.html>. | Bookkeeping only. |

## `experiments/compress_gpt2.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `save_results(rows)` | Writes `results.csv`. | Python `csv.DictWriter`, <https://docs.python.org/3/library/csv.html>. | Bookkeeping only. |
| main block | `GPT2LMHeadModel.from_pretrained("gpt2")`, `GPT2TokenizerFast.from_pretrained("gpt2")`; saves `torch.linalg.svdvals` of each original matrix. | <https://huggingface.co/docs/transformers/model_doc/gpt2>; <https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html>. | Seed 42 is set as in the paper's §5, but nothing in the pipeline is random (note 4). |

## `experiments/diagnostics.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| `bug_check(model, input_ids)` | Step 1 (H3). Writes one near-lossless matrix (HadNet r = 300, SVD rank 600), reads it back with `get_weight`, compares the weight error before and after, asserts `conv1d(x) == x @ W_hat + bias` (Ŵ in the (in, out) layout), measures perplexity, restores the original. | `Conv1D.forward` in `transformers/pytorch_utils.py` (`torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)`) for the forward identity. Uses `get_weight`, `set_weight`, `compress_matrix`, `perplexity` above. | A test, not a method. |
| `save_rows(rows)` | Writes `diagnostics.csv`. | Python `csv.DictWriter`. | Bookkeeping only. |

## `experiments/activation_aware.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| main block | Gram matrices from the original model (`src/gpt2.py`, `collect_gram_matrices`, 64 × 512 train tokens); blocks 2–11 compressed together for each (method, r); one test perplexity per run; weights restored after each run. | `src/`. | Blocks 0 and 1 are left uncompressed, following the single-matrix results in `RESULTS.md`. Output error is reported with the undamped G. |

## `experiments/cumulative.py`

| Function | What it does | Source | Difference from the source, and why |
|---|---|---|---|
| main block | For each (method, r): compresses blocks 2, 3, …, 11 one after another (earlier blocks stay compressed) with `src/methods.py`, `compress`, and measures test perplexity after each block. Gram matrices from the original model. Restores the weights after each sequence. | `src/`, `experiments/activation_aware.py`. | An experiment, not a method. Its last step is the same configuration as one `activation_aware.py` run. |

## `experiments/plot.py`

| Function | What it does | Source | Difference |
|---|---|---|---|
| `read_perplexities`, `plot_perplexity`, `plot_spectra`, `plot_error_vs_perplexity`, `plot_activation_aware`, `plot_cumulative`, `plot_per_matrix_errors`, `size_table`, `plot_size_vs_perplexity` | Read the files in `results/` and draw the eight figures in `figures/` (the spectra and the model-size plot at the top level, the rest in `weight_error_fits/` and `activation_aware_fits/`). | matplotlib pyplot API, <https://matplotlib.org/stable/api/pyplot_summary.html>. | Plotting only; no method from the paper. |

## `tests/checks.py`

| Function | What it does | Source | Difference |
|---|---|---|---|
| `random_factors` | Gaussian test factors. | `torch.randn`. | Helper. |
| `check_khatri_rao` | Asserts (W1H1)∘(W2H2) = E F. | README §M1. The paper states the consequence, rank ≤ r², as Prop. 3.3. | None. |
| `check_rank_bound` | r = 3, m = n = 20: factor products have rank 3, their Hadamard product rank ≤ 9. | Paper Prop. 3.3. | None. |
| `check_svd_init_exact` | At r = min(m, n) the SVD init reproduces X. | Paper §3.3 (X = mat_simple ∘ mat_signed). | None. |
| `check_eckart_young` | Truncated-SVD error = √(Σ discarded σᵢ²). | README §M2; paper Theorem 3.1 for one slice (the paper calls it the Eckart–Young theorem). | None. |
| `check_param_count` | Both methods store 2r(m+n) numbers. | README §M3. | None. |
| `compare_on`, `check_spectrum_dependence` | HadNet(16) vs SVD(32) on a 128×128 Gaussian matrix (asserted) and on σᵢ = 1/i (printed). | README §M4, a reading of paper Prop. 3.5. | None. |
| `check_output_error` | `output_error` from G = XᵀX equals ‖XE‖_F / ‖XW‖_F computed directly on a small random X. | The identity ‖XE‖²_F = trace(EᵀXᵀXE), derived in the `output_error` docstring. | None. |
| `check_spearman` | `spearman` gives +1, −1 and 0.8 on three hand-checkable examples. | Definition in the `scipy.stats.spearmanr` docs, <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html>. | None. |
| `check_weighted` | `weighted_error` equals ‖E Xᵀ‖_F computed directly and ‖E‖_F when G = I; `whitened_svd` with G = I equals `svd_truncate`; with a real G its weighted error equals the Eckart–Young bound for X S and is no larger than plain SVD's. | SVD-LLM §3.1 (whitening); README §M2 (Eckart–Young). | None. |

## Notes: paper text vs. repo vs. this project

1. **"Keep the best factors seen" is in neither the paper nor the repo.** Algorithm 7 tracks `best_err`, but only to drive the learning-rate schedule; its line 21 returns the factors as they stand after the last epoch, and so does the repo (`HadamardNet_train` returns `model.W1` etc. after the loop). This project returns the best factors because the project brief asks for it. On GPT-2 the two differ very little: in the smoke test (`h[0].mlp.c_fc`, r = 16) the best loss was at epoch 1972 of 2000 and the last-epoch loss was larger by a relative 5·10⁻⁵.
2. **Dead code in the repo.** The repo's schedule test is `if wait == 0 or updatecount == 100`, but `updatecount` is never incremented, so it behaves exactly as Algorithm 7's `if wait == 0`. We follow Algorithm 7.
3. **SVD routine.** Paper §5 says SVD used `np.linalg.svd`. The repo uses `torch.linalg.svd` both for the SVD baseline and inside `svd_initialize`. We use `torch.linalg.svd`. Both compute the same decomposition.
4. **Seed 42.** In the repo the seed is set once in `__main__`, before the random test matrices are generated. HadNet itself draws no random numbers (SVD start, full-matrix steps), and neither does anything else here, so the seed has no effect on the GPT-2 results. It matters only for the random matrices in `checks.py`.
5. **Variable name in the repo.** `svd_initialize` names the third output of `torch.linalg.svd` `V1`, but it is Vᵀ (`Vh`). The repo's expression `(torch.sqrt(S1[:r]) * V1[:r, :].T).T` therefore equals √S_r V_rᵀ, as in the paper. Ours is the same quantity written as `torch.sqrt(S[:r])[:, None] * Vh[:r, :]`.
6. **Precision.** The repo casts the input to float32. GPT-2 weights are float32, so the experiment runs in float32 too.
