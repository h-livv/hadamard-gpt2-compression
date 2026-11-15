"""
Diagnostics: why did a lower weight error give a worse perplexity?

Three hypotheses are measured here. Nothing is fitted differently.
  H3 (bug)             the compressed weight in the model is not the one we fitted.
  H1 (error direction) the network feels the OUTPUT error ||X (W - W_hat)|| on real
                       activations X, not the weight error ||W - W_hat||.
  H2 (outlier entries) the Hadamard reconstruction has heavier-tailed entrywise errors.

Step 1  bug_check: one near-lossless matrix is written, read back and evaluated.
Step 2  gentler budgets: each matrix compressed alone (r = 64, 128), and all 24
        together (r = 64, 128, 192, 256); Hadamard at rank r against SVD at rank 2r.
Step 3  for every compressed matrix: relative output error on calibration
        activations, and entrywise error statistics.

ORIENTATION. In this file W, W_hat and E = W - W_hat are in GPT-2's own
(in_features, out_features) layout, so the layer computes X @ W for inputs
X (tokens, in_features). `get_weight` returns the transposed (out, in) layout,
so we transpose once, right after calling it.

Run:  python diagnostics.py      (writes diagnostics.csv; about 75 minutes on an RTX 4050)
"""

import csv
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from compress_gpt2 import (DEVICE, compress_matrix, get_weight, load_test_tokens,
                           perplexity, set_weight)

# ---- settings -------------------------------------------------------------
RUN_BUG_CHECK = True
BUG_CHECK_MATRICES = [(0, "c_fc"), (5, "c_fc")]   # (layer, name)
BUG_CHECK_R = 300   # break-even is r* = mn / (2(m+n)) = 307.2 for 768x3072; SVD gets rank 2r = 600

SINGLE_RANKS = [64, 128]            # Step 2.1: one matrix compressed, the other 23 untouched
ALL_RANKS = [64, 128, 192, 256]     # Step 2.2: all 24 matrices compressed together
METHODS = ["svd", "hadamard"]
LAYERS = range(12)
MATRICES = ["c_fc", "c_proj"]

N_CALIBRATION = 64                  # calibration sequences, from the TRAIN split
CALIBRATION_LENGTH = 512            # tokens per calibration sequence
SEED = 42
# ---------------------------------------------------------------------------


def relative_error(W, W_hat):
    """Relative Frobenius error ||W - W_hat||_F / ||W||_F.

    Args:    W, W_hat of the same shape (either layout; the value is the same).
    Returns: float.
    Source:  the paper reports this as "Relative error" (repo random-tensor/main.py,
             `err / norm_tensor`); torch.linalg.norm,
             https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
    """
    return (torch.linalg.norm(W - W_hat, "fro") / torch.linalg.norm(W, "fro")).item()


def bug_check(model, input_ids):
    """Step 1 (H3): write one near-lossless matrix, read it back, measure perplexity.

    For each matrix and method it prints: the weight error before writing,
    the weight error of what `get_weight` reads back from the model, and the
    perplexity with only that matrix replaced. It also asserts that
      - the two errors agree (the model holds what we fitted),
      - the layer really computes x @ W_hat + bias for the matrix we fitted
        (so orientation, dtype and device are right),
      - the original weight is restored exactly afterwards.

    Args:    model (GPT2LMHeadModel), input_ids (1, seq_len) test tokens.
    Returns: list of dict rows (matrix, method, r, error_before, error_read_back, perplexity).
    Source:  Conv1D.forward in transformers/pytorch_utils.py
             (`torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)`) for the
             forward identity; get_weight / set_weight / perplexity as in VERIFICATION.md.
    """
    rows = []
    for layer, name in BUG_CHECK_MATRICES:
        W_out_in = get_weight(model, layer, name)                # (out, in), a copy of the original
        conv1d = getattr(model.transformer.h[layer].mlp, name)
        x = torch.randn(4, W_out_in.shape[1], device=DEVICE)     # (4, in) random probe inputs
        for method in METHODS:
            W_hat_out_in = compress_matrix(W_out_in, method, BUG_CHECK_R)   # (out, in)
            error_before = relative_error(W_out_in, W_hat_out_in)

            set_weight(model, layer, name, W_hat_out_in)
            W_read = get_weight(model, layer, name)              # (out, in), read back from the model
            error_read_back = relative_error(W_out_in, W_read)
            assert abs(error_before - error_read_back) < 1e-7
            assert conv1d.weight.dtype == torch.float32
            assert conv1d.weight.shape == (W_out_in.shape[1], W_out_in.shape[0])
            with torch.no_grad():
                assert torch.allclose(conv1d(x), x @ W_hat_out_in.T + conv1d.bias, atol=1e-4)

            ppl = perplexity(model, input_ids)

            set_weight(model, layer, name, W_out_in)             # restore before the next test
            assert torch.equal(get_weight(model, layer, name), W_out_in)

            rank = 2 * BUG_CHECK_R if method == "svd" else BUG_CHECK_R
            rows.append(dict(matrix=f"h[{layer}].mlp.{name}", method=method, r=rank,
                             error_before=error_before, error_read_back=error_read_back, perplexity=ppl))
            print(f"h[{layer}].mlp.{name}  {method:9s} rank {rank:3d}  error before {error_before:.6f}  "
                  f"read back {error_read_back:.6f}  perplexity {ppl:.2f}", flush=True)
    return rows


def load_calibration_tokens(tokenizer):
    """N_CALIBRATION random windows of CALIBRATION_LENGTH tokens from the WikiText-2 TRAIN split.

    The test split is never touched here, so the diagnostics cannot leak
    into the perplexity measurement.

    Returns: calibration_ids (N_CALIBRATION, CALIBRATION_LENGTH) = (64, 512).
    Source:  SVD-LLM, utils/data_utils.py, `get_calib_train_data`
             (https://github.com/AIoT-MLSys-Lab/SVD-LLM): `random.seed(seed)`,
             the "wikitext-2-raw-v1" train split joined with "\\n\\n", and
             `random.randint(0, length - seqlen - 1)` for each window start.
    Differs: SVD-LLM draws a random CHARACTER offset and tokenises from there;
             we tokenise the whole split once and draw a random TOKEN offset,
             which gives windows of exactly 512 tokens without a retry loop.
             Seed 42 (the paper's) and 64 x 512 are this project's settings.
    """
    random.seed(SEED)
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    train_ids = tokenizer("\n\n".join(train["text"]), return_tensors="pt").input_ids   # (1, n_train_tokens)
    n_train_tokens = train_ids.size(1)
    windows = []
    for _ in range(N_CALIBRATION):
        i = random.randint(0, n_train_tokens - CALIBRATION_LENGTH - 1)
        windows.append(train_ids[:, i:i + CALIBRATION_LENGTH])  # (1, 512)
    return torch.cat(windows, dim=0)                             # (64, 512)


def add_to_gram(module, args, output):
    """Forward hook: add X^T X of this call's input to `module.G`.

    PyTorch calls a forward hook as hook(module, args, output) after every
    forward pass of `module`; args[0] is the layer's input.

    Args:    module (Conv1D with attribute G (in, in)), args[0] (batch, seq, in).
    Source:  torch.nn.Module.register_forward_hook,
             https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
             SVD-LLM, SVDLLM.py, `profle_svdllm`, inner function `hook`
             (`adds = torch.matmul(inp.transpose(1,2), inp)`, summed into
             `module.raw_scaling_diag_matrix`).
    Differs: we flatten batch and sequence first (same sum) and accumulate in
             float64, because G sums 32,768 float32 outer products.
    """
    X = args[0].detach().double().reshape(-1, args[0].size(-1))  # (batch * seq, in)
    module.G += X.T @ X                                          # (in, in)


def collect_gram_matrices(model, calibration_ids):
    """Gram matrix G = X^T X of the inputs to every MLP matrix, over all calibration tokens.

    X is the (tokens, in) matrix whose rows are the inputs the layer sees.
    Only G is kept, so memory does not grow with the number of tokens.

    Hook points, from GPT2MLP.forward in transformers/models/gpt2/modeling_gpt2.py:
        hidden_states = self.c_fc(hidden_states)     <- input of c_fc: 768 numbers per token
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)   <- input of c_proj: 3072 numbers per token
    A forward hook on the c_fc / c_proj module receives exactly that input.

    Args:    model (GPT2LMHeadModel, original weights), calibration_ids (N, L).
    Returns: dict {(layer, name): G} with G (768, 768) for c_fc, (3072, 3072) for c_proj, float64.
    Source:  as add_to_gram; https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
    """
    handles = []
    for layer in LAYERS:
        for name in MATRICES:
            conv1d = getattr(model.transformer.h[layer].mlp, name)
            conv1d.G = torch.zeros(conv1d.nx, conv1d.nx, dtype=torch.float64, device=DEVICE)   # (in, in)
            handles.append(conv1d.register_forward_hook(add_to_gram))

    with torch.no_grad():
        for sequence in calibration_ids:                         # one (L,) sequence at a time
            model(sequence[None, :].to(DEVICE))

    for handle in handles:
        handle.remove()                                          # the model is back to normal
    grams = {}
    for layer in LAYERS:
        for name in MATRICES:
            conv1d = getattr(model.transformer.h[layer].mlp, name)
            grams[(layer, name)] = conv1d.G                      # (in, in)
            del conv1d.G
    return grams


def output_error(W, W_hat, G):
    """Relative output error ||X E||_F / ||X W||_F with E = W - W_hat, from the Gram matrix G = X^T X.

    This is the damage the layer's OUTPUT suffers on real inputs, relative to
    the size of the true output. Identity used (so X itself is not needed):
        ||X E||_F^2 = trace((X E)^T (X E)) = trace(E^T X^T X E) = trace(E^T G E),
    and trace(E^T G E) = sum of all entries of E * (G @ E).

    Args:    W, W_hat (in, out) in GPT-2's layout; G (in, in).
    Returns: float.
    Source:  the calibration objective of SVD-LLM (Wang, Zheng, Wan, Zhang,
             ICLR 2025, https://openreview.net/forum?id=LNYIUouhdt; code
             https://github.com/AIoT-MLSys-Lab/SVD-LLM) and ASVD (Yuan et al.,
             arXiv:2312.05821): the compression loss is
             ||W X - W' X||_F on calibration activations X. Their W is (out, in)
             with X (in, tokens); ours is the transpose of both.
    Differs: they MINIMISE this quantity; we only MEASURE it, and divide by ||X W||_F.
    """
    W = W.double()                                               # (in, out)
    E = W - W_hat.double()                                       # (in, out)
    error_squared = torch.sum(E * (G @ E))                       # trace(E^T G E) = ||X E||_F^2
    signal_squared = torch.sum(W * (G @ W))                      # trace(W^T G W) = ||X W||_F^2
    return torch.sqrt(error_squared / signal_squared).item()


def entry_stats(E):
    """Three summaries of how unevenly the error E is spread over its entries.

    Args:    E (in, out) in GPT-2's layout, so column j of E is the error in
             all the weights feeding output unit j.
    Returns: max_abs    largest |E_ij|
             p999_abs   99.9th percentile of |E_ij|
             col_ratio  largest column norm of E / median column norm of E
    Source:  plain descriptive statistics. torch.quantile,
             https://pytorch.org/docs/stable/generated/torch.quantile.html
             torch.linalg.vector_norm,
             https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
    """
    abs_E = torch.abs(E).flatten()                               # (in * out,)
    max_abs = abs_E.max().item()
    p999_abs = torch.quantile(abs_E, 0.999).item()
    column_norms = torch.linalg.vector_norm(E, dim=0)            # (out,)
    col_ratio = (column_norms.max() / column_norms.median()).item()
    return max_abs, p999_abs, col_ratio


def spearman(a, b):
    """Spearman rank correlation: the Pearson correlation of the ranks of a and b.

    +1 means b always rises when a rises, -1 the opposite, 0 no monotone relation.

    Args:    a, b sequences of the same length, no tied values.
    Returns: float in [-1, 1].
    Source:  definition as in scipy.stats.spearmanr,
             https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
             numpy.corrcoef for the Pearson step.
    Differs: no handling of ties (scipy averages tied ranks). Our values are
             floating-point errors and perplexities, where ties do not occur.
    """
    rank_a = np.argsort(np.argsort(a))                           # (len,) rank of each a[i], 0 = smallest
    rank_b = np.argsort(np.argsort(b))                           # (len,)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def save_rows(rows):
    """Write all rows collected so far to diagnostics.csv (rewritten after every run)."""
    with open("diagnostics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = load_test_tokens(tokenizer)
    baseline = perplexity(model, input_ids)
    print(f"uncompressed perplexity {baseline:.2f}", flush=True)

    if RUN_BUG_CHECK:
        bug_check(model, input_ids)

    # Gram matrices must come from the ORIGINAL model, before anything is compressed.
    grams = collect_gram_matrices(model, load_calibration_tokens(tokenizer))

    m, n = 768, 3072
    rows = []
    for r in ALL_RANKS:
        for method in METHODS:
            new_rows = []
            compressed = {}
            for layer in LAYERS:
                for name in MATRICES:
                    W_out_in = get_weight(model, layer, name)                    # (out, in) original
                    W_hat_out_in = compress_matrix(W_out_in, method, r)          # (out, in)
                    compressed[(layer, name)] = (W_out_in, W_hat_out_in)

                    W = W_out_in.T                                               # (in, out) GPT-2 layout
                    W_hat = W_hat_out_in.T                                       # (in, out)
                    max_abs, p999_abs, col_ratio = entry_stats(W - W_hat)
                    row = dict(method=method, r=r, layer=layer, matrix=name,
                               budget_percent=100 * 2 * r * (m + n) / (m * n),
                               weight_error=relative_error(W, W_hat),
                               output_error=output_error(W, W_hat, grams[(layer, name)]),
                               max_abs_E=max_abs, p999_abs_E=p999_abs, col_norm_ratio=col_ratio,
                               perplexity_single="", perplexity_all="")

                    if r in SINGLE_RANKS:                                        # Step 2.1
                        set_weight(model, layer, name, W_hat_out_in)
                        row["perplexity_single"] = perplexity(model, input_ids)
                        set_weight(model, layer, name, W_out_in)                 # restore
                        print(f"  {method:9s} r={r:3d} h[{layer}].{name:6s} alone: perplexity "
                              f"{row['perplexity_single']:9.2f}  weight error {row['weight_error']:.4f}  "
                              f"output error {row['output_error']:.4f}", flush=True)
                    new_rows.append(row)

            for (layer, name), (W_out_in, W_hat_out_in) in compressed.items():   # Step 2.2
                set_weight(model, layer, name, W_hat_out_in)
            ppl_all = perplexity(model, input_ids)
            for (layer, name), (W_out_in, W_hat_out_in) in compressed.items():
                set_weight(model, layer, name, W_out_in)                         # restore
            for row in new_rows:
                row["perplexity_all"] = ppl_all                  # same value on all 24 rows of this run
            rows.extend(new_rows)
            save_rows(rows)
            print(f"{method:9s} r={r:3d} all 24 matrices: perplexity {ppl_all:.2f}  "
                  f"mean weight error {np.mean([row['weight_error'] for row in new_rows]):.4f}  "
                  f"mean output error {np.mean([row['output_error'] for row in new_rows]):.4f}", flush=True)

    print(f"uncompressed perplexity after restoring {perplexity(model, input_ids):.2f}")
