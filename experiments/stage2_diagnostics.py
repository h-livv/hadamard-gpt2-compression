"""
Stage 2, diagnostics: why did a lower weight error give a worse perplexity (stage 1)?

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

Reads     GPT-2 small and WikiText-2 from the Hugging Face Hub (cached after first use).
Writes    results/stage2/diagnostics.csv  one row per (method, r, matrix); the bug check is printed only
Runtime   about 75 minutes on an RTX 4050 laptop GPU.

Run:  python experiments/stage2_diagnostics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import (DEVICE, LAYERS, MATRICES, SEED, collect_gram_matrices, get_weight,
                                     load_calibration_tokens, load_test_tokens, perplexity, set_weight)
from src.methods import compress_matrix
from src.metrics import entry_stats, output_error, relative_error
from src.paths import RESULTS_DIR, write_csv


# ---- settings -------------------------------------------------------------
RUN_BUG_CHECK = True
BUG_CHECK_MATRICES = [(0, "c_fc"), (5, "c_fc")]   # (layer, name)
BUG_CHECK_R = 300   # break-even is r* = mn / (2(m+n)) = 307.2 for 768x3072; SVD gets rank 2r = 600

SINGLE_RANKS = [64, 128]            # Step 2.1: one matrix compressed, the other 23 untouched
ALL_RANKS = [64, 128, 192, 256]     # Step 2.2: all 24 matrices compressed together
METHODS = ["svd", "hadamard"]
# ---------------------------------------------------------------------------


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
             forward identity; get_weight / set_weight / perplexity from src/gpt2.py.
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
            write_csv(rows, RESULTS_DIR / "stage2" / "diagnostics.csv")
            print(f"{method:9s} r={r:3d} all 24 matrices: perplexity {ppl_all:.2f}  "
                  f"mean weight error {np.mean([row['weight_error'] for row in new_rows]):.4f}  "
                  f"mean output error {np.mean([row['output_error'] for row in new_rows]):.4f}", flush=True)

    print(f"uncompressed perplexity after restoring {perplexity(model, input_ids):.2f}")
