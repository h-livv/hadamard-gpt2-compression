"""
Stage 3, activation-aware fits, with blocks 0 and 1 left uncompressed.

Two changes from stages 1 and 2, suggested by their results (docs/REPORT.md):
  1. Blocks 0 and 1 are not compressed (they caused the six worst single-matrix
     runs in stage 2). Only the 20 MLP matrices of blocks 2-11 are.
  2. Besides the plain weight-error fits, each method is also fitted to the
     activation-weighted error ||X (W - W_hat)||_F on calibration inputs X,
     as SVD-LLM and ASVD do for SVD:
        "svd"          truncated SVD at rank 2r                (weight error)
        "hadamard"     HadNet at rank r                        (weight error)
        "svd_w"        whitened SVD at rank 2r (SVD-LLM)       (activation-weighted, optimal)
        "hadamard_w"   HadNet at rank r, weighted loss         (activation-weighted)
All four store 2r(m+n) numbers per compressed matrix. For each (method, r) the
20 matrices are compressed together and perplexity is measured once; no
matrix is evaluated on its own.

The Gram matrices G = X^T X come from the ORIGINAL model on 64 x 512 tokens of
the WikiText-2 TRAIN split (src/gpt2.py, load_calibration_tokens); perplexity is
on the TEST split. Before fitting, G is damped: G + DAMP * mean(diag G) * I.

Reads     GPT-2 small and WikiText-2 from the Hugging Face Hub (cached after first use).
Writes    results/stage3/activation_aware.csv  one row per (method, r, matrix)
Runtime   about 35 minutes on an RTX 4050 laptop GPU.
If that CSV already exists, the (method, r) runs it contains are
skipped, so an interrupted run can be resumed. Delete it to start afresh.

Run:  python experiments/stage3_activation_aware.py
"""

import csv
import os
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import (DEVICE, MATRICES, SEED, collect_gram_matrices, get_weight,
                                     load_calibration_tokens, load_test_tokens, perplexity, set_weight)
from src.methods import compress, damped
from src.metrics import output_error, relative_error
from src.paths import RESULTS_DIR, write_csv

# ---- settings -------------------------------------------------------------
RANKS = [64, 128, 256]                              # Hadamard rank r; SVD rank 2r
METHODS = ["svd", "hadamard", "svd_w", "hadamard_w"]
COMPRESSED_LAYERS = range(2, 12)                    # blocks 0 and 1 stay uncompressed
OUT = RESULTS_DIR / "stage3" / "activation_aware.csv"
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(SEED)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = load_test_tokens(tokenizer)
    baseline = perplexity(model, input_ids)
    print(f"uncompressed perplexity {baseline:.2f}", flush=True)

    grams = collect_gram_matrices(model, load_calibration_tokens(tokenizer))   # original model

    m, n = 768, 3072
    rows = []
    if os.path.exists(OUT):                                 # resume an interrupted run
        with open(OUT, newline="") as f:
            rows = list(csv.DictReader(f))
    done = {(row["method"], int(row["r"])) for row in rows}
    for r in RANKS:
        matrix_share = 2 * r * (m + n) / (m * n)
        mlp_share = (2 * 2 + 2 * len(COMPRESSED_LAYERS) * matrix_share) / 24   # blocks 0, 1 kept whole
        for method in METHODS:
            if (method, r) in done:
                print(f"{method:10s} r={r:3d}  already in {OUT.name}, skipped", flush=True)
                continue
            new_rows = []
            originals = {}
            for layer in COMPRESSED_LAYERS:
                for name in MATRICES:
                    X = get_weight(model, layer, name)                         # (out, in) original
                    G = damped(grams[(layer, name)])                           # (in, in)
                    start = time.time()
                    X_hat = compress(X, method, r, G)                          # (out, in)
                    seconds = time.time() - start
                    originals[(layer, name)] = X
                    set_weight(model, layer, name, X_hat)
                    new_rows.append(dict(
                        method=method, r=r, layer=layer, matrix=name,
                        matrix_budget_percent=100 * matrix_share,
                        mlp_budget_percent=100 * mlp_share,
                        weight_error=relative_error(X, X_hat),
                        # GPT-2 layout (in, out); undamped G, i.e. the calibration (train) activations
                        output_error=output_error(X.T, X_hat.T, grams[(layer, name)]),
                        seconds=seconds, perplexity=""))
            ppl = perplexity(model, input_ids)
            for (layer, name), X in originals.items():
                set_weight(model, layer, name, X)                              # restore
            for row in new_rows:
                row["perplexity"] = ppl                                        # same on all 20 rows
            rows.extend(new_rows)
            write_csv(rows, OUT)
            print(f"{method:10s} r={r:3d}  perplexity {ppl:10.2f}  "
                  f"mean weight error {np.mean([x['weight_error'] for x in new_rows]):.4f}  "
                  f"mean output error {np.mean([x['output_error'] for x in new_rows]):.4f}  "
                  f"fit time {sum(x['seconds'] for x in new_rows):.0f}s", flush=True)

    print(f"uncompressed perplexity after restoring {perplexity(model, input_ids):.2f}")
