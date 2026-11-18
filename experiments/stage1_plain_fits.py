"""
Stage 1, plain fits: compress every MLP weight matrix of GPT-2 small and measure WikiText-2 test perplexity.

For each (method, r) below, all 24 MLP matrices are replaced by a compressed
version holding 2r(m+n) numbers each:
    "svd"       truncated SVD at rank 2r          (src/hadamard.py, svd_truncate)
    "hadamard"  Hadamard decomposition at rank r  (src/hadamard.py, hadnet_fit)
The dense matrix is rebuilt from the factors and copied back into the model,
so this measures STORAGE compression only; the model runs at its usual speed.

Reads     GPT-2 small and WikiText-2 from the Hugging Face Hub (cached after first use).
Writes    results/stage1/results.csv  one row per compressed matrix, plus one baseline row
          results/stage1/spectra.npz  singular values of the 24 original matrices
Runtime   about 11 minutes on an RTX 4050 laptop GPU.

Run:  python experiments/stage1_plain_fits.py
"""

import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import (DEVICE, LAYERS, MATRICES, SEED, get_weight, load_test_tokens,
                                     perplexity, set_weight)
from src.methods import compress_matrix
from src.paths import RESULTS_DIR, write_csv

# ---- settings -------------------------------------------------------------
METHODS = ["svd", "hadamard"]
RANKS = [8, 16, 32, 64]          # r. Hadamard uses rank r, SVD uses rank 2r.
OUT = RESULTS_DIR / "stage1"
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # https://huggingface.co/docs/transformers/model_doc/gpt2
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = load_test_tokens(tokenizer)

    # Keep the original matrices: every run must start from the true weights.
    originals = {}
    spectra = {}
    for layer in LAYERS:
        for name in MATRICES:
            X = get_weight(model, layer, name)               # (out, in)
            originals[(layer, name)] = X
            spectra[f"h{layer}_{name}"] = torch.linalg.svdvals(X).cpu().numpy()   # (768,)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "spectra.npz", **spectra)

    m, n = 768, 3072                                         # every MLP matrix is 768x3072 or 3072x768
    baseline = perplexity(model, input_ids)
    print(f"baseline perplexity {baseline:.2f}")
    rows = [dict(method="baseline", r=0, layer=-1, matrix="", params_per_matrix=m * n,
                 rel_error=0.0, seconds=0.0, perplexity=baseline)]
    write_csv(rows, OUT / "results.csv")

    for method in METHODS:
        for r in RANKS:
            new_rows = []
            for (layer, name), X in originals.items():
                start = time.time()
                X_approx = compress_matrix(X, method, r)     # (out, in)
                seconds = time.time() - start
                rel_error = (torch.linalg.norm(X - X_approx, "fro") / torch.linalg.norm(X, "fro")).item()
                set_weight(model, layer, name, X_approx)
                new_rows.append(dict(method=method, r=r, layer=layer, matrix=name,
                                     params_per_matrix=2 * r * (m + n),
                                     rel_error=rel_error, seconds=seconds))
            ppl = perplexity(model, input_ids)
            for row in new_rows:
                row["perplexity"] = ppl                      # same value on all 24 rows of this run
            rows.extend(new_rows)
            write_csv(rows, OUT / "results.csv")
            mean_error = np.mean([row["rel_error"] for row in new_rows])
            total_seconds = sum(row["seconds"] for row in new_rows)
            print(f"{method:9s} r={r:3d}  perplexity {ppl:12.2f}  mean rel. error {mean_error:.4f}  "
                  f"fit time {total_seconds:.0f}s")
