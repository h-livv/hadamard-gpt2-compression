"""
How damage accumulates: compress blocks 2, 3, ..., 11 one after another and
measure perplexity after each block is added.

For each (method, r), the model starts uncompressed. Then block 2's two MLP
matrices are compressed and perplexity is measured; then block 3's are
compressed as well (block 2 stays compressed) and perplexity is measured
again; and so on until blocks 2-11 are all compressed. Blocks 0 and 1 are
never compressed. The last step is the same configuration as one run of
activation_aware.py, so its perplexity should match that run's.

Methods are the two activation-weighted ones from activation_aware.py:
    "svd_w"       whitened SVD at rank 2r (SVD-LLM)
    "hadamard_w"  HadNet at rank r with the activation-weighted loss
Each matrix is fitted once, with Gram matrices from the ORIGINAL model (as in
activation_aware.py), so later blocks do not see the errors of earlier ones.

Run:  python experiments/cumulative.py      (writes results/cumulative.csv; resumes like activation_aware.py)
"""

import csv
import os
import time

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import (DEVICE, MATRICES, SEED, collect_gram_matrices, get_weight,
                                     load_calibration_tokens, load_test_tokens, perplexity, set_weight)
from src.methods import compress, damped
from src.metrics import output_error, relative_error
from src.paths import RESULTS_DIR

# ---- settings -------------------------------------------------------------
RANKS = [64, 128, 256]
METHODS = ["svd_w", "hadamard_w"]
COMPRESSED_LAYERS = range(2, 12)    # as in activation_aware.py: blocks 0 and 1 stay uncompressed
OUT = RESULTS_DIR / "cumulative.csv"
# ---------------------------------------------------------------------------


def save_rows(rows):
    """Write all rows collected so far to OUT (rewritten after every step)."""
    with open(OUT, "w", newline="") as f:
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

    grams = collect_gram_matrices(model, load_calibration_tokens(tokenizer))   # original model

    m, n = 768, 3072
    rows = []
    if os.path.exists(OUT):                                                    # resume
        with open(OUT, newline="") as f:
            rows = list(csv.DictReader(f))
    done = {(row["method"], int(row["r"])) for row in rows if int(row["last_block"]) == COMPRESSED_LAYERS[-1]}
    rows = [row for row in rows if (row["method"], int(row["r"])) in done]    # drop half-finished sequences

    for r in RANKS:
        matrix_share = 2 * r * (m + n) / (m * n)
        for method in METHODS:
            if (method, r) in done:
                print(f"{method:10s} r={r:3d}  already in {OUT.name}, skipped", flush=True)
                continue
            originals = {}
            previous = baseline
            for step, layer in enumerate(COMPRESSED_LAYERS, start=1):
                weight_errors, output_errors = [], []
                start = time.time()
                for name in MATRICES:
                    X = get_weight(model, layer, name)                         # (out, in) original
                    X_hat = compress(X, method, r, damped(grams[(layer, name)]))
                    originals[(layer, name)] = X
                    set_weight(model, layer, name, X_hat)
                    weight_errors.append(relative_error(X, X_hat))
                    output_errors.append(output_error(X.T, X_hat.T, grams[(layer, name)]))
                seconds = time.time() - start
                ppl = perplexity(model, input_ids)
                rows.append(dict(
                    method=method, r=r, blocks_compressed=step, last_block=layer,
                    mlp_budget_percent=100 * (24 - 2 * step + 2 * step * matrix_share) / 24,
                    c_fc_weight_error=weight_errors[0], c_proj_weight_error=weight_errors[1],
                    c_fc_output_error=output_errors[0], c_proj_output_error=output_errors[1],
                    perplexity=ppl, increase_from_previous=ppl - previous, fit_seconds=seconds))
                save_rows(rows)
                print(f"{method:10s} r={r:3d}  blocks 2..{layer:2d} compressed: perplexity {ppl:9.2f} "
                      f"(+{ppl - previous:8.2f})", flush=True)
                previous = ppl
            for (layer, name), X in originals.items():
                set_weight(model, layer, name, X)                              # restore for the next sequence

    print(f"uncompressed perplexity after restoring {perplexity(model, input_ids):.2f}")
