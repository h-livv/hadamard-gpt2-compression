"""
Compress every MLP weight matrix of GPT-2 small and measure WikiText-2 test perplexity.

For each (method, r) below, all 24 MLP matrices are replaced by a compressed
version holding 2r(m+n) numbers each:
    "svd"       truncated SVD at rank 2r          (hadamard.svd_truncate)
    "hadamard"  Hadamard decomposition at rank r  (hadamard.hadnet_fit)
The dense matrix is rebuilt from the factors and copied back into the model,
so this measures STORAGE compression only; the model runs at its usual speed.

Run:  python compress_gpt2.py
Writes results.csv (one row per compressed matrix, plus one baseline row)
and spectra.npz (singular values of the 24 original matrices).
"""

import csv
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from hadamard import hadnet_fit, reconstruct, svd_truncate

# ---- settings -------------------------------------------------------------
METHODS = ["svd", "hadamard"]
RANKS = [8, 16, 32, 64]          # r. Hadamard uses rank r, SVD uses rank 2r.
LAYERS = range(12)               # GPT-2 small has 12 transformer blocks
MATRICES = ["c_fc", "c_proj"]    # the two linear maps inside each block's MLP
STRIDE = 512                     # sliding-window stride of the perplexity guide
SEED = 42                        # paper §5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


def get_weight(model, layer, name):
    """Return one MLP weight matrix of GPT-2 in the usual (out, in) layout.

    THE ONE PLACE (with set_weight) THAT DEALS WITH GPT-2's TRANSPOSED STORAGE.
    GPT-2's MLP layers are `Conv1D` modules, not `nn.Linear`. Conv1D computes
    y = x @ weight + bias with weight stored as (in_features, out_features),
    the transpose of nn.Linear's (out_features, in_features). We transpose
    here so that the rest of the code sees an ordinary linear-layer matrix.

    Args:    model (GPT2LMHeadModel), layer in 0..11, name "c_fc" or "c_proj".
    Returns: X (out, in): (3072, 768) for c_fc, (768, 3072) for c_proj. A copy.
    Source:  transformers/pytorch_utils.py, class Conv1D
             (`self.weight = nn.Parameter(torch.empty(nx, nf))`, nx = inputs,
             nf = outputs; forward is `torch.addmm(self.bias, x, self.weight)`),
             https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py
             transformers/models/gpt2/modeling_gpt2.py, class GPT2MLP (c_fc, c_proj).
    """
    conv1d = getattr(model.transformer.h[layer].mlp, name)
    return conv1d.weight.detach().T.clone()                  # (out, in)


def set_weight(model, layer, name, X):
    """Copy a dense (out, in) matrix into a GPT-2 MLP layer. Inverse of get_weight.

    Args:    X (out, in), same layout as get_weight returns.
    Source:  as get_weight.
    """
    conv1d = getattr(model.transformer.h[layer].mlp, name)
    with torch.no_grad():
        conv1d.weight.copy_(X.T)                             # stored as (in, out)


def compress_matrix(X, method, r):
    """Compress one matrix and return its dense reconstruction.

    Both methods store 2r(m+n) numbers (checked in checks.py, check 5).

    Args:    X (m, n), method "svd" or "hadamard", rank r.
    Returns: X_approx (m, n).
    Source:  hadamard.py (paper §3.3, §4.1; torch.linalg.svd).
    """
    if method == "svd":
        A, B = svd_truncate(X, 2 * r)                        # (m, 2r), (2r, n)
        return A @ B                                         # (m, n)
    W1, H1, W2, H2, hist = hadnet_fit(X, r)                  # (m, r), (r, n), (m, r), (r, n)
    return reconstruct(W1, H1, W2, H2)                       # (m, n)


def load_test_tokens(tokenizer):
    """WikiText-2 test set as one long row of token ids.

    Returns: input_ids (1, seq_len).
    Source:  https://huggingface.co/docs/transformers/perplexity
             (the `load_dataset` / `"\\n\\n".join(test["text"])` block, unchanged).
    """
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    return encodings.input_ids                               # (1, seq_len)


def perplexity(model, input_ids):
    """Perplexity = exp(average negative log-likelihood per token), strided sliding window.

    Each window is up to 1024 tokens and windows start every STRIDE tokens.
    Only the tokens not already scored by the previous window count towards
    the loss (label -100 means "ignore"); the earlier ones are context only.

    Args:    model (GPT2LMHeadModel), input_ids (1, seq_len).
    Returns: float.
    Source:  https://huggingface.co/docs/transformers/perplexity, the loop
             under "Example: Calculating perplexity with GPT-2", stride 512.
    Differs: no tqdm progress bar and no accelerate (a plain device string);
             the guide's example uses gpt2-large, we use gpt2 (small).
    """
    max_length = model.config.n_positions                    # 1024 for GPT-2
    seq_len = input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, STRIDE):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc                     # may differ from STRIDE on the last loop
        window = input_ids[:, begin_loc:end_loc].to(DEVICE)  # (1, <=1024)
        target_ids = window.clone()                          # (1, <=1024)
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            # loss = mean cross-entropy over the labels that are not -100
            neg_log_likelihood = model(window, labels=target_ids).loss

        # The model shifts labels left by one internally, so one fewer token
        # is actually scored than is unmasked.
        num_valid_tokens = (target_ids != -100).sum().item()
        num_loss_tokens = num_valid_tokens - target_ids.size(0)
        nll_sum += neg_log_likelihood.item() * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    return float(np.exp(nll_sum / n_tokens))


def save_results(rows):
    """Write all result rows collected so far to results.csv (rewritten after every run)."""
    with open("results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    np.savez("spectra.npz", **spectra)

    m, n = 768, 3072                                         # every MLP matrix is 768x3072 or 3072x768
    baseline = perplexity(model, input_ids)
    print(f"baseline perplexity {baseline:.2f}")
    rows = [dict(method="baseline", r=0, layer=-1, matrix="", params_per_matrix=m * n,
                 rel_error=0.0, seconds=0.0, perplexity=baseline)]
    save_results(rows)

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
            save_results(rows)
            mean_error = np.mean([row["rel_error"] for row in new_rows])
            total_seconds = sum(row["seconds"] for row in new_rows)
            print(f"{method:9s} r={r:3d}  perplexity {ppl:12.2f}  mean rel. error {mean_error:.4f}  "
                  f"fit time {total_seconds:.0f}s")
