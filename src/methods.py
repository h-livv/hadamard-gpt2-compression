"""
The compression methods as the experiments call them: one function that takes
a weight matrix and returns its dense reconstruction.

The maths is in hadamard.py; this file only chooses which function to call
and with what rank, so that every method stores 2r(m+n) numbers.
"""

import torch

from src.hadamard import hadnet_fit, reconstruct, svd_truncate, whitened_svd

DAMP = 0.01   # relative damping of G, as GPTQ's `percdamp` (Frantar et al., arXiv:2210.17323)


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


def damped(G):
    """G + DAMP * mean(diag G) * I, divided by mean(diag G), in float32.

    The damping keeps G invertible (whitened_svd divides by sqrt of its
    eigenvalues). The division only rescales the loss, which neither SVD nor
    Adam is sensitive to, and keeps float32 well inside its range.

    Args:    G (in, in) float64.  Returns: (in, in) float32.
    Source:  GPTQ's relative damping (`percdamp`); used by stages 3, 4 and 5.5.
    """
    scale = torch.mean(torch.diagonal(G))
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return ((G + DAMP * scale * eye) / scale).float()


def compress(X, method, r, G):
    """Dense reconstruction of X (out, in) by "svd", "hadamard", "svd_w" or "hadamard_w"; G (in, in) damped, float32.

    hadamard.py puts G on the ROW index (weighted error ||S^T E||_F), so the
    weighted methods fit X^T, GPT-2's own (in, out) layout, and transpose back.
    Stages 3 and 4 were run with an earlier form of this function that fitted X
    with G on the column index: the same loss, written transposed. For the exact
    float32 arithmetic of those runs, check out tag pre-restructure-main
    (docs/REPORT.md, Reproduction).
    """
    if method == "svd":
        A, B = svd_truncate(X, 2 * r)
        return A @ B
    if method == "svd_w":
        A, B = whitened_svd(X.T.double(), 2 * r, G.double())  # float64: G's eigenvalues span many decades
        return (A @ B).T.float()
    if method == "hadamard_w":
        W1, H1, W2, H2, hist = hadnet_fit(X.T, r, G=G)
        return reconstruct(W1, H1, W2, H2).detach().T
    W1, H1, W2, H2, hist = hadnet_fit(X, r)
    return reconstruct(W1, H1, W2, H2).detach()


# ---- storage accounting ----------------------------------------------------
M, N = 768, 3072                      # every GPT-2 small MLP matrix is 768 x 3072 or 3072 x 768
MATRIX_PARAMS = M * N                 # 2,359,296 numbers in one uncompressed MLP matrix
GPT2_TOTAL_PARAMS = 124_439_808       # all parameters of GPT-2 small (tied embedding counted once),
                                      # sum(p.numel() for p in GPT2LMHeadModel.from_pretrained("gpt2").parameters())
BYTES_PER_PARAM = 4                   # float32, as everywhere in this project


def model_params(r, n_compressed):
    """Numbers stored for the whole model when n_compressed MLP matrices are kept as rank-r factors.

    Every method here stores 2r(m+n) numbers per compressed matrix (Hadamard at rank r,
    SVD at rank 2r), so the size depends only on r and on how many matrices are
    compressed, not on the method. Biases, attention, embeddings and LayerNorms are
    never compressed. This counts STORED numbers: the experiments rebuild each dense
    matrix before running the model, so memory in use while running is unchanged.

    Args:    r (int), n_compressed (0..24).
    Returns: int.
    """
    return GPT2_TOTAL_PARAMS - n_compressed * MATRIX_PARAMS + n_compressed * 2 * r * (M + N)


def model_megabytes(r, n_compressed):
    """model_params(r, n_compressed) in megabytes (10^6 bytes) at float32."""
    return model_params(r, n_compressed) * BYTES_PER_PARAM / 1e6
