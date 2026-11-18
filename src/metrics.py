"""
Error measures and statistics used to compare an approximation W_hat with W.

Pure functions of tensors and arrays; nothing here knows about GPT-2.
v2 adds relative_weighted_error, psnr and condition_number.
"""

import numpy as np
import torch


def relative_error(W, W_hat):
    """Relative Frobenius error ||W - W_hat||_F / ||W||_F.

    Args:    W, W_hat of the same shape (either layout; the value is the same).
    Returns: float.
    Source:  the paper reports this as "Relative error" (repo random-tensor/main.py,
             `err / norm_tensor`); torch.linalg.norm,
             https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
    """
    return (torch.linalg.norm(W - W_hat, "fro") / torch.linalg.norm(W, "fro")).item()


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


def relative_weighted_error(A, A_hat, G):
    """Relative weighted error ||S^T (A - A_hat)||_F / ||S^T A||_F with G = S S^T on the row index.

    Same quantity as output_error (there G = X^T X); written for any G.
    Args:    A, A_hat (m, n), G (m, m).  Returns: float.
    Source:  hadamard.weighted_error (SVD-LLM Eq. 1 with the Gram matrix).
    """
    A, A_hat, G = A.double(), A_hat.double(), G.double()
    E = A - A_hat
    return torch.sqrt(torch.sum(E * (G @ E)) / torch.sum(A * (G @ A))).item()


def psnr(A, A_hat):
    """PSNR = 10 log10(1 / MSE) in dB, for pixel values on [0, 1], no clipping of A_hat.

    Args:    A, A_hat (m, n).  Returns: float.
    Source:  paper §6 (performance measures): PSNR = 10 log10(MPP / MSE), MPP the
             maximum possible pixel value. MPP = 1 here, so MPP and MPP^2 (the
             usual definition) agree.
    """
    mse = torch.mean((A.double() - A_hat.double()) ** 2).item()
    return 10.0 * np.log10(1.0 / mse)


def condition_number(G):
    """lambda_max / lambda_min of a symmetric positive definite G (float64 eigvalsh).

    Source:  torch.linalg.eigvalsh, https://pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html
    """
    lam = torch.linalg.eigvalsh(G.double())                      # ascending
    return (lam[-1] / lam[0]).item()
