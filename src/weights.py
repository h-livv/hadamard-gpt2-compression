"""
The weightings G of the single-matrix experiments (docs/REPORT.md §2).

Every G is (n, n), symmetric positive definite, acts on the ROW index of the
image (weighted error ||S^T E||_F = sqrt(trace(E^T G E)), G = S S^T), and is
normalised so that mean(diag G) = 1. All in float64.

    identity            G = I                              condition number 1
    region              G = D^2, d_i = 10 on the middle    condition number 100
                        third of the rows, 1 elsewhere
    freq{kappa}         G = C^T diag(w) C, C the orthonormal condition number kappa
                        DCT-II, w_k = kappa^(-k/(n-1))

Source: the task specification for v2 (this project's design). The DCT matrix
is scipy.fft.dct(np.eye(n), norm="ortho", axis=0),
https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
"""

import numpy as np
import scipy.fft
import torch

KAPPAS = [10, 100, 1000]
WEIGHTINGS = ["identity", "region"] + [f"freq{k}" for k in KAPPAS]


def normalise(G):
    """G divided by mean(diag G), so that mean(diag G) = 1 (does not change the condition number)."""
    return G / np.mean(np.diag(G))


def region_d(n):
    """d (n,): 10 on rows n//3 .. 2n//3 - 1 (the middle third), 1 elsewhere."""
    d = np.ones(n)
    d[n // 3: 2 * n // 3] = 10.0
    return d


def dct_matrix(n):
    """Orthonormal DCT-II matrix C (n, n): C @ x is the DCT of x, row k is frequency k, C^T C = I."""
    return scipy.fft.dct(np.eye(n), norm="ortho", axis=0)


def make_G(name, n):
    """The weighting `name` (one of WEIGHTINGS) at size n, as a float64 torch tensor (n, n).

    For "region" the normalised G is still diagonal, G = D^2 with D = diag(d / sqrt(mean(d^2))).
    """
    if name == "identity":
        G = np.eye(n)
    elif name == "region":
        G = np.diag(region_d(n) ** 2)
    elif name.startswith("freq"):
        kappa = float(name[4:])
        w = kappa ** (-np.arange(n) / (n - 1))             # (n,) 1 down to 1/kappa
        C = dct_matrix(n)
        G = C.T @ np.diag(w) @ C
        G = (G + G.T) / 2                                  # exact symmetry against rounding
    else:
        raise ValueError(name)
    return torch.from_numpy(normalise(G))
