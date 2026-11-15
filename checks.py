"""
Correctness checks for hadamard.py. Run:  python checks.py   (CPU, well under a minute)

Checks 1-5, 7 and 8 are exact identities, so they use float64 and assert.
Check 6 runs the real optimiser with the paper's settings in float32.
"paper" as in hadamard.py. "README §M1..M4" are the short derivations in the
README section "Maths behind the checks".
"""

import torch

from diagnostics import output_error, spearman
from hadamard import hadnet_fit, reconstruct, svd_init, svd_truncate

torch.manual_seed(42)  # paper §5: seed 42


def random_factors(m, n, r):
    """Gaussian factors W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n) in float64."""
    W1 = torch.randn(m, r, dtype=torch.float64)              # (m, r)
    H1 = torch.randn(r, n, dtype=torch.float64)              # (r, n)
    W2 = torch.randn(m, r, dtype=torch.float64)              # (m, r)
    H2 = torch.randn(r, n, dtype=torch.float64)              # (r, n)
    return W1, H1, W2, H2


def check_khatri_rao():
    """(W1 H1) * (W2 H2) = E @ F, with E, F the Khatri-Rao products (README §M1).

    This identity is why the rank can reach r^2 (paper Prop. 3.3).
    """
    m, n, r = 7, 5, 3
    W1, H1, W2, H2 = random_factors(m, n, r)
    # Row-wise Khatri-Rao: row i of E lists all products W1[i, k] * W2[i, l].
    E = torch.einsum("ik,il->ikl", W1, W2).reshape(m, r * r)   # (m, r^2)
    # Column-wise Khatri-Rao: column j of F lists all products H1[k, j] * H2[l, j].
    F = torch.einsum("kj,lj->klj", H1, H2).reshape(r * r, n)   # (r^2, n)
    assert torch.allclose(reconstruct(W1, H1, W2, H2), E @ F)
    print("1. Khatri-Rao identity (W1 H1)*(W2 H2) = E F ............ OK")


def check_rank_bound():
    """Each factor product has rank r; their Hadamard product has rank <= r^2 (paper Prop. 3.3)."""
    m, n, r = 20, 20, 3
    W1, H1, W2, H2 = random_factors(m, n, r)
    rank_M1 = torch.linalg.matrix_rank(W1 @ H1).item()
    rank_M2 = torch.linalg.matrix_rank(W2 @ H2).item()
    rank_X = torch.linalg.matrix_rank(reconstruct(W1, H1, W2, H2)).item()
    assert rank_M1 == 3 and rank_M2 == 3
    assert rank_X <= 9
    print(f"2. Rank bound: rank(M1)={rank_M1}, rank(M2)={rank_M2}, rank(M1*M2)={rank_X} <= 9 ... OK")


def check_svd_init_exact():
    """With r = min(m, n) nothing is truncated, so the SVD init reproduces X (paper §3.3)."""
    m, n = 12, 9
    X = torch.randn(m, n, dtype=torch.float64)               # (m, n)
    W1, H1, W2, H2 = svd_init(X, min(m, n))
    assert torch.allclose(reconstruct(W1, H1, W2, H2), X)
    print("3. SVD init is exact at full rank ....................... OK")


def check_eckart_young():
    """Truncated-SVD error = sqrt(sum of discarded sigma_i^2) (README §M2; paper Thm 3.1, one slice)."""
    m, n, k = 12, 9, 4
    X = torch.randn(m, n, dtype=torch.float64)               # (m, n)
    A, B = svd_truncate(X, k)                                # (m, k), (k, n)
    error = torch.linalg.norm(X - A @ B, "fro")
    sigma = torch.linalg.svdvals(X)                          # (min(m, n),), descending
    predicted = torch.sqrt(torch.sum(sigma[k:] ** 2))
    assert torch.allclose(error, predicted)
    print(f"4. Eckart-Young: error {error:.6f} = sqrt(tail) {predicted:.6f} ... OK")


def check_param_count():
    """Hadamard at rank r and SVD at rank 2r both store 2r(m+n) numbers (README §M3)."""
    m, n, r = 30, 50, 4
    X = torch.randn(m, n, dtype=torch.float64)               # (m, n)
    hadamard_count = sum(f.numel() for f in svd_init(X, r))
    svd_count = sum(f.numel() for f in svd_truncate(X, 2 * r))
    assert hadamard_count == svd_count == 2 * r * (m + n)
    print(f"5. Parameter count: Hadamard(r={r}) = SVD(2r={2 * r}) = {hadamard_count} = 2r(m+n) ... OK")


def compare_on(X, r):
    """Relative Frobenius errors of SVD at rank 2r and HadNet at rank r on X (m, n)."""
    A, B = svd_truncate(X, 2 * r)                            # (m, 2r), (2r, n)
    W1, H1, W2, H2, hist = hadnet_fit(X, r)                  # paper's default settings
    norm_X = torch.linalg.norm(X, "fro")
    svd_error = (torch.linalg.norm(X - A @ B, "fro") / norm_X).item()
    had_error = (torch.linalg.norm(X - reconstruct(W1, H1, W2, H2), "fro") / norm_X).item()
    return svd_error, had_error


def check_spectrum_dependence():
    """Hadamard wins on a flat spectrum; SVD is expected to win on a fast-decaying one.

    README §M4 (reading of paper Prop. 3.5): the gain over SVD(2r) is at
    most sum_{i=2r+1}^{r^2} sigma_i^2, which is large for a flat spectrum and
    small for a fast-decaying one.
    """
    n, r = 128, 16

    # Flat spectrum: i.i.d. Gaussian entries.
    X_flat = torch.randn(n, n)                               # (128, 128)
    svd_error, had_error = compare_on(X_flat, r)
    print(f"6a. Flat spectrum:   SVD(rank {2 * r}) rel. error {svd_error:.4f}   HadNet(r={r}) rel. error {had_error:.4f}")
    assert had_error < svd_error
    print("    HadNet beats SVD at equal budget on a flat spectrum .... OK")

    # Fast decay: X = U diag(1/i) V^T with random orthogonal U, V
    # (QR of a Gaussian matrix gives a random orthogonal matrix).
    U, _ = torch.linalg.qr(torch.randn(n, n))                # (128, 128)
    V, _ = torch.linalg.qr(torch.randn(n, n))                # (128, 128)
    sigma = 1.0 / torch.arange(1, n + 1)                     # (128,)  sigma_i = 1/i
    X_decay = (U * sigma) @ V.T                              # (128, 128)
    svd_error, had_error = compare_on(X_decay, r)
    winner = "SVD" if svd_error < had_error else "HadNet"
    print(f"6b. sigma_i ~ 1/i:   SVD(rank {2 * r}) rel. error {svd_error:.4f}   HadNet(r={r}) rel. error {had_error:.4f}")
    print(f"    winner: {winner} (printed only, not asserted; SVD is expected to win)")


def check_output_error():
    """output_error (computed from G = X^T X) equals ||X E||_F / ||X W||_F computed directly.

    Identity: ||X E||_F^2 = trace(E^T X^T X E) = trace(E^T G E)  (diagnostics.output_error).
    """
    tokens, n_in, n_out = 50, 6, 9
    X = torch.randn(tokens, n_in, dtype=torch.float64)       # (tokens, in)  layer inputs
    W = torch.randn(n_in, n_out, dtype=torch.float64)        # (in, out)     GPT-2 layout: output = X @ W
    W_hat = W + 0.1 * torch.randn(n_in, n_out, dtype=torch.float64)   # (in, out)
    G = X.T @ X                                              # (in, in)
    direct = torch.linalg.norm(X @ (W - W_hat), "fro") / torch.linalg.norm(X @ W, "fro")
    from_gram = output_error(W, W_hat, G)
    assert abs(direct.item() - from_gram) < 1e-12
    print(f"7. Output error from Gram matrix {from_gram:.6f} = direct {direct:.6f} ... OK")


def check_spearman():
    """Spearman correlation is +1 / -1 for increasing / decreasing relations, and matches a hand example."""
    a = [0.1, 0.5, 0.2, 0.9, 0.7]
    assert abs(spearman(a, [x ** 3 for x in a]) - 1.0) < 1e-12      # monotone increasing, not linear
    assert abs(spearman(a, [-x for x in a]) + 1.0) < 1e-12          # monotone decreasing
    # Hand example: ranks (1,2,3,4,5) vs (2,1,4,3,5); d^2 sums to 4; rho = 1 - 6*4/(5*24) = 0.8.
    assert abs(spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]) - 0.8) < 1e-12
    print("8. Spearman rank correlation (+1, -1, hand example 0.8) .. OK")


if __name__ == "__main__":
    check_khatri_rao()
    check_rank_bound()
    check_svd_init_exact()
    check_eckart_young()
    check_param_count()
    check_spectrum_dependence()
    check_output_error()
    check_spearman()
    print("All checks passed.")
