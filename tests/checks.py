"""
Correctness checks for src/hadamard.py, src/metrics.py and src/weights.py.
Run:  python tests/checks.py   (CPU, about a minute)

Checks 1-9 cover the library used by every stage (checks 7-9 the weighted
error of stages 2-5). Checks 10-15 cover the weighted fitting of stage 5.
Check 16 re-reads the numbers in the appendix of docs/REPORT.md from results/.
Exact identities use float64 and assert. Check 6 runs the real optimiser with
the paper's settings in float32.
"paper" as in hadamard.py. "M1..M4" are the short derivations in
docs/REPORT.md, section "Background".
"""

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

from src.hadamard import (embed_rank_r, hadnet_fit, reconstruct, svd_init, svd_truncate,
                          weighted_error, whitened_svd)
from src.metrics import condition_number, output_error, spearman
from src.weights import WEIGHTINGS, make_G, region_d

torch.manual_seed(42)  # paper §5: seed 42


def random_factors(m, n, r):
    """Gaussian factors W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n) in float64."""
    W1 = torch.randn(m, r, dtype=torch.float64)              # (m, r)
    H1 = torch.randn(r, n, dtype=torch.float64)              # (r, n)
    W2 = torch.randn(m, r, dtype=torch.float64)              # (m, r)
    H2 = torch.randn(r, n, dtype=torch.float64)              # (r, n)
    return W1, H1, W2, H2


def check_khatri_rao():
    """(W1 H1) * (W2 H2) = E @ F, with E, F the Khatri-Rao products (REPORT, Background M1).

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
    """Truncated-SVD error = sqrt(sum of discarded sigma_i^2) (REPORT, Background M2; paper Thm 3.1, one slice)."""
    m, n, k = 12, 9, 4
    X = torch.randn(m, n, dtype=torch.float64)               # (m, n)
    A, B = svd_truncate(X, k)                                # (m, k), (k, n)
    error = torch.linalg.norm(X - A @ B, "fro")
    sigma = torch.linalg.svdvals(X)                          # (min(m, n),), descending
    predicted = torch.sqrt(torch.sum(sigma[k:] ** 2))
    assert torch.allclose(error, predicted)
    print(f"4. Eckart-Young: error {error:.6f} = sqrt(tail) {predicted:.6f} ... OK")


def check_param_count():
    """Hadamard at rank r and SVD at rank 2r both store 2r(m+n) numbers (REPORT, Background M3)."""
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

    REPORT, Background M4 (reading of paper Prop. 3.5): the gain over SVD(2r) is at
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


def check_weighted():
    """Weighted error and whitened SVD (hadamard.weighted_error, whitened_svd), G on the row index.

    (a) weighted_error(E, X^T X) = ||X E||_F computed directly; with G = I it is ||E||_F.
    (b) whitened_svd with G = I is the plain truncated SVD.
    (c) whitened_svd attains the Eckart-Young bound in the weighted error: its error is
        sqrt(sum of discarded sigma_i^2 of S^T W), and it beats plain SVD in that error.
    """
    inp, out, tokens, k = 20, 30, 200, 5
    W = torch.randn(inp, out, dtype=torch.float64)                       # (in, out), GPT-2 layout
    X = torch.randn(tokens, inp, dtype=torch.float64) * torch.linspace(0.1, 3, inp, dtype=torch.float64)
    G = X.T @ X                                                          # (in, in)
    E = torch.randn(inp, out, dtype=torch.float64)                       # (in, out)
    assert abs(weighted_error(E, G) - torch.linalg.norm(X @ E)) < 1e-9
    assert abs(weighted_error(E, torch.eye(inp, dtype=torch.float64)) - torch.linalg.norm(E)) < 1e-12

    I = torch.eye(inp, dtype=torch.float64)
    A, B = whitened_svd(W, k, I)
    A0, B0 = svd_truncate(W, k)
    assert torch.allclose(A @ B, A0 @ B0, atol=1e-10)

    A, B = whitened_svd(W, k, G)
    err_white = weighted_error(W - A @ B, G).item()
    lam, Q = torch.linalg.eigh(G)
    sigma = torch.linalg.svdvals(torch.sqrt(lam)[:, None] * Q.T @ W)    # singular values of S^T W
    assert abs(err_white - torch.sqrt(torch.sum(sigma[k:] ** 2)).item()) < 1e-8
    err_plain = weighted_error(W - A0 @ B0, G).item()
    assert err_white <= err_plain
    print(f"9. Weighted error identity; whitened SVD at Eckart-Young bound ({err_white:.4f} <= plain {err_plain:.4f}) ... OK")


# ---- weighted fitting (stage 5) ---------------------------------------------

def check_identity_weighting():
    """With G = I: whitened SVD = SVD, weighted loss = plain loss, and
    weighted HadNet follows plain HadNet's loss curve (same start, same loss)."""
    n, k, r = 32, 6, 3
    A = torch.rand(n, n, dtype=torch.float64)
    G = make_G("identity", n)
    Aw, Bw = whitened_svd(A, k, G)
    As, Bs = svd_truncate(A, k)
    gap_svd = torch.max(torch.abs(Aw @ Bw - As @ Bs)).item()
    E = torch.randn(n, n, dtype=torch.float64)
    gap_loss = abs(weighted_error(E, G).item() - torch.linalg.norm(E).item())
    *_, hist_w = hadnet_fit(A, r, max_epochs=300, G=G)
    *_, hist_p = hadnet_fit(A, r, max_epochs=300)
    gap_curve = max(abs(a - b) / b for a, b in zip(hist_w, hist_p))
    assert gap_svd < 1e-10 and gap_loss < 1e-12 and gap_curve < 1e-8
    print(f"10. G = I: |whitened - SVD| {gap_svd:.1e}, |weighted - plain loss| {gap_loss:.1e}, "
          f"HadNet curves rel. gap {gap_curve:.1e} ... OK")


def check_diagonal_identities():
    """For diagonal G = D^2 (REPORT, stage 5.4):
    (a) ||D(A - (W1 H1)*(W2 H2))||_F = ||DA - ((D W1) H1)*(W2 H2)||_F, from D(X * Y) = (DX) * Y;
    (b) whitened_svd(A, k, D^2) = D^{-1} [DA]_k."""
    n, r, k = 24, 3, 5
    A = torch.rand(n, n, dtype=torch.float64)
    d = torch.from_numpy(region_d(n)) * (1 + torch.rand(n, dtype=torch.float64))  # any positive d
    D, G = torch.diag(d), torch.diag(d ** 2)
    W1, H1, W2, H2 = random_factors(n, n, r)
    lhs = weighted_error(A - reconstruct(W1, H1, W2, H2), G).item()
    rhs = torch.linalg.norm(D @ A - reconstruct(D @ W1, H1, W2, H2)).item()
    gap_had = abs(lhs - rhs) / rhs
    Aw, Bw = whitened_svd(A, k, G)
    As, Bs = svd_truncate(D @ A, k)
    gap_svd = torch.max(torch.abs(Aw @ Bw - torch.diag(1 / d) @ As @ Bs)).item()
    assert gap_had < 1e-10 and gap_svd < 1e-10
    print(f"11. Diagonal G: Hadamard identity rel. gap {gap_had:.1e}, whitened SVD = D^-1[DA]_k gap {gap_svd:.1e} ... OK")


def check_weightings():
    """Every weighting is symmetric positive definite with mean(diag G) = 1;
    freq{kappa} has condition number kappa (to 1e-6 relative), region 100, identity 1."""
    worst = 0.0
    for n in [64, 128, 256, 512]:
        for name in WEIGHTINGS:
            G = make_G(name, n)
            assert torch.equal(G, G.T)
            assert torch.linalg.eigvalsh(G)[0] > 0
            assert abs(torch.mean(torch.diagonal(G)).item() - 1) < 1e-12
            target = {"identity": 1.0, "region": 100.0}.get(name) or float(name[4:])
            rel = abs(condition_number(G) - target) / target
            worst = max(worst, rel)
            assert rel < 1e-6, (n, name, rel)
    print(f"12. Weightings SPD, mean diag 1, condition numbers exact (worst rel. gap {worst:.1e}) ... OK")


def check_embedding():
    """embed_rank_r reproduces the rank-r whitened SVD A_r exactly, and its zero
    columns of W2 / zero rows of H2 get exactly zero gradient (so gradient descent from it stays at rank r)."""
    n, r = 40, 4
    A = torch.rand(n, n, dtype=torch.float64)
    G = make_G("freq100", n)
    Aw, Bw = whitened_svd(A, r, G)
    W1, H1, W2, H2 = embed_rank_r(Aw, Bw)
    gap = torch.max(torch.abs(reconstruct(W1, H1, W2, H2) - Aw @ Bw)).item()
    assert gap < 1e-12
    for f in (W1, H1, W2, H2):
        f.requires_grad_(True)
    weighted_error(A - reconstruct(W1, H1, W2, H2), G).backward()
    dead = torch.abs(W2.grad[:, 1:]).max().item() + torch.abs(H2.grad[1:, :]).max().item()
    assert dead == 0.0
    print(f"13. Rank-r embedding reproduces A_r (max gap {gap:.1e}); unused W2/H2 gradient exactly {dead} ... OK")


def check_warm_start():
    """Weighted HadNet warm-started from plain HadNet's factors never ends above
    its starting weighted loss (best-seen tracking), here with lr 0.1 on the worst-conditioned G."""
    n, r = 48, 4
    A = torch.rand(n, n, dtype=torch.float64)
    G = make_G("freq1000", n)
    plain = hadnet_fit(A, r, max_epochs=300)[:4]
    start = weighted_error(A - reconstruct(*plain), G).item()
    *fit, hist = hadnet_fit(A, r, max_epochs=300, G=G, init=plain)
    end = weighted_error(A - reconstruct(*fit), G).item()
    assert abs(hist[0] - start) < 1e-12 and end <= start and abs(end - min(hist)) < 1e-12
    print(f"14. Warm start: weighted loss {start:.5f} -> {end:.5f} (never above start) ... OK")


def check_whitened_eckart_young():
    """On a small case, whitened SVD's weighted error equals the Eckart-Young error
    of S^T A (sqrt of the discarded sigma_i^2), and no random rank-k matrix does better."""
    n, k = 16, 3
    A = torch.rand(n, n, dtype=torch.float64)
    G = make_G("freq1000", n)
    Aw, Bw = whitened_svd(A, k, G)
    err = weighted_error(A - Aw @ Bw, G).item()
    S = torch.linalg.cholesky(G)                                         # a different S with S S^T = G
    sigma = torch.linalg.svdvals(S.T @ A)
    ey = torch.sqrt(torch.sum(sigma[k:] ** 2)).item()
    others = min(weighted_error(A - (Aw + 0.01 * torch.randn(n, k, dtype=torch.float64))
                                @ (Bw + 0.01 * torch.randn(k, n, dtype=torch.float64)), G).item()
                 for _ in range(200))
    assert abs(err - ey) < 1e-10 and others > err
    print(f"15. Whitened SVD error {err:.8f} = Eckart-Young of S^T A {ey:.8f}; 200 nearby rank-{k} "
          f"matrices all worse (best {others:.6f}) ... OK")


def check_report_numbers():
    """Every number in the appendix table of docs/REPORT.md equals its CSV cell, to the digits shown.

    Each appendix row names a file, a row (key=value pairs) and a column. The
    value printed in the report must equal the file's value formatted with the
    same number of decimals.
    """
    import csv
    import re
    from src.paths import ROOT
    text = (ROOT / "docs" / "REPORT.md").read_text()
    appendix = text[text.index("## Appendix"):]
    rows = re.findall(r"^\| (\d+) \| ([-0-9.]+) \| (results/\S+) \| (.+?) \| (\w+) \|$", appendix, re.M)
    assert len(rows) >= 20, len(rows)
    for number, value, path, where, column in rows:
        keys = dict(pair.split("=") for pair in where.split("; "))
        with open(ROOT / path, newline="") as f:
            matches = [row for row in csv.DictReader(f) if all(row[k] == v for k, v in keys.items())]
        assert matches, (number, path, where)
        decimals = len(value.split(".")[1]) if "." in value else 0
        got = f"{float(matches[0][column]):.{decimals}f}"
        assert got == value, (number, value, got)
    print(f"16. Report appendix: {len(rows)} numbers re-read from results/ ... OK")


if __name__ == "__main__":
    check_khatri_rao()
    check_rank_bound()
    check_svd_init_exact()
    check_eckart_young()
    check_param_count()
    check_spectrum_dependence()
    check_output_error()
    check_spearman()
    check_weighted()
    check_identity_weighting()
    check_diagonal_identities()
    check_weightings()
    check_embedding()
    check_warm_start()
    check_whitened_eckart_young()
    check_report_numbers()
    print("All checks passed.")
