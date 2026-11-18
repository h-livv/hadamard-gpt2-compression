"""
Hadamard decomposition of a single matrix, and the truncated-SVD baseline.

Paper ("the paper" everywhere below):
    Panchal, Sahoo, Behera, "Tensor decomposition and neural architectures
    through generalized Hadamard products", Neurocomputing 699 (2026) 134372.
Paper's code ("the repo"):
    https://github.com/PanchalAneesh/HadNet-HadDNN  (file random-tensor/main.py)

The paper works with tensors (stacks of matrices) and treats every matrix in
the stack independently. We only ever need ONE matrix, so every function here
is the paper's method with the loop over slices removed.

Notation (the paper's):  X is the (m, n) matrix to approximate, r is the rank,
    W1, W2 : (m, r)        H1, H2 : (r, n)
    M1 = W1 @ H1           M2 = W2 @ H2           X_pred = M1 * M2
where @ is matrix multiplication and * is the entrywise (Hadamard) product.
The four factors hold 2r(m+n) numbers in total.
"""

import torch


def svd_init(X, r):
    """Starting guess for the four Hadamard factors (paper §3.3).

    Idea: every real number x satisfies x = sqrt|x| * (sqrt|x| * sign(x)).
    Applied entrywise, X = P * Q exactly, with P = sqrt|X| and
    Q = sqrt|X| * sign(X). P and Q are full rank, so we cut each one down to
    rank r with a truncated SVD and split sqrt(S) evenly between the factors.

    Args:    X (m, n), r with 1 <= r <= min(m, n).
    Returns: W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n).
    Source:  paper §3.3 (mat_simple, mat_signed and the W-tilde, H-tilde
             formulas); repo function `svd_initialize`.
    """
    P = torch.sqrt(torch.abs(X))                             # (m, n)  paper's mat_simple
    Q = P * torch.sign(X)                                    # (m, n)  paper's mat_signed

    U, S, Vh = torch.linalg.svd(P, full_matrices=False)      # (m, p), (p,), (p, n) with p = min(m, n)
    W1 = U[:, :r] * torch.sqrt(S[:r])                        # (m, r)  = U_r sqrt(S_r)
    H1 = torch.sqrt(S[:r])[:, None] * Vh[:r, :]              # (r, n)  = sqrt(S_r) V_r^T

    U, S, Vh = torch.linalg.svd(Q, full_matrices=False)      # (m, p), (p,), (p, n)
    W2 = U[:, :r] * torch.sqrt(S[:r])                        # (m, r)
    H2 = torch.sqrt(S[:r])[:, None] * Vh[:r, :]              # (r, n)
    return W1, H1, W2, H2


def reconstruct(W1, H1, W2, H2):
    """Dense matrix represented by the four Hadamard factors.

    Args:    W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n).
    Returns: X_pred (m, n) = (W1 @ H1) * (W2 @ H2).
    Source:  paper §4.1 forward pass (M1, M2, X_pred); also Eq. (5).
    """
    M1 = W1 @ H1                                             # (m, n), rank <= r
    M2 = W2 @ H2                                             # (m, n), rank <= r
    X_pred = M1 * M2                                         # (m, n), rank <= r^2 (paper Prop. 3.3)
    return X_pred


def weighted_error(E, G):
    """Weighted error ||S^T E||_F = sqrt(trace(E^T G E)) for E (m, n) and G = S S^T (m, m).

    G acts on the ROW index of E (docs/REPORT.md, Background). In GPT-2's
    layout W (in, out) with layer inputs X (tokens, in), G = X^T X and
        ||X E||_F^2 = trace(E^T X^T X E) = trace(E^T G E) = sum(E * (G @ E)),
    i.e. the error of the layer's output on those inputs. With G = I this is
    the plain Frobenius norm ||E||_F.

    Args:    E (m, n), G (m, m) symmetric positive semidefinite.
    Returns: scalar tensor (differentiable).
    Source:  the loss ||W X - W' X||_F of SVD-LLM (Wang et al., ICLR 2025,
             arXiv:2403.07378, Eq. 1), written with the Gram matrix; the trace
             identity is SVD-LLM's Lemma 3.1 (Eq. 2). Checked in tests/checks.py.
    Differs: stages 3 and 4 were run with G on the column index of an (out, in)
             matrix. Same quantity for the transposed matrix; the row form lets
             images and GPT-2's native (in, out) layout be used without transposes.
    """
    return torch.sqrt(torch.sum(E * (G @ E)))


def hadnet_fit(X, r, max_epochs=2000, lr=0.1, patience=100, G=None, init=None):
    """Fit the Hadamard decomposition of X with HadNet (paper §4.1, Algorithm 7).

    HadNet has no layers: the four factors are the trainable parameters and
    Adam minimises the Frobenius norm ||X - X_pred||_F (the norm itself, not
    its square; paper Eq. 6) on the whole matrix at once, so one "epoch" is
    one Adam step. Nothing here is random: the start is `svd_init` (or `init`)
    and there are no batches.

    Defaults are the paper's (§5): lr 0.1, 2000 epochs, patience ("wait") 100.

    Args:    X (m, n), r with 1 <= r <= min(m, n).
             G (m, m) optional. If given, the loss is the weighted error
             weighted_error(X - X_pred, G) = ||S^T (X - X_pred)||_F instead of
             ||X - X_pred||_F. Everything else is unchanged. (This project's
             extension, not in the paper.)
             init optional (W1, H1, W2, H2) to start from instead of
             svd_init(X, r); copied, never modified. (This project's extension,
             used for the warm-start and embedding starts of REPORT stage 5.3.)
    Returns: W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n), hist
             The factors are the best ones seen (lowest loss).
             hist is a list of max_epochs floats, the loss at each epoch;
             hist[0] is the loss of the starting factors and min(hist) is
             the loss of the returned factors.
    Source:  paper Algorithm 7; repo random-tensor/main.py `HadamardNet_train`
             (lines 89-125 at commit 4aabe3d).
    Differs: Algorithm 7 (line 21) and the repo return the factors after the
             LAST epoch. We return the BEST ones (project requirement). The
             repo's extra `updatecount == 100` test (line 113) is dead code.
    """
    if init is None:
        W1, H1, W2, H2 = svd_init(X, r)                      # (m, r), (r, n), (m, r), (r, n)
    else:
        W1, H1, W2, H2 = [f.detach().clone() for f in init]
    params = [W1, H1, W2, H2]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)                    # Algorithm 7, line 3

    wait = patience                                          # Algorithm 7, line 4
    best_err = float("inf")
    best_params = None
    hist = []

    for epoch in range(max_epochs):                          # Algorithm 7, line 5
        opt.zero_grad()
        X_pred = reconstruct(W1, H1, W2, H2)                 # (m, n)   lines 6-8
        if G is None:
            loss = torch.linalg.norm(X - X_pred, "fro")      # scalar   line 9
        else:
            loss = weighted_error(X - X_pred, G)             # scalar   line 9, activation-weighted
        loss.backward()                                      # line 10 (backpropagate)
        hist.append(loss.item())

        if loss.item() < best_err:                           # lines 11-12
            best_err = loss.item()
            wait = patience
            # `loss` was computed from the factors as they are NOW, before
            # the Adam step below changes them, so this is the moment to
            # copy them if we want "the factors that achieved best_err".
            best_params = [p.detach().clone() for p in params]
        else:                                                # lines 13-14
            wait = wait - 1

        opt.step()                                           # line 10 (ADAM step)

        if wait == 0:                                        # lines 16-18
            for group in opt.param_groups:
                group["lr"] = 0.1 * group["lr"]
            wait = patience

    W1, H1, W2, H2 = best_params                             # (m, r), (r, n), (m, r), (r, n)
    return W1, H1, W2, H2, hist


def svd_truncate(X, k):
    """Best rank-k approximation of X, as two thin factors.

    By the Eckart-Young theorem no rank-k matrix has a smaller Frobenius
    error, and that error is sqrt(sum of the discarded sigma_i^2).

    Args:    X (m, n), k with 1 <= k <= min(m, n).
    Returns: A (m, k) = U_k S_k   and   B (k, n) = V_k^T.
             The approximation is A @ B; it is stored in k(m+n) numbers.
    Source:  torch.linalg.svd, https://pytorch.org/docs/stable/generated/torch.linalg.svd.html
             repo function `svd_transform`; paper Theorem 3.1 for one slice.
    Differs: the repo returns the dense product (U_r * S_r) @ Vh_r. We return
             the two halves so that the parameter count can be read off.
    """
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)      # (m, p), (p,), (p, n) with p = min(m, n)
    A = U[:, :k] * S[:k]                                     # (m, k)  singular values folded into U
    B = Vh[:k, :]                                            # (k, n)
    return A, B


def whitened_svd(X, k, G):
    """Rank-k approximation minimising the weighted error ||S^T (X - X_hat)||_F (SVD-LLM's whitening).

    Write G = S S^T. Then weighted_error(E, G) = ||S^T E||_F, and S^T is
    invertible, so X_hat ranges over all rank-k matrices exactly when S^T X_hat
    does. The best one is therefore X_hat = (S^T)^{-1} [S^T X]_k, with [.]_k
    the truncated SVD (Eckart-Young applied to S^T X), and its weighted error
    is sqrt(sum of the discarded sigma_i^2 of S^T X). S comes from an
    eigendecomposition G = Q diag(lam) Q^T, S = Q diag(sqrt(lam)), so that
    S^T = diag(sqrt(lam)) Q^T and (S^T)^{-1} = Q diag(1/sqrt(lam)).

    Args:    X (m, n), k, G (m, m) symmetric positive definite.
    Returns: A (m, k), B (k, n) with X_hat = A @ B; k(m + n) numbers, as svd_truncate.
    Source:  SVD-LLM (Wang, Zheng, Wan, Zhang, ICLR 2025, arXiv:2403.07378),
             §3.1 "truncation-aware data whitening", Theorem 3.2 and
             Corollary 3.3 (Eq. 6: loss^2 = sum of truncated sigma_i^2);
             code SVDLLM.py `whitening`, lines 219-224 at commit 7538cca:
             SVD of W S, truncate, multiply by S^{-1}.
    Differs: SVD-LLM's W is (out, in) with S on the right; ours has G on the
             row index, so S^T multiplies from the left (the transpose of the
             same construction). SVD-LLM uses a Cholesky factor of X X^T; we use
             the symmetric eigendecomposition. Any S with S S^T = G gives the
             same X_hat (when sigma_k > sigma_{k+1}).
    """
    lam, Q = torch.linalg.eigh(G)                            # (m,), (m, m)
    St = torch.sqrt(lam)[:, None] * Q.T                      # (m, m), S^T = diag(sqrt(lam)) Q^T
    St_inv = Q / torch.sqrt(lam)                             # (m, m), (S^T)^{-1} = Q diag(1/sqrt(lam))
    A, B = svd_truncate(St @ X, k)                           # (m, k), (k, n)
    return St_inv @ A, B


def embed_rank_r(A, B):
    """Hadamard factors that reproduce a given rank-r matrix A @ B exactly.

    W1 H1 = A @ B and W2 H2 = all-ones, so (W1 H1) * (W2 H2) = A @ B.
    The all-ones matrix is rank 1: W2 = [1, 0, ..., 0] (column of ones, then
    zeros), H2 = the matching row of ones above zero rows. Singular values
    are split evenly between W1 and H1 (as svd_init does).

    Caution: the zero columns of W2 and zero rows of H2 receive exactly zero
    gradient (dL/dW2[:, k] = (dL/dM2) H2[k, :]^T = 0 and vice versa), and
    Adam turns a zero gradient into a zero step, so gradient descent from this
    start can never use them: W2 H2 stays rank 1 and the fit stays rank r.

    Args:    A (m, r), B (r, n).
    Returns: W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n).
    Source:  this project (REPORT stage 5.3); the rank-1 all-ones factor is the
             identity X * 1 = X of the Hadamard product.
    """
    U, s, Vh = torch.linalg.svd(A @ B, full_matrices=False)  # balanced split of A @ B
    r = A.shape[1]
    W1 = U[:, :r] * torch.sqrt(s[:r])                        # (m, r)
    H1 = torch.sqrt(s[:r])[:, None] * Vh[:r, :]              # (r, n)
    W2 = torch.zeros_like(W1)
    H2 = torch.zeros_like(H1)
    W2[:, 0] = 1.0
    H2[0, :] = 1.0
    return W1, H1, W2, H2
