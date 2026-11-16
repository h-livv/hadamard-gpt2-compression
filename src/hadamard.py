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
    """Activation-weighted error ||E X^T||_F for E (out, in) and G = X^T X (in, in).

    X is the (tokens, in) matrix of the layer's real inputs. In the (out, in)
    layout the layer's output error on those inputs is E X^T, and
        ||E X^T||_F^2 = trace(E X^T X E^T) = trace(E G E^T) = sum(E * (E @ G)).
    With G = I this is the plain Frobenius norm ||E||_F.

    Args:    E (out, in), G (in, in) symmetric positive semidefinite.
    Returns: scalar tensor (differentiable).
    Source:  the loss ||W X - W' X||_F of SVD-LLM (Wang et al., ICLR 2025) and
             ASVD (Yuan et al., arXiv:2312.05821), written with the Gram matrix.
    """
    return torch.sqrt(torch.sum(E * (E @ G)))


def hadnet_fit(X, r, max_epochs=2000, lr=0.1, patience=100, G=None):
    """Fit the Hadamard decomposition of X with HadNet (paper §4.1, Algorithm 7).

    HadNet has no layers: the four factors are the trainable parameters and
    Adam minimises the Frobenius norm ||X - X_pred||_F (the norm itself, not
    its square) on the whole matrix at once, so one "epoch" is one Adam step.
    Nothing here is random: the start is `svd_init` and there are no batches.

    Defaults are the paper's (§5): lr 0.1, 2000 epochs, patience ("wait") 100.

    Args:    X (m, n), r with 1 <= r <= min(m, n).
             G (n, n) optional. If given, the loss is the activation-weighted
             error weighted_error(X - X_pred, G) instead of ||X - X_pred||_F
             (X is then a weight in the (out, in) layout and G the Gram matrix
             of the layer's inputs). Everything else is unchanged.
    Returns: W1 (m, r), H1 (r, n), W2 (m, r), H2 (r, n), hist
             The factors are the best ones seen (lowest loss).
             hist is a list of max_epochs floats, the loss at each epoch;
             hist[0] is the loss of the SVD initialisation and min(hist) is
             the loss of the returned factors.
    Source:  paper Algorithm 7; repo function `HadamardNet_train`.
    Differs: Algorithm 7 and the repo return the factors after the LAST
             epoch. We return the BEST ones (project requirement). See
             VERIFICATION.md.
    """
    W1, H1, W2, H2 = svd_init(X, r)                          # (m, r), (r, n), (m, r), (r, n)
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
    """Rank-k approximation minimising the activation-weighted error (SVD-LLM's truncation-aware whitening).

    Write G = S S^T. Then weighted_error(E, G) = ||E S||_F, so the best rank-k
    X_hat in that error is X_hat = [X S]_k S^{-1}, where [.]_k is the truncated
    SVD (Eckart-Young applied to X S). S comes from an eigendecomposition
    G = Q diag(lam) Q^T, S = Q diag(sqrt(lam)), which, unlike Cholesky, does
    not fail when G is nearly singular.

    Args:    X (out, in), k, G (in, in) symmetric positive definite (damped by the caller).
    Returns: A (out, k), B (k, in) with X_hat = A @ B; k(out + in) numbers, as svd_truncate.
    Source:  SVD-LLM (Wang, Zheng, Wan, Zhang, ICLR 2025), §3.1 "truncation-aware
             data whitening": SVD of W S, truncate, multiply by S^{-1}.
             https://github.com/AIoT-MLSys-Lab/SVD-LLM, SVDLLM.py `whitening`.
    Differs: SVD-LLM uses a Cholesky factor of X X^T (adding to the diagonal if
             it fails); we use the symmetric eigendecomposition. Any S with
             S S^T = G gives the same X_hat.
    """
    lam, Q = torch.linalg.eigh(G)                            # (in,), (in, in)
    S = Q * torch.sqrt(lam)                                  # (in, in), S S^T = G
    S_inv = (Q / torch.sqrt(lam)).T                          # (in, in), S^{-1} = diag(1/sqrt(lam)) Q^T
    A, B = svd_truncate(X @ S, k)                            # (out, k), (k, in)
    return A, B @ S_inv
