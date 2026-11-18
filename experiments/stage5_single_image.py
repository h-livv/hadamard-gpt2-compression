"""
Stage 5, single matrices (image part): docs/REPORT.md, stage 5.1-5.4.

What it does
    The image is skimage.data.astronaut() in grayscale, values in [0, 1],
    downsampled with anti-aliasing to n = 64, 128, 256, 512. The control is a
    256 x 256 standard Gaussian matrix (seed 42). Hadamard rank r is set by the
    storage share 4r/n in {6.25, 12.5, 25, 50}%, i.e. r = n/64, n/32, n/16, n/8.
    Every fit is float64 on the CPU.

      5.1  plain error: SVD(2r), HadNet(r), SVD(r) (paper's equal-rank reference),
           plus the Prop 3.5 bound and the actual gain over SVD(2r).
      5.2  each weighting (src/weights.py): SVD(2r), HadNet(r), whitened SVD(2r),
           weighted HadNet(r) at the paper's settings.
      5.3  n = 256, region and freq{10,100,1000}: weighted HadNet from three starts
           (paper; warm start from plain HadNet; rank-r whitened SVD embedded)
           x three learning rates (0.1, 0.03, 0.01). Then (optional part) the
           best two of the nine configurations repeated at n = 512.
      5.4  region weighting (diagonal G = D^2): weighted HadNet on (A, D^2) against
           plain HadNet on DA mapped back (W1 <- D^-1 W1).

Reads     nothing (the image ships with scikit-image).
Writes    results/stage5/conditions.csv       condition number of every G
          results/stage5/spectra.npz          singular values of every matrix
          results/stage5/s51_spectrum.csv     5.1 metrics, bound and gain
          results/stage5/s52_weightings.csv   5.2 metrics
          results/stage5/s53_fix.csv          5.3 metrics (n = 256 and the n = 512 repeat)
          results/stage5/s54_diagonal.csv     5.4 metrics and the two identities' gaps
          results/stage5/curves_s5x.npz       every HadNet loss curve, key = run_id column
          results/stage5/reconstructions.npz  n = 256, 12.5% share, for figure stage5/fig6
Runtime   about 27 minutes on 8 CPU threads (laptop).

Run:  python experiments/stage5_single_image.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from skimage import color, data, transform

from src.hadamard import embed_rank_r, hadnet_fit, reconstruct, svd_truncate, whitened_svd
from src.metrics import condition_number, psnr, relative_error, relative_weighted_error
from src.paths import RESULTS_DIR, write_csv

OUT = RESULTS_DIR / "stage5"
from src.weights import WEIGHTINGS, make_G

# ---- settings -------------------------------------------------------------
SEED = 42                               # paper §5
THREADS = 8                             # fixed so that float64 BLAS sums are reproducible
SIZES = [64, 128, 256, 512]
SHARES = [6.25, 12.5, 25.0, 50.0]       # percent of n*n stored; r = share * n / 400
CONTROL_N = 256
FIX_N = 256                             # 5.3
FIX_WEIGHTINGS = ["region", "freq10", "freq100", "freq1000"]
FIX_INITS = ["paper", "warm", "embed"]
FIX_LRS = [0.1, 0.03, 0.01]
REPEAT_N = 512                          # optional 5.3 repeat
RECON_N, RECON_SHARE = 256, 12.5        # figure 6
# ---------------------------------------------------------------------------


def load_image(n):
    """astronaut() in grayscale on [0, 1], resized to (n, n) with anti-aliasing, float64 tensor."""
    gray = color.rgb2gray(data.astronaut())                          # (512, 512) float64 in [0, 1]
    if n != gray.shape[0]:
        gray = transform.resize(gray, (n, n), anti_aliasing=True)    # Gaussian pre-filter, bilinear
    assert gray.min() >= 0 and gray.max() <= 1
    return torch.from_numpy(np.ascontiguousarray(gray))


def load_control(n):
    """Standard Gaussian (n, n) matrix, seed SEED: a flat-spectrum control."""
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(n, n, generator=g, dtype=torch.float64)


def rank_for(n, share):
    """Hadamard rank r with 2r(n+n)/(n*n) = share percent."""
    r = share * n / 400
    assert r == int(r)
    return int(r)


def measure(A, A_hat, G, is_image):
    """The metrics of every fit: relative plain and weighted error, PSNR (image only)."""
    return dict(rel_err=relative_error(A, A_hat),
                rel_werr=relative_weighted_error(A, A_hat, G),
                psnr=psnr(A, A_hat) if is_image else float("nan"))


def timed(f, *args, **kwargs):
    """(result, seconds) of f(*args, **kwargs)."""
    start = time.perf_counter()
    out = f(*args, **kwargs)
    return out, time.perf_counter() - start


def fit_svd(A, k):
    (Ak, Bk), sec = timed(svd_truncate, A, k)
    return Ak @ Bk, sec


def fit_wsvd(A, k, G):
    (Ak, Bk), sec = timed(whitened_svd, A, k, G)
    return Ak @ Bk, sec


def fit_hadnet(A, r, G=None, lr=0.1, init=None):
    """HadNet; returns (factors, A_hat, seconds, hist)."""
    (W1, H1, W2, H2, hist), sec = timed(hadnet_fit, A, r, lr=lr, G=G, init=init)
    factors = (W1, H1, W2, H2)
    return factors, reconstruct(*factors).detach(), sec, hist


def curve_fields(run_id, hist, curves):
    """Store a loss curve and return the CSV fields that describe it."""
    curves[run_id] = np.asarray(hist)
    return dict(run_id=run_id, steps=len(hist), best_step=int(np.argmin(hist)),
                start_loss=hist[0], best_loss=min(hist))


# ---- 5.1 ------------------------------------------------------------------

def section_51(matrices, plain_cache, curves):
    """Plain error: SVD(2r), HadNet(r), SVD(r); Prop 3.5 bound against actual gain."""
    rows = []
    for (kind, n), A in matrices.items():
        sigma = torch.linalg.svdvals(A)                                       # (n,) descending
        norm2 = torch.sum(A ** 2).item()
        I = torch.eye(n, dtype=torch.float64)
        for share in SHARES:
            r = rank_for(n, share)
            base = dict(matrix=kind, n=n, share=share, r=r)
            A2r, sec2 = fit_svd(A, 2 * r)
            Ar, sec1 = fit_svd(A, r)
            factors, A_had, sec_h, hist = fit_hadnet(A, r)
            plain_cache[(kind, n, share)] = (factors, A_had, sec_h)
            # Prop 3.5: ||A - A_2r||^2 - ||A - A_hat||^2 <= sum_{i=2r+1}^{min(r^2, n)} sigma_i^2
            bound = torch.sum(sigma[2 * r:min(r * r, n)] ** 2).item()          # empty (0) if r^2 <= 2r
            gain = torch.sum((A - A2r) ** 2).item() - torch.sum((A - A_had) ** 2).item()
            extra = dict(bound=bound, gain=gain, bound_rel=bound / norm2, gain_rel=gain / norm2,
                         tail_r2=torch.sum(sigma[min(r * r, n):] ** 2).item() / norm2)
            for method, rank, A_hat, sec in [("svd_2r", 2 * r, A2r, sec2), ("hadnet", r, A_had, sec_h),
                                             ("svd_r", r, Ar, sec1)]:
                row = dict(**base, method=method, rank=rank, **measure(A, A_hat, I, kind == "image"),
                           seconds=sec, run_id="", steps="", best_step="", start_loss="", best_loss="")
                if method == "hadnet":
                    row.update(curve_fields(f"s51_{kind}_n{n}_r{r}", hist, curves))
                rows.append(dict(**row, **extra))
            print(f"5.1 {kind:7s} n={n:3d} r={r:2d}  SVD(2r) {rows[-3]['rel_err']:.4f}  "
                  f"HadNet {rows[-2]['rel_err']:.4f}  SVD(r) {rows[-1]['rel_err']:.4f}  "
                  f"bound {bound / norm2:.2e} gain {gain / norm2:+.2e}", flush=True)
    return rows


# ---- 5.2 ------------------------------------------------------------------

def section_52(images, plain_cache, weighted_cache, curves, recon):
    """The four methods under each weighting; weighted HadNet at the paper's settings."""
    rows = []
    for n, A in images.items():
        for share in SHARES:
            r = rank_for(n, share)
            A2r, sec2 = fit_svd(A, 2 * r)
            _, A_had, sec_h = plain_cache[("image", n, share)]
            for wname in WEIGHTINGS:
                G = make_G(wname, n)
                A_w, sec_w = fit_wsvd(A, 2 * r, G)
                factors, A_wh, sec_wh, hist = fit_hadnet(A, r, G=G)
                weighted_cache[(n, share, wname)] = (factors, A_wh)
                base = dict(n=n, share=share, r=r, weighting=wname)
                for method, A_hat, sec in [("svd_2r", A2r, sec2), ("hadnet", A_had, sec_h),
                                           ("wsvd_2r", A_w, sec_w), ("whadnet", A_wh, sec_wh)]:
                    row = dict(**base, method=method, **measure(A, A_hat, G, True), seconds=sec,
                               run_id="", steps="", best_step="", start_loss="", best_loss="")
                    if method == "whadnet":
                        row.update(curve_fields(f"s52_n{n}_r{r}_{wname}", hist, curves))
                    rows.append(row)
                    if n == RECON_N and share == RECON_SHARE:
                        recon[f"{wname}_{method}"] = A_hat.numpy()
                wsvd, had, wh = rows[-2]["rel_werr"], rows[-3]["rel_werr"], rows[-1]["rel_werr"]
                print(f"5.2 n={n:3d} r={r:2d} {wname:9s}  wSVD {wsvd:.4f}  wHadNet {wh:.4f}  "
                      f"gap {(wh - wsvd) / wsvd:+.3f}  HadNet {had:.4f}"
                      f"{'  <-- worse than plain HadNet' if wh > had else ''}", flush=True)
    return rows


# ---- 5.3 ------------------------------------------------------------------

def start_factors(init, A, r, G, plain_factors):
    """Starting factors for weighted HadNet; None means the paper's svd_init(A, r)."""
    if init == "paper":
        return None
    if init == "warm":
        return plain_factors
    return embed_rank_r(*whitened_svd(A, r, G))                              # rank-r whitened SVD


def fix_runs(A, n, shares, weightings, configs, plain_cache, curves, tag):
    """Weighted HadNet for every (share, weighting, (init, lr)) in the given lists."""
    rows = []
    for share in shares:
        r = rank_for(n, share)
        plain_factors, A_had, _ = plain_cache[("image", n, share)]
        for wname in weightings:
            G = make_G(wname, n)
            norm_w = torch.sqrt(torch.sum(A * (G @ A))).item()                 # ||S^T A||_F
            ref_wsvd = relative_weighted_error(A, fit_wsvd(A, 2 * r, G)[0], G)
            ref_wsvd_r = relative_weighted_error(A, fit_wsvd(A, r, G)[0], G)
            ref_had = relative_weighted_error(A, A_had, G)
            for init, lr in configs:
                start = start_factors(init, A, r, G, plain_factors)
                factors, A_hat, sec, hist = fit_hadnet(A, r, G=G, lr=lr, init=start)
                row = dict(n=n, share=share, r=r, weighting=wname, init=init, lr=lr,
                           **measure(A, A_hat, G, True), seconds=sec,
                           ref_wsvd_2r=ref_wsvd, ref_wsvd_r=ref_wsvd_r, ref_hadnet=ref_had,
                           **curve_fields(f"{tag}_n{n}_r{r}_{wname}_{init}_lr{lr}", hist, curves))
                row["start_rel_werr"] = hist[0] / norm_w                    # hist[0]: loss of the start
                rows.append(row)
                print(f"{tag} n={n} r={r:2d} {wname:9s} {init:12s} lr={lr:<5}  "
                      f"{row['rel_werr']:.4f}  (wSVD(2r) {ref_wsvd:.4f}, HadNet {ref_had:.4f})", flush=True)
    return rows


def best_two(rows):
    """The two (init, lr) configurations with the lowest mean rel_werr / ref_wsvd_2r."""
    scores = {}
    for row in rows:
        scores.setdefault((row["init"], row["lr"]), []).append(row["rel_werr"] / row["ref_wsvd_2r"])
    ranked = sorted(scores, key=lambda c: np.mean(scores[c]))
    for c in ranked:
        print(f"5.3 mean ratio to wSVD(2r)  {c[0]:12s} lr={c[1]:<5} {np.mean(scores[c]):.4f}")
    return ranked[:2]


# ---- 5.4 ------------------------------------------------------------------

def section_54(images, weighted_cache, curves):
    """Region weighting G = D^2: weighted HadNet on (A, G) vs plain HadNet on DA mapped back."""
    rows = []
    for n, A in images.items():
        G = make_G("region", n)
        d = torch.sqrt(torch.diagonal(G))                                     # (n,) G = D^2
        D, D_inv = torch.diag(d), torch.diag(1 / d)
        for share in SHARES:
            r = rank_for(n, share)
            _, A_wh = weighted_cache[(n, share, "region")]
            (W1, H1, W2, H2), _, sec, hist = fit_hadnet(D @ A, r)              # plain fit of DA
            A_fold = reconstruct(D_inv @ W1, H1, W2, H2).detach()             # W1 <- D^-1 W1
            # Identity (a): the folded factors' weighted error on A = plain error on DA.
            werr_fold = torch.linalg.norm(D @ (A - A_fold)).item()
            plain_DA = torch.linalg.norm(D @ A - reconstruct(W1, H1, W2, H2)).item()
            # Identity (b): whitened SVD with D^2 = D^-1 [DA]_2r.
            A_w = fit_wsvd(A, 2 * r, G)[0]
            As, Bs = svd_truncate(D @ A, 2 * r)
            rows.append(dict(
                n=n, share=share, r=r,
                rel_werr_whadnet=relative_weighted_error(A, A_wh, G),
                rel_werr_fold=relative_weighted_error(A, A_fold, G),
                rel_err_whadnet=relative_error(A, A_wh), rel_err_fold=relative_error(A, A_fold),
                identity_a_rel_gap=abs(werr_fold - plain_DA) / plain_DA,
                identity_b_max_gap=torch.max(torch.abs(A_w - D_inv @ As @ Bs)).item(),
                seconds_fold=sec, **curve_fields(f"s54_n{n}_r{r}_fold", hist, curves)))
            row = rows[-1]
            print(f"5.4 n={n:3d} r={r:2d}  weighted HadNet {row['rel_werr_whadnet']:.5f}  "
                  f"plain on DA {row['rel_werr_fold']:.5f}  identity gaps {row['identity_a_rel_gap']:.1e} "
                  f"{row['identity_b_max_gap']:.1e}", flush=True)
    return rows


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    images = {n: load_image(n) for n in SIZES}
    matrices = {("image", n): A for n, A in images.items()}
    matrices[("control", CONTROL_N)] = load_control(CONTROL_N)

    np.savez(OUT / "spectra.npz",
             **{f"{kind}_{n}": torch.linalg.svdvals(A).numpy() for (kind, n), A in matrices.items()})
    write_csv([dict(n=n, weighting=w, cond=condition_number(make_G(w, n)))
              for n in SIZES for w in WEIGHTINGS], OUT / "conditions.csv")

    plain_cache, weighted_cache, curves, recon = {}, {}, {}, {}
    write_csv(section_51(matrices, plain_cache, curves), OUT / "s51_spectrum.csv")
    write_csv(section_52(images, plain_cache, weighted_cache, curves, recon), OUT / "s52_weightings.csv")
    recon["original"] = images[RECON_N].numpy()
    np.savez(OUT / "reconstructions.npz", **recon)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    configs = [(init, lr) for init in FIX_INITS for lr in FIX_LRS]
    rows53 = fix_runs(images[FIX_N], FIX_N, SHARES, FIX_WEIGHTINGS, configs, plain_cache, curves, "s53")
    write_csv(rows53, OUT / "s53_fix.csv")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    # Optional part of 5.3: the best two of the nine configurations at n = 512.
    rows53 += fix_runs(images[REPEAT_N], REPEAT_N, SHARES, FIX_WEIGHTINGS, best_two(rows53),
                       plain_cache, curves, "s53")
    write_csv(rows53, OUT / "s53_fix.csv")

    write_csv(section_54(images, weighted_cache, curves), OUT / "s54_diagonal.csv")
    np.savez_compressed(OUT / "curves_s5x.npz", **curves)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
