"""
GPT-2 single matrix, docs/REPORT.md §5.5 (task §6): block 5's c_proj at Hadamard rank r = 128.

What it does
    Uses v1's calibration and damping exactly: G = X^T X of the layer's inputs on
    64 x 512 WikiText-2 train tokens (seed 42) from the original model, and
    G_fit = (G + 0.01 s I) / s with s = mean(diag G) (src/gpt2.py `damped`).
    The matrix is used in GPT-2's own (in, out) layout, A = W (3072, 768), so that
    G (3072, 3072) acts on the row index, as everywhere in v2. Float32 on the GPU,
    as v1 (whitened SVD in float64, as v1).

      1. SVD(2r), HadNet(r), whitened SVD(2r), weighted HadNet(r) at the paper's
         settings, compared with v1's per-matrix numbers in
         archive/v1-gpt2/results/activation_aware.csv.
      2. Weighted HadNet from the starts of REPORT §5.3 (paper, warm, embed)
         x learning rates 0.1, 0.03, 0.01.
      3. Weighted error (under the raw G, v1's "output error", and under G_fit,
         the fitted loss) and plain error of every run, plus loss curves.
      4. Condition numbers of G and G_fit.

Reads     GPT-2 small and WikiText-2 from the Hugging Face Hub (cached after first use);
          archive/v1-gpt2/results/activation_aware.csv.
Writes    results/s6_gpt2.csv, results/s6_conditions.csv, results/curves_s6.npz
Runtime   about 3.5 minutes on an RTX 4050 laptop GPU (calibration ~1 min).

Run:  python experiments/gpt2_single_matrix.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import DEVICE, SEED, collect_gram_matrices, damped, get_weight, load_calibration_tokens
from src.hadamard import embed_rank_r, hadnet_fit, reconstruct, svd_truncate, whitened_svd
from src.metrics import condition_number, relative_error, relative_weighted_error
from src.paths import RESULTS_DIR, ROOT

# ---- settings -------------------------------------------------------------
LAYER, NAME, R = 5, "c_proj", 128
INITS = ["paper", "warm", "embed"]
LRS = [0.1, 0.03, 0.01]
V1_CSV = ROOT / "archive/v1-gpt2/results/activation_aware.csv"
# ---------------------------------------------------------------------------


def v1_numbers():
    """v1's (weight_error, output_error) for this matrix at r = 128, by method."""
    with open(V1_CSV, newline="") as f:
        return {row["method"]: (float(row["weight_error"]), float(row["output_error"]))
                for row in csv.DictReader(f)
                if int(row["r"]) == R and int(row["layer"]) == LAYER and row["matrix"] == NAME}


def save_csv(rows, name):
    with open(RESULTS_DIR / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    torch.manual_seed(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    G = collect_gram_matrices(model, load_calibration_tokens(tokenizer))[(LAYER, NAME)]   # (3072, 3072) float64
    G_fit = damped(G)                                                                     # float32
    A = get_weight(model, LAYER, NAME).T.contiguous()                                     # (in, out) = (3072, 768)
    del model

    conds = [dict(matrix="G", cond=condition_number(G),
                  lam_min=torch.linalg.eigvalsh(G)[0].item(), lam_max=torch.linalg.eigvalsh(G)[-1].item()),
             dict(matrix="G_fit", cond=condition_number(G_fit),
                  lam_min=torch.linalg.eigvalsh(G_fit.double())[0].item(),
                  lam_max=torch.linalg.eigvalsh(G_fit.double())[-1].item())]
    for c in conds:
        print(f"cond({c['matrix']}) = {c['cond']:.4g}", flush=True)
    save_csv(conds, "s6_conditions.csv")

    v1 = v1_numbers()
    rows, curves = [], {}

    def record(method, init, lr, A_hat, seconds, hist=None):
        row = dict(method=method, init=init, lr=lr, r=R,
                   rel_err=relative_error(A, A_hat),
                   rel_werr_G=relative_weighted_error(A, A_hat, G),          # v1's "output error"
                   rel_werr_Gfit=relative_weighted_error(A, A_hat, G_fit),   # the fitted loss
                   seconds=seconds, run_id="", steps="", best_step="", start_loss="", best_loss="",
                   v1_rel_err="", v1_rel_werr_G="")
        if hist is not None:
            run_id = f"s6_{method}_{init}_lr{lr}"
            curves[run_id] = np.asarray(hist)
            row.update(run_id=run_id, steps=len(hist), best_step=int(np.argmin(hist)),
                       start_loss=hist[0], best_loss=min(hist))
        rows.append(row)
        print(f"{method:8s} {init:12s} lr={lr:<5}  rel_err {row['rel_err']:.4f}  "
              f"rel_werr(G) {row['rel_werr_G']:.4f}  rel_werr(G_fit) {row['rel_werr_Gfit']:.4f}  "
              f"{seconds:.1f}s", flush=True)
        return row

    def timed(f, *args, **kwargs):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = f(*args, **kwargs)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - start

    # ---- 1. the four methods at the paper's settings (v1 reproduction) ----
    (Ak, Bk), sec = timed(svd_truncate, A, 2 * R)
    row = record("svd", "", "", Ak @ Bk, sec)
    row["v1_rel_err"], row["v1_rel_werr_G"] = v1["svd"]

    (W1, H1, W2, H2, hist), sec = timed(hadnet_fit, A, R)
    plain = (W1, H1, W2, H2)
    row = record("hadnet", "paper", 0.1, reconstruct(*plain).detach(), sec, hist)
    row["v1_rel_err"], row["v1_rel_werr_G"] = v1["hadamard"]

    (Ak, Bk), sec = timed(whitened_svd, A.double(), 2 * R, G_fit.double())   # float64, as v1
    row = record("wsvd", "", "", (Ak @ Bk).float(), sec)
    row["v1_rel_err"], row["v1_rel_werr_G"] = v1["svd_w"]

    # ---- 2. weighted HadNet: starts x learning rates ----
    Ar, Br = whitened_svd(A.double(), R, G_fit.double())                       # rank-r whitened SVD
    record("wsvd_r", "", "", (Ar @ Br).float(), 0.0)
    embed = [f.float() for f in embed_rank_r(Ar, Br)]
    starts = dict(paper=None, warm=plain, embed=embed)
    for init in INITS:
        for lr in LRS:
            (W1, H1, W2, H2, hist), sec = timed(hadnet_fit, A, R, lr=lr, G=G_fit, init=starts[init])
            row = record("whadnet", init, lr, reconstruct(W1, H1, W2, H2).detach(), sec, hist)
            if init == "paper" and lr == 0.1:
                row["v1_rel_err"], row["v1_rel_werr_G"] = v1["hadamard_w"]

    save_csv(rows, "s6_gpt2.csv")
    np.savez_compressed(RESULTS_DIR / "curves_s6.npz", **curves)
