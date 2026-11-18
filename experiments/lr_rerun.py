"""
v2.1: learning-rate re-run of v1's activation-aware GPT-2 experiment (docs/REPORT.md §10).

What it does
    The 20 MLP matrices of GPT-2 small's blocks 2-11 (blocks 0 and 1 stay whole,
    as in v1) are fitted at r = 64, 128, 256 by
        svd       truncated SVD, rank 2r                       (plain error)
        wsvd      whitened SVD, rank 2r (SVD-LLM)              (weighted error, exact optimum)
        hadnet    HadNet, rank r, lr in {0.1, 0.03, 0.01}      (plain error)
        whadnet   weighted HadNet, rank r, lr in {0.1, 0.03, 0.01}   (weighted error)
    HadNet fits use the paper's start (§3.3), schedule (lr x 0.1 after 100 steps
    without a new best) and 2000 steps, with best-seen tracking. A fit is
    "converged" if its best loss fell by less than 0.1% (relative) over the last
    200 steps; every HadNet fit that is not converged at 2000 steps is re-run from
    the same start for 5000 steps, and both results are kept. Trying lr 0.03 and
    0.01 is this project's choice, not the paper's.

    Calibration and damping are v1's: G = X^T X of each layer's inputs on 64 x 512
    WikiText-2 train tokens (seed 42) from the uncompressed model, and
    G_fit = (G + 0.01 s I) / s, s = mean(diag G). Metrics use GPT-2's (in, out)
    layout (G on the row index); the HadNet fits themselves run in v1's (out, in)
    layout with G on the column index, which reproduces v1 bit for bit (the
    transposed fit does not, REPORT §10.2). Float32 on the GPU (whitened SVD in
    float64), as v1.

    Order (no test-set tuning: every choice is made from calibration losses):
      1. v1 reproduction: wsvd and whadnet at lr 0.1, then test perplexity, compared
         with archive/v1-gpt2/results/activation_aware.csv. Stops on any mismatch.
      2. All other fits, then the 5000-step re-runs. No perplexity is looked at.
      3. Policies (one fit per matrix, chosen by fixed rules on calibration losses),
         then WikiText-2 test perplexity for each, then a check that the restored
         model scores 25.17 again.

Reads     GPT-2 small, WikiText-2 (Hugging Face Hub, cached after first use);
          archive/v1-gpt2/results/activation_aware.csv.
Writes    results/lr_rerun_fits.csv        one row per fit
          results/lr_rerun_perplexity.csv  one row per (policy, r)
          results/lr_rerun_curves.npz      every HadNet loss curve, key = fit_id column
          cache/lr_rerun/                  fitted factors (gitignored; lets an interrupted run resume)
Runtime   about 3 hours on an RTX 4050 laptop GPU (see docs/REPORT.md §9 for the measured time).

Run:  python experiments/lr_rerun.py
"""

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from src.gpt2 import (DEVICE, SEED, collect_gram_matrices, damped, get_weight, load_calibration_tokens,
                      load_test_tokens, perplexity, set_weight)
from src.hadamard import hadnet_fit, reconstruct, svd_truncate, weighted_error, whitened_svd
from src.metrics import relative_error, relative_weighted_error
from src.paths import RESULTS_DIR, ROOT

# ---- settings -------------------------------------------------------------
RANKS = [64, 128, 256]
LAYERS = range(2, 12)                    # blocks 0 and 1 stay uncompressed, as in v1
NAMES = ["c_fc", "c_proj"]
LRS = [0.1, 0.03, 0.01]                  # 0.1 is the paper's; 0.03 and 0.01 are this project's
STEPS, LONG_STEPS = 2000, 5000
WINDOW, TOLERANCE = 200, 1e-3            # converged: best loss fell < 0.1% over the last 200 steps
CACHE = ROOT / "cache" / "lr_rerun"
V1_CSV = ROOT / "archive/v1-gpt2/results/activation_aware.csv"
BASELINE = 25.17                         # uncompressed test perplexity (v1)
# Model size (v1's formula, archive/v1-gpt2/src/methods.py `model_params`)
M, N = 768, 3072
GPT2_TOTAL_PARAMS = 124_439_808
# ---------------------------------------------------------------------------


def model_megabytes(r, n_compressed=20):
    """Stored float32 size of GPT-2 small with n_compressed MLP matrices kept as 2r(m+n) numbers (v1)."""
    params = GPT2_TOTAL_PARAMS - n_compressed * M * N + n_compressed * 2 * r * (M + N)
    return params * 4 / 1e6


def converged(hist):
    """True if the best loss improved by less than TOLERANCE (relative) over the last WINDOW steps."""
    best = np.minimum.accumulate(np.asarray(hist))
    return bool((best[-WINDOW - 1] - best[-1]) / best[-WINDOW - 1] < TOLERANCE)


def fit_id(layer, name, r, method, lr="", steps=""):
    return f"h{layer}.{name}_r{r}_{method}" + (f"_lr{lr}_s{steps}" if method in ("hadnet", "whadnet") else "")


def dense(fit):
    """Dense (in, out) reconstruction of a cached fit, on DEVICE (factors are kept on the CPU to save GPU memory)."""
    f = [x.to(DEVICE) for x in fit["factors"]]
    A_hat = f[0] @ f[1] if len(f) == 2 else reconstruct(*f)
    return A_hat.T if fit.get("out_in") else A_hat                         # HadNet factors are (out, in)


def sync():
    """Wait for the GPU, so that fit times are real."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def fit_one(A, G, G_fit, layer, name, r, method, lr="", steps=""):
    """Fit (or load from CACHE) one matrix with one method; returns {'row', 'factors', 'hist'}."""
    key = fit_id(layer, name, r, method, lr, steps)
    path = CACHE / f"{key}.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    sync()
    start = time.perf_counter()
    hist = None
    if method == "svd":
        factors = svd_truncate(A, 2 * r)
    elif method == "wsvd":
        Aw, Bw = whitened_svd(A.double(), 2 * r, G_fit.double())          # float64, as v1
        factors = (Aw.float(), Bw.float())
    else:
        # Fitted in v1's (out, in) layout with G on the column index, exactly as v1 did: at lr 0.1
        # the optimiser path is sensitive to float32 operation order, and the transposed
        # (row-convention) fit does not reproduce v1's numbers (REPORT §10.2).
        *factors, hist = hadnet_fit(A.T.contiguous(), r, max_epochs=steps, lr=lr,
                                    G=G_fit if method == "whadnet" else None, G_side="col")
    sync()
    seconds = time.perf_counter() - start
    fit = dict(factors=tuple(f.detach().cpu() for f in factors), hist=hist,
               out_in=method in ("hadnet", "whadnet"))
    A_hat = dense(fit)
    fit["row"] = dict(
        fit_id=key, block=layer, matrix=name, r=r, method=method, lr=lr, steps=steps,
        best_step=int(np.argmin(hist)) if hist else "", converged=converged(hist) if hist else "",
        rel_weight_error=relative_error(A, A_hat),
        rel_output_error=relative_weighted_error(A, A_hat, G),               # undamped G (v1's output error)
        weighted_loss_Gfit=weighted_error(A - A_hat, G_fit).item(),          # the loss weighted HadNet minimises
        rel_weighted_error_Gfit=relative_weighted_error(A, A_hat, G_fit),
        plain_loss=torch.linalg.norm(A - A_hat).item(),                     # the loss plain HadNet minimises
        fit_seconds=seconds)
    torch.save(fit, path)
    print(f"  {key:40s} rel_err {fit['row']['rel_weight_error']:.4f}  out_err "
          f"{fit['row']['rel_output_error']:.4f}  conv {fit['row']['converged']!s:5s} {seconds:6.1f}s", flush=True)
    return fit


def fit_all(model, grams, specs):
    """Run fit_one for every (r, method, lr, steps) in specs on all 20 matrices; returns {fit_id: fit}."""
    fits = {}
    for layer in LAYERS:
        for name in NAMES:
            A = get_weight(model, layer, name).T.contiguous()               # (in, out)
            G = grams[(layer, name)]                                         # (in, in) float64
            G_fit = damped(G)                                                # (in, in) float32
            for r, method, lr, steps in specs:
                fit = fit_one(A, G, G_fit, layer, name, r, method, lr, steps)
                fits[fit["row"]["fit_id"]] = fit
    return fits


def evaluate(model, input_ids, fits, choice, r):
    """Write the 20 chosen fits into the model, measure test perplexity, restore. choice(layer, name) -> fit_id."""
    originals, chosen = {}, []
    for layer in LAYERS:
        for name in NAMES:
            originals[(layer, name)] = get_weight(model, layer, name)
            fit = fits[choice(layer, name)]
            set_weight(model, layer, name, dense(fit).T)                     # back to (out, in)
            chosen.append(fit["row"])
    ppl = perplexity(model, input_ids)
    for (layer, name), X in originals.items():
        set_weight(model, layer, name, X)
    return ppl, chosen


def policy_row(policy, r, ppl, chosen):
    lrs = sorted({str(c["lr"]) for c in chosen})
    steps = sorted({str(c["steps"]) for c in chosen})
    return dict(policy=policy, r=r, perplexity=ppl, model_mb=model_megabytes(r),
                mean_output_error=float(np.mean([c["rel_output_error"] for c in chosen])),
                mean_weight_error=float(np.mean([c["rel_weight_error"] for c in chosen])),
                lrs_used="/".join(lrs), steps_used="/".join(steps),
                fit_ids=";".join(c["fit_id"] for c in chosen))


def save_csv(rows, name):
    with open(RESULTS_DIR / name, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote results/{name} ({len(rows)} rows)", flush=True)


def v1_perplexities():
    """{(method, r): perplexity} from v1, with v1's method names mapped to ours."""
    names = {"svd": "svd", "hadamard": "hadnet", "svd_w": "wsvd", "hadamard_w": "whadnet"}
    with open(V1_CSV, newline="") as f:
        return {(names[row["method"]], int(row["r"])): float(row["perplexity"]) for row in csv.DictReader(f)}


if __name__ == "__main__":
    torch.manual_seed(SEED)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = load_test_tokens(tokenizer)
    grams = collect_gram_matrices(model, load_calibration_tokens(tokenizer))   # uncompressed model
    v1 = v1_perplexities()

    # ---- 1. v1 reproduction ----
    repro_specs = [(r, m, lr, STEPS if lr else "") for r in RANKS for m, lr in [("wsvd", ""), ("whadnet", 0.1)]]
    fits = fit_all(model, grams, repro_specs)
    ppl_rows = []
    for r in RANKS:
        for policy, method, lr in [("wsvd", "wsvd", ""), ("A_whadnet_lr0.1", "whadnet", 0.1)]:
            ppl, chosen = evaluate(model, input_ids, fits,
                                   lambda l, n: fit_id(l, n, r, method, lr, STEPS if lr else ""), r)
            ppl_rows.append(policy_row(policy, r, ppl, chosen))
            match = f"{ppl:.2f}" == f"{v1[(method, r)]:.2f}"
            print(f"REPRO r={r:3d} {policy:16s} {ppl:9.2f}  v1 {v1[(method, r)]:9.2f}  "
                  f"{'match' if match else 'MISMATCH'}", flush=True)
            if not match:
                sys.exit("v1 reproduction failed; see docs/REPORT.md §10 before changing anything")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- 2. the remaining fits, then 5000-step re-runs of unconverged HadNet fits ----
    specs = [(r, "svd", "", "") for r in RANKS]
    specs += [(r, m, lr, STEPS) for r in RANKS for m in ("hadnet", "whadnet") for lr in LRS]
    fits.update(fit_all(model, grams, specs))
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    unconverged = sorted({(f["row"]["r"], f["row"]["method"], f["row"]["lr"]) for f in fits.values()
                          if f["row"]["converged"] is False})
    long_specs = [(r, m, lr, LONG_STEPS) for r, m, lr in unconverged]
    print(f"{len(long_specs)} (r, method, lr) groups have unconverged fits; re-running those groups' "
          f"unconverged matrices at {LONG_STEPS} steps", flush=True)
    for layer in LAYERS:
        for name in NAMES:
            A = get_weight(model, layer, name).T.contiguous()
            G, G_fit = grams[(layer, name)], damped(grams[(layer, name)])
            for r, m, lr, steps in long_specs:
                if not fits[fit_id(layer, name, r, m, lr, STEPS)]["row"]["converged"]:
                    fit = fit_one(A, G, G_fit, layer, name, r, m, lr, steps)
                    fits[fit["row"]["fit_id"]] = fit
    rows = [f["row"] for f in fits.values()]
    save_csv(rows, "lr_rerun_fits.csv")
    np.savez_compressed(RESULTS_DIR / "lr_rerun_curves.npz",
                        **{f["row"]["fit_id"]: np.asarray(f["hist"], dtype=np.float32)
                           for f in fits.values() if f["hist"] is not None})
    print(f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- 3. policies, chosen from calibration losses only; then perplexity ----
    def lowest_whadnet(layer, name, r):
        """Policy C: the weighted HadNet fit (any lr, 2000 or 5000 steps) with the lowest G_fit loss."""
        cands = [f["row"] for f in fits.values() if f["row"]["block"] == layer and f["row"]["matrix"] == name
                 and f["row"]["r"] == r and f["row"]["method"] == "whadnet"]
        return min(cands, key=lambda row: row["weighted_loss_Gfit"])["fit_id"]

    for r in RANKS:
        # Policy D: the plain-HadNet lr with the lowest mean relative plain error over the 20 matrices (2000 steps).
        mean_err = {lr: np.mean([fits[fit_id(l, n, r, "hadnet", lr, STEPS)]["row"]["rel_weight_error"]
                                 for l in LAYERS for n in NAMES]) for lr in LRS}
        lr_d = min(LRS, key=lambda lr: mean_err[lr])
        policies = [
            ("B_whadnet_lr0.03", lambda l, n: fit_id(l, n, r, "whadnet", 0.03, STEPS)),
            ("B_whadnet_lr0.01", lambda l, n: fit_id(l, n, r, "whadnet", 0.01, STEPS)),
            ("C_whadnet_select", lambda l, n: lowest_whadnet(l, n, r)),
            (f"D_hadnet_lr{lr_d}", lambda l, n: fit_id(l, n, r, "hadnet", lr_d, STEPS)),
            ("svd", lambda l, n: fit_id(l, n, r, "svd")),
        ]
        for policy, choice in policies:
            ppl, chosen = evaluate(model, input_ids, fits, choice, r)
            ppl_rows.append(policy_row(policy, r, ppl, chosen))
            print(f"PPL r={r:3d} {policy:18s} {ppl:9.2f}", flush=True)
    save_csv(ppl_rows, "lr_rerun_perplexity.csv")

    restored = perplexity(model, input_ids)
    print(f"restored model perplexity {restored:.2f} (expected {BASELINE})", flush=True)
    assert f"{restored:.2f}" == f"{BASELINE:.2f}"
    print(f"done in {(time.time() - t0) / 60:.1f} min")
