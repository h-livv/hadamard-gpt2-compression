"""
All v2 figures, from the CSV and .npz files in results/.

Reads     results/spectra.npz, s51_spectrum.csv, s52_weightings.csv, s53_fix.csv,
          curves_s5x.npz, reconstructions.npz, s6_gpt2.csv, curves_s6.npz
Writes    figures/v2/fig1_spectra.png ... fig7_gpt2_curves.png
Runtime   about 10 seconds.

Run:  python experiments/plot_v2.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.paths import FIGURES_DIR, RESULTS_DIR

OUT = FIGURES_DIR / "v2"

# Fixed colour per method (categorical slots in fixed order; a method keeps its colour in every figure).
METHOD_STYLE = {
    "svd_2r":  dict(color="#2a78d6", marker="o", label="SVD, rank 2r"),
    "hadnet":  dict(color="#eb6834", marker="s", label="HadNet, rank r"),
    "wsvd_2r": dict(color="#1baf7a", marker="^", label="whitened SVD, rank 2r"),
    "whadnet": dict(color="#6250d6", marker="D", label="weighted HadNet, rank r"),
    "svd_r":   dict(color="#8a8984", marker="v", label="SVD, rank r (equal-rank ref.)", linestyle="--"),
}
LR_COLORS = {0.1: "#2a78d6", 0.03: "#eb6834", 0.01: "#1baf7a"}
SHARE_COLORS = {6.25: "#c6dbf3", 12.5: "#7fb0e8", 25.0: "#2a78d6", 50.0: "#0d3f80"}   # one hue, light -> dark
INIT_LABEL = {"paper": "paper init (§3.3)", "warm": "warm start (plain HadNet)",
              "embed": "whitened-SVD embedding"}
W_LABEL = {"identity": "identity (κ = 1)", "region": "region (κ = 100, diagonal)",
           "freq10": "frequency, κ = 10", "freq100": "frequency, κ = 100", "freq1000": "frequency, κ = 1000"}


def read(name):
    """Rows of results/<name> with numeric fields converted to float where possible."""
    def conv(v):
        try:
            return float(v)
        except ValueError:
            return v
    with open(RESULTS_DIR / name, newline="") as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]


def pick(rows, **where):
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def style_axes(ax):
    ax.grid(True, color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/v2/{name}")


def fig1_spectra():
    spectra = np.load(RESULTS_DIR / "spectra.npz")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, color in zip([64, 128, 256, 512], ["#c6dbf3", "#7fb0e8", "#2a78d6", "#0d3f80"]):
        s = spectra[f"image_{n}"]
        ax.semilogy(np.arange(1, n + 1) / n, s / s[0], color=color, linewidth=2, label=f"image, n = {n}")
    s = spectra["control_256"]
    ax.semilogy(np.arange(1, 257) / 256, s / s[0], color="#eb6834", linewidth=2, linestyle="--",
                label="Gaussian control, n = 256")
    ax.set_xlabel("singular value index i / n (dimensionless)")
    ax.set_ylabel("σᵢ / σ₁ (dimensionless, log scale)")
    ax.set_title("The image's top 10% of singular values hold 96–99% of its energy; the Gaussian control's hold 31%", fontsize=10)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig1_spectra.png")


def fig2_plain_error():
    rows = read("s51_spectrum.csv")
    panels = [("image", n) for n in [64, 128, 256, 512]] + [("control", 256)]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True)
    for ax, (kind, n) in zip(axes, panels):
        for method in ["svd_2r", "hadnet", "svd_r"]:
            sel = sorted(pick(rows, matrix=kind, n=n, method=method), key=lambda r: r["share"])
            st = METHOD_STYLE[method]
            ax.plot([r["share"] for r in sel], [r["rel_err"] for r in sel], color=st["color"],
                    marker=st["marker"], linestyle=st.get("linestyle", "-"), linewidth=2, markersize=6,
                    label=st["label"])
        ax.set_xscale("log", base=2)
        ax.set_xticks([6.25, 12.5, 25, 50], ["6.25", "12.5", "25", "50"])
        ax.set_title(f"{kind}, n = {n}", fontsize=10)
        ax.set_xlabel("storage share 4r/n (%)")
        style_axes(ax)
    axes[0].set_ylabel("relative plain error ‖A − Â‖/‖A‖")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("At equal storage SVD(2r) beats HadNet(r) on the image at every n and share; "
                 "on the flat-spectrum control HadNet(r) wins at every share", fontsize=11, y=1.03)
    save(fig, "fig2_plain_error.png")


def fig3_weighted_n256():
    rows = read("s52_weightings.csv")
    names = ["identity", "region", "freq10", "freq100", "freq1000"]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharex=True)
    for ax, w in zip(axes, names):
        for method in ["svd_2r", "hadnet", "wsvd_2r", "whadnet"]:
            sel = sorted(pick(rows, n=256, weighting=w, method=method), key=lambda r: r["share"])
            st = METHOD_STYLE[method]
            ax.plot([r["share"] for r in sel], [r["rel_werr"] for r in sel], color=st["color"],
                    marker=st["marker"], linewidth=2, markersize=6, label=st["label"])
        ax.set_xscale("log", base=2)
        ax.set_xticks([6.25, 12.5, 25, 50], ["6.25", "12.5", "25", "50"])
        ax.set_title(W_LABEL[w], fontsize=10)
        ax.set_xlabel("storage share 4r/n (%)")
        style_axes(ax)
    axes[0].set_ylabel("relative weighted error ‖Sᵀ(A − Â)‖/‖SᵀA‖")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("n = 256: whitened SVD has the lowest weighted error under every weighting and share",
                 fontsize=11, y=1.03)
    save(fig, "fig3_weighted_n256.png")


def fig4_gap_vs_kappa():
    rows = read("s52_weightings.csv")
    kappa = {"identity": 1, "freq10": 10, "freq100": 100, "freq1000": 1000}
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    for ax, n in zip(axes, [64, 128, 256, 512]):
        for share, color in SHARE_COLORS.items():
            xs, ys = [], []
            for w, k in kappa.items():
                wh = pick(rows, n=n, share=share, weighting=w, method="whadnet")[0]["rel_werr"]
                ws = pick(rows, n=n, share=share, weighting=w, method="wsvd_2r")[0]["rel_werr"]
                xs.append(k)
                ys.append(100 * (wh - ws) / ws)
            ax.plot(xs, ys, color=color, marker="o", linewidth=2, markersize=6, label=f"share {share:g}%")
        ax.set_xscale("log")
        ax.set_title(f"image, n = {n}", fontsize=10)
        ax.set_xlabel("condition number κ of G (frequency weighting; κ = 1 is G = I)")
        style_axes(ax)
    axes[0].set_ylabel("weighted HadNet − whitened SVD (% of whitened SVD's error)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Weighted HadNet (paper settings) never reaches whitened SVD: the gap is set mainly by the share, ~100% at 50%",
                 fontsize=11, y=1.03)
    save(fig, "fig4_gap_vs_kappa.png")


def loss_curve_panels(rows, curves, refs, title, name, ylabel, norm_of):
    """One panel per start, one line per learning rate, horizontal reference lines."""
    inits = ["paper", "warm", "embed"]
    fig, axes = plt.subplots(1, len(inits), figsize=(4 * len(inits), 3.6), sharey=True)
    for ax, init in zip(axes, inits):
        for lr, color in LR_COLORS.items():
            row = pick(rows, init=init, lr=lr)[0]
            hist = curves[row["run_id"]] / norm_of(row)
            ax.plot(np.arange(len(hist)), hist, color=color, linewidth=1.5, label=f"lr {lr:g}")
        for value, label, ls in refs:
            ax.axhline(value, color="#52514e", linestyle=ls, linewidth=1, label=label)
        ax.set_yscale("log")
        ax.set_title(INIT_LABEL[init], fontsize=10)
        ax.set_xlabel("Adam step (full matrix)")
        style_axes(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, y=1.03)
    save(fig, name)


def fig5_hardest_curves():
    rows = pick(read("s53_fix.csv"), n=256, share=50.0, weighting="freq1000")
    curves = np.load(RESULTS_DIR / "curves_s5x.npz")
    refs = [(rows[0]["ref_wsvd_2r"], "whitened SVD, rank 2r", "--"),
            (rows[0]["ref_hadnet"], "plain HadNet (its weighted error)", ":")]
    loss_curve_panels(rows, curves, refs,
                      "n = 256, 50% share, κ = 1000: every start and learning rate stays ≥ 2× above whitened SVD; the embedding never leaves rank r",
                      "fig5_loss_curves_hardest.png", "relative weighted error (dimensionless)",
                      lambda r: r["start_loss"] / r["start_rel_werr"])


def fig6_reconstructions():
    recon = np.load(RESULTS_DIR / "reconstructions.npz")
    rows = pick(read("s52_weightings.csv"), n=256, share=12.5)
    methods = ["svd_2r", "hadnet", "wsvd_2r", "whadnet"]
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6))
    for i, w in enumerate(["region", "freq1000"]):
        axes[i, 0].imshow(recon["original"], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(f"original, 256 × 256\n(weighting: {W_LABEL[w]})", fontsize=9)
        for j, m in enumerate(methods, start=1):
            row = pick(rows, weighting=w, method=m)[0]
            axes[i, j].imshow(recon[f"{w}_{m}"], cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{METHOD_STYLE[m]['label']}\nweighted err {row['rel_werr']:.3f}, "
                                 f"PSNR {row['psnr']:.1f} dB", fontsize=9)
        if w == "region":
            for ax in axes[i]:
                for y in (256 // 3, 2 * 256 // 3):
                    ax.axhline(y - 0.5, color="#eb6834", linewidth=0.8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("n = 256, 12.5% share (r = 8): whitened SVD has the lowest weighted error under both weightings "
                 "(display clipped to [0, 1]; orange lines bound the region's middle third)", fontsize=11)
    save(fig, "fig6_reconstructions.png")


def fig7_gpt2_curves():
    rows = read("s6_gpt2.csv")
    curves = np.load(RESULTS_DIR / "curves_s6.npz")
    wh = pick(rows, method="whadnet")
    refs = [(pick(rows, method="wsvd")[0]["rel_werr_Gfit"], "whitened SVD, rank 2r", "--"),
            (pick(rows, method="hadnet")[0]["rel_werr_Gfit"], "plain HadNet (its weighted error)", ":")]
    # rel_werr_Gfit of the best factors = best_loss / ||S^T A||, so the norm is best_loss / rel_werr_Gfit
    loss_curve_panels(wh, curves, refs,
                      "GPT-2 block 5 c_proj, r = 128: at lr 0.1 no start improves; at lr ≤ 0.03 the paper and warm starts beat whitened SVD",
                      "fig7_gpt2_curves.png", "relative weighted error under G_fit (dimensionless)",
                      lambda r: r["best_loss"] / r["rel_werr_Gfit"])


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    fig1_spectra()
    fig2_plain_error()
    fig3_weighted_n256()
    fig4_gap_vs_kappa()
    fig5_hardest_curves()
    fig6_reconstructions()
    if (RESULTS_DIR / "s6_gpt2.csv").exists():
        fig7_gpt2_curves()
