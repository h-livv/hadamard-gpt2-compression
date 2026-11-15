"""
Plots from the output of compress_gpt2.py and diagnostics.py. Run:  python plot.py

From results.csv and spectra.npz (compress_gpt2.py):
  perplexity.png         WikiText-2 test perplexity against parameter budget, r = 8..64.
  spectra.png            normalised singular values sigma_i / sigma_1 of every MLP matrix.
From diagnostics.csv (diagnostics.py):
  perplexity_gentle.png  the same perplexity plot for the gentler budgets, r = 64..256.
  error_vs_perplexity.png  one matrix compressed at a time: weight error and output
                         error against perplexity, to see which one predicts the damage.

Source: matplotlib pyplot API, https://matplotlib.org/stable/api/pyplot_summary.html
(plain plotting; no method from the paper is involved).
"""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from diagnostics import spearman

FULL_PARAMS = 768 * 3072         # numbers in one uncompressed MLP matrix
COLORS = {"svd": "#2a78d6", "hadamard": "#eb6834"}           # blue, orange
LABELS = {"svd": "Truncated SVD, rank 2r", "hadamard": "Hadamard (HadNet), rank r"}
MARKERS = {"svd": "o", "hadamard": "s"}


def read_perplexities(filename, column):
    """Read the all-matrices perplexity of every run into {(method, r): perplexity}.

    Both CSV files have one row per compressed matrix; the perplexity of a run
    is repeated on each of its rows, so later rows simply overwrite earlier ones.

    Args: filename "results.csv" (column "perplexity") or
          "diagnostics.csv" (column "perplexity_all").
    """
    table = {}
    with open(filename) as f:
        for row in csv.DictReader(f):
            table[(row["method"], int(row["r"]))] = float(row[column])
    return table


def plot_perplexity(table, baseline, filename):
    """Perplexity (log scale) against the share of each MLP matrix that is stored.

    Args: table {(method, r): perplexity}, baseline = uncompressed perplexity,
          filename of the PNG to write.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axhline(baseline, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(f"uncompressed: {baseline:.1f}", xy=(0, baseline), xycoords=("axes fraction", "data"),
                xytext=(4, 4), textcoords="offset points", ha="left", color="#52514e")

    for method in ["svd", "hadamard"]:
        ranks = sorted(r for (name, r) in table if name == method)
        budget = [100 * 2 * r * (768 + 3072) / FULL_PARAMS for r in ranks]   # 2r(m+n) as percent of full size
        ppl = [table[(method, r)] for r in ranks]
        ax.plot(budget, ppl, color=COLORS[method], marker=MARKERS[method], markersize=7,
                linewidth=2, label=LABELS[method])
        # SVD labels go above their point and Hadamard labels below, so the two never collide.
        label_offset = (6, 7) if method == "svd" else (6, -13)
        for r, x, y in zip(ranks, budget, ppl):
            ax.annotate(f"r={r}", xy=(x, y), xytext=label_offset, textcoords="offset points",
                        fontsize=8, color="#52514e")

    # Perplexities span several orders of magnitude, hence the log axis.
    ax.set_yscale("log")
    ax.set_xlabel("Numbers stored per MLP matrix, 2r(m+n), as % of the full 768×3072")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("GPT-2 small, all 24 MLP matrices compressed, equal parameter budget")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)


def plot_error_vs_perplexity():
    """One matrix compressed at a time: which error measure predicts the perplexity?

    Left panel: relative weight error ||E||_F / ||W||_F on the x axis.
    Right panel: relative output error ||X E||_F / ||X W||_F on calibration inputs.
    Each point is one (matrix, r, method) run from Step 2.1 of diagnostics.py.
    The legend gives each method's Spearman rank correlation with perplexity
    (the same as with log perplexity, because log keeps the order).
    """
    runs = []
    with open("diagnostics.csv") as f:
        for row in csv.DictReader(f):
            if row["perplexity_single"] != "":               # only the single-matrix runs
                runs.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    panels = [("weight_error", "Relative weight error  ‖E‖ / ‖W‖"),
              ("output_error", "Relative output error  ‖XE‖ / ‖XW‖")]
    for ax, (column, xlabel) in zip(axes, panels):
        for method in ["svd", "hadamard"]:
            x = [float(row[column]) for row in runs if row["method"] == method]
            y = [float(row["perplexity_single"]) for row in runs if row["method"] == method]
            ax.scatter(x, y, s=36, color=COLORS[method], marker=MARKERS[method], alpha=0.8,
                       edgecolors="white", linewidths=0.6,
                       label=f"{LABELS[method]} (Spearman {spearman(x, y):+.2f})")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("Perplexity with this one matrix compressed (log scale)")
    fig.suptitle("One MLP matrix compressed at a time (r = 64 and 128)")
    fig.tight_layout()
    fig.savefig("error_vs_perplexity.png", dpi=150)


def plot_spectra():
    """sigma_i / sigma_1 for all 24 matrices: one panel per matrix type, one line per layer."""
    spectra = np.load("spectra.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    # One hue, light to dark with depth, because layer index is an ordered quantity.
    layer_colors = plt.cm.Blues(np.linspace(0.3, 1.0, 12))

    for ax, name in zip(axes, ["c_fc", "c_proj"]):
        for layer in range(12):
            sigma = spectra[f"h{layer}_{name}"]              # (768,), descending
            ax.plot(np.arange(1, len(sigma) + 1), sigma / sigma[0], color=layer_colors[layer], linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel("Singular value index i")
        ax.set_title(f"mlp.{name}")
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("σᵢ / σ₁ (log scale)")

    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.matplotlib.colors.ListedColormap(layer_colors),
                                                  norm=plt.Normalize(-0.5, 11.5)),
                            ax=axes, ticks=range(12), pad=0.02)
    colorbar.set_label("Transformer block (0 = first)")
    fig.suptitle("Normalised singular-value spectra of the GPT-2 small MLP matrices")
    fig.savefig("spectra.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    harsh = read_perplexities("results.csv", "perplexity")
    baseline = harsh[("baseline", 0)]
    del harsh[("baseline", 0)]
    plot_perplexity(harsh, baseline, "perplexity.png")
    plot_spectra()
    print("wrote perplexity.png and spectra.png")

    # The diagnostic figures need the output of diagnostics.py, a separate, longer run.
    if os.path.exists("diagnostics.csv"):
        plot_perplexity(read_perplexities("diagnostics.csv", "perplexity_all"), baseline, "perplexity_gentle.png")
        plot_error_vs_perplexity()
        print("wrote perplexity_gentle.png and error_vs_perplexity.png")
    else:
        print("no diagnostics.csv; run diagnostics.py first for the diagnostic figures")
