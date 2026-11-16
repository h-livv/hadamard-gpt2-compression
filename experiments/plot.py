"""
Plots from the output of compress_gpt2.py, diagnostics.py, activation_aware.py and cumulative.py. Run:  python experiments/plot.py

Writes, under figures/:
  singular_value_spectra_original.png
        from results.csv / spectra.npz: sigma_i / sigma_1 of every original MLP matrix.
  model_size_vs_perplexity_all_runs.png
        every all-together run of every experiment: stored size of the whole model
        (src/methods.py, model_megabytes) against perplexity.
  weight_error_fits/      both methods fitted to plain weight error ||W - W_hat||_F
    perplexity_vs_storage_all24_r8-64.png       all 24 matrices compressed (compress_gpt2.py)
    perplexity_vs_storage_all24_r64-256.png     same, gentler budgets (diagnostics.py)
    single_matrix_errors_vs_perplexity_r64-128.png
                                                one matrix compressed at a time: weight and output
                                                error against perplexity (diagnostics.py)
  activation_aware_fits/  runs of activation_aware.py and cumulative.py (blocks 2-11 compressed)
    perplexity_vs_storage_blocks2-11_weighted_vs_plain.png
                                                the four fits (SVD / Hadamard x weight error /
                                                activation-weighted), r = 64..256
    cumulative_perplexity_by_block_r64-256.png  blocks 2, 3, ..., 11 compressed one after another
    per_matrix_errors_by_block_r64-256.png      weight and output error of each compressed matrix,
                                                activation-weighted fits only

Source: matplotlib pyplot API, https://matplotlib.org/stable/api/pyplot_summary.html
(plain plotting; no method from the paper is involved).
Reads results/ and writes figures/ (locations in src/paths.py).
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

from src.methods import GPT2_TOTAL_PARAMS, model_megabytes, model_params
from src.metrics import spearman
from src.paths import FIGURES_DIR, RESULTS_DIR

FULL_PARAMS = 768 * 3072         # numbers in one uncompressed MLP matrix
COLORS = {"svd": "#2a78d6", "hadamard": "#eb6834"}           # blue, orange
LABELS = {"svd": "Truncated SVD, rank 2r", "hadamard": "Hadamard (HadNet), rank r"}
MARKERS = {"svd": "o", "hadamard": "s"}


def read_perplexities(filename, column):
    """Read the all-matrices perplexity of every run into {(method, r): perplexity}.

    Both CSV files have one row per compressed matrix; the perplexity of a run
    is repeated on each of its rows, so later rows simply overwrite earlier ones.

    Args: filename results/results.csv (column "perplexity") or
          results/diagnostics.csv (column "perplexity_all").
    """
    table = {}
    with open(filename) as f:
        for row in csv.DictReader(f):
            table[(row["method"], int(row["r"]))] = float(row[column])
    return table


def plot_perplexity(table, baseline, filename):
    """Perplexity (log scale) against the share of each MLP matrix that is stored.

    Args: table {(method, r): perplexity}, baseline = uncompressed perplexity,
          filename = path of the PNG to write.
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

    # Each r has one storage share, shared by both methods, so r is labelled once on the
    # x-axis under its percentage instead of beside every point (where the labels collided).
    ranks = sorted({r for (_, r) in table})
    budget = [100 * 2 * r * (768 + 3072) / FULL_PARAMS for r in ranks]
    ax.set_xticks(budget)
    ax.set_xticklabels([f"{b:.1f}%\nr = {r}" for r, b in zip(ranks, budget)], fontsize=9)
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # Perplexities span several orders of magnitude, hence the log axis.
    ax.set_yscale("log")
    ax.set_xlabel("Share of each MLP matrix stored, 2r(m+n) / (768×3072); Hadamard rank r, SVD rank 2r")
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
    with open(RESULTS_DIR / "diagnostics.csv") as f:
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
    fig.savefig(FIGURES_DIR / "weight_error_fits/single_matrix_errors_vs_perplexity_r64-128.png", dpi=150)


def plot_spectra():
    """sigma_i / sigma_1 for all 24 matrices: one panel per matrix type, one line per layer."""
    spectra = np.load(RESULTS_DIR / "spectra.npz")
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
    fig.savefig(FIGURES_DIR / "singular_value_spectra_original.png", dpi=150, bbox_inches="tight")


# Activation-aware runs: color = method family, line style = fitting loss.
FAMILY = {"svd": "svd", "hadamard": "hadamard", "svd_w": "svd", "hadamard_w": "hadamard"}
AA_LABELS = {"svd": "SVD, weight error", "hadamard": "Hadamard, weight error",
             "svd_w": "SVD, activation-weighted", "hadamard_w": "Hadamard, activation-weighted"}


def plot_activation_aware(baseline):
    """Perplexity (log) against the share of all MLP weights stored, for the four fits of activation_aware.py."""
    table, share = {}, {}
    with open(RESULTS_DIR / "activation_aware.csv") as f:
        for row in csv.DictReader(f):
            key = (row["method"], int(row["r"]))
            table[key] = float(row["perplexity"])
            share[key] = float(row["mlp_budget_percent"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(baseline, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(f"uncompressed: {baseline:.1f}", xy=(0, baseline), xycoords=("axes fraction", "data"),
                xytext=(4, 4), textcoords="offset points", ha="left", color="#52514e")
    for method in ["svd", "hadamard", "svd_w", "hadamard_w"]:
        ranks = sorted(r for (name, r) in table if name == method)
        x = [share[(method, r)] for r in ranks]
        y = [table[(method, r)] for r in ranks]
        weighted = method.endswith("_w")
        color = COLORS[FAMILY[method]]
        ax.plot(x, y, color=color, linewidth=2, linestyle="-" if weighted else (0, (4, 3)),
                marker=MARKERS[FAMILY[method]], markersize=8,
                markerfacecolor=color if weighted else "white", markeredgecolor=color, markeredgewidth=2,
                label=AA_LABELS[method])
        nudge = {"hadamard": -6, "hadamard_w": 6}.get(method, 0)  # end labels 38.5 / 41.9 sit close together
        ax.annotate(f"{y[-1]:.1f}", xy=(x[-1], y[-1]), xytext=(6, nudge), textcoords="offset points",
                    va="center", fontsize=9, color="#1f1e1c")
    ax.set_yscale("log")
    ax.set_xlabel("Share of all MLP weights stored (%); blocks 0-1 kept whole")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("Blocks 2-11 compressed: weight-error vs activation-weighted fits")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "activation_aware_fits/perplexity_vs_storage_blocks2-11_weighted_vs_plain.png", dpi=150)
    plt.close(fig)


RANK_STYLES = {64: (0, (1, 2)), 128: (0, (4, 3)), 256: "-"}   # dotted, dashed, solid


def plot_cumulative(baseline):
    """Perplexity (log) as blocks 2..11 are compressed one after another (cumulative.py)."""
    series = {}
    with open(RESULTS_DIR / "cumulative.csv") as f:
        for row in csv.DictReader(f):
            series.setdefault((row["method"], int(row["r"])), []).append(
                (int(row["last_block"]), float(row["perplexity"])))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(baseline, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(f"uncompressed: {baseline:.1f}", xy=(1, baseline), xycoords=("axes fraction", "data"),
                xytext=(-4, -12), textcoords="offset points", ha="right", color="#52514e")
    for (method, r), points in sorted(series.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        points.sort()
        x = [1] + [layer for layer, _ in points]                     # x = 1: nothing compressed yet
        y = [baseline] + [ppl for _, ppl in points]
        color = COLORS[FAMILY[method]]
        style = RANK_STYLES.get(r, "-")                               # line style encodes r
        filled = r == max(rr for (_, rr) in series)                   # largest r: filled markers
        ax.plot(x, y, color=color, linewidth=2, linestyle=style,
                marker=MARKERS[FAMILY[method]], markersize=7,
                markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=2,
                label=f"{'SVD' if FAMILY[method] == 'svd' else 'Hadamard'}, activation-weighted, r = {r}")
        ax.annotate(f"{y[-1]:.1f}", xy=(x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color="#1f1e1c")
    ax.set_yscale("log")
    ticks = [25, 30, 40, 60, 100, 150, 250]                            # plain numbers, not 2x10^2
    ax.set_ylim(bottom=22)                                             # room for the baseline label below the line
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(range(1, 12))
    ax.set_xticklabels(["none"] + [f"2..{b}" for b in range(2, 12)], fontsize=8)
    ax.set_xlabel("Blocks whose MLP matrices are compressed")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("How the damage accumulates, block by block")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "activation_aware_fits/cumulative_perplexity_by_block_r64-256.png", dpi=150)
    plt.close(fig)


def plot_per_matrix_errors():
    """Weight and output error of each compressed matrix, for the two activation-weighted fits.

    Small multiples: rows c_fc / c_proj, columns r. One point per block (2..11).
    Only errors are shown; no single-matrix perplexity was measured for these fits.
    """
    data = {}
    with open(RESULTS_DIR / "activation_aware.csv") as f:
        for row in csv.DictReader(f):
            if row["method"] in ("svd_w", "hadamard_w"):
                data[(row["method"], int(row["r"]), row["matrix"], int(row["layer"]))] = (
                    float(row["weight_error"]), float(row["output_error"]))
    ranks = sorted({r for (_, r, _, _) in data})
    blocks = sorted({b for (_, _, _, b) in data})
    fig, axes = plt.subplots(2, len(ranks), figsize=(3.6 * len(ranks), 6.2), sharex=True, sharey=True)
    for i, name in enumerate(["c_fc", "c_proj"]):
        for j, r in enumerate(ranks):
            ax = axes[i, j]
            for method in ["svd_w", "hadamard_w"]:
                color = COLORS[FAMILY[method]]
                weight = [data[(method, r, name, b)][0] for b in blocks]
                output = [data[(method, r, name, b)][1] for b in blocks]
                ax.plot(blocks, output, color=color, linewidth=2, marker=MARKERS[FAMILY[method]], markersize=6,
                        label=f"{AA_LABELS[method]}: output error")
                ax.plot(blocks, weight, color=color, linewidth=1.5, linestyle=(0, (4, 3)),
                        marker=MARKERS[FAMILY[method]], markersize=6, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=1.5,
                        label=f"{AA_LABELS[method]}: weight error")
            ax.set_ylim(0, 1)
            ax.set_xticks(blocks)
            ax.grid(True, color="#e4e3df", linewidth=0.8)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if i == 0:
                share = 100 * 2 * r * (768 + 3072) / FULL_PARAMS
                ax.set_title(f"r = {r}  ({share:.1f}% of each matrix)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{name}: relative error")
            if i == 1:
                ax.set_xlabel("Block")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Each compressed MLP matrix: error of the activation-weighted fits (blocks 2-11)", fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES_DIR / "activation_aware_fits/per_matrix_errors_by_block_r64-256.png", dpi=150)
    plt.close(fig)


def size_table():
    """Every all-together run of every experiment, with its stored size.

    Returns: list of dict(setting, method, r, n_compressed, megabytes, percent, perplexity).
    settings: "all24_weight"   all 24 matrices, weight-error fits (results.csv r = 8..64,
                               diagnostics.csv r = 64..256; r = 64 is in both, with the same value)
              "b2-11_weight"   blocks 2-11, weight-error fits (activation_aware.csv)
              "b2-11_weighted" blocks 2-11, activation-weighted fits (activation_aware.csv)
    """
    runs = {}
    with open(RESULTS_DIR / "results.csv") as f:
        for row in csv.DictReader(f):
            if row["method"] != "baseline":
                runs[("all24_weight", row["method"], int(row["r"]))] = (24, float(row["perplexity"]))
    with open(RESULTS_DIR / "diagnostics.csv") as f:
        for row in csv.DictReader(f):
            runs[("all24_weight", row["method"], int(row["r"]))] = (24, float(row["perplexity_all"]))
    with open(RESULTS_DIR / "activation_aware.csv") as f:
        for row in csv.DictReader(f):
            setting = "b2-11_weighted" if row["method"].endswith("_w") else "b2-11_weight"
            runs[(setting, FAMILY[row["method"]], int(row["r"]))] = (20, float(row["perplexity"]))
    table = []
    for (setting, method, r), (n, ppl) in sorted(runs.items()):
        table.append(dict(setting=setting, method=method, r=r, n_compressed=n,
                          megabytes=model_megabytes(r, n),
                          percent=100 * model_params(r, n) / GPT2_TOTAL_PARAMS, perplexity=ppl))
    return table


SETTING_STYLES = {   # line style, filled markers, label
    "all24_weight": ((0, (1, 2)), False, "all 24 matrices, weight-error fit"),
    "b2-11_weight": ((0, (4, 3)), False, "blocks 2-11, weight-error fit"),
    "b2-11_weighted": ("-", True, "blocks 2-11, activation-weighted fit"),
}


def plot_size_vs_perplexity(baseline):
    """Perplexity (log) against the stored size of the whole model, for every all-together run."""
    table = size_table()
    full_mb = model_megabytes(0, 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(baseline, color="#52514e", linewidth=1, linestyle="--")
    ax.annotate(f"uncompressed: {baseline:.1f} at {full_mb:.0f} MB", xy=(0, baseline),
                xycoords=("axes fraction", "data"), xytext=(4, 4), textcoords="offset points",
                ha="left", color="#52514e")
    for setting, (style, filled, label) in SETTING_STYLES.items():
        for method in ["svd", "hadamard"]:
            rows = sorted((t for t in table if t["setting"] == setting and t["method"] == method),
                          key=lambda t: t["percent"])
            color = COLORS[method]
            name = {"svd": "SVD", "hadamard": "Hadamard"}[method]
            if setting == "b2-11_weighted":
                name = {"svd": "Whitened SVD", "hadamard": "Weighted Hadamard"}[method]
            ax.plot([t["percent"] for t in rows], [t["perplexity"] for t in rows], color=color,
                    linewidth=2, linestyle=style, marker=MARKERS[method], markersize=7,
                    markerfacecolor=color if filled else "white", markeredgecolor=color,
                    markeredgewidth=2, label=f"{name}, {label}")
    ax.set_yscale("log")
    ax.set_xlabel(f"Stored size of the whole model, % of the original {full_mb:.1f} MB (float32)")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("Model size against perplexity, every run (Hadamard rank r, SVD rank 2r)")
    ax.grid(True, which="major", color="#e4e3df", linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_size_vs_perplexity_all_runs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for folder in ["weight_error_fits", "activation_aware_fits"]:
        (FIGURES_DIR / folder).mkdir(parents=True, exist_ok=True)
    harsh = read_perplexities(RESULTS_DIR / "results.csv", "perplexity")
    baseline = harsh[("baseline", 0)]
    del harsh[("baseline", 0)]
    plot_perplexity(harsh, baseline, FIGURES_DIR / "weight_error_fits/perplexity_vs_storage_all24_r8-64.png")
    plot_spectra()
    print("wrote the all-24 r = 8..64 perplexity plot and the spectra")

    # Each later experiment is a separate run; plot it only if its results exist.
    if os.path.exists(RESULTS_DIR / "diagnostics.csv"):
        plot_perplexity(read_perplexities(RESULTS_DIR / "diagnostics.csv", "perplexity_all"),
                        baseline, FIGURES_DIR / "weight_error_fits/perplexity_vs_storage_all24_r64-256.png")
        plot_error_vs_perplexity()
        print("wrote the r = 64..256 perplexity plot and the single-matrix error plot")
    for csv_name, plot in [("activation_aware.csv", plot_activation_aware), ("cumulative.csv", plot_cumulative)]:
        if os.path.exists(RESULTS_DIR / csv_name):
            plot(baseline)
            print(f"wrote the {csv_name.replace('.csv', '')} plot")
    if os.path.exists(RESULTS_DIR / "activation_aware.csv"):
        plot_per_matrix_errors()
        print("wrote the per-matrix error plot")
    if os.path.exists(RESULTS_DIR / "diagnostics.csv") and os.path.exists(RESULTS_DIR / "activation_aware.csv"):
        plot_size_vs_perplexity(baseline)
        print("wrote the model size plot")
