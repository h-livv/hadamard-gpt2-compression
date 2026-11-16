"""
Every figure, drawn from the files in results/. Run:  python experiments/plot.py

Writes, under figures/:
  singular_value_spectra_original.png          sigma_i / sigma_1 of every original MLP matrix (spectra.npz)
  weight_error_fits/                           both methods fitted to plain weight error
    perplexity_vs_storage_all24.png            all 24 matrices compressed, r = 8..256
                                               (results.csv, diagnostics.csv)
    single_matrix_damage_by_block.png          perplexity with ONE matrix compressed, by block and
                                               matrix (diagnostics.csv)
  activation_aware_fits/                       blocks 2-11 compressed (activation_aware.csv, cumulative.csv)
    perplexity_vs_model_size_blocks2-11.png    the four fits against the stored size of the whole model
    output_error_vs_perplexity_compressed_together.png
                                               mean output error against perplexity, every run that
                                               compresses many matrices at once (also diagnostics.csv)
    per_matrix_errors_by_block_r64-256.png     weight and output error of each compressed matrix
    cumulative_cost_per_block_r64-256.png      perplexity increase caused by each block as blocks
                                               2, 3, ..., 11 are compressed one after another

Source: matplotlib pyplot API, https://matplotlib.org/stable/api/pyplot_summary.html
(plain plotting; no method from the paper is involved).
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

from src.methods import GPT2_TOTAL_PARAMS, model_megabytes, model_params
from src.metrics import spearman
from src.paths import FIGURES_DIR, RESULTS_DIR

FULL_PARAMS = 768 * 3072         # numbers in one uncompressed MLP matrix
COLORS = {"svd": "#2a78d6", "hadamard": "#eb6834"}           # blue, orange
MARKERS = {"svd": "o", "hadamard": "s"}
FAMILY = {"svd": "svd", "hadamard": "hadamard", "svd_w": "svd", "hadamard_w": "hadamard"}
NAMES = {"svd": "SVD, weight error", "hadamard": "Hadamard, weight error",
         "svd_w": "Whitened SVD (activation-weighted)", "hadamard_w": "Weighted Hadamard (activation-weighted)"}
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#e4e3df"


def share(r):
    """Percent of one MLP matrix stored at rank r: 2r(m+n) / (m n)."""
    return 100 * 2 * r * (768 + 3072) / FULL_PARAMS


def style(ax):
    """Recessive grid, no top/right spines."""
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def baseline_line(ax, baseline, text=None):
    """Dashed line at the uncompressed perplexity, labelled at the right end, under the line."""
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(text or f"uncompressed: {baseline:.1f}", xy=(1, baseline), xycoords=("axes fraction", "data"),
                xytext=(-4, -12), textcoords="offset points", ha="right", color=MUTED)


def read_rows(name):
    with open(RESULTS_DIR / name) as f:
        return list(csv.DictReader(f))


def plot_benchmark(baseline):
    """All 24 matrices, weight-error fits: perplexity against share stored, r = 8..256 (log x: r doubles)."""
    table = {}
    for row in read_rows("results.csv"):
        if row["method"] != "baseline":
            table[(row["method"], int(row["r"]))] = float(row["perplexity"])
    for row in read_rows("diagnostics.csv"):               # r = 64 is in both files, with the same value
        table[(row["method"], int(row["r"]))] = float(row["perplexity_all"])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    baseline_line(ax, baseline)
    labels = {"svd": "Truncated SVD, rank 2r", "hadamard": "Hadamard (HadNet), rank r"}
    for method in ["svd", "hadamard"]:
        ranks = sorted(r for (name, r) in table if name == method)
        ax.plot([share(r) for r in ranks], [table[(method, r)] for r in ranks], color=COLORS[method],
                marker=MARKERS[method], markersize=7, linewidth=2, label=labels[method])
    ranks = sorted({r for (_, r) in table})
    ax.set_xscale("log")
    ax.set_xticks([share(r) for r in ranks])
    ax.set_xticklabels([f"{share(r):.1f}%\nr = {r}" for r in ranks], fontsize=8)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yscale("log")
    ax.set_ylim(bottom=13)
    ax.set_xlabel("Share of each MLP matrix stored (log scale)")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("All 24 MLP matrices compressed, weight-error fits, equal storage")
    style(ax)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "weight_error_fits/perplexity_vs_storage_all24.png", dpi=150)
    plt.close(fig)


def plot_single_matrix_damage(baseline):
    """Heatmap: perplexity with only one matrix compressed, rows = block, columns = method x r."""
    ppl = {}
    for row in read_rows("diagnostics.csv"):
        if row["perplexity_single"]:
            ppl[(row["matrix"], row["method"], int(row["r"]), int(row["layer"]))] = float(row["perplexity_single"])
    columns = [(m, r) for m in ["svd", "hadamard"] for r in sorted({r for (_, _, r, _) in ppl})]
    vmax = max(ppl.values())
    norm = LogNorm(vmin=baseline, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 6.2), sharey=True)
    for ax, name in zip(axes, ["c_fc", "c_proj"]):
        grid = np.array([[ppl[(name, m, r, layer)] for (m, r) in columns] for layer in range(12)])
        image = ax.imshow(grid, cmap="Blues", norm=norm, aspect="auto")
        for i in range(12):
            for j in range(len(columns)):
                value = grid[i, j]
                dark = norm(value) > 0.55
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else INK)
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels([f"{'SVD' if m == 'svd' else 'Hadamard'}\nr = {r}" for m, r in columns], fontsize=8)
        ax.set_yticks(range(12))
        ax.set_title(f"mlp.{name} compressed alone")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Block")
    colorbar = fig.colorbar(image, ax=axes, pad=0.02, shrink=0.85)
    ticks = [30, 40, 60, 100]
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([str(t) for t in ticks])
    colorbar.ax.yaxis.set_minor_formatter(plt.NullFormatter())
    colorbar.set_label(f"Perplexity with only this matrix compressed (log; uncompressed {baseline:.2f})")
    fig.suptitle("One matrix compressed at a time: the damage sits in blocks 0 and 1")
    fig.savefig(FIGURES_DIR / "weight_error_fits/single_matrix_damage_by_block.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


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
        style(ax)
    axes[0].set_ylabel("σᵢ / σ₁ (log scale)")

    colorbar = fig.colorbar(plt.cm.ScalarMappable(cmap=plt.matplotlib.colors.ListedColormap(layer_colors),
                                                  norm=plt.Normalize(-0.5, 11.5)),
                            ax=axes, ticks=range(12), pad=0.02)
    colorbar.set_label("Transformer block (0 = first)")
    fig.suptitle("Normalised singular-value spectra of the GPT-2 small MLP matrices")
    fig.savefig(FIGURES_DIR / "singular_value_spectra_original.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_four_fits(baseline):
    """Blocks 2-11: the four fits, perplexity against the stored size of the whole model."""
    table = {(row["method"], int(row["r"])): float(row["perplexity"]) for row in read_rows("activation_aware.csv")}
    ranks = sorted({r for (_, r) in table})
    size = {r: model_megabytes(r, 20) for r in ranks}
    full = model_megabytes(0, 0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    baseline_line(ax, baseline, f"uncompressed: {baseline:.1f} at {full:.0f} MB")
    for method in ["svd", "hadamard", "svd_w", "hadamard_w"]:
        weighted = method.endswith("_w")
        color = COLORS[FAMILY[method]]
        y = [table[(method, r)] for r in ranks]
        ax.plot([size[r] for r in ranks], y, color=color, linewidth=2, linestyle="-" if weighted else (0, (4, 3)),
                marker=MARKERS[FAMILY[method]], markersize=8, markerfacecolor=color if weighted else "white",
                markeredgecolor=color, markeredgewidth=2, label=NAMES[method])
        nudge = {"hadamard": -6, "hadamard_w": 6}.get(method, 0)  # end labels 38.5 / 41.9 sit close together
        ax.annotate(f"{y[-1]:.1f}", xy=(size[ranks[-1]], y[-1]), xytext=(6, nudge), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks([size[r] for r in ranks])
    ax.set_xticklabels([f"{size[r]:.0f} MB ({100 * model_params(r, 20) / GPT2_TOTAL_PARAMS:.0f}%)\nr = {r}"
                        for r in ranks], fontsize=8)
    ax.set_xlim(size[ranks[0]] - 10, size[ranks[-1]] + 18)
    ax.set_yscale("log")
    ax.set_ylim(bottom=18)
    ax.set_xlabel(f"Stored size of the whole model (float32; original {full:.1f} MB); blocks 0-1 kept whole")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("Blocks 2-11 compressed: weight-error vs activation-weighted fits")
    style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "activation_aware_fits/perplexity_vs_model_size_blocks2-11.png", dpi=150)
    plt.close(fig)


def run_means(rows, ppl_column):
    """{(method, r): (mean output error over the run's matrices, perplexity)} for all-together runs."""
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], int(row["r"])), []).append(row)
    return {key: (np.mean([float(row["output_error"]) for row in group]), float(group[0][ppl_column]))
            for key, group in groups.items()}


def plot_output_error_vs_perplexity(baseline):
    """Mean output error against perplexity, for every run that compresses many matrices at once.

    Left: all 24 matrices, weight-error fits (diagnostics.csv). Right: blocks 2-11, four fits
    (activation_aware.csv). The title of each panel gives the Spearman rank correlation.
    Returns {panel: Spearman} so the numbers can be quoted.
    """
    panels = [("All 24 matrices, weight-error fits", run_means(read_rows("diagnostics.csv"), "perplexity_all")),
              ("Blocks 2-11, four fits", run_means(read_rows("activation_aware.csv"), "perplexity"))]
    correlations = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (title, runs) in zip(axes, panels):
        baseline_line(ax, baseline)
        for method in ["svd", "hadamard", "svd_w", "hadamard_w"]:
            points = sorted((r, err, ppl) for (m, r), (err, ppl) in runs.items() if m == method)
            if not points:
                continue
            color = COLORS[FAMILY[method]]
            weighted = method.endswith("_w")
            ax.scatter([p[1] for p in points], [p[2] for p in points], s=70, marker=MARKERS[FAMILY[method]],
                       color=color if weighted else "white", edgecolors=color, linewidths=2, label=NAMES[method],
                       zorder=3)
        errors = [v[0] for v in runs.values()]
        ppls = [v[1] for v in runs.values()]
        correlations[title] = spearman(errors, ppls)
        ax.set_title(f"{title}\nSpearman {correlations[title]:+.2f} over {len(runs)} runs", fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(bottom=13)
        ax.set_xlabel("Mean output error  ‖XE‖ / ‖XW‖ over the compressed matrices")
        style(ax)
    axes[0].set_ylabel("WikiText-2 test perplexity (log scale)")
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Many matrices compressed together: output error predicts perplexity (each point one run; r = 64..256)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES_DIR / "activation_aware_fits/output_error_vs_perplexity_compressed_together.png", dpi=150)
    plt.close(fig)
    return correlations


def plot_per_matrix_errors():
    """Weight and output error of each compressed matrix, for the two activation-weighted fits.

    Small multiples: rows c_fc / c_proj, columns r. One point per block (2..11).
    Only errors are shown; no single-matrix perplexity was measured for these fits.
    """
    data = {}
    for row in read_rows("activation_aware.csv"):
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
                ax.plot(blocks, output, color=color, linewidth=2, marker=MARKERS[FAMILY[method]], markersize=6)
                ax.plot(blocks, weight, color=color, linewidth=1.5, linestyle=(0, (4, 3)),
                        marker=MARKERS[FAMILY[method]], markersize=6, markerfacecolor="white",
                        markeredgecolor=color, markeredgewidth=1.5)
            ax.set_ylim(0, 1)
            ax.set_xticks(blocks)
            style(ax)
            if i == 0:
                ax.set_title(f"r = {r}  ({share(r):.1f}% of each matrix)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{name}: relative error")
            if i == 1:
                ax.set_xlabel("Block")
    # Two-part legend: color = method, line style = which error is measured.
    handles = [Line2D([], [], color=COLORS["svd"], linewidth=2, marker="o", label="Whitened SVD"),
               Line2D([], [], color=COLORS["hadamard"], linewidth=2, marker="s", label="Weighted Hadamard"),
               Line2D([], [], color=MUTED, linewidth=2, marker="o", label="Output error  ‖X(W − Ŵ)‖ / ‖XW‖"),
               Line2D([], [], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), marker="o", markerfacecolor="white",
                      markeredgewidth=1.5, label="Weight error  ‖W − Ŵ‖ / ‖W‖")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Each compressed MLP matrix: error of the two activation-weighted fits (blocks 2-11)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURES_DIR / "activation_aware_fits/per_matrix_errors_by_block_r64-256.png", dpi=150)
    plt.close(fig)


def plot_cumulative_cost(baseline):
    """Bars: % increase in perplexity when each block is added (cumulative.py), one panel per r."""
    series = {}
    for row in read_rows("cumulative.csv"):
        series.setdefault((row["method"], int(row["r"])), {})[int(row["last_block"])] = float(row["perplexity"])
    ranks = sorted({r for (_, r) in series})
    fig, axes = plt.subplots(1, len(ranks), figsize=(4.2 * len(ranks), 4.2), sharex=True)
    width = 0.4
    for ax, r in zip(axes, ranks):
        ax.axvspan(2.5, 8.5, color="#f0efec", zorder=0)
        for k, method in enumerate(["svd_w", "hadamard_w"]):
            ppl = series[(method, r)]
            blocks = sorted(ppl)
            previous = [baseline] + [ppl[b] for b in blocks[:-1]]
            cost = [100 * (ppl[b] / p - 1) for b, p in zip(blocks, previous)]
            ax.bar([b + (k - 0.5) * width for b in blocks], cost, width=width * 0.9, color=COLORS[FAMILY[method]],
                   label={"svd_w": "Whitened SVD", "hadamard_w": "Weighted Hadamard"}[method], zorder=2)
        ax.set_title(f"r = {r}   (final perplexity {series[('svd_w', r)][11]:.1f} vs "
                     f"{series[('hadamard_w', r)][11]:.1f})", fontsize=10)
        ax.set_xticks(range(2, 12))
        ax.set_xlabel("Block added")
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Perplexity increase when the block is added (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Compressing blocks 2, 3, ..., 11 one after another: what each block costs (shaded: blocks 3-8)")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES_DIR / "activation_aware_fits/cumulative_cost_per_block_r64-256.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for folder in ["weight_error_fits", "activation_aware_fits"]:
        (FIGURES_DIR / folder).mkdir(parents=True, exist_ok=True)
    baseline = next(float(row["perplexity"]) for row in read_rows("results.csv") if row["method"] == "baseline")
    plot_spectra()
    plot_benchmark(baseline)
    plot_single_matrix_damage(baseline)
    plot_four_fits(baseline)
    for panel, rho in plot_output_error_vs_perplexity(baseline).items():
        print(f"Spearman, mean output error vs perplexity, {panel}: {rho:+.2f}")
    plot_per_matrix_errors()
    plot_cumulative_cost(baseline)
    print("wrote every figure")
