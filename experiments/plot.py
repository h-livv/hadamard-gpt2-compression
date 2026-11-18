"""
Every figure of docs/REPORT.md, drawn from the files in results/, organised by stage.

Reads     results/stage1/results.csv, spectra.npz          (stage 1)
          results/stage2/diagnostics.csv                    (stages 1, 2, 3)
          results/stage3/activation_aware.csv               (stage 3)
          results/stage4/cumulative.csv                     (stage 4)
          results/stage5/*.csv, *.npz                        (stage 5)
Writes    figures/stage1/singular_value_spectra_original.png, perplexity_vs_storage_all24.png
          figures/stage2/single_matrix_damage_by_block.png
          figures/stage3/perplexity_vs_model_size_blocks2-11.png,
                         output_error_vs_perplexity_compressed_together.png,
                         per_matrix_errors_by_block_r64-256.png
          figures/stage4/cumulative_cost_per_block_r64-256.png
          figures/stage5/fig1_spectra.png ... fig7_gpt2_curves.png
          It also prints the Spearman correlations quoted in the report.
Runtime   about 20 seconds.

Source: matplotlib pyplot API, https://matplotlib.org/stable/api/pyplot_summary.html
(plain plotting; no method from the paper is involved).

Run:  python experiments/plot.py
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repository root, so `src` imports

from src.methods import GPT2_TOTAL_PARAMS, model_megabytes, model_params
from src.metrics import spearman
from src.paths import FIGURES_DIR, RESULTS_DIR

# ---- stages 1-4 (GPT-2) ---------------------------------------------------
FULL_PARAMS = 768 * 3072         # numbers in one uncompressed MLP matrix
COLORS = {"svd": "#2a78d6", "hadamard": "#eb6834"}           # blue, orange
MARKERS = {"svd": "o", "hadamard": "s"}
FAMILY = {"svd": "svd", "hadamard": "hadamard", "svd_w": "svd", "hadamard_w": "hadamard"}
NAMES = {"svd": "SVD, weight error", "hadamard": "Hadamard, weight error",
         "svd_w": "Whitened SVD (activation-weighted)", "hadamard_w": "Weighted Hadamard (activation-weighted)"}
INK, MUTED, GRID = "#1f1e1c", "#52514e", "#e4e3df"


# ---- stage 5 (single matrices) ----------------------------------------------
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

def share(r):
    """Percent of one MLP matrix stored at rank r: 2r(m+n) / (m n)."""
    return 100 * 2 * r * (768 + 3072) / FULL_PARAMS


def style(ax):
    """Recessive grid, no top/right spines."""
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def baseline_line(ax, baseline, text=None):
    """Dashed line at the uncompressed perplexity, labelled at the right end, under the line."""
    ax.axhline(baseline, color=MUTED, linewidth=1, linestyle="--")
    ax.annotate(text or f"uncompressed: {baseline:.1f}", xy=(1, baseline), xycoords=("axes fraction", "data"),
                xytext=(-4, -12), textcoords="offset points", ha="right", color=MUTED)

def result(stage, name):
    """Path of a result file of a stage."""
    return RESULTS_DIR / f"stage{stage}" / name


def read_rows(stage, name):
    """Rows of a stage's CSV, all values as strings."""
    with open(result(stage, name), newline="") as f:
        return list(csv.DictReader(f))


def read_numeric(stage, name):
    """Rows of a stage's CSV, with every value that parses as a number converted to float."""
    def conv(v):
        try:
            return float(v)
        except ValueError:
            return v
    return [{k: conv(v) for k, v in row.items()} for row in read_rows(stage, name)]


def pick(rows, **where):
    return [r for r in rows if all(r[k] == v for k, v in where.items())]


def save(fig, stage, name):
    """Save a figure to figures/stage<stage>/<name> and close it."""
    folder = FIGURES_DIR / f"stage{stage}"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/stage{stage}/{name}")


# ---- stages 1-4 figures -----------------------------------------------------

def plot_benchmark(baseline):
    """All 24 matrices, weight-error fits: perplexity against share stored, r = 8..256 (log x: r doubles)."""
    table = {}
    for row in read_rows(1, "results.csv"):
        if row["method"] != "baseline":
            table[(row["method"], int(row["r"]))] = float(row["perplexity"])
    for row in read_rows(2, "diagnostics.csv"):               # r = 64 is in both files, with the same value
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
    save(fig, 1, "perplexity_vs_storage_all24.png")


def plot_single_matrix_damage(baseline):
    """Heatmap: perplexity with only one matrix compressed, rows = block, columns = method x r."""
    ppl = {}
    for row in read_rows(2, "diagnostics.csv"):
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
    save(fig, 2, "single_matrix_damage_by_block.png")


def plot_spectra():
    """sigma_i / sigma_1 for all 24 matrices: one panel per matrix type, one line per layer."""
    spectra = np.load(result(1, "spectra.npz"))
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
    save(fig, 1, "singular_value_spectra_original.png")


def plot_four_fits(baseline):
    """Blocks 2-11: the four fits, perplexity against the stored size of the whole model."""
    table = {(row["method"], int(row["r"])): float(row["perplexity"]) for row in read_rows(3, "activation_aware.csv")}
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
    save(fig, 3, "perplexity_vs_model_size_blocks2-11.png")


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
    panels = [("All 24 matrices, weight-error fits", run_means(read_rows(2, "diagnostics.csv"), "perplexity_all")),
              ("Blocks 2-11, four fits", run_means(read_rows(3, "activation_aware.csv"), "perplexity"))]
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
    save(fig, 3, "output_error_vs_perplexity_compressed_together.png")
    return correlations


def plot_per_matrix_errors():
    """Weight and output error of each compressed matrix, for the two activation-weighted fits.

    Small multiples: rows c_fc / c_proj, columns r. One point per block (2..11).
    Only errors are shown; no single-matrix perplexity was measured for these fits.
    """
    data = {}
    for row in read_rows(3, "activation_aware.csv"):
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
    save(fig, 3, "per_matrix_errors_by_block_r64-256.png")


def plot_cumulative_cost(baseline):
    """Bars: % increase in perplexity when each block is added (stage 4), one panel per r."""
    series = {}
    for row in read_rows(4, "cumulative.csv"):
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
    save(fig, 4, "cumulative_cost_per_block_r64-256.png")


# ---- stage 5 figures --------------------------------------------------------

def stage5_spectra():
    spectra = np.load(result(5, "spectra.npz"))
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
    style(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, 5, "fig1_spectra.png")


def stage5_plain_error():
    rows = read_numeric(5, "s51_spectrum.csv")
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
        style(ax)
    axes[0].set_ylabel("relative plain error ‖A − Â‖/‖A‖")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("At equal storage SVD(2r) beats HadNet(r) on the image at every n and share; "
                 "on the flat-spectrum control HadNet(r) wins at every share", fontsize=11, y=1.03)
    save(fig, 5, "fig2_plain_error.png")


def stage5_weighted_n256():
    rows = read_numeric(5, "s52_weightings.csv")
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
        style(ax)
    axes[0].set_ylabel("relative weighted error ‖Sᵀ(A − Â)‖/‖SᵀA‖")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("n = 256: whitened SVD has the lowest weighted error under every weighting and share",
                 fontsize=11, y=1.03)
    save(fig, 5, "fig3_weighted_n256.png")


def stage5_gap_vs_kappa():
    rows = read_numeric(5, "s52_weightings.csv")
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
        style(ax)
    axes[0].set_ylabel("weighted HadNet − whitened SVD (% of whitened SVD's error)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Weighted HadNet (paper settings) never reaches whitened SVD: the gap is set mainly by the share, ~100% at 50%",
                 fontsize=11, y=1.03)
    save(fig, 5, "fig4_gap_vs_kappa.png")


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
        style(ax)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=11, y=1.03)
    save(fig, 5, name)


def stage5_hardest_curves():
    rows = pick(read_numeric(5, "s53_fix.csv"), n=256, share=50.0, weighting="freq1000")
    curves = np.load(result(5, "curves_s5x.npz"))
    refs = [(rows[0]["ref_wsvd_2r"], "whitened SVD, rank 2r", "--"),
            (rows[0]["ref_hadnet"], "plain HadNet (its weighted error)", ":")]
    loss_curve_panels(rows, curves, refs,
                      "n = 256, 50% share, κ = 1000: every start and learning rate stays ≥ 2× above whitened SVD; the embedding never leaves rank r",
                      "fig5_loss_curves_hardest.png", "relative weighted error (dimensionless)",
                      lambda r: r["start_loss"] / r["start_rel_werr"])


def stage5_reconstructions():
    recon = np.load(result(5, "reconstructions.npz"))
    rows = pick(read_numeric(5, "s52_weightings.csv"), n=256, share=12.5)
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
    save(fig, 5, "fig6_reconstructions.png")


def stage5_gpt2_curves():
    rows = read_numeric(5, "s55_gpt2.csv")
    curves = np.load(result(5, "curves_s55.npz"))
    wh = pick(rows, method="whadnet")
    refs = [(pick(rows, method="wsvd")[0]["rel_werr_Gfit"], "whitened SVD, rank 2r", "--"),
            (pick(rows, method="hadnet")[0]["rel_werr_Gfit"], "plain HadNet (its weighted error)", ":")]
    # rel_werr_Gfit of the best factors = best_loss / ||S^T A||, so the norm is best_loss / rel_werr_Gfit
    loss_curve_panels(wh, curves, refs,
                      "GPT-2 block 5 c_proj, r = 128: at lr 0.1 no start improves; at lr ≤ 0.03 the paper and warm starts beat whitened SVD",
                      "fig7_gpt2_curves.png", "relative weighted error under G_fit (dimensionless)",
                      lambda r: r["best_loss"] / r["rel_werr_Gfit"])


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9})
    baseline = next(float(row["perplexity"]) for row in read_rows(1, "results.csv") if row["method"] == "baseline")
    plot_spectra()                                   # stage 1
    plot_benchmark(baseline)                         # stage 1 (with stage 2's larger budgets)
    plot_single_matrix_damage(baseline)              # stage 2
    plot_four_fits(baseline)                         # stage 3
    for panel, rho in plot_output_error_vs_perplexity(baseline).items():
        print(f"Spearman, mean output error vs perplexity, {panel}: {rho:+.2f}")
    plot_per_matrix_errors()                         # stage 3
    plot_cumulative_cost(baseline)                   # stage 4
    stage5_spectra()
    stage5_plain_error()
    stage5_weighted_n256()
    stage5_gap_vs_kappa()
    stage5_hardest_curves()
    stage5_reconstructions()
    stage5_gpt2_curves()
