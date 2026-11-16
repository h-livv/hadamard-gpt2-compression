"""Where the experiment scripts read and write their files.

    results/  CSV and .npz output of the experiments
    figures/  PNG output of experiments/plot.py
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the repository root (this file is src/paths.py)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

