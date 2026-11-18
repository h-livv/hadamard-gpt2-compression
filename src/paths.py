"""Where the experiment scripts read and write their files, and the one CSV writer they share.

    results/stage1 .. stage5/  CSV and .npz output of the experiments, one folder per stage
    figures/stage1 .. stage5/  PNG output of experiments/plot.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the repository root (this file is src/paths.py)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def write_csv(rows, path):
    """Write a list of dicts to a CSV file (header from the first row), creating the folder if needed.

    The experiments call this after every run with all rows so far, so an
    interrupted run leaves a valid file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

