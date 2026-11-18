# Weighted Hadamard decomposition on single matrices (v2)

An independent study of the method of Panchal, Sahoo and Behera, [*Tensor decomposition and neural architectures through generalized Hadamard products*](https://doi.org/10.1016/j.neucom.2026.134372), Neurocomputing 699 (2026) 134372. Not affiliated with the authors.

**Version 2** compares four ways of compressing a single matrix at **equal storage**: truncated SVD at rank 2r, the paper's Hadamard decomposition (HadNet) at rank r, and the same two fitted to a *weighted* error ‖Sᵀ(A − Â)‖_F (whitened SVD, from SVD-LLM, and weighted HadNet, this project's extension). The main test matrix is a natural image, where every quantity can be inspected directly, alongside a random-matrix control and one GPT-2 matrix from v1.

- **Report:** [`docs/REPORT.md`](docs/REPORT.md), with method, verification, results, conclusions and how to reproduce each number.
- **Version 1** (GPT-2 small, perplexity): frozen in [`archive/v1-gpt2/`](archive/v1-gpt2/), and reproduced exactly by `git checkout v1-weight-error`.

## How to run it

Python 3.12. The image experiments run on the CPU in float64. The GPT-2 part needs about 6 GB of GPU memory, and downloads GPT-2 small and WikiText-2 from the Hugging Face Hub on first use. Times are for a laptop (8 CPU threads, RTX 4050).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/checks.py                      # 15 checks, ~6 s; must end with "All checks passed."
python experiments/single_image.py          # REPORT §5.1-5.4, all image fits      (~27 min)
python experiments/gpt2_single_matrix.py    # REPORT §5.5, block 5 c_proj, r = 128 (~3 min)
python experiments/plot_v2.py               # figures/v2/, every figure            (~10 s)
```

Settings are constants at the top of each script. Every number in the report comes from a CSV in `results/`.

## Layout

```
src/
  hadamard.py   the methods: SVD init (paper §3.3), HadNet (Algorithm 7, optional weighted loss and start),
                truncated SVD, whitened SVD (SVD-LLM), rank-r embedding
  weights.py    the weightings G: identity, diagonal region, DCT frequency with condition number κ
  metrics.py    relative plain and weighted error, PSNR, condition number (plus v1's measures)
  gpt2.py       GPT-2 weights, calibration Gram matrices, damping (v1 code)
  paths.py      locations of results/ and figures/
experiments/    single_image.py, gpt2_single_matrix.py, plot_v2.py
tests/          checks.py
results/        one CSV per experiment, loss curves (.npz), spectra, reconstructions
figures/v2/     the seven figures of the report
docs/           REPORT.md
archive/v1-gpt2/  version 1, frozen
```

## Sources

- **Method:** Panchal, Sahoo, Behera, Neurocomputing 699 (2026) 134372, <https://doi.org/10.1016/j.neucom.2026.134372>; code <https://github.com/PanchalAneesh/HadNet-HadDNN>. `src/hadamard.py` re-implements the single-matrix case; no file is copied.
- **Whitened SVD:** SVD-LLM (Wang, Zheng, Wan, Zhang, ICLR 2025), <https://arxiv.org/abs/2403.07378>; code <https://github.com/AIoT-MLSys-Lab/SVD-LLM>.
- **Image:** `skimage.data.astronaut()`, a public-domain NASA photograph shipped with scikit-image.
