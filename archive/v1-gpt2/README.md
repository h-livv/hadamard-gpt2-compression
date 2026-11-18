# Archive: v1, GPT-2 small MLP compression (frozen)

This folder holds version 1 of the project: Hadamard decomposition (HadNet) against truncated SVD
on the 24 MLP matrices of GPT-2 small, first fitted to plain weight error and then to an
activation-weighted error, with WikiText-2 perplexity as the outcome.

**v1 is frozen.** Nothing here is maintained, and the scripts are not runnable from this folder:
they import `src.methods`, which now lives here as `src/methods.py`, and they write to the old
top-level `results/` and `figures/`. To reproduce v1 exactly, check out the tag:

```
git checkout v1-weight-error
```

and follow that version's `README.md` (kept here as `README_v1.md`).

| Path | What it was |
|---|---|
| `README_v1.md` | v1's top-level README |
| `docs/RESULTS.md`, `docs/VERIFICATION.md` | v1 results and per-function sources |
| `experiments/` | the five v1 scripts (benchmark, diagnostics, activation-aware, cumulative, plots) |
| `results/`, `figures/` | v1 outputs |
| `src/methods.py` | v1's dispatchers and GPT-2 size accounting; `damped` moved to `src/gpt2.py` in v2 |

Files were moved with `git mv`, so `git log --follow <file>` shows their full history.
Version 2 (single matrices) is described in the top-level `README.md` and `docs/REPORT.md`.
