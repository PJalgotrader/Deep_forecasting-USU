# Python migration - running the PyCaret notebooks in 2026

**Why this folder exists.** The course notebooks (Modules 3, 4, 5, "All in one", and the stock-market examples) are built on `pycaret.time_series`. The last official release, PyCaret 3.3.2 (April 2024), hard-codes a check that raises `RuntimeError` on Python >= 3.12. Google Colab now runs Python 3.13, so `!pip install pycaret` no longer works there. The full write-up, with the timeline and the traps we hit, is [`tutorial.html`](tutorial.html).

**The fix.** Install the community fork **pycaret-core 3.5.0** (maintained by the sktime organisation, MIT). It is a drop-in: same `import pycaret`, same `TSForecastingExperiment`, same `setup / create_model / compare_models / plot_model / predict_model / finalize_model / save_model` calls. The only change in every notebook is the install line, plus one temporary pin, `statsmodels<0.15`.

**Not** PyCaret 4.0 (pre-release): it is a redesign that drops the functional API, renames the experiment class, and removes `plot_model` and `check_stats`.

## Pick one way to run the notebooks

| Option | Folder | Best for |
|---|---|---|
| 1. Google Colab | [`colab/`](colab/) | zero install, any laptop |
| 2. uv (locked) | [`uv/`](uv/) | local work, exact reproducibility; JupyterLab or VS Code |
| 3. conda | [`conda/`](conda/) | students already on Anaconda; JupyterLab or VS Code |

Options 2 and 3 both work in **JupyterLab and VS Code** (Python + Jupyter extensions); each README has the kernel-selection steps.

All three install the same thing: Python 3.13, `pycaret-core==3.5.0`, `statsmodels<0.15`, `lightgbm`, `xgboost`, `catboost`.


## Folder map

- `colab/`, `uv/`, `conda/` - the three options above (`uv/` is the locked environment; `uv/check_environment.py` verifies it; `colab/colab_tests/` holds the executed Colab test notebooks)
- `colab/colab_tests/` - the three Colab notebooks used to test official 3.3.2 vs pycaret-core 3.5.0 vs pycaret 4.0 pre-release (executed outputs included)
- `tutorial.html` - the tutorial page: what broke, what we tested, the traps, how to run
- New to uv? [`Platforms and tools/uv/conda_to_uv_student_cheatsheet.html`](../Platforms%20and%20tools/uv/conda_to_uv_student_cheatsheet.html)
