# Option 2 - uv (recommended for local work)

**The uv environment for this course is the repository root.** There is nothing to install from this folder.

New to uv? Read [`Platforms and tools/uv/conda_to_uv_student_cheatsheet.html`](../../Platforms%20and%20tools/uv/conda_to_uv_student_cheatsheet.html) first.

From the top folder of the repository:

```bash
uv sync                                        # downloads Python 3.13 if needed + installs the locked packages (~1.3 GB)
uv run python scripts/check_environment.py     # should end with "Your course environment is ready."
uv run jupyter lab
```

It covers Modules 2–5: PyCaret (`pycaret-core`), statsmodels, sktime, LightGBM, XGBoost, CatBoost and the notebook utilities. Modules 6–7 run on Google Colab; Module 8 has its own setup (see the root [`README.md`](../../README.md)).

## VS Code

Register the environment once as a named kernel, then choose it from **Select Kernel → Jupyter Kernel** in any notebook:

```bash
uv run python -m ipykernel install --user --name deep-forecasting --display-name "Python 3.13 (Deep Forecasting)"
```

## Maintenance

- `uv.lock` pins every package; do not edit it by hand. To move one package: `uv lock --upgrade-package <name>`, `uv sync`, then re-run `scripts/check_environment.py`.
- `python-preference = "only-managed"` in `pyproject.toml` is deliberate: on Windows, uv would otherwise build the environment on an Anaconda Python found on PATH, which breaks compiled packages inside notebooks.
- To remove the environment, delete `.venv/` at the repository root and, if you registered the kernel, run `jupyter kernelspec uninstall deep-forecasting`.
