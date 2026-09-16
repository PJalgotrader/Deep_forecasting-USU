# Option 2 - uv (recommended for local work)

New to uv? Read [`Platforms and tools/uv/conda_to_uv_student_cheatsheet.html`](../../Platforms%20and%20tools/uv/conda_to_uv_student_cheatsheet.html) first.

Requires [uv](https://docs.astral.sh/uv/): `winget install astral-sh.uv` (Windows), `brew install uv` (macOS), or `curl -LsSf https://astral.sh/uv/install.sh | sh` (Linux/macOS).

## 1. Create the environment (once)

```bash
cd python_migration/uv
uv sync
```

`uv sync` downloads Python 3.13 if you do not have it, creates `.venv/` here, and installs exactly the versions in `uv.lock`. Everyone gets the same packages.

## 2a. Run notebooks in JupyterLab

```bash
uv run jupyter lab
```

## 2b. Run notebooks in VS Code

1. Install the **Python** and **Jupyter** extensions (Microsoft).
2. Register the environment as a Jupyter kernel (once):

   ```bash
   uv run python -m ipykernel install --user --name df-pycaret --display-name "Python (df-pycaret)"
   ```

3. Open any course notebook, click **Select Kernel** (top right), choose **Jupyter Kernel**, then **Python (df-pycaret)**.

The kernel shows up in every VS Code window from now on, whatever folder the notebook is in.

Alternative without registering a kernel: **Select Kernel** > **Python Environments** > **Enter interpreter path** and pick `python_migration/uv/.venv/Scripts/python.exe` (Windows) or `python_migration/uv/.venv/bin/python` (macOS/Linux). VS Code also auto-detects `.venv` if you open the `python_migration/uv` folder itself as the workspace.

## Maintenance

- `uv.lock` pins every package; do not edit it by hand. To move one package: `uv lock --upgrade-package <name>`, then rerun the smoke tests in `../evidence/`.
- To remove the environment, delete `.venv/` and, if you registered the kernel, run `jupyter kernelspec uninstall df-pycaret`.
