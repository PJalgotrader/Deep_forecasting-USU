# Option 3 - conda / Anaconda

## 1. Create the environment (once)

```bash
cd python_migration/conda
conda env create -f environment.yml
conda activate df_pycaret
```

Everything except Python and Jupyter is installed with pip inside the env, because `pycaret-core` is not packaged on conda-forge. The pins match the uv option, but conda does not lock transitive packages, so the uv option is the reproducible one.

## 2a. Run notebooks in JupyterLab

```bash
conda activate df_pycaret
jupyter lab
```

## 2b. Run notebooks in VS Code

1. Install the **Python** and **Jupyter** extensions (Microsoft).
2. Open any course notebook, click **Select Kernel** (top right), choose **Python Environments**, then **df_pycaret**. VS Code discovers conda environments automatically; `ipykernel` is already installed in the env.

If `df_pycaret` does not appear, reload the window (Ctrl+Shift+P > *Developer: Reload Window*) or register the kernel explicitly:

```bash
conda activate df_pycaret
python -m ipykernel install --user --name df_pycaret --display-name "Python (df_pycaret)"
```

Then pick **Jupyter Kernel** > **Python (df_pycaret)** in the kernel picker.

## Maintenance

- Remove with `conda env remove -n df_pycaret` (and `jupyter kernelspec uninstall df_pycaret` if you registered it).
