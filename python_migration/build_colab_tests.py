"""Builds the three Colab test notebooks (run each in a FRESH Colab runtime)."""
import nbformat as nbf, pathlib
out = pathlib.Path(__file__).parent / "colab_tests"

ENV_CELL = '''import sys, importlib.metadata as md
print("Python:", sys.version.split()[0])
for p in ["numpy","pandas","scikit-learn","scipy","statsmodels","sktime","matplotlib","lightgbm","xgboost","catboost","pycaret","pycaret-core"]:
    try: print(f"  {p:14s} {md.version(p)}")
    except md.PackageNotFoundError: print(f"  {p:14s} (not installed)")'''

SMOKE = '''import warnings; warnings.filterwarnings("ignore")
import pycaret; print("pycaret.__version__ =", pycaret.__version__)
from pycaret.utils import version; print("pycaret.utils.version() =", version())
from pycaret.datasets import get_data
from pycaret.time_series import *

data = get_data("airline")
exp = TSForecastingExperiment()
exp.setup(data=data, fh=12, coverage=0.90, session_id=42)
print("models available:", len(exp.models()), "| catboost_cds_dt:", "catboost_cds_dt" in exp.models().index)
ets = exp.create_model("ets", cross_validation=False)          # the statsmodels-0.15 trap
best = exp.compare_models()                                     # every available model (turbo=True skips the slowest)
exp.plot_model(best, plot="forecast", data_kwargs={"fh": 24})
exp.predict_model(best)
final = exp.finalize_model(best)
exp.save_model(final, "colab_test_model"); _ = load_model("colab_test_model")
print()
print("ALL STEPS PASSED")'''

def nb(title, intro, cells, fname):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}\n\n**Run this in a fresh Colab runtime** (Runtime → Disconnect and delete runtime, then Run all).")]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nbf.write(n, out / fname); print("wrote", out / fname)

nb("Colab test A — official pycaret 3.3.2",
   "Expected: install succeeds, `import pycaret` raises RuntimeError on Python 3.12.",
   [("md","## 1. Runtime before install"),("code",ENV_CELL),
    ("md","## 2. Install the official stable release"),("code","!pip install -q pycaret==3.3.2"),
    ("md","## 3. Import — this is where Colab students get stuck"),("code","import pycaret\nprint(pycaret.__version__)")],
   "A_pycaret_3.3.2.ipynb")

nb("Colab test B — pycaret-core 3.5.0 (community fork)",
   "Expected: install succeeds without touching numpy/pandas/scikit-learn on Python 3.13 (no restart), all notebook API calls work unchanged, and the lightgbm / xgboost / catboost forecasters train.",
   [("md","## 1. Runtime before install"),("code",ENV_CELL),
    ("md","## 2. Install the fork\n\n`statsmodels<0.15` is required until sktime ships the fix merged 2026-09-13 (sktime PR 10972)."),
    ("code",'!pip install -q pycaret-core "statsmodels<0.15" lightgbm xgboost catboost'),
    ("md","## 3. Runtime after install\n\nIf Colab shows a *Restart session* banner, restart and continue from here."),("code",ENV_CELL),
    ("md","## 4. The same API the course notebooks use"),("code",SMOKE)],
   "B_pycaret_core_3.5.0.ipynb")

nb("Colab test C — pycaret 4.0 pre-release",
   "Expected: install succeeds, but `TSForecastingExperiment` and the functional API no longer exist.",
   [("md","## 1. Runtime before install"),("code",ENV_CELL),
    ("md","## 2. Install the pre-release"),("code",'!pip install -q --pre "pycaret[timeseries]"'),
    ("md","## 3. What survived from 3.x?"),
    ("code",'import pycaret, pycaret.time_series as ts\nprint("pycaret", pycaret.__version__)\nprint("exports:", [n for n in dir(ts) if not n.startswith("_")])\nprint("TSForecastingExperiment:", hasattr(ts, "TSForecastingExperiment"))\nprint("functional setup():   ", hasattr(ts, "setup"))'),
    ("md","## 4. First cell of every course notebook"),("code","from pycaret.time_series import *\nexp = TSForecastingExperiment()")],
   "C_pycaret_4.0_pre.ipynb")
