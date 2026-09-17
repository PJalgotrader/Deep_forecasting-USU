"""Student-friendly check for the course's single local environment (Modules 2-5).

Run from the repository root:

    uv run python scripts/check_environment.py
"""

from importlib import import_module
from sys import version_info
import warnings

warnings.filterwarnings("ignore")

PACKAGES = {
    "JupyterLab": "jupyterlab",
    "pandas": "pandas",
    "NumPy": "numpy",
    "scikit-learn": "sklearn",
    "statsmodels": "statsmodels",
    "sktime": "sktime",
    "PyCaret (pycaret-core)": "pycaret",
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "seaborn": "seaborn",
    "yfinance": "yfinance",
}


def main() -> None:
    print(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    if (version_info.major, version_info.minor) != (3, 13):
        raise RuntimeError("This environment should run Python 3.13. Run `uv sync` from the repository root.")
    for label, module_name in PACKAGES.items():
        module = import_module(module_name)
        print(f"[OK] {label}: {getattr(module, '__version__', 'installed')}")

    import numpy as np
    import pandas as pd
    from pycaret.time_series import TSForecastingExperiment

    observations = pd.Series(
        20 + np.arange(48) * 0.4 + np.sin(np.arange(48) * 2 * np.pi / 12),
        index=pd.period_range("2022-01", periods=48, freq="M"),
    )
    experiment = TSForecastingExperiment()
    experiment.setup(data=observations, fh=3, session_id=123, verbose=False)
    # 'ets' is the model that breaks when statsmodels >= 0.15 sneaks in, so it is the one we test.
    model = experiment.create_model("ets", cross_validation=False, verbose=False)
    forecast = experiment.predict_model(model, verbose=False)
    if len(forecast) != 3 or not np.isfinite(forecast["y_pred"]).all():
        raise RuntimeError("The PyCaret sample forecast did not complete correctly.")
    print(f"[OK] Sample ETS forecast: {forecast['y_pred'].round(2).tolist()}")

    available = set(experiment.models().index)
    for name in ("lightgbm_cds_dt", "xgboost_cds_dt", "catboost_cds_dt"):
        if name not in available:
            raise RuntimeError(f"{name} is not available. Delete .venv and run `uv sync` again.")
    print("[OK] LightGBM / XGBoost / CatBoost forecasters available")
    print("\nYour course environment is ready.")


if __name__ == "__main__":
    main()
