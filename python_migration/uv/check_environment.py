"""Small, student-friendly check for the separate PyCaret environment (pycaret-core)."""

from sys import version_info
import warnings

import numpy as np
import pandas as pd
import pycaret
from pycaret.time_series import TSForecastingExperiment

warnings.filterwarnings("ignore")


def main() -> None:
    print(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    print(f"[OK] PyCaret (pycaret-core): {pycaret.__version__}")
    print(f"[OK] Time-series tools: {TSForecastingExperiment.__name__}")
    for label, module_name in {"lightgbm": "lightgbm", "xgboost": "xgboost", "catboost": "catboost", "statsmodels": "statsmodels", "sktime": "sktime"}.items():
        module = __import__(module_name)
        print(f"[OK] {label}: {module.__version__}")

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
            raise RuntimeError(f"{name} is not available - was the environment built before the boosting libraries were installed?")
    print("[OK] lightgbm / xgboost / catboost forecasters available")
    print("\nYour PyCaret environment is ready.")


if __name__ == "__main__":
    main()
