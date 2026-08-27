"""Small, student-friendly check for the separate PyCaret environment."""

from sys import version_info

import numpy as np
import pandas as pd
import pycaret
from pycaret.time_series import TSForecastingExperiment


def main() -> None:
    print(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    print(f"[OK] PyCaret: {pycaret.__version__}")
    print(f"[OK] Time-series tools: {TSForecastingExperiment.__name__}")

    observations = pd.Series(
        20 + np.arange(36) * 0.4 + np.sin(np.arange(36) * 2 * np.pi / 12),
        index=pd.date_range("2023-01-01", periods=36, freq="MS"),
    )
    experiment = TSForecastingExperiment()
    experiment.setup(data=observations, fh=3, session_id=123, verbose=False)
    model = experiment.create_model("naive", verbose=False)
    forecast = experiment.predict_model(model, verbose=False)
    if len(forecast) != 3 or not np.isfinite(forecast["y_pred"]).all():
        raise RuntimeError("The PyCaret sample forecast did not complete correctly.")
    print(f"[OK] Sample forecast: {forecast['y_pred'].round(2).tolist()}")
    print("\nYour PyCaret environment is ready.")


if __name__ == "__main__":
    main()
