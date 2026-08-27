"""Small, student-friendly check for the main course environment."""

from importlib import import_module
from sys import version_info

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing


PACKAGES = {
    "JupyterLab": "jupyterlab",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "statsmodels": "statsmodels",
    "StatsForecast": "statsforecast",
    "MLForecast": "mlforecast",
    "NeuralForecast": "neuralforecast",
    "TensorFlow": "tensorflow",
    "Keras": "keras",
    "Prophet": "prophet",
    "NeuralProphet": "neuralprophet",
}


def main() -> None:
    print(f"Python {version_info.major}.{version_info.minor}.{version_info.micro}")
    for label, module_name in PACKAGES.items():
        module = import_module(module_name)
        package_version = getattr(module, "__version__", "installed")
        print(f"[OK] {label}: {package_version}")

    observations = np.array([10, 12, 13, 15, 16, 18, 20, 21], dtype=float)
    model = ExponentialSmoothing(observations, trend="add").fit()
    forecast = model.forecast(2)
    if not np.isfinite(forecast).all():
        raise RuntimeError("The sample forecast did not produce finite values.")
    print(f"[OK] Sample forecast: {forecast.round(2).tolist()}")
    print("\nYour main course environment is ready.")


if __name__ == "__main__":
    main()
