"""
df_mlforecast.py
Production-ready MLForecast forecasting module.

This module provides three forecasting strategies for ML models:
1. One-step ahead forecasting (optimistic backtesting)
2. Multi-step recursive forecasting (iterative predictions)
3. Multi-output direct forecasting (one model per horizon - via max_horizon)

Author: Generated for Deep Forecasting Course
Models: XGBoost, LightGBM, RandomForest, CatBoost (if available)
"""

from typing import Dict, List, Literal, Optional, Tuple, Union, Any
import pandas as pd
import numpy as np
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Try to import optional ML libraries
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class MLForecastForecaster:
    """
    Production-ready forecaster using MLForecast with tree-based models.

    Supports XGBoost, LightGBM, RandomForest, CatBoost with three forecasting modes:
    - one_step: One-step ahead forecasting (h=1 iterative)
    - multi_step: Multi-step recursive forecasting (default MLForecast behavior)
    - multi_output: Multi-output direct forecasting (via max_horizon parameter)

    Parameters
    ----------
    model_type : str
        Type of model to use. Options: 'xgboost', 'lightgbm', 'random_forest',
        'catboost', 'linear'
    freq : str
        Frequency of the time series (e.g., 'MS', 'D', 'H')
    lags : list of int, optional
        List of lags to use as features (e.g., [1, 12])
    lag_transforms : dict, optional
        Dictionary of lag transformations (e.g., {'lag': [1], 'rolling_mean': 3})
    date_features : list, optional
        List of date features to extract (e.g., ['month', 'dayofweek'])
    target_transforms : list, optional
        List of target transformations (e.g., [Differences([1])])
    **model_params : dict
        Additional model-specific parameters
    """

    def __init__(
        self,
        model_type: Literal['xgboost', 'lightgbm', 'random_forest', 'catboost', 'linear'],
        freq: str = 'MS',
        lags: Optional[List[int]] = None,
        lag_transforms: Optional[Dict] = None,
        date_features: Optional[List[str]] = None,
        target_transforms: Optional[List] = None,
        **model_params
    ):
        self.model_type = model_type.lower()
        self.freq = freq
        self.lags = lags if lags is not None else [1, 12]
        self.lag_transforms = lag_transforms
        self.date_features = date_features if date_features is not None else []
        self.target_transforms = target_transforms if target_transforms is not None else []
        self.model_params = model_params
        self.mlf = None
        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Validate that the model type is supported and available."""
        if self.model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        elif self.model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
        elif self.model_type == 'catboost' and not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is not installed. Install with: pip install catboost")

    def _create_model(self):
        """Create the ML model instance."""
        if self.model_type == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            return XGBRegressor(**default_params)

        elif self.model_type == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'verbose': -1
            }
            default_params.update(self.model_params)
            return LGBMRegressor(**default_params)

        elif self.model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(self.model_params)
            return RandomForestRegressor(**default_params)

        elif self.model_type == 'catboost':
            default_params = {
                'iterations': 100,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbose': 0
            }
            default_params.update(self.model_params)
            return CatBoostRegressor(**default_params)

        elif self.model_type == 'linear':
            return LinearRegression(**self.model_params)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> pd.DataFrame:
        """
        Prepare data in MLForecast format.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Unique identifier for the series

        Returns
        -------
        pd.DataFrame
            Data in MLForecast format with columns [unique_id, ds, y]
        """
        df = data.copy()

        # Handle index as date column if needed
        if date_col not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: 'ds'})
        elif date_col in df.columns:
            df = df.rename(columns={date_col: 'ds'})

        # Rename target column
        if target_col in df.columns:
            df = df.rename(columns={target_col: 'y'})

        # Add unique_id
        df['unique_id'] = unique_id

        # Ensure correct column order
        df = df[['unique_id', 'ds', 'y']]

        # Ensure ds is datetime
        df['ds'] = pd.to_datetime(df['ds'])

        return df

    def one_step_forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        One-step ahead forecasting (h=1 iterative).

        For each timestamp in test set:
        1. Train on actuals up to t-1
        2. Predict ŷ(t)
        3. Prediction does NOT feed into next step
        4. Proceed iteratively

        This is "optimistic backtesting" - most accurate but not real deployment.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        test_df : pd.DataFrame
            Test data
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Series identifier

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE
        """
        # Prepare data
        train_long = self._prepare_data(train_df, target_col, date_col, unique_id)
        test_long = self._prepare_data(test_df, target_col, date_col, unique_id)

        predictions = []
        actuals = test_long['y'].values
        dates = test_long['ds'].values

        # Iterative one-step forecasting
        for i in range(len(test_long)):
            # Use all actual data up to current point
            if i == 0:
                current_train = train_long.copy()
            else:
                current_train = pd.concat([
                    train_long,
                    test_long.iloc[:i]
                ], ignore_index=True)

            # Create MLForecast model
            model = self._create_model()
            mlf = MLForecast(
                models=[model],
                freq=self.freq,
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=self.date_features,
                target_transforms=self.target_transforms
            )

            # Fit and predict h=1
            mlf.fit(df=current_train)
            forecast = mlf.predict(h=1)

            # Extract prediction (column name depends on model type)
            pred_col = forecast.columns[-1]  # Last column is the prediction
            predictions.append(forecast[pred_col].values[0])

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'unique_id': [unique_id] * len(predictions),
            'ds': dates,
            'y_true': actuals,
            'y_pred': predictions
        })

        # Calculate metrics
        errors = actuals - np.array(predictions)
        metrics = {
            'mae': float(np.mean(np.abs(errors))),
            'rmse': float(np.sqrt(np.mean(errors ** 2))),
            'mape': float(np.mean(np.abs(errors / actuals)) * 100) if np.all(actuals != 0) else np.nan
        }

        return {
            'forecasts': forecast_df,
            'metrics': metrics
        }

    def multi_step_forecast(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1',
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-step recursive forecasting.

        For each timestamp t:
        1. Predict using model
        2. Use previous predictions as input
        3. Iterate through entire horizon

        This simulates real deployment where future actuals are unknown.
        This is the DEFAULT MLForecast behavior when you call predict(h).

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        horizon : int
            Forecast horizon
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Series identifier
        test_df : pd.DataFrame, optional
            Test data for metric calculation

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE (if test_df provided)
        """
        # Prepare data
        train_long = self._prepare_data(train_df, target_col, date_col, unique_id)

        # Create MLForecast model
        model = self._create_model()
        mlf = MLForecast(
            models=[model],
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=self.target_transforms
        )

        # Fit model
        mlf.fit(df=train_long)

        # Recursive forecasting (default behavior)
        forecast = mlf.predict(h=horizon)

        # Extract predictions
        pred_col = forecast.columns[-1]
        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={pred_col: 'y_pred'})

        # Calculate metrics if test data provided
        metrics = None
        if test_df is not None:
            test_long = self._prepare_data(test_df, target_col, date_col, unique_id)

            # Merge with actuals
            forecast_df = forecast_df.merge(
                test_long[['ds', 'y']],
                on='ds',
                how='left'
            )
            forecast_df = forecast_df.rename(columns={'y': 'y_true'})

            # Calculate metrics
            actuals = forecast_df['y_true'].values
            predictions = forecast_df['y_pred'].values

            # Only calculate where we have actuals
            mask = ~pd.isna(actuals)
            if mask.sum() > 0:
                errors = actuals[mask] - predictions[mask]
                metrics = {
                    'mae': float(np.mean(np.abs(errors))),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'mape': float(np.mean(np.abs(errors / actuals[mask])) * 100) if np.all(actuals[mask] != 0) else np.nan
                }

        return {
            'forecasts': forecast_df,
            'metrics': metrics
        }

    def multi_output_forecast(
        self,
        train_df: pd.DataFrame,
        horizon: int,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1',
        test_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Multi-output direct forecasting (one model per horizon).

        Trains H separate models, each specialized to predict t+k horizon directly.
        This is implemented via MLForecast's max_horizon parameter.

        Advantages:
        - No error accumulation (each step predicted independently)
        - Better accuracy for longer horizons

        Disadvantages:
        - More training time (H models vs 1)
        - Requires more data

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data
        horizon : int
            Forecast horizon
        target_col : str
            Name of target column
        date_col : str
            Name of date column
        unique_id : str
            Series identifier
        test_df : pd.DataFrame, optional
            Test data for metric calculation

        Returns
        -------
        dict
            Dictionary with keys:
            - 'forecasts': DataFrame with predictions
            - 'metrics': Dictionary with MAE, MAPE, RMSE (if test_df provided)
        """
        # Prepare data
        train_long = self._prepare_data(train_df, target_col, date_col, unique_id)

        # Create MLForecast model
        model = self._create_model()
        mlf = MLForecast(
            models=[model],
            freq=self.freq,
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=self.target_transforms
        )

        # Fit with max_horizon to train one model per step
        mlf.fit(df=train_long, max_horizon=horizon)

        # Predict using direct strategy
        forecast = mlf.predict(h=horizon)

        # Extract predictions
        pred_col = forecast.columns[-1]
        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={pred_col: 'y_pred'})

        # Calculate metrics if test data provided
        metrics = None
        if test_df is not None:
            test_long = self._prepare_data(test_df, target_col, date_col, unique_id)

            # Merge with actuals
            forecast_df = forecast_df.merge(
                test_long[['ds', 'y']],
                on='ds',
                how='left'
            )
            forecast_df = forecast_df.rename(columns={'y': 'y_true'})

            # Calculate metrics
            actuals = forecast_df['y_true'].values
            predictions = forecast_df['y_pred'].values

            # Only calculate where we have actuals
            mask = ~pd.isna(actuals)
            if mask.sum() > 0:
                errors = actuals[mask] - predictions[mask]
                metrics = {
                    'mae': float(np.mean(np.abs(errors))),
                    'rmse': float(np.sqrt(np.mean(errors ** 2))),
                    'mape': float(np.mean(np.abs(errors / actuals[mask])) * 100) if np.all(actuals[mask] != 0) else np.nan
                }

        return {
            'forecasts': forecast_df,
            'metrics': metrics
        }


def train_test_split_ts(
    data: pd.DataFrame,
    test_size: Union[int, float] = 0.2,
    target_col: str = 'y'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    test_size : int or float
        If int, number of observations for test set
        If float, proportion of data for test set
    target_col : str
        Name of target column

    Returns
    -------
    tuple
        (train_df, test_df)
    """
    if isinstance(test_size, float):
        test_size = int(len(data) * test_size)

    train_df = data.iloc[:-test_size].copy()
    test_df = data.iloc[-test_size:].copy()

    return train_df, test_df


def evaluate_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate forecast evaluation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary with MAE, MAPE, RMSE
    """
    errors = y_true - y_pred

    metrics = {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mape': float(np.mean(np.abs(errors / y_true)) * 100) if np.all(y_true != 0) else np.nan
    }

    return metrics


# Example usage functions
def example_one_step():
    """Example of one-step ahead forecasting with RandomForest."""
    # Load sample data
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    # Split data
    train, test = train_test_split_ts(data, test_size=12)

    # Create forecaster
    forecaster = MLForecastForecaster(
        model_type='random_forest',
        freq='MS',
        lags=[1, 12],
        target_transforms=[Differences([1])],
        n_estimators=200,
        random_state=42
    )

    # One-step ahead forecast
    results = forecaster.one_step_forecast(train, test)

    print("One-step Ahead Forecasting Results (Random Forest):")
    print(f"MAE: {results['metrics']['mae']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")

    return results


def example_multi_step():
    """Example of multi-step recursive forecasting with XGBoost."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available. Using RandomForest instead.")
        model_type = 'random_forest'
    else:
        model_type = 'xgboost'

    # Load sample data
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    # Split data
    train, test = train_test_split_ts(data, test_size=12)

    # Create forecaster
    forecaster = MLForecastForecaster(
        model_type=model_type,
        freq='MS',
        lags=[1, 12],
        target_transforms=[Differences([1])],
        n_estimators=100,
        random_state=42
    )

    # Multi-step forecast
    results = forecaster.multi_step_forecast(
        train,
        horizon=12,
        test_df=test
    )

    print(f"Multi-step Recursive Forecasting Results ({model_type.upper()}):")
    print(f"MAE: {results['metrics']['mae']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")

    return results


def example_multi_output():
    """Example of multi-output direct forecasting with LightGBM."""
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available. Using RandomForest instead.")
        model_type = 'random_forest'
    else:
        model_type = 'lightgbm'

    # Load sample data
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    # Split data
    train, test = train_test_split_ts(data, test_size=12)

    # Create forecaster
    forecaster = MLForecastForecaster(
        model_type=model_type,
        freq='MS',
        lags=[1, 12],
        target_transforms=[Differences([1])],
        n_estimators=100,
        random_state=42
    )

    # Multi-output direct forecast
    results = forecaster.multi_output_forecast(
        train,
        horizon=12,
        test_df=test
    )

    print(f"Multi-output Direct Forecasting Results ({model_type.upper()}):")
    print(f"MAE: {results['metrics']['mae']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")

    return results


if __name__ == "__main__":
    print("="*80)
    print("MLForecast Forecasting Module - Examples")
    print("="*80)

    print("\n1. One-Step Ahead Forecasting:")
    print("-" * 80)
    example_one_step()

    print("\n2. Multi-Step Recursive Forecasting:")
    print("-" * 80)
    example_multi_step()

    print("\n3. Multi-Output Direct Forecasting:")
    print("-" * 80)
    example_multi_output()
