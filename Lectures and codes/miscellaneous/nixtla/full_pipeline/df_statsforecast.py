"""
df_statsforecast.py
Production-ready StatsForecast forecasting module.

This module provides three forecasting strategies for statistical models:
1. One-step ahead forecasting (optimistic backtesting)
2. Multi-step recursive forecasting (real deployment simulation)
3. Multi-output forecasting (NOT supported for ARIMA/ETS - raises NotImplementedError)

Author: Generated for Deep Forecasting Course
Models: ARIMA, AutoARIMA, AutoETS, Naive, SeasonalNaive, RandomWalkWithDrift
"""

from typing import Dict, List, Literal, Optional, Tuple, Union
import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import (
    ARIMA,
    AutoARIMA,
    AutoETS,
    Naive,
    SeasonalNaive,
    RandomWalkWithDrift,
)
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mae, mape, rmse
from functools import partial


class StatsforecastForecaster:
    """
    Production-ready forecaster using StatsForecast models.

    Supports ARIMA, AutoETS, and baseline models with three forecasting modes:
    - one_step: One-step ahead forecasting (h=1 iterative)
    - multi_step: Multi-step recursive forecasting (default StatsForecast behavior)
    - multi_output: NOT SUPPORTED for statistical models (raises NotImplementedError)

    Parameters
    ----------
    model_type : str
        Type of model to use. Options: 'arima', 'auto_arima', 'auto_ets',
        'naive', 'seasonal_naive', 'rw_drift'
    freq : str
        Frequency of the time series (e.g., 'MS', 'D', 'H')
    season_length : int, optional
        Seasonal period length (e.g., 12 for monthly data)
    **model_params : dict
        Additional model-specific parameters
    """

    def __init__(
        self,
        model_type: Literal['arima', 'auto_arima', 'auto_ets', 'naive', 'seasonal_naive', 'rw_drift'],
        freq: str = 'MS',
        season_length: int = 12,
        **model_params
    ):
        self.model_type = model_type.lower()
        self.freq = freq
        self.season_length = season_length
        self.model_params = model_params
        self.sf_model = None
        self._validate_model_type()

    def _validate_model_type(self) -> None:
        """Validate that the model type is supported."""
        valid_models = ['arima', 'auto_arima', 'auto_ets', 'naive', 'seasonal_naive', 'rw_drift']
        if self.model_type not in valid_models:
            raise ValueError(
                f"model_type must be one of {valid_models}, got '{self.model_type}'"
            )

    def _create_model(self, alias: str = 'Model'):
        """Create the StatsForecast model instance."""
        if self.model_type == 'arima':
            # Manual ARIMA with user-specified order
            order = self.model_params.get('order', (1, 1, 1))
            seasonal_order = self.model_params.get('seasonal_order', (0, 0, 0))
            include_mean = self.model_params.get('include_mean', True)

            return ARIMA(
                order=order,
                season_length=self.season_length,
                seasonal_order=seasonal_order,
                include_mean=include_mean,
                alias=alias
            )

        elif self.model_type == 'auto_arima':
            # AutoARIMA for automatic model selection
            return AutoARIMA(
                season_length=self.season_length,
                seasonal=self.model_params.get('seasonal', True),
                d=self.model_params.get('d', None),
                D=self.model_params.get('D', None),
                max_p=self.model_params.get('max_p', 5),
                max_q=self.model_params.get('max_q', 5),
                max_P=self.model_params.get('max_P', 2),
                max_Q=self.model_params.get('max_Q', 2),
                stepwise=self.model_params.get('stepwise', True),
                alias=alias
            )

        elif self.model_type == 'auto_ets':
            # AutoETS for exponential smoothing
            model = self.model_params.get('model', 'ZZZ')
            return AutoETS(
                model=model,
                season_length=self.season_length,
                alias=alias
            )

        elif self.model_type == 'naive':
            return Naive(alias=alias)

        elif self.model_type == 'seasonal_naive':
            return SeasonalNaive(
                season_length=self.season_length,
                alias=alias
            )

        elif self.model_type == 'rw_drift':
            return RandomWalkWithDrift(alias=alias)

    def _prepare_data(
        self,
        data: pd.DataFrame,
        target_col: str = 'y',
        date_col: str = 'ds',
        unique_id: str = 'series_1'
    ) -> pd.DataFrame:
        """
        Prepare data in StatsForecast format.

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
            Data in StatsForecast format with columns [unique_id, ds, y]
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

            # Create model and forecast h=1
            model = self._create_model(alias='Model')
            sf = StatsForecast(models=[model], freq=self.freq)

            forecast = sf.forecast(df=current_train, h=1)
            predictions.append(forecast['Model'].values[0])

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

        # Create model and forecast
        model = self._create_model(alias='Model')
        sf = StatsForecast(models=[model], freq=self.freq)

        # This is the default StatsForecast behavior - recursive forecasting
        forecast = sf.forecast(df=train_long, h=horizon)

        forecast_df = forecast.copy()
        forecast_df = forecast_df.rename(columns={'Model': 'y_pred'})

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
        unique_id: str = 'series_1'
    ) -> None:
        """
        Multi-output forecasting - NOT SUPPORTED for statistical models.

        ARIMA/ETS models do NOT support multi-output forecasting.
        They can only do recursive forecasting.

        This method raises NotImplementedError to handle gracefully.

        Raises
        ------
        NotImplementedError
            Always raised - statistical models don't support multi-output
        """
        raise NotImplementedError(
            f"Multi-output forecasting is NOT supported for statistical models "
            f"like {self.model_type.upper()}. Statistical models (ARIMA/ETS) can only "
            f"perform recursive forecasting. Use multi_step_forecast() instead, or "
            f"switch to ML models (mlforecast) or neural models (neuralforecast) "
            f"which support true multi-output forecasting."
        )


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
    """Example of one-step ahead forecasting."""
    # Load sample data
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    # Split data
    train, test = train_test_split_ts(data, test_size=12)

    # Create forecaster
    forecaster = StatsforecastForecaster(
        model_type='auto_arima',
        freq='MS',
        season_length=12
    )

    # One-step ahead forecast
    results = forecaster.one_step_forecast(train, test)

    print("One-step Ahead Forecasting Results:")
    print(f"MAE: {results['metrics']['mae']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")

    return results


def example_multi_step():
    """Example of multi-step recursive forecasting."""
    # Load sample data
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    # Split data
    train, test = train_test_split_ts(data, test_size=12)

    # Create forecaster
    forecaster = StatsforecastForecaster(
        model_type='auto_ets',
        freq='MS',
        season_length=12,
        model='ZZZ'  # Auto selection
    )

    # Multi-step forecast
    results = forecaster.multi_step_forecast(
        train,
        horizon=12,
        test_df=test
    )

    print("Multi-step Recursive Forecasting Results:")
    print(f"MAE: {results['metrics']['mae']:.2f}")
    print(f"RMSE: {results['metrics']['rmse']:.2f}")
    print(f"MAPE: {results['metrics']['mape']:.2f}%")

    return results


def example_multi_output_error():
    """Example showing multi-output is not supported."""
    data = pd.DataFrame({
        'ds': pd.date_range('2020-01-01', periods=144, freq='MS'),
        'y': np.random.randn(144).cumsum() + 100
    })

    train, test = train_test_split_ts(data, test_size=12)

    forecaster = StatsforecastForecaster(
        model_type='arima',
        freq='MS',
        season_length=12,
        order=(1, 1, 1)
    )

    try:
        forecaster.multi_output_forecast(train, horizon=12)
    except NotImplementedError as e:
        print(f"Expected error: {e}")
        return None


if __name__ == "__main__":
    print("="*80)
    print("StatsForecast Forecasting Module - Examples")
    print("="*80)

    print("\n1. One-Step Ahead Forecasting:")
    print("-" * 80)
    example_one_step()

    print("\n2. Multi-Step Recursive Forecasting:")
    print("-" * 80)
    example_multi_step()

    print("\n3. Multi-Output Forecasting (Not Supported):")
    print("-" * 80)
    example_multi_output_error()
