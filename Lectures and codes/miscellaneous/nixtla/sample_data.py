"""
Sample Time Series Datasets for Streamlit Forecasting App

This module provides pre-loaded example datasets for testing and demonstration
of the forecasting modules (StatsForecast, MLForecast, NeuralForecast).
"""

import pandas as pd
import numpy as np
from typing import Tuple


def get_airpassengers() -> pd.DataFrame:
    """
    Classic AirPassengers dataset (1949-1960)
    Monthly totals of international airline passengers.

    Characteristics:
    - Frequency: Monthly (MS)
    - Season length: 12
    - Trend: Strong upward
    - Seasonality: Strong yearly pattern
    - 144 observations

    Returns:
        DataFrame with columns: ds (datetime), y (passengers count)
    """
    # AirPassengers data (1949-1960)
    passengers = [
        112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,  # 1949
        115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,  # 1950
        145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,  # 1951
        171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,  # 1952
        196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,  # 1953
        204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,  # 1954
        242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,  # 1955
        284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,  # 1956
        315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,  # 1957
        340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,  # 1958
        360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,  # 1959
        417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432   # 1960
    ]

    dates = pd.date_range(start='1949-01-01', periods=len(passengers), freq='MS')

    return pd.DataFrame({
        'ds': dates,
        'y': passengers
    })


def get_energy_consumption() -> pd.DataFrame:
    """
    Synthetic Daily Energy Consumption Dataset
    Simulates household energy usage patterns.

    Characteristics:
    - Frequency: Daily (D)
    - Season length: 7 (weekly pattern)
    - Trend: Slight upward
    - Seasonality: Weekly pattern (higher on weekdays)
    - 365 observations (1 year)

    Returns:
        DataFrame with columns: ds (datetime), y (kWh consumption)
    """
    np.random.seed(42)

    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')

    # Base consumption
    base = 50

    # Trend component (slight increase over time)
    trend = np.linspace(0, 10, 365)

    # Weekly seasonality (higher on weekdays, lower on weekends)
    weekly_pattern = np.array([1.1, 1.15, 1.2, 1.15, 1.1, 0.8, 0.75])  # Mon-Sun multipliers
    seasonality = np.tile(weekly_pattern, 53)[:365]  # Tile 53 times to ensure we have enough

    # Random noise
    noise = np.random.normal(0, 5, 365)

    # Combine components
    consumption = base + trend + (base * (seasonality - 1)) + noise

    return pd.DataFrame({
        'ds': dates,
        'y': consumption
    })


def get_retail_sales() -> pd.DataFrame:
    """
    Synthetic Weekly Retail Sales Dataset
    Simulates retail store sales with seasonal patterns.

    Characteristics:
    - Frequency: Weekly (W)
    - Season length: 52 (yearly pattern)
    - Trend: Strong upward
    - Seasonality: Yearly pattern (higher in Q4 for holidays)
    - 156 observations (3 years)

    Returns:
        DataFrame with columns: ds (datetime), y (sales amount)
    """
    np.random.seed(123)

    dates = pd.date_range(start='2021-01-01', periods=156, freq='W')

    # Base sales
    base = 10000

    # Trend component (growing business)
    trend = np.linspace(0, 5000, 156)

    # Yearly seasonality (higher in Nov-Dec for holidays)
    # Create pattern for one year (52 weeks) and repeat
    yearly_pattern = np.concatenate([
        np.ones(39) * 0.9,      # Weeks 1-39: normal
        np.ones(4) * 1.2,       # Weeks 40-43: back to school bump
        np.ones(5) * 1.0,       # Weeks 44-48: normal
        np.ones(4) * 1.5        # Weeks 49-52: holiday season
    ])
    seasonality = np.tile(yearly_pattern, 3)[:156]

    # Random noise
    noise = np.random.normal(0, 800, 156)

    # Combine components
    sales = base + trend + (base * (seasonality - 1)) + noise

    return pd.DataFrame({
        'ds': dates,
        'y': sales
    })


def get_temperature() -> pd.DataFrame:
    """
    Synthetic Daily Temperature Dataset
    Simulates daily average temperature with strong seasonality.

    Characteristics:
    - Frequency: Daily (D)
    - Season length: 365 (yearly pattern)
    - Trend: None (stationary mean)
    - Seasonality: Strong yearly pattern (summer peaks, winter lows)
    - 730 observations (2 years)

    Returns:
        DataFrame with columns: ds (datetime), y (temperature in °F)
    """
    np.random.seed(456)

    dates = pd.date_range(start='2022-01-01', periods=730, freq='D')

    # Base temperature
    base = 55

    # Yearly seasonality using sine wave
    days = np.arange(730)
    seasonality = 25 * np.sin(2 * np.pi * days / 365 - np.pi/2)  # Peak in summer

    # Random noise (weather variability)
    noise = np.random.normal(0, 5, 730)

    # Combine components
    temperature = base + seasonality + noise

    return pd.DataFrame({
        'ds': dates,
        'y': temperature
    })


# Dataset metadata for Streamlit UI
DATASETS = {
    'AirPassengers (Monthly, 1949-1960)': {
        'function': get_airpassengers,
        'freq': 'MS',
        'season_length': 12,
        'description': 'Classic airline passengers dataset with strong trend and seasonality',
        'recommended_horizon': 12
    },
    'Energy Consumption (Daily, 2023)': {
        'function': get_energy_consumption,
        'freq': 'D',
        'season_length': 7,
        'description': 'Synthetic household energy usage with weekly patterns',
        'recommended_horizon': 14
    },
    'Retail Sales (Weekly, 2021-2023)': {
        'function': get_retail_sales,
        'freq': 'W',
        'season_length': 52,
        'description': 'Synthetic retail sales with yearly seasonality and holiday spikes',
        'recommended_horizon': 8
    },
    'Temperature (Daily, 2022-2023)': {
        'function': get_temperature,
        'freq': 'D',
        'season_length': 365,
        'description': 'Synthetic temperature data with strong yearly patterns',
        'recommended_horizon': 30
    }
}


def get_sample_data(dataset_name: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load a sample dataset by name.

    Args:
        dataset_name: Name of the dataset (must be in DATASETS keys)

    Returns:
        Tuple of (DataFrame, metadata dict)

    Raises:
        ValueError: If dataset_name is not found
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DATASETS.keys())}")

    dataset_info = DATASETS[dataset_name]
    df = dataset_info['function']()

    metadata = {
        'freq': dataset_info['freq'],
        'season_length': dataset_info['season_length'],
        'description': dataset_info['description'],
        'recommended_horizon': dataset_info['recommended_horizon']
    }

    return df, metadata


if __name__ == '__main__':
    """Test all sample datasets"""
    print("Testing Sample Datasets\n" + "="*50)

    for name in DATASETS.keys():
        df, meta = get_sample_data(name)
        print(f"\n{name}")
        print(f"  Frequency: {meta['freq']}")
        print(f"  Season Length: {meta['season_length']}")
        print(f"  Observations: {len(df)}")
        print(f"  Date Range: {df['ds'].min()} to {df['ds'].max()}")
        print(f"  Value Range: {df['y'].min():.2f} to {df['y'].max():.2f}")
        print(f"  Description: {meta['description']}")
