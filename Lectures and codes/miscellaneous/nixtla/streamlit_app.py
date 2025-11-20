"""
Time Series Forecasting App with Streamlit
Supports StatsForecast, MLForecast, and NeuralForecast modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

# Add the full_pipeline directory to path to import forecasting modules
pipeline_path = Path(__file__).parent / 'full_pipeline'
sys.path.insert(0, str(pipeline_path))

# Import forecasting modules
try:
    from df_statsforecast import StatsforecastForecaster, train_test_split_ts
    STATSFORECAST_AVAILABLE = True
except ImportError as e:
    STATSFORECAST_AVAILABLE = False
    st.error(f"StatsForecast module not available: {e}")

try:
    from df_mlforecast import MLForecastForecaster
    MLFORECAST_AVAILABLE = True
except ImportError as e:
    MLFORECAST_AVAILABLE = False
    st.error(f"MLForecast module not available: {e}")

try:
    from df_neuralforecast import NeuralForecastForecaster
    NEURALFORECAST_AVAILABLE = True
except ImportError as e:
    NEURALFORECAST_AVAILABLE = False
    st.error(f"NeuralForecast module not available: {e}")

# Import sample data
try:
    from sample_data import DATASETS, get_sample_data
    SAMPLE_DATA_AVAILABLE = True
except ImportError:
    SAMPLE_DATA_AVAILABLE = False
    st.warning("Sample data module not available. Only file upload will work.")


# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model configurations
STATSFORECAST_MODELS = [
    'arima', 'auto_arima', 'auto_ets', 'naive',
    'seasonal_naive', 'random_walk_with_drift', 'window_average', 'seasonal_window_average'
]

MLFORECAST_MODELS = [
    'xgboost', 'lightgbm', 'random_forest', 'catboost', 'linear'
]

NEURALFORECAST_MODELS = [
    'mlp', 'rnn', 'lstm', 'gru'
]

RECURRENT_MODELS = ['rnn', 'lstm', 'gru']

# Frequency options
FREQUENCY_OPTIONS = {
    'Hourly': 'H',
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly (Start)': 'MS',
    'Monthly (End)': 'M',
    'Quarterly (Start)': 'QS',
    'Quarterly (End)': 'Q',
    'Yearly (Start)': 'YS',
    'Yearly (End)': 'Y'
}


def initialize_session_state():
    """Initialize session state variables"""
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load and validate uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def validate_and_prepare_data(df: pd.DataFrame, date_col: str, value_col: str) -> Optional[pd.DataFrame]:
    """Validate and prepare data in Nixtla format"""
    try:
        # Check if columns exist
        if date_col not in df.columns:
            st.error(f"Column '{date_col}' not found in data")
            return None
        if value_col not in df.columns:
            st.error(f"Column '{value_col}' not found in data")
            return None

        # Create standardized dataframe
        prepared_df = pd.DataFrame({
            'ds': pd.to_datetime(df[date_col]),
            'y': pd.to_numeric(df[value_col])
        })

        # Sort by date
        prepared_df = prepared_df.sort_values('ds').reset_index(drop=True)

        # Check for missing values
        if prepared_df['y'].isna().any():
            st.warning(f"Found {prepared_df['y'].isna().sum()} missing values. They will be removed.")
            prepared_df = prepared_df.dropna()

        return prepared_df

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None


def plot_time_series(df: pd.DataFrame, title: str = "Time Series Data"):
    """Plot time series using Plotly"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['ds'],
        y=df['y'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=2)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_forecast_results(results: Dict[str, Any], title: str = "Forecast Results"):
    """Plot forecast vs actual values"""
    forecasts_df = results['forecasts'].copy()

    fig = go.Figure()

    # Only plot if we have actual values (not NaN)
    if 'y_true' in forecasts_df.columns and not forecasts_df['y_true'].isna().all():
        # Plot actual values
        fig.add_trace(go.Scatter(
            x=forecasts_df['ds'],
            y=forecasts_df['y_true'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

    # Plot predictions
    fig.add_trace(go.Scatter(
        x=forecasts_df['ds'],
        y=forecasts_df['y_pred'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )

    return fig


def display_metrics(metrics: Dict[str, float]):
    """Display metrics in columns"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value=f"{metrics['mae']:.4f}",
            help="Average absolute difference between predicted and actual values"
        )

    with col2:
        st.metric(
            label="RMSE (Root Mean Squared Error)",
            value=f"{metrics['rmse']:.4f}",
            help="Square root of average squared differences (penalizes larger errors)"
        )

    with col3:
        st.metric(
            label="MAPE (Mean Absolute % Error)",
            value=f"{metrics['mape']:.2f}%",
            help="Average percentage error (scale-independent metric)"
        )


def fix_forecast_actuals(results: Dict[str, Any], test_df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """
    Fix actual values in forecast results by properly aligning with test data.
    This completely rebuilds the forecasts DataFrame to ensure correct alignment.
    """
    if results is None or test_df is None:
        return results

    forecasts_df = results['forecasts'].copy()

    # Get the first 'horizon' rows from test data
    test_subset = test_df.head(horizon).reset_index(drop=True)

    # Determine how many values we can match
    n_match = min(len(forecasts_df), len(test_subset))

    # Create a new dataframe with correct alignment
    new_forecasts = pd.DataFrame({
        'ds': test_subset['ds'].iloc[:n_match].values,
        'y_true': test_subset['y'].iloc[:n_match].values,
        'y_pred': forecasts_df['y_pred'].iloc[:n_match].values
    })

    # Add unique_id if it exists
    if 'unique_id' in forecasts_df.columns:
        new_forecasts['unique_id'] = forecasts_df['unique_id'].iloc[:n_match].values

    # Recalculate metrics with correct alignment
    actuals = new_forecasts['y_true'].values
    predictions = new_forecasts['y_pred'].values

    errors = actuals - predictions
    metrics = {
        'mae': float(np.mean(np.abs(errors))),
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mape': float(np.mean(np.abs(errors / actuals)) * 100) if np.all(actuals != 0) else np.nan
    }

    return {
        'forecasts': new_forecasts,
        'metrics': metrics
    }


def run_forecast(
    module_type: str,
    model_type: str,
    strategy: str,
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    params: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Run forecasting based on selected module and strategy"""

    try:
        # Create forecaster based on module type
        if module_type == "StatsForecast":
            if not STATSFORECAST_AVAILABLE:
                st.error("StatsForecast module is not available")
                return None

            forecaster = StatsforecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                season_length=params['season_length'],
                **params.get('model_params', {})
            )

        elif module_type == "MLForecast":
            if not MLFORECAST_AVAILABLE:
                st.error("MLForecast module is not available")
                return None

            forecaster = MLForecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                lags=params.get('lags'),
                **params.get('model_params', {})
            )

        elif module_type == "NeuralForecast":
            if not NEURALFORECAST_AVAILABLE:
                st.error("NeuralForecast module is not available")
                return None

            forecaster = NeuralForecastForecaster(
                model_type=model_type,
                freq=params['freq'],
                input_size=params.get('input_size', 12),
                horizon=params['horizon'],
                **params.get('model_params', {})
            )

        else:
            st.error(f"Unknown module type: {module_type}")
            return None

        # Run forecast based on strategy
        # Note: Data is already in standard format with columns 'ds' and 'y'
        if strategy == "One-step forecast":
            if test_df is None:
                st.error("Test data is required for one-step forecasting")
                return None
            results = forecaster.one_step_forecast(
                train_df,
                test_df,
                target_col='y',
                date_col='ds',
                unique_id='1'
            )

        elif strategy == "Multi-step recursive":
            if module_type == "NeuralForecast" and model_type in RECURRENT_MODELS:
                results = forecaster.multi_step_forecast(
                    train_df,
                    params['horizon'],
                    target_col='y',
                    date_col='ds',
                    unique_id='1',
                    test_df=test_df,
                    use_recurrent=True
                )
            else:
                results = forecaster.multi_step_forecast(
                    train_df,
                    params['horizon'],
                    target_col='y',
                    date_col='ds',
                    unique_id='1',
                    test_df=test_df
                )

        elif strategy == "Multi-output direct":
            # Check if multi-output is supported
            if module_type == "StatsForecast":
                st.error("❌ Multi-output forecasting is not supported for StatsForecast models. "
                        "Statistical models (ARIMA/ETS) can only do recursive forecasting. "
                        "Please select 'Multi-step recursive' or use ML/Neural models.")
                return None

            results = forecaster.multi_output_forecast(
                train_df,
                params['horizon'],
                target_col='y',
                date_col='ds',
                unique_id='1',
                test_df=test_df
            )

        else:
            st.error(f"Unknown strategy: {strategy}")
            return None

        # Fix actual values alignment for multi-step and multi-output strategies
        if strategy in ["Multi-step recursive", "Multi-output direct"] and test_df is not None:
            results = fix_forecast_actuals(results, test_df, params['horizon'])

        return results

    except NotImplementedError as e:
        st.error(f"❌ {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error during forecasting: {e}")
        st.exception(e)
        return None


def main():
    """Main Streamlit app"""
    initialize_session_state()

    # Header
    st.title("📈 Time Series Forecasting App")
    st.markdown("""
    This app supports three forecasting paradigms using Nixtla's ecosystem:
    - **StatsForecast**: Statistical models (ARIMA, ETS, etc.)
    - **MLForecast**: Machine learning models (XGBoost, LightGBM, etc.)
    - **NeuralForecast**: Deep learning models (LSTM, NBEATS, etc.)
    """)

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    # Data Source Selection
    st.sidebar.subheader("1. Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Sample Dataset"],
        help="Upload your own CSV file or use a pre-loaded example dataset"
    )

    data = None
    metadata = {}

    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV file with datetime and numeric columns"
        )

        if uploaded_file is not None:
            data = load_uploaded_file(uploaded_file)

            if data is not None:
                st.sidebar.success(f"✅ Loaded {len(data)} rows")

                # Column selection
                date_col = st.sidebar.selectbox(
                    "Select date column:",
                    options=data.columns.tolist()
                )
                value_col = st.sidebar.selectbox(
                    "Select value column:",
                    options=data.columns.tolist()
                )

                # Prepare data
                data = validate_and_prepare_data(data, date_col, value_col)

                if data is not None:
                    st.session_state.current_data = data
                    st.session_state.data_loaded = True

    else:  # Sample Dataset
        if SAMPLE_DATA_AVAILABLE:
            dataset_name = st.sidebar.selectbox(
                "Select dataset:",
                options=list(DATASETS.keys()),
                help="Pre-loaded example datasets"
            )

            if dataset_name:
                data, metadata = get_sample_data(dataset_name)
                st.sidebar.success(f"✅ Loaded {len(data)} rows")
                st.sidebar.info(f"📝 {metadata['description']}")
                st.session_state.current_data = data
                st.session_state.data_loaded = True
        else:
            st.sidebar.error("Sample data module not available")

    # Only show configuration if data is loaded
    if st.session_state.data_loaded and st.session_state.current_data is not None:
        data = st.session_state.current_data

        # Module Selection
        st.sidebar.subheader("2. Forecasting Module")
        available_modules = []
        if STATSFORECAST_AVAILABLE:
            available_modules.append("StatsForecast")
        if MLFORECAST_AVAILABLE:
            available_modules.append("MLForecast")
        if NEURALFORECAST_AVAILABLE:
            available_modules.append("NeuralForecast")

        if not available_modules:
            st.error("No forecasting modules available. Please install required packages.")
            return

        module_type = st.sidebar.radio(
            "Select module:",
            options=available_modules,
            help="StatsForecast=Statistical, MLForecast=ML, NeuralForecast=DL"
        )

        # Model Selection
        st.sidebar.subheader("3. Model Selection")
        if module_type == "StatsForecast":
            model_type = st.sidebar.selectbox("Select model:", STATSFORECAST_MODELS)
        elif module_type == "MLForecast":
            model_type = st.sidebar.selectbox("Select model:", MLFORECAST_MODELS)
        else:  # NeuralForecast
            model_type = st.sidebar.selectbox("Select model:", NEURALFORECAST_MODELS)

        # Strategy Selection
        st.sidebar.subheader("4. Forecasting Strategy")
        strategy = st.sidebar.radio(
            "Select strategy:",
            [
                "One-step forecast",
                "Multi-step recursive",
                "Multi-output direct"
            ],
            help="One-step=iterative (slowest, most accurate), Recursive=default, Direct=all at once"
        )

        # Show warning for StatsForecast + Multi-output
        if module_type == "StatsForecast" and strategy == "Multi-output direct":
            st.sidebar.warning("⚠️ Multi-output is NOT supported for StatsForecast models")

        # Parameters
        st.sidebar.subheader("5. Parameters")

        # Train/Test Split
        test_size = st.sidebar.slider(
            "Test set size (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            help="Percentage of data to use for testing"
        )

        test_size_n = int(len(data) * test_size / 100)
        train_df, test_df = train_test_split_ts(data, test_size=test_size_n)

        st.sidebar.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

        # Frequency
        if metadata and 'freq' in metadata:
            default_freq = metadata['freq']
        else:
            default_freq = 'MS'

        # Find the index of default frequency
        freq_values = list(FREQUENCY_OPTIONS.values())
        try:
            freq_index = freq_values.index(default_freq)
        except ValueError:
            freq_index = 3  # Default to Monthly Start

        freq_label = st.sidebar.selectbox(
            "Frequency:",
            options=list(FREQUENCY_OPTIONS.keys()),
            index=freq_index,
            help="Time series frequency"
        )
        freq = FREQUENCY_OPTIONS[freq_label]

        # Season length
        if metadata and 'season_length' in metadata:
            default_season = metadata['season_length']
        else:
            default_season = 12

        season_length = st.sidebar.number_input(
            "Season length:",
            min_value=1,
            max_value=365,
            value=default_season,
            help="Number of periods in a season (e.g., 12 for monthly data with yearly seasonality)"
        )

        # Forecast horizon
        if metadata and 'recommended_horizon' in metadata:
            default_horizon = metadata['recommended_horizon']
        else:
            default_horizon = min(12, len(test_df))

        horizon = st.sidebar.number_input(
            "Forecast horizon (h):",
            min_value=1,
            max_value=len(test_df),
            value=default_horizon,
            help="Number of periods to forecast ahead"
        )

        # Module-specific parameters
        model_params = {}

        if module_type == "MLForecast":
            with st.sidebar.expander("ML-Specific Parameters"):
                lags_input = st.text_input(
                    "Lags (comma-separated):",
                    value="1,12",
                    help="Lag features to use (e.g., 1,12 for lag-1 and lag-12)"
                )
                try:
                    lags = [int(x.strip()) for x in lags_input.split(',')]
                except:
                    lags = [1, 12]

        elif module_type == "NeuralForecast":
            with st.sidebar.expander("Neural-Specific Parameters"):
                input_size = st.number_input(
                    "Input size (lookback window):",
                    min_value=1,
                    max_value=min(100, len(train_df)),
                    value=min(12, len(train_df) // 2),
                    help="Number of past observations to use"
                )

                hidden_size = st.number_input(
                    "Hidden size:",
                    min_value=4,
                    max_value=128,
                    value=16,
                    help="Size of hidden layers"
                )

                max_steps = st.number_input(
                    "Max training steps:",
                    min_value=10,
                    max_value=1000,
                    value=100,
                    help="Number of training epochs"
                )

                model_params = {
                    'encoder_hidden_size': hidden_size,
                    'max_steps': max_steps
                }

        # Prepare parameters dictionary
        params = {
            'freq': freq,
            'season_length': season_length,
            'horizon': horizon,
            'model_params': model_params
        }

        if module_type == "MLForecast":
            params['lags'] = lags

        if module_type == "NeuralForecast":
            params['input_size'] = input_size

        # Main area - Data Preview
        st.header("📊 Data Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Summary")
            st.write(f"**Total observations:** {len(data)}")
            st.write(f"**Date range:** {data['ds'].min().date()} to {data['ds'].max().date()}")
            st.write(f"**Value range:** {data['y'].min():.2f} to {data['y'].max():.2f}")
            st.write(f"**Mean:** {data['y'].mean():.2f}")
            st.write(f"**Std Dev:** {data['y'].std():.2f}")

        with col2:
            st.subheader("Train/Test Split")
            st.write(f"**Training set:** {len(train_df)} observations")
            st.write(f"**Test set:** {len(test_df)} observations")
            st.write(f"**Split ratio:** {100-test_size}/{test_size}")

        # Plot time series
        st.subheader("Time Series Plot")
        fig = plot_time_series(data)
        st.plotly_chart(fig, use_container_width=True)

        # Show data table
        with st.expander("View Data Table"):
            col_left, col_right = st.columns(2)
            with col_left:
                st.write("**First 10 rows:**")
                st.dataframe(data.head(10))
            with col_right:
                st.write("**Last 10 rows:**")
                st.dataframe(data.tail(10))

        # Run Forecast Button
        st.header("🚀 Run Forecast")

        if st.button("Run Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Running {strategy} with {module_type} - {model_type}..."):
                results = run_forecast(
                    module_type=module_type,
                    model_type=model_type,
                    strategy=strategy,
                    train_df=train_df,
                    test_df=test_df if strategy == "One-step forecast" else test_df,
                    params=params
                )

                if results is not None:
                    st.session_state.forecast_results = results
                    st.success("✅ Forecast completed successfully!")

        # Display Results
        if st.session_state.forecast_results is not None:
            results = st.session_state.forecast_results

            st.header("📈 Forecast Results")

            # Metrics
            st.subheader("Performance Metrics")
            display_metrics(results['metrics'])

            # Forecast Plot
            st.subheader("Forecast vs Actual")
            fig_forecast = plot_forecast_results(
                results,
                title=f"{module_type} - {model_type} ({strategy})"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Results table
            with st.expander("View Forecast Table"):
                st.dataframe(results['forecasts'])

            # Debug view - show alignment verification
            with st.expander("🔍 Debug: Verify Data Alignment"):
                st.write("**First 5 rows of Test Set:**")
                st.dataframe(test_df.head(5))
                st.write("**First 5 rows of Forecast Results:**")
                st.dataframe(results['forecasts'].head(5))
                st.write("**Note:** The 'y' values from Test Set should match 'y_true' values in Forecast Results")

            # Download results
            st.subheader("💾 Download Results")
            csv = results['forecasts'].to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{module_type}_{model_type}_{strategy.replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    else:
        # Show instructions if no data loaded
        st.info("👈 Please load data from the sidebar to get started")

        st.markdown("""
        ### Getting Started

        1. **Choose a data source:**
           - Upload your own CSV file with datetime and value columns
           - Or select a pre-loaded sample dataset

        2. **Configure forecasting:**
           - Select forecasting module (Statistical, ML, or Neural)
           - Choose a specific model
           - Pick a forecasting strategy
           - Adjust parameters as needed

        3. **Run forecast and analyze results:**
           - View performance metrics (MAE, RMSE, MAPE)
           - Visualize forecast vs actual values
           - Download results as CSV

        ### Forecasting Strategies

        - **One-step forecast**: Refits model for each prediction (slowest, most accurate)
        - **Multi-step recursive**: Uses previous predictions to forecast ahead (balanced)
        - **Multi-output direct**: Generates all predictions at once (fastest)

        ### Module Capabilities

        | Module | One-step | Recursive | Direct |
        |--------|----------|-----------|--------|
        | StatsForecast | ✅ | ✅ | ❌ |
        | MLForecast | ✅ | ✅ | ✅ |
        | NeuralForecast | ✅ | ✅ | ✅ |
        """)


if __name__ == "__main__":
    main()
