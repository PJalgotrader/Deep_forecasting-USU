# Author: Prof. Pedram Jahangiry
# Date: 2024-10-10
# Enhanced with Advanced UI by AI Assistant

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from theme_manager import ThemeManager
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]

def run_forecast(y_train, y_test, model, fh, **kwargs):
    if model == 'ETS':
        forecaster = AutoETS(**kwargs)
    elif model == 'ARIMA':
        forecaster = AutoARIMA(**kwargs)
    else:
        raise ValueError("Unsupported model")
    
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh=ForecastingHorizon(y_test.index, is_relative=False))
    
    last_date = y_test.index[-1]
    future_dates = pd.period_range(start=last_date + 1, periods=fh, freq=y_train.index.freq)
    future_horizon = ForecastingHorizon(future_dates, is_relative=False)
    y_forecast = forecaster.predict(fh=future_horizon)
    
    return forecaster, y_pred, y_forecast

def plot_time_series_plotly(y_train, y_test, y_pred, y_forecast, title, theme):
    fig = go.Figure()
    
    # Add training data
    fig.add_trace(go.Scatter(
        x=y_train.index.to_timestamp(),
        y=y_train.values,
        mode='lines',
        name='Training Data',
        line=dict(color=theme['info_color'], width=2)
    ))
    
    # Add test data
    fig.add_trace(go.Scatter(
        x=y_test.index.to_timestamp(),
        y=y_test.values,
        mode='lines+markers',
        name='Actual Test Data',
        line=dict(color=theme['success_color'], width=2),
        marker=dict(size=6)
    ))
    
    # Add predictions
    fig.add_trace(go.Scatter(
        x=y_pred.index.to_timestamp(),
        y=y_pred.values,
        mode='lines+markers',
        name='Test Predictions',
        line=dict(color=theme['warning_color'], width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=y_forecast.index.to_timestamp(),
        y=y_forecast.values,
        mode='lines+markers',
        name='Future Forecast',
        line=dict(color=theme['accent_color'], width=3),
        marker=dict(size=10, symbol='star')
    ))
    
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'font': {'size': 20, 'color': theme['text_color']}
        },
        xaxis_title='Date',
        yaxis_title='Value',
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color']),
        legend=dict(
            bgcolor=theme['card_bg'],
            bordercolor=theme['border_color'],
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor=theme['border_color'],
            linecolor=theme['border_color']
        ),
        yaxis=dict(
            gridcolor=theme['border_color'],
            linecolor=theme['border_color']
        ),
        hovermode='x unified'
    )
    
    return fig

def plot_time_series(y_train, y_test, y_pred, y_forecast, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index.to_timestamp(), y_train.values, label="Train")
    ax.plot(y_test.index.to_timestamp(), y_test.values, label="Test")
    ax.plot(y_pred.index.to_timestamp(), y_pred.values, label="Test Predictions")
    ax.plot(y_forecast.index.to_timestamp(), y_forecast.values, label="Forecast")
    plt.legend()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    return fig



def render_theme_selector():
    theme_manager = ThemeManager()
    
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        
        theme_names = list(theme_manager.themes.keys())
        selected_theme_name = st.selectbox(
            "Choose Theme",
            theme_names,
            index=0 if 'selected_theme' not in st.session_state else theme_names.index(st.session_state.selected_theme),
            key="theme_selector"
        )
        
        st.session_state.selected_theme = selected_theme_name
        current_theme = theme_manager.get_theme(selected_theme_name)
        
        # Custom theme editor
        if selected_theme_name == "🎨 Custom":
            st.markdown("#### Customize Colors")
            
            custom_theme = {
                "bg_color": st.color_picker("Background Color", current_theme["bg_color"]),
                "secondary_bg": st.color_picker("Secondary Background", current_theme["secondary_bg"]),
                "text_color": st.color_picker("Text Color", current_theme["text_color"]),
                "accent_color": st.color_picker("Accent Color", current_theme["accent_color"]),
                "success_color": st.color_picker("Success Color", current_theme["success_color"]),
                "warning_color": st.color_picker("Warning Color", current_theme["warning_color"]),
                "error_color": st.color_picker("Error Color", current_theme["error_color"]),
                "info_color": st.color_picker("Info Color", current_theme["info_color"]),
                "card_bg": st.color_picker("Card Background", current_theme["card_bg"]),
                "border_color": st.color_picker("Border Color", current_theme["border_color"]),
                "gradient_start": st.color_picker("Gradient Start", current_theme["gradient_start"]),
                "gradient_end": st.color_picker("Gradient End", current_theme["gradient_end"])
            }
            
            current_theme = custom_theme
            
            if st.button("💾 Save Custom Theme"):
                theme_manager.save_custom_theme(custom_theme)
                st.success("Custom theme saved!")
        
        # Theme preview
        st.markdown("#### Theme Preview")
        preview_html = f"""
        <div style="
            background: linear-gradient(135deg, {current_theme['gradient_start']}, {current_theme['gradient_end']});
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid {current_theme['border_color']};
            margin: 0.5rem 0;
        ">
            <div style="color: {current_theme['text_color']}; font-weight: 600;">Sample Text</div>
            <div style="color: {current_theme['accent_color']}; font-size: 0.9rem;">Accent Color</div>
            <div style="
                background: {current_theme['card_bg']};
                padding: 0.5rem;
                margin-top: 0.5rem;
                border-radius: 4px;
                border: 1px solid {current_theme['border_color']};
            ">
                <span style="color: {current_theme['text_color']};">Card Content</span>
            </div>
        </div>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # UI Settings
        st.markdown("### ⚙️ UI Settings")
        use_plotly = st.checkbox("🚀 Use Interactive Charts (Plotly)", value=True)
        show_animations = st.checkbox("✨ Enable Animations", value=True)
        compact_mode = st.checkbox("📱 Compact Mode", value=False)
        
        return current_theme, use_plotly, show_animations, compact_mode

def main():
    st.set_page_config(
        page_title="Advanced Forecasting Studio",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize theme manager and render theme selector
    current_theme, use_plotly, show_animations, compact_mode = render_theme_selector()
    theme_manager = ThemeManager()
    
    # Apply theme
    st.markdown(theme_manager.apply_theme(current_theme), unsafe_allow_html=True)
    
    # Enhanced header with gradient and animations
    header_html = f"""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;">
        <h1 style="
            font-size: 3.5rem;
            background: linear-gradient(90deg, {current_theme['accent_color']}, {current_theme['info_color']}, {current_theme['success_color']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            {'animation: fadeInScale 1s ease-out;' if show_animations else ''}
        ">📈 Advanced Forecasting Studio</h1>
        <p style="
            color: {current_theme['text_color']};
            font-size: 1.2rem;
            opacity: 0.8;
            margin: 0;
        ">Powered by AI • Enhanced UI • Professional Analytics</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Adjust column ratios based on compact mode
    col_ratios = [1.2, 3, 5.8] if compact_mode else [1.5, 3.5, 5]
    col1, col2, col3 = st.columns(col_ratios)

    with col1:
        st.markdown("### 🔧 Model Configuration")
        
        # Model selection with enhanced UI
        model_options = {
            "ETS": "📊 Exponential Smoothing (ETS)",
            "ARIMA": "📈 AutoRegressive Integrated Moving Average"
        }
        
        model_choice = st.selectbox(
            "🤖 Select Forecasting Model",
            list(model_options.keys()),
            format_func=lambda x: model_options[x]
        )
        
        # Enhanced train size slider with visual feedback
        train_size = st.slider(
            "🎯 Training Data Split (%)",
            min_value=50,
            max_value=95,
            value=80,
            help="Percentage of data used for training the model"
        ) / 100
        
        # Display train/test split info
        split_info = f"""
        <div style="
            background: {current_theme['card_bg']};
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid {current_theme['border_color']};
            margin: 1rem 0;
        ">
            <div style="color: {current_theme['info_color']}; font-weight: 600; margin-bottom: 0.5rem;">📊 Data Split</div>
            <div style="color: {current_theme['text_color']}; font-size: 0.9rem;">
                🎯 Training: {int(train_size*100)}%<br>
                🧪 Testing: {int((1-train_size)*100)}%
            </div>
        </div>
        """
        st.markdown(split_info, unsafe_allow_html=True)

        if model_choice == "ETS":
            st.markdown("#### 📊 ETS Parameters")
            
            error_options = {
                "add": "➕ Additive",
                "mul": "✖️ Multiplicative"
            }
            error = st.selectbox("🎯 Error Type", list(error_options.keys()), format_func=lambda x: error_options[x])
            
            trend_options = {
                "add": "📈 Additive Trend",
                "mul": "📊 Multiplicative Trend",
                None: "➖ No Trend"
            }
            trend = st.selectbox("📈 Trend Type", list(trend_options.keys()), format_func=lambda x: trend_options[x])
            
            seasonal_options = {
                "add": "🌊 Additive Seasonal",
                "mul": "🌊 Multiplicative Seasonal",
                None: "➖ No Seasonality"
            }
            seasonal = st.selectbox("🌊 Seasonal Type", list(seasonal_options.keys()), format_func=lambda x: seasonal_options[x])
            
            damped_trend = st.checkbox("🔄 Damped Trend", value=False, help="Apply damping to trend component")
            seasonal_periods = st.number_input(
                "📅 Seasonal Periods",
                min_value=1,
                value=1,
                help="Number of periods in a complete seasonal cycle"
            )
            model_params = {
                "error": error,
                "trend": trend,
                "seasonal": seasonal,
                "damped_trend": damped_trend,
                "sp": seasonal_periods,
            }
        elif model_choice == "ARIMA":
            st.markdown("#### 📈 ARIMA Parameters")
            
            # Non-seasonal parameters in an expander
            with st.expander("🔧 Non-Seasonal Parameters (p,d,q)", expanded=True):
                col_p, col_d, col_q = st.columns(3)
                with col_p:
                    start_p = st.number_input("Min p", min_value=0, value=0, help="Autoregressive order")
                    max_p = st.number_input("Max p", min_value=0, value=5)
                with col_d:
                    d = st.number_input("d", min_value=0, value=1, help="Differencing order")
                with col_q:
                    start_q = st.number_input("Min q", min_value=0, value=0, help="Moving average order")
                    max_q = st.number_input("Max q", min_value=0, value=5)
            
            # Seasonal parameters
            seasonal = st.checkbox("🌊 Enable Seasonal ARIMA", value=True)
            
            if seasonal:
                with st.expander("🌊 Seasonal Parameters (P,D,Q,s)", expanded=True):
                    col_P, col_D, col_Q = st.columns(3)
                    with col_P:
                        start_P = st.number_input("Min P", min_value=0, value=0, help="Seasonal AR order")
                        max_P = st.number_input("Max P", min_value=0, value=2)
                    with col_D:
                        D = st.number_input("D", min_value=0, value=1, help="Seasonal differencing")
                    with col_Q:
                        start_Q = st.number_input("Min Q", min_value=0, value=0, help="Seasonal MA order")
                        max_Q = st.number_input("Max Q", min_value=0, value=2)
                    
                    sp = st.number_input("📅 Seasonal Periods", min_value=1, value=12, help="Length of seasonal cycle")
            
            model_params = {
                "start_p": start_p,
                "max_p": max_p,
                "start_q": start_q,
                "max_q": max_q,
                "d": d,
                "seasonal": seasonal,
            }
            if seasonal:
                model_params.update({
                    "start_P": start_P,
                    "max_P": max_P,
                    "start_Q": start_Q,
                    "max_Q": max_Q,
                    "D": D,
                    "sp": sp
                })

    with col2:
        st.markdown("### 📊 Data Management")
        
        # Enhanced file uploader with drag & drop styling
        uploaded_file = st.file_uploader(
            "📁 Upload Your Time Series Data",
            type=["csv"],
            help="Drag and drop your CSV file here or click to browse"
        )
        
        # Sample datasets section
        with st.expander("📚 Or Try Sample Datasets", expanded=False):
            sample_datasets = {
                "Airline Passengers": "airline_passengers.csv",
                "US Macro (Monthly)": "US_macro_monthly.csv", 
                "US Macro (Quarterly)": "US_macro_Quarterly.csv"
            }
            
            selected_sample = st.selectbox(
                "Choose a sample dataset",
                ["None"] + list(sample_datasets.keys())
            )
            
            if selected_sample != "None" and st.button(f"📈 Load {selected_sample}"):
                try:
                    uploaded_file = sample_datasets[selected_sample]
                    st.success(f"✅ Loaded {selected_sample}")
                except:
                    st.error("❌ Could not load sample dataset")
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Enhanced frequency selection with emojis and descriptions
                freq_options = {
                    'D': '📅 Daily',
                    'W': '📆 Weekly', 
                    'M': '🗓️ Monthly',
                    'Q': '📊 Quarterly',
                    'Y': '🗓️ Yearly'
                }
                
                freq = st.selectbox(
                    "📊 Data Frequency",
                    list(freq_options.keys()),
                    format_func=lambda x: freq_options[x],
                    help="Select the time interval between observations"
                )
                
                # Convert the index to datetime and then to PeriodIndex
                df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                df = df.set_index('date')
                df = df.sort_index()  # Ensure the index is sorted
                df.index = df.index.to_period(freq)
                
                # Remove any rows with NaT in the index
                df = df.loc[df.index.notnull()]
                
                # Enhanced data preview with metrics
                st.markdown("#### 👀 Data Overview")
                
                # Data metrics in columns
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("📊 Rows", len(df))
                with metric_col2:
                    st.metric("📈 Columns", len(df.columns))
                with metric_col3:
                    st.metric("📅 Start Date", str(df.index[0]))
                with metric_col4:
                    st.metric("📅 End Date", str(df.index[-1]))
                
                # Data preview with enhanced styling
                with st.expander("🔍 Raw Data Preview", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)

                # Filter out non-numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if not numeric_columns:
                    st.error("No numeric columns found in the uploaded data. Please ensure your CSV contains numeric data for forecasting.")
                else:
                    target_variable = st.selectbox("Select your target variable", numeric_columns)

                    # Enhanced time series plot
                    st.markdown(f"#### 📈 Time Series: **{target_variable}**")
                    
                    if use_plotly:
                        # Interactive Plotly chart
                        fig = px.line(
                            x=df.index.to_timestamp(),
                            y=df[target_variable],
                            title=f"{target_variable} Time Series Analysis",
                            labels={'x': 'Date', 'y': target_variable}
                        )
                        
                        fig.update_layout(
                            plot_bgcolor=current_theme['bg_color'],
                            paper_bgcolor=current_theme['bg_color'],
                            font=dict(color=current_theme['text_color']),
                            title_font_size=16,
                            showlegend=False,
                            hovermode='x'
                        )
                        
                        fig.update_traces(
                            line_color=current_theme['accent_color'],
                            line_width=2,
                            hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> %{y:.2f}<extra></extra>'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Traditional matplotlib chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(df.index.to_timestamp(), df[target_variable], color=current_theme['accent_color'], linewidth=2)
                        ax.set_title(f"{target_variable} Time Series", fontsize=14, color=current_theme['text_color'])
                        ax.set_xlabel("Date", color=current_theme['text_color'])
                        ax.set_ylabel("Value", color=current_theme['text_color'])
                        fig.patch.set_facecolor(current_theme['bg_color'])
                        ax.set_facecolor(current_theme['bg_color'])
                        st.pyplot(fig)
                    
                    # Basic statistics
                    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                    with stats_col1:
                        st.metric("📊 Mean", f"{df[target_variable].mean():.2f}")
                    with stats_col2:
                        st.metric("📈 Max", f"{df[target_variable].max():.2f}")
                    with stats_col3:
                        st.metric("📉 Min", f"{df[target_variable].min():.2f}")
                    with stats_col4:
                        st.metric("📊 Std Dev", f"{df[target_variable].std():.2f}")

            except Exception as e:
                st.error(f"An error occurred while processing the file: {str(e)}")
                st.error("Please ensure your CSV file is properly formatted with a date column and numeric data for forecasting.")

    with col3:
        st.markdown("### 🎯 Forecasting Results")
        
        # Enhanced forecast configuration
        forecast_col1, forecast_col2 = st.columns([2, 1])
        with forecast_col1:
            fh = st.number_input(
                "🔮 Forecast Horizon",
                min_value=1,
                max_value=100,
                value=10,
                help="Number of future periods to predict"
            )
        with forecast_col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Spacer
            run_forecast_button = st.button(
                "🚀 **Generate Forecast**",
                type="primary",
                use_container_width=True
            )
        
        # Forecast confidence settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            show_confidence = st.checkbox("📊 Show Confidence Intervals", value=True)
            export_results = st.checkbox("💾 Enable Result Export", value=True)
            detailed_metrics = st.checkbox("📈 Show Detailed Metrics", value=True)
        
        if run_forecast_button:
            if 'df' in locals() and 'target_variable' in locals():
                try:
                    y = df[target_variable]
                    
                    # Perform train-test split
                    y_train, y_test = manual_train_test_split(y, train_size)

                    # Progress bar for forecast generation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("🔄 Preparing data...")
                    progress_bar.progress(25)
                    
                    status_text.text("🤖 Training model...")
                    progress_bar.progress(50)
                    
                    forecaster, y_pred, y_forecast = run_forecast(y_train, y_test, model_choice, fh, **model_params)
                    
                    status_text.text("📊 Generating visualizations...")
                    progress_bar.progress(75)
                    
                    # Enhanced plotting
                    if use_plotly:
                        fig = plot_time_series_plotly(y_train, y_test, y_pred, y_forecast, 
                                                    f"{model_choice} Forecast for {target_variable}", current_theme)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = plot_time_series(y_train, y_test, y_pred, y_forecast, f"{model_choice} Forecast for {target_variable}")
                        st.pyplot(fig)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Forecast completed!")
                    
                    # Enhanced results display
                    results_tab1, results_tab2, results_tab3 = st.tabs(["📊 Test Predictions", "🔮 Future Forecast", "📈 Model Info"])
                    
                    with results_tab1:
                        st.markdown("#### 🧪 Test Set Performance")
                        test_results_df = pd.DataFrame({
                            'Date': y_test.index.to_timestamp(),
                            'Actual': y_test.values,
                            'Predicted': y_pred.values,
                            'Error': y_test.values - y_pred.values,
                            'Error %': ((y_test.values - y_pred.values) / y_test.values * 100)
                        })
                        st.dataframe(test_results_df, use_container_width=True)
                        
                        if detailed_metrics:
                            # Performance metrics
                            mae = np.mean(np.abs(test_results_df['Error']))
                            mse = np.mean(test_results_df['Error']**2)
                            rmse = np.sqrt(mse)
                            mape = np.mean(np.abs(test_results_df['Error %']))
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            with metric_col1:
                                st.metric("📊 MAE", f"{mae:.2f}")
                            with metric_col2:
                                st.metric("📈 RMSE", f"{rmse:.2f}")
                            with metric_col3:
                                st.metric("📉 MAPE", f"{mape:.2f}%")
                            with metric_col4:
                                st.metric("📊 MSE", f"{mse:.2f}")
                    
                    with results_tab2:
                        st.markdown("#### 🔮 Future Predictions")
                        forecast_df = pd.DataFrame({
                            'Date': y_forecast.index.to_timestamp(),
                            'Forecast': y_forecast.values
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                        
                        if export_results:
                            # Export functionality
                            csv = forecast_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast as CSV",
                                data=csv,
                                file_name=f"forecast_{target_variable}_{model_choice}.csv",
                                mime="text/csv"
                            )
                    
                    with results_tab3:
                        st.markdown("#### 🤖 Model Information")
                        model_info = {
                            "Model Type": model_choice,
                            "Training Size": f"{int(train_size*100)}%",
                            "Test Size": f"{int((1-train_size)*100)}%",
                            "Forecast Horizon": f"{fh} periods",
                            "Data Frequency": freq_options[freq]
                        }
                        
                        for key, value in model_info.items():
                            st.write(f"**{key}:** {value}")
                        
                        # Model parameters
                        with st.expander("🔧 Model Parameters", expanded=False):
                            st.json(model_params)
                except Exception as e:
                    st.error(f"An error occurred during forecasting: {str(e)}")
            else:
                st.warning("⚠️ Please upload data and select a target variable before running the forecast.")
                
                # Quick start guide
                st.info("""
                **🚀 Quick Start Guide:**
                1. 📁 Upload your CSV file or select a sample dataset
                2. 📊 Choose your target variable for forecasting
                3. 🔧 Configure model parameters
                4. 🚀 Click 'Generate Forecast' to begin
                """)

# Add footer with credits and stats
def render_footer():
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**👨‍🏫 Original Author:** Prof. Pedram Jahangiry")
    with footer_col2:
        st.markdown("**🤖 Enhanced by:** AI Assistant")
    with footer_col3:
        st.markdown("**🏫 Institution:** Utah State University")

if __name__ == "__main__":
    main()
    render_footer()