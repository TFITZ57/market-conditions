"""
Forecast Page

This module renders the Forecast page of the application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

from src.utils.config import load_config
from src.data_processing.time_series import forecast_arima, forecast_exponential_smoothing, forecast_linear_regression, generate_ensemble_forecast
from src.visualization.charts import create_forecast_chart
from src.visualization.exporters import export_visualizations, create_download_link

def show_forecast_page(filters: Dict[str, Any]):
    """
    Display the forecast page.
    
    Args:
        filters: Dictionary containing filter selections.
    """
    # Load configuration
    config = load_config()
    
    # Set page title
    st.title("Market Forecast")
    st.write("Time series forecasting for key housing market metrics")
    
    # Display current filter selections
    st.write(f"**Selected Time Period:** {filters['date_range']['start_quarter']} to {filters['date_range']['end_quarter']}")
    if filters['location']['towns'] != config['locations']['towns']:
        st.write(f"**Selected Towns:** {', '.join(filters['location']['towns'])}")
    if filters['property_types'] != config['property_types']:
        st.write(f"**Selected Property Types:** {', '.join(filters['property_types'])}")
    
    # Select metric to forecast
    selected_metric = st.selectbox(
        "Select metric to forecast",
        options=config["metrics"]["housing"],
        index=0  # Default to first metric
    )
    
    # Forecast settings
    st.subheader("Forecast Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        forecast_periods = st.number_input(
            "Forecast periods (quarters)",
            min_value=1,
            max_value=12,
            value=4  # Default to 1 year (4 quarters)
        )
    
    with col2:
        forecast_model = st.selectbox(
            "Forecast model",
            options=["ARIMA", "Exponential Smoothing", "Linear Regression", "Ensemble"],
            index=3  # Default to Ensemble
        )
    
    with col3:
        confidence_level = st.slider(
            "Confidence level (%)",
            min_value=50,
            max_value=99,
            value=95  # Default to 95%
        )
    
    # Additional model parameters based on selected model
    if forecast_model == "ARIMA":
        st.subheader("ARIMA Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p_param = st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
        
        with col2:
            d_param = st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
        
        with col3:
            q_param = st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
        
        model_params = (p_param, d_param, q_param)
    
    elif forecast_model == "Exponential Smoothing":
        st.subheader("Exponential Smoothing Parameters")
        
        use_seasonality = st.checkbox("Include seasonality", value=True)
        
        if use_seasonality:
            seasonal_periods = st.number_input(
                "Seasonal periods",
                min_value=2,
                max_value=12,
                value=4  # Default to quarterly seasonality
            )
        else:
            seasonal_periods = None
        
        model_params = {
            'seasonal': use_seasonality,
            'seasonal_periods': seasonal_periods
        }
    
    elif forecast_model == "Linear Regression":
        # No additional parameters needed for linear regression
        model_params = {}
    
    else:  # Ensemble
        # No additional parameters needed for ensemble
        model_params = {}
    
    # In a real implementation, we would load actual data from the processed data directory
    # For this demonstration, we'll generate synthetic data
    data = generate_synthetic_data(
        selected_metric, 
        filters['date_range']['quarters_list'],
        filters['location']['towns']
    )
    
    # Generate forecast when button is clicked
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            # Filter data based on selected towns
            if len(filters['location']['towns']) < len(config['locations']['towns']):
                # If specific towns are selected, filter and aggregate
                town_data = data[data['town'].isin(filters['location']['towns'])]
                
                # Group by date and calculate mean
                agg_data = town_data.groupby('date')['value'].mean().reset_index()
            else:
                # If all towns are selected, use county-level aggregate
                agg_data = data.groupby('date')['value'].mean().reset_index()
            
            # Sort by date
            agg_data = agg_data.sort_values('date')
            
            # Generate forecast based on selected model
            forecast_df, model, confidence_intervals = generate_forecast(
                agg_data,
                forecast_model,
                selected_metric,
                forecast_periods,
                confidence_level / 100,
                model_params
            )
            
            # Display forecast results
            display_forecast_results(agg_data, forecast_df, selected_metric, forecast_model)
            
            # Display model details
            display_model_details(model, forecast_model, selected_metric)
            
            # Display scenario analysis
            display_scenario_analysis(agg_data, selected_metric, forecast_periods)

def generate_forecast(data: pd.DataFrame, 
                     model_type: str, 
                     metric: str, 
                     periods: int, 
                     confidence: float, 
                     params: Any) -> Tuple[pd.DataFrame, Any, Dict[str, np.ndarray]]:
    """
    Generate forecast for time series data.
    
    Args:
        data: DataFrame with time series data
        model_type: Type of forecasting model
        metric: Name of the metric being forecasted
        periods: Number of periods to forecast
        confidence: Confidence level for prediction intervals
        params: Model-specific parameters
        
    Returns:
        Tuple containing forecast DataFrame, model object, and confidence intervals
    """
    # In a real implementation, this would use actual forecasting models
    # For this demonstration, we'll create a simple synthetic forecast
    
    # Get last date in the data
    last_date = data['date'].max()
    
    # Create forecast dates (quarterly)
    forecast_dates = [last_date + pd.DateOffset(months=3*i) for i in range(1, periods+1)]
    
    # Create trend based on recent data
    if len(data) > 4:
        # Calculate average growth over recent periods
        recent_values = data['value'].iloc[-4:].values
        avg_growth = (recent_values[-1] / recent_values[0]) ** (1/4) - 1
    else:
        # Default growth if insufficient data
        avg_growth = 0.02
    
    # Last value
    last_value = data['value'].iloc[-1]
    
    # Generate forecast with trend
    forecast_values = [last_value * (1 + avg_growth) ** (i+1) for i in range(periods)]
    
    # Add some randomness
    forecast_values = [v * (1 + np.random.normal(0, 0.02)) for v in forecast_values]
    
    # Create confidence intervals
    lower_ci = [v * (1 - 0.05 * (i+1)) for i, v in enumerate(forecast_values)]
    upper_ci = [v * (1 + 0.05 * (i+1)) for i, v in enumerate(forecast_values)]
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecast_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    })
    
    # Dummy model object
    model = {
        'type': model_type,
        'parameters': params,
        'accuracy': {
            'mape': 4.5,
            'rmse': last_value * 0.03
        }
    }
    
    # Confidence intervals
    confidence_intervals = {
        'lower': np.array(lower_ci),
        'upper': np.array(upper_ci)
    }
    
    return forecast_df, model, confidence_intervals

def display_forecast_results(historical_data: pd.DataFrame, 
                            forecast_data: pd.DataFrame, 
                            metric: str, 
                            model_type: str):
    """
    Display forecast results with visualizations.
    
    Args:
        historical_data: DataFrame with historical data
        forecast_data: DataFrame with forecast data
        metric: Name of the metric
        model_type: Type of forecasting model used
    """
    st.subheader(f"{metric} Forecast ({model_type})")
    
    # Display forecast chart
    st.info("Forecast chart will be displayed here.")
    
    # Show data table
    with st.expander("View forecast data"):
        st.write(forecast_data)

def display_model_details(model: Any, 
                         model_type: str, 
                         metric: str):
    """
    Display details about the forecasting model.
    
    Args:
        model: Model object or information
        model_type: Type of forecasting model used
        metric: Name of the metric
    """
    st.subheader("Model Details")
    
    # Show model details
    st.info("Model details will be displayed here.")

def display_scenario_analysis(historical_data: pd.DataFrame, 
                             metric: str, 
                             periods: int):
    """
    Display scenario analysis for the forecast.
    
    Args:
        historical_data: DataFrame with historical data
        metric: Name of the metric
        periods: Number of forecast periods
    """
    st.subheader("Scenario Analysis")
    
    # Show scenario analysis
    st.info("Scenario analysis will be displayed here.")

def generate_synthetic_data(metric: str, 
                          quarters_list: List[str], 
                          towns: List[str]) -> pd.DataFrame:
    """
    Generate synthetic data for forecast demonstration.
    
    Args:
        metric: Metric name
        quarters_list: List of quarters
        towns: List of towns
        
    Returns:
        DataFrame with synthetic time series data
    """
    # Convert quarters list to proper dates
    dates = []
    for quarter in quarters_list:
        year, q = quarter.split()
        quarter_num = int(q[1])
        month = ((quarter_num - 1) * 3) + 1
        dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    # Base parameters based on metric
    if metric == "Median Sale Price":
        base = 500000
        trend = 10000
        noise = 20000
    elif metric == "Average Sale Price":
        base = 600000
        trend = 12000
        noise = 25000
    elif metric == "Days on Market":
        base = 30
        trend = -0.2
        noise = 5
    elif metric == "Sales Volume":
        base = 100
        trend = 0.5
        noise = 15
    elif metric == "Months of Supply":
        base = 3.5
        trend = -0.01
        noise = 0.5
    else:
        # Generic metric
        base = 100
        trend = 1
        noise = 10
    
    # Create data for each town and date
    rows = []
    for town in towns:
        # Each town has different baseline values
        town_factor = 0.7 + (hash(town) % 10) / 10  # Random factor between 0.7-1.6
        
        for i, date in enumerate(dates):
            # Generate value with trend, seasonality, and noise
            town_base = base * town_factor
            trend_component = trend * i / len(dates)
            seasonal_component = town_base * 0.05 * np.sin(i * np.pi / 2)
            noise_component = np.random.normal(0, noise)
            
            value = town_base + trend_component + seasonal_component + noise_component
            
            # Ensure non-negative values
            value = max(0, value)
            
            rows.append({
                'date': date,
                'town': town,
                'value': value
            })
    
    return pd.DataFrame(rows)