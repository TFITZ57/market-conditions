"""
Economic Indicators Page

This module renders the Economic Indicators page of the application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.utils.config import load_config
from src.visualization.charts import create_metric_chart, create_combined_chart, create_correlation_chart
from src.visualization.dashboards import create_economic_dashboard
from src.visualization.exporters import export_visualizations, create_download_link

def show_economic_page(filters: Dict[str, Any]):
    """
    Display the economic indicators page.
    
    Args:
        filters: Dictionary containing filter selections.
    """
    # Load configuration
    config = load_config()
    
    # Set page title
    st.title("Economic Indicators")
    st.write("Analysis of economic indicators affecting the Fairfield County housing market")
    
    # Display current filter selections
    st.write(f"**Selected Time Period:** {filters['date_range']['start_quarter']} to {filters['date_range']['end_quarter']}")
    if filters['location']['towns'] != config['locations']['towns']:
        st.write(f"**Selected Towns:** {', '.join(filters['location']['towns'])}")
    
    # Display selected metrics
    selected_metrics = st.multiselect(
        "Select economic metrics to display",
        options=config["metrics"]["economic"],
        default=["Local Job Growth", "Unemployment Rate", "Median Household Income"]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one economic metric to display.")
        return
    
    # In a real implementation, we would load actual data from the processed data directory
    # For this demonstration, we'll generate synthetic data
    data_dict = generate_synthetic_economic_data(
        selected_metrics, 
        filters['date_range']['quarters_list']
    )
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Economic Trends", "Correlation Analysis", "Housing Impact"])
    
    with tab1:
        show_trends_tab(data_dict, selected_metrics, filters)
    
    with tab2:
        show_correlation_tab(data_dict, selected_metrics, filters, config)
    
    with tab3:
        show_housing_impact_tab(data_dict, selected_metrics, filters, config)

def show_trends_tab(data_dict: Dict[str, pd.DataFrame], 
                    selected_metrics: List[str], 
                    filters: Dict[str, Any]):
    """
    Display the economic trends tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Economic Trends")
    
    # Create metric selector for trend analysis
    metric = st.selectbox(
        "Select metric for trend analysis",
        options=selected_metrics,
        index=0 if selected_metrics else None
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Create trend chart
        fig = create_metric_chart(
            df, 
            metric, 
            chart_type='line', 
            filters=filters
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download link for the chart
        if st.button("Download Chart"):
            chart_data = {metric: fig}
            exported_files = export_visualizations(chart_data, formats=['png'])
            
            if exported_files:
                for name, path in exported_files.items():
                    st.success(f"Chart exported to {path}")
        
        # Add year-over-year comparison if there's sufficient data
        if 'date' in df.columns and len(df['date'].unique()) > 4:
            st.subheader("Year-over-Year Change")
            
            # Calculate year-over-year change
            from src.data_processing.transformers import calculate_year_over_year_change
            
            yoy_df = calculate_year_over_year_change(df, date_col='date', value_col='value')
            
            # Filter out rows with missing YoY values
            yoy_df = yoy_df.dropna(subset=[f'value_yoy'])
            
            if not yoy_df.empty:
                # Create YoY chart
                fig_yoy = go.Figure()
                
                # Add YoY change line
                fig_yoy.add_trace(
                    go.Bar(
                        x=yoy_df['date'],
                        y=yoy_df[f'value_yoy'],
                        name="YoY Change (%)",
                        marker_color='royalblue'
                    )
                )
                
                # Update layout
                fig_yoy.update_layout(
                    title=f"{metric} Year-over-Year Change",
                    xaxis_title="Date",
                    yaxis_title="YoY Change (%)",
                    hovermode="x unified"
                )
                
                # Display chart
                st.plotly_chart(fig_yoy, use_container_width=True)
    
    # Add section for multiple metric comparison
    st.subheader("Multiple Metric Comparison")
    
    # Select metrics to compare
    compare_metrics = st.multiselect(
        "Select metrics to compare",
        options=selected_metrics,
        default=selected_metrics[:min(3, len(selected_metrics))]
    )
    
    if compare_metrics:
        # Filter data dictionary to include only selected metrics
        compare_data = {metric: data_dict[metric] for metric in compare_metrics if metric in data_dict}
        
        # Create combined chart
        fig_combined = create_combined_chart(
            compare_data, 
            compare_metrics, 
            chart_type='line',
            filters=filters
        )
        
        # Display chart
        st.plotly_chart(fig_combined, use_container_width=True)
        
        # Add download link for the combined chart
        if st.button("Download Combined Chart"):
            chart_data = {"Combined Economic Metrics": fig_combined}
            exported_files = export_visualizations(chart_data, formats=['png'])
            
            if exported_files:
                for name, path in exported_files.items():
                    st.success(f"Chart exported to {path}")

def show_correlation_tab(data_dict: Dict[str, pd.DataFrame], 
                         selected_metrics: List[str], 
                         filters: Dict[str, Any], 
                         config: Dict[str, Any]):
    """
    Display the correlation analysis tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
        config (dict): Application configuration.
    """
    st.subheader("Correlation Analysis")
    
    # Create metric selectors for correlation
    col1, col2 = st.columns(2)
    
    with col1:
        metric1 = st.selectbox(
            "Select first metric",
            options=selected_metrics,
            index=0 if selected_metrics else None,
            key="corr_metric1"
        )
    
    with col2:
        metric2 = st.selectbox(
            "Select second metric",
            options=selected_metrics,
            index=min(1, len(selected_metrics) - 1) if len(selected_metrics) > 1 else None,
            key="corr_metric2"
        )
    
    if metric1 and metric2 and metric1 != metric2 and metric1 in data_dict and metric2 in data_dict:
        # Create correlation data
        corr_data = {
            metric1: data_dict[metric1],
            metric2: data_dict[metric2]
        }
        
        # Create correlation chart
        fig_corr = create_correlation_chart(
            corr_data,
            [metric1, metric2],
            filters=filters
        )
        
        # Display chart
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Calculate and display correlation coefficient
        if 'date' in data_dict[metric1].columns and 'date' in data_dict[metric2].columns:
            # Merge data on date
            df1 = data_dict[metric1][['date', 'value']].rename(columns={'value': 'value1'})
            df2 = data_dict[metric2][['date', 'value']].rename(columns={'value': 'value2'})
            
            merged_df = pd.merge(df1, df2, on='date')
            
            # Calculate correlation coefficient
            corr = merged_df['value1'].corr(merged_df['value2'])
            
            # Display correlation coefficient with interpretation
            st.metric("Correlation Coefficient", f"{corr:.3f}")
            
            if corr > 0.7:
                st.info(f"Strong positive correlation: As {metric1} increases, {metric2} tends to strongly increase as well.")
            elif corr > 0.3:
                st.info(f"Moderate positive correlation: As {metric1} increases, {metric2} tends to moderately increase as well.")
            elif corr > -0.3:
                st.info(f"Weak or no correlation: {metric1} and {metric2} do not show a strong relationship.")
            elif corr > -0.7:
                st.info(f"Moderate negative correlation: As {metric1} increases, {metric2} tends to moderately decrease.")
            else:
                st.info(f"Strong negative correlation: As {metric1} increases, {metric2} tends to strongly decrease.")
    
    # Add correlation matrix for all selected metrics
    if len(selected_metrics) > 2:
        st.subheader("Correlation Matrix")
        
        # Create correlation matrix
        corr_matrix = pd.DataFrame(index=selected_metrics, columns=selected_metrics)
        
        for i, metric_i in enumerate(selected_metrics):
            if metric_i not in data_dict:
                continue
                
            df_i = data_dict[metric_i]
            
            for j, metric_j in enumerate(selected_metrics):
                if metric_j not in data_dict:
                    continue
                    
                df_j = data_dict[metric_j]
                
                # Calculate correlation coefficient
                if i == j:
                    corr_matrix.loc[metric_i, metric_j] = 1.0
                else:
                    # Merge data on date
                    merged = pd.merge(
                        df_i[['date', 'value']].rename(columns={'value': 'value_i'}),
                        df_j[['date', 'value']].rename(columns={'value': 'value_j'}),
                        on='date'
                    )
                    
                    # Calculate correlation
                    corr_matrix.loc[metric_i, metric_j] = merged['value_i'].corr(merged['value_j'])
        
        # Create heatmap
        fig_matrix = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            title="Correlation Matrix"
        )
        
        # Update layout
        fig_matrix.update_layout(
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(tickangle=45)
        )
        
        # Display heatmap
        st.plotly_chart(fig_matrix, use_container_width=True)

def show_housing_impact_tab(data_dict: Dict[str, pd.DataFrame], 
                           selected_metrics: List[str], 
                           filters: Dict[str, Any], 
                           config: Dict[str, Any]):
    """
    Display the housing impact tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
        config (dict): Application configuration.
    """
    st.subheader("Housing Market Impact")
    
    # Select economic metric
    eco_metric = st.selectbox(
        "Select economic indicator",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="eco_impact_metric"
    )
    
    # Select housing metric
    housing_metric = st.selectbox(
        "Select housing metric to compare",
        options=config["metrics"]["housing"],
        index=0,
        key="housing_impact_metric"
    )
    
    if eco_metric and housing_metric and eco_metric in data_dict:
        # In a real implementation, we would load the housing data
        # For this demonstration, we'll generate synthetic housing data
        housing_data = generate_synthetic_housing_metric(
            housing_metric,
            filters['date_range']['quarters_list']
        )
        
        if housing_metric not in data_dict:
            data_dict[housing_metric] = housing_data
        
        # Create correlation chart between economic and housing metric
        impact_data = {
            eco_metric: data_dict[eco_metric],
            housing_metric: housing_data
        }
        
        # Create correlation chart
        fig_impact = create_correlation_chart(
            impact_data,
            [eco_metric, housing_metric],
            filters=filters
        )
        
        # Display chart
        st.plotly_chart(fig_impact, use_container_width=True)
        
        # Create overlay chart showing both metrics over time
        st.subheader("Time Series Comparison")
        
        # Create figure with secondary y-axis
        fig_overlay = go.Figure()
        
        # Add economic metric line
        eco_df = data_dict[eco_metric]
        
        fig_overlay.add_trace(
            go.Scatter(
                x=eco_df['date'],
                y=eco_df['value'],
                name=eco_metric,
                line=dict(color='royalblue', width=3)
            )
        )
        
        # Add housing metric line on secondary axis
        housing_df = housing_data
        
        fig_overlay.add_trace(
            go.Scatter(
                x=housing_df['date'],
                y=housing_df['value'],
                name=housing_metric,
                line=dict(color='firebrick', width=3),
                yaxis='y2'
            )
        )
        
        # Set up layout with secondary y-axis
        fig_overlay.update_layout(
            title=f"{eco_metric} vs. {housing_metric} Over Time",
            xaxis=dict(title="Date"),
            yaxis=dict(
                title=eco_metric,
                titlefont=dict(color='royalblue'),
                tickfont=dict(color='royalblue')
            ),
            yaxis2=dict(
                title=housing_metric,
                titlefont=dict(color='firebrick'),
                tickfont=dict(color='firebrick'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        # Display chart
        st.plotly_chart(fig_overlay, use_container_width=True)
        
        # Add lead-lag analysis
        st.subheader("Lead-Lag Analysis")
        
        # Explain lead-lag analysis
        st.write("""
        Lead-lag analysis examines whether changes in one metric (the leading indicator) 
        predict changes in another metric (the lagging indicator) after a time delay.
        """)
        
        # Calculate cross-correlation for different lags
        if 'date' in eco_df.columns and 'date' in housing_df.columns:
            # Merge data on date
            merged_df = pd.merge(
                eco_df[['date', 'value']].rename(columns={'value': 'eco_value'}),
                housing_df[['date', 'value']].rename(columns={'value': 'housing_value'}),
                on='date'
            )
            
            if len(merged_df) > 3:  # Need at least 4 data points for meaningful analysis
                # Sort by date
                merged_df = merged_df.sort_values('date')
                
                # Calculate correlations for different lags
                max_lag = min(4, len(merged_df) - 2)  # Maximum lag is either 4 quarters or length - 2
                lags = list(range(-max_lag, max_lag + 1))
                corrs = []
                
                for lag in lags:
                    if lag < 0:
                        # Housing leads economy (housing shifted forward)
                        shifted_housing = merged_df['housing_value'].shift(-lag)
                        corr = merged_df['eco_value'].corr(shifted_housing)
                    else:
                        # Economy leads housing (economy shifted forward)
                        shifted_eco = merged_df['eco_value'].shift(lag)
                        corr = shifted_eco.corr(merged_df['housing_value'])
                    
                    corrs.append(corr)
                
                # Create lag correlation chart
                lag_df = pd.DataFrame({
                    'Lag (Quarters)': lags,
                    'Correlation': corrs
                })
                
                fig_lag = px.bar(
                    lag_df,
                    x='Lag (Quarters)',
                    y='Correlation',
                    title="Lead-Lag Correlation Analysis",
                    labels={'Correlation': 'Correlation Coefficient'},
                    color='Correlation',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0
                )
                
                # Update layout
                fig_lag.update_layout(
                    xaxis=dict(tickvals=lags),
                    yaxis=dict(range=[-1, 1]),
                    hovermode='x'
                )
                
                # Display chart
                st.plotly_chart(fig_lag, use_container_width=True)
                
                # Find highest correlation and interpret
                max_corr_idx = np.argmax(np.abs(corrs))
                max_corr_lag = lags[max_corr_idx]
                max_corr = corrs[max_corr_idx]
                
                if max_corr_lag < 0:
                    lag_interpretation = f"{housing_metric} appears to lead {eco_metric} by {abs(max_corr_lag)} quarters"
                elif max_corr_lag > 0:
                    lag_interpretation = f"{eco_metric} appears to lead {housing_metric} by {max_corr_lag} quarters"
                else:
                    lag_interpretation = f"{eco_metric} and {housing_metric} appear to move contemporaneously (no lead/lag)"
                
                st.info(f"**Lead-Lag Relationship:** {lag_interpretation} with correlation of {max_corr:.3f}")
            else:
                st.info("Insufficient data for lead-lag analysis. Requires at least 4 data points.")

def generate_synthetic_economic_data(selected_metrics: List[str], 
                                    quarters_list: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic economic data for demonstration purposes.
    
    Args:
        selected_metrics: List of selected economic metrics
        quarters_list: List of quarters in the selected date range
        
    Returns:
        Dictionary mapping metric names to DataFrames
    """
    data_dict = {}
    
    # Convert quarters list to proper dates
    dates = []
    for quarter in quarters_list:
        year, q = quarter.split()
        quarter_num = int(q[1])
        month = ((quarter_num - 1) * 3) + 1
        dates.append(pd.Timestamp(f"{year}-{month:02d}-01"))
    
    for metric in selected_metrics:
        # Generate random trend with some seasonality
        base = 100 if "Rate" not in metric else 5
        
        # Different metrics have different trends
        if metric == "Local Job Growth":
            values = np.linspace(2, 4, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.5 + np.random.normal(0, 0.2, len(dates))
        elif metric == "Unemployment Rate":
            values = np.linspace(5, 3, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.3 + np.random.normal(0, 0.1, len(dates))
        elif metric == "Median Household Income":
            values = np.linspace(85000, 105000, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 2000 + np.random.normal(0, 1000, len(dates))
        elif metric == "GDP Growth":
            values = np.linspace(2.5, 3.5, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.4 + np.random.normal(0, 0.2, len(dates))
        elif metric == "Population Growth":
            values = np.linspace(0.8, 1.2, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.1 + np.random.normal(0, 0.05, len(dates))
        elif metric == "Consumer Price Index":
            values = np.linspace(250, 300, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 5 + np.random.normal(0, 2, len(dates))
        elif metric == "Interest Rates":
            values = np.linspace(4, 7, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.3 + np.random.normal(0, 0.1, len(dates))
        else:
            # Generic trend for other metrics
            values = np.linspace(base, base * 1.3, len(dates)) + np.sin(np.linspace(0, 4*np.pi, len(dates))) * (base * 0.05) + np.random.normal(0, base * 0.02, len(dates))
        
        # Create dataframe
        df = pd.DataFrame({
            'date': dates,
            'value': values
        })
        
        df = df.set_index('date')
        data_dict[metric] = df
    
    return data_dict

def generate_synthetic_housing_metric(metric, quarters):
    """
    Generate synthetic housing metric data for demonstration.
    
    Args:
        metric (str): Housing metric to generate.
        quarters (list): List of quarters.
    
    Returns:
        pd.DataFrame: DataFrame with synthetic housing data.
    """
    # Parameters for the housing metric
    metric_params = {
        "Median Sale Price": {"base": 750000, "trend": 10000, "volatility": 20000, "seasonality": [0.02, 0.05, -0.01, -0.05]},
        "Average Sale Price": {"base": 900000, "trend": 12000, "volatility": 30000, "seasonality": [0.02, 0.05, -0.01, -0.05]},
        "Sales Volume": {"base": 1000, "trend": -10, "volatility": 100, "seasonality": [-0.2, 0.3, 0.1, -0.2]},
        "Days on Market": {"base": 45, "trend": -0.5, "volatility": 5, "seasonality": [0.1, -0.1, -0.2, 0.2]},
        "Listing Inventory": {"base": 2500, "trend": -20, "volatility": 150, "seasonality": [-0.1, 0.2, 0.1, -0.2]},
        "Months of Supply": {"base": 3.5, "trend": -0.03, "volatility": 0.3, "seasonality": [-0.1, 0.1, 0.15, -0.15]},
        "New Listings": {"base": 800, "trend": -5, "volatility": 80, "seasonality": [-0.2, 0.4, 0.1, -0.3]},
        "Absorption Rate": {"base": 30, "trend": 0.2, "volatility": 3, "seasonality": [-0.1, 0.2, 0.1, -0.2]},
        "Pending Home Sales": {"base": 700, "trend": -8, "volatility": 70, "seasonality": [-0.1, 0.3, 0.1, -0.3]},
        "LP/SP Ratio": {"base": 98, "trend": -0.1, "volatility": 1, "seasonality": [0.01, 0.02, -0.01, -0.02]},
        "Home Price-to-Income Ratio": {"base": 5.8, "trend": 0.05, "volatility": 0.2, "seasonality": [0.01, 0.02, -0.01, -0.02]},
        "Mortgage Rates": {"base": 6.5, "trend": 0.05, "volatility": 0.2, "seasonality": [-0.05, 0.1, 0.05, -0.1]},
        "Housing Affordability Index": {"base": 95, "trend": -0.3, "volatility": 3, "seasonality": [0.02, -0.02, -0.01, 0.01]},
        "Vacancy Rates": {"base": 3.8, "trend": -0.02, "volatility": 0.3, "seasonality": [0.05, -0.05, -0.05, 0.05]},
        "Seller Concessions": {"base": 1.2, "trend": 0.03, "volatility": 0.2, "seasonality": [0.1, -0.05, -0.1, 0.05]}
    }
    
    # Default to Median Sale Price if metric not available
    params = metric_params.get(metric, metric_params["Median Sale Price"])
    
    base = params["base"]
    trend = params["trend"]
    volatility = params["volatility"]
    seasonality = params["seasonality"]
    
    data = []
    
    # Convert quarters to datetime
    quarters_dt = [pd.to_datetime(q.replace(" Q", "-")) for q in quarters]
    
    for i, quarter in enumerate(quarters_dt):
        # Calculate quarter index (0-3)
        quarter_idx = quarter.quarter - 1
        
        # Calculate trend component
        trend_value = trend * i
        
        # Calculate seasonal component
        seasonal_value = base * seasonality[quarter_idx]
        
        # Calculate random component
        random_value = np.random.normal(0, volatility)
        
        # Combined value
        value = base + trend_value + seasonal_value + random_value
        
        # Ensure non-negative values for appropriate metrics
        if metric in ["Sales Volume", "Days on Market", "Listing Inventory", "Months of Supply", "New Listings", "Absorption Rate", "Pending Home Sales", "Vacancy Rates"]:
            value = max(value, 0)
        
        # Add data point
        data.append({
            "date": quarter,
            "value": value,
            "metric": metric
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df
