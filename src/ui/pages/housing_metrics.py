import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

from src.utils.config import load_config
from src.data_processing.metrics_calculator import calculate_derived_metrics
from src.visualization.charts import create_metric_chart, create_combined_chart, create_comparison_chart
from src.visualization.dashboards import create_housing_metrics_dashboard
from src.visualization.exporters import export_visualizations, create_download_link

def show_housing_metrics_page(filters):
    """
    Display the housing metrics page.
    
    Args:
        filters (dict): Dictionary containing filter selections.
    """
    # Load configuration
    config = load_config()
    
    # Set page title
    st.title("Housing Market Metrics")
    st.write("Detailed analysis of housing market metrics in Fairfield County")
    
    # Display current filter selections
    st.write(f"**Selected Time Period:** {filters['date_range']['start_quarter']} to {filters['date_range']['end_quarter']}")
    if filters['location']['towns'] != config['locations']['towns']:
        st.write(f"**Selected Towns:** {', '.join(filters['location']['towns'])}")
    if filters['property_types'] != config['property_types']:
        st.write(f"**Selected Property Types:** {', '.join(filters['property_types'])}")
    
    # Display selected metrics
    selected_metrics = st.multiselect(
        "Select metrics to display",
        options=config["metrics"]["housing"],
        default=["Median Sale Price", "Sales Volume", "Days on Market"]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to display.")
        return
    
    # In a real implementation, we would load actual data from the processed data directory
    # For this demonstration, we'll generate synthetic data
    data_dict = generate_synthetic_housing_data(
        selected_metrics, 
        filters['date_range']['quarters_list'],
        filters['location']['towns']
    )
    
    # Calculate derived metrics if needed
    calculated_data = calculate_derived_metrics(data_dict)
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Trends", "Comparison", "Details"])
    
    with tab1:
        show_trends_tab(calculated_data, selected_metrics, filters)
    
    with tab2:
        show_comparison_tab(calculated_data, selected_metrics, filters)
    
    with tab3:
        show_details_tab(calculated_data, selected_metrics, filters)

def show_trends_tab(data_dict, selected_metrics, filters):
    """
    Display the trends tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Market Trends")
    
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
            chart_data = {"Combined Metrics": fig_combined}
            exported_files = export_visualizations(chart_data, formats=['png'])
            
            if exported_files:
                for name, path in exported_files.items():
                    st.success(f"Chart exported to {path}")

def show_comparison_tab(data_dict, selected_metrics, filters):
    """
    Display the comparison tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Town Comparison")
    
    # Create metric selector for comparison
    metric = st.selectbox(
        "Select metric for town comparison",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="comparison_metric"
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Check if town column exists
        if 'town' in df.columns:
            # Create comparison chart
            fig = create_comparison_chart(
                df, 
                metric, 
                'town', 
                chart_type='bar',
                filters=filters
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Create town comparison over time
            st.subheader("Town Comparison Over Time")
            
            # Select towns to compare
            towns = st.multiselect(
                "Select towns to compare",
                options=filters['location']['towns'],
                default=filters['location']['towns'][:min(5, len(filters['location']['towns']))]
            )
            
            if towns and 'date' in df.columns:
                # Filter data for selected towns
                town_df = df[df['town'].isin(towns)]
                
                # Create line chart by town
                fig_towns = px.line(
                    town_df,
                    x='date',
                    y='value',
                    color='town',
                    title=f"{metric} by Town",
                    labels={'value': metric, 'date': 'Date'}
                )
                
                # Format y-axis if metric is price or dollar amount
                if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
                    fig_towns.update_layout(yaxis=dict(tickprefix='$', tickformat=','))
                
                # Format y-axis if metric is percentage
                if any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
                    fig_towns.update_layout(yaxis=dict(ticksuffix='%'))
                
                # Display chart
                st.plotly_chart(fig_towns, use_container_width=True)
        else:
            st.info(f"The {metric} data does not have town-level granularity for comparison.")
    
    # Add property type comparison section
    st.subheader("Property Type Comparison")
    
    # Create metric selector for property type comparison
    prop_metric = st.selectbox(
        "Select metric for property type comparison",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="property_metric"
    )
    
    if prop_metric and prop_metric in data_dict:
        # Get data for the selected metric
        df = data_dict[prop_metric]
        
        # Check if property_type column exists
        if 'property_type' in df.columns:
            # Create comparison chart
            fig = create_comparison_chart(
                df, 
                prop_metric, 
                'property_type', 
                chart_type='bar',
                filters=filters
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Create property type comparison over time
            st.subheader("Property Type Comparison Over Time")
            
            if 'date' in df.columns:
                # Create line chart by property type
                fig_prop = px.line(
                    df,
                    x='date',
                    y='value',
                    color='property_type',
                    title=f"{prop_metric} by Property Type",
                    labels={'value': prop_metric, 'date': 'Date'}
                )
                
                # Format y-axis if metric is price or dollar amount
                if any(term in prop_metric.lower() for term in ['price', 'value', 'dollar', 'income']):
                    fig_prop.update_layout(yaxis=dict(tickprefix='$', tickformat=','))
                
                # Format y-axis if metric is percentage
                if any(term in prop_metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
                    fig_prop.update_layout(yaxis=dict(ticksuffix='%'))
                
                # Display chart
                st.plotly_chart(fig_prop, use_container_width=True)
        else:
            st.info(f"The {prop_metric} data does not have property type granularity for comparison.")

def show_details_tab(data_dict, selected_metrics, filters):
    """
    Display the details tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Detailed Data")
    
    # Create metric selector for detailed data
    metric = st.selectbox(
        "Select metric for detailed data",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="detail_metric"
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Display the data table
        st.dataframe(df)
        
        # Add download link for the data
        if st.button("Download Data as CSV"):
            # Create a CSV string
            csv = df.to_csv(index=False)
            
            # Create a download link
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{metric.lower().replace(' ', '_')}_data.csv",
                mime="text/csv"
            )
        
        # Display summary statistics
        st.subheader("Summary Statistics")
        
        # Calculate basic statistics
        if 'value' in df.columns:
            stats_df = df['value'].describe().reset_index()
            stats_df.columns = ['Statistic', 'Value']
            
            # Display statistics
            st.table(stats_df)
            
            # Create histogram of values
            fig_hist = px.histogram(
                df,
                x='value',
                title=f"Distribution of {metric} Values",
                labels={'value': metric},
                nbins=20
            )
            
            # Format x-axis if metric is price or dollar amount
            if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
                fig_hist.update_layout(xaxis=dict(tickprefix='$', tickformat=','))
            
            # Format x-axis if metric is percentage
            if any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
                fig_hist.update_layout(xaxis=dict(ticksuffix='%'))
            
            # Display histogram
            st.plotly_chart(fig_hist, use_container_width=True)

def generate_synthetic_housing_data(metrics, quarters, towns):
    """
    Generate synthetic housing market data for demonstration.
    
    Args:
        metrics (list): List of metrics to generate.
        quarters (list): List of quarters.
        towns (list): List of towns.
    
    Returns:
        dict: Dictionary of synthetic DataFrames keyed by metric.
    """
    # Parameters for each metric
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
    
    # Town characteristics (relative to base)
    town_factors = {
        "Bridgeport": 0.6,
        "Danbury": 0.8,
        "Darien": 1.7,
        "Easton": 1.2,
        "Fairfield": 1.1,
        "Greenwich": 2.0,
        "Monroe": 0.9,
        "New Canaan": 1.8,
        "New Fairfield": 0.8,
        "Newtown": 0.9,
        "Norwalk": 0.9,
        "Redding": 1.1,
        "Ridgefield": 1.3,
        "Shelton": 0.7,
        "Stamford": 1.1,
        "Stratford": 0.7,
        "Trumbull": 0.9,
        "Weston": 1.5,
        "Westport": 1.8,
        "Wilton": 1.4
    }
    
    # Property type factors (relative to base)
    property_type_factors = {
        "Single-family": 1.1,
        "Multi-unit": 1.5,
        "Condo/Townhouse": 0.7
    }
    
    # Generate data for each metric
    data_dict = {}
    
    for metric in metrics:
        if metric not in metric_params:
            continue
        
        params = metric_params[metric]
        base = params["base"]
        trend = params["trend"]
        volatility = params["volatility"]
        seasonality = params["seasonality"]
        
        data = []
        
        # Convert quarters to datetime
        quarters_dt = [pd.to_datetime(q.replace(" Q", "-")) for q in quarters]
        
        # Generate data for each town and property type
        for town in towns:
            town_factor = town_factors.get(town, 1.0)
            
            for prop_type in ["Single-family", "Multi-unit", "Condo/Townhouse"]:
                prop_factor = property_type_factors.get(prop_type, 1.0)
                
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
                    value = base * town_factor * prop_factor + trend_value + seasonal_value + random_value
                    
                    # Ensure non-negative values for appropriate metrics
                    if metric in ["Sales Volume", "Days on Market", "Listing Inventory", "Months of Supply", "New Listings", "Absorption Rate", "Pending Home Sales", "Vacancy Rates"]:
                        value = max(value, 0)
                    
                    # Add data point
                    data.append({
                        "date": quarter,
                        "value": value,
                        "town": town,
                        "property_type": prop_type,
                        "metric": metric
                    })
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Store in dictionary
        data_dict[metric] = df
    
    return data_dict
