"""
Comparison Page

This module renders the Comparison page of the application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.utils.config import load_config
from src.visualization.charts import create_comparison_chart
from src.visualization.maps import create_town_comparison_map
from src.visualization.dashboards import create_comparison_dashboard
from src.visualization.exporters import export_visualizations, create_download_link

def show_comparison_page(filters: Dict[str, Any]):
    """
    Display the comparison page.
    
    Args:
        filters: Dictionary containing filter selections.
    """
    # Load configuration
    config = load_config()
    
    # Set page title
    st.title("Town Comparison")
    st.write("Compare housing market metrics across different towns in Fairfield County")
    
    # Display current filter selections
    st.write(f"**Selected Time Period:** {filters['date_range']['start_quarter']} to {filters['date_range']['end_quarter']}")
    if filters['location']['towns'] != config['locations']['towns']:
        st.write(f"**Selected Towns:** {', '.join(filters['location']['towns'])}")
    if filters['property_types'] != config['property_types']:
        st.write(f"**Selected Property Types:** {', '.join(filters['property_types'])}")
    
    # Create location and metric selectors
    col1, col2 = st.columns(2)
    
    with col1:
        selected_towns = st.multiselect(
            "Select towns to compare",
            options=filters['location']['towns'],
            default=filters['location']['towns'][:min(5, len(filters['location']['towns']))]
        )
    
    with col2:
        selected_metrics = st.multiselect(
            "Select metrics to compare",
            options=config["metrics"]["housing"],
            default=["Median Sale Price", "Sales Volume", "Days on Market"]
        )
    
    if not selected_towns:
        st.warning("Please select at least one town to compare.")
        return
    
    if not selected_metrics:
        st.warning("Please select at least one metric to compare.")
        return
    
    # In a real implementation, we would load actual data from the processed data directory
    # For this demonstration, we'll generate synthetic data
    data_dict = generate_synthetic_comparison_data(
        selected_metrics, 
        filters['date_range']['quarters_list'],
        selected_towns
    )
    
    # Create tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs(["Town Rankings", "Trend Comparison", "Map View", "Summary Table"])
    
    with tab1:
        show_rankings_tab(data_dict, selected_metrics, selected_towns, filters)
    
    with tab2:
        show_trend_comparison_tab(data_dict, selected_metrics, selected_towns, filters)
    
    with tab3:
        show_map_view_tab(data_dict, selected_metrics, selected_towns, filters)
    
    with tab4:
        show_summary_table_tab(data_dict, selected_metrics, selected_towns, filters)

def show_rankings_tab(data_dict: Dict[str, pd.DataFrame],
                     selected_metrics: List[str],
                     selected_towns: List[str],
                     filters: Dict[str, Any]):
    """
    Display the rankings tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        selected_towns (list): List of selected towns.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Town Rankings")
    
    # Create metric selector for ranking
    metric = st.selectbox(
        "Select metric for ranking",
        options=selected_metrics,
        index=0 if selected_metrics else None
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Filter for selected towns
        df = df[df['town'].isin(selected_towns)]
        
        # Get the most recent period
        latest_date = df['date'].max()
        latest_df = df[df['date'] == latest_date]
        
        # Create ranking bar chart
        fig = px.bar(
            latest_df.sort_values('value', ascending=False),
            x='town',
            y='value',
            title=f"{metric} by Town (Latest: {latest_date.strftime('%Y-%m-%d')})",
            labels={'value': metric, 'town': 'Town'},
            color='value',
            color_continuous_scale='Blues'
        )
        
        # Format y-axis if metric is price or dollar amount
        if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
            fig.update_layout(yaxis=dict(tickprefix='$', tickformat=','))
        
        # Format y-axis if metric is percentage
        if any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
            fig.update_layout(yaxis=dict(ticksuffix='%'))
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate and display growth ranking
        st.subheader("Growth Ranking")
        
        # Calculate year-over-year growth for each town
        if len(df['date'].unique()) >= 5:  # Need at least 5 quarters for meaningful YoY growth
            # Calculate year-ago date (approximately 4 quarters ago)
            date_list = sorted(df['date'].unique())
            if len(date_list) >= 5:
                current_date = date_list[-1]
                year_ago_date = date_list[-5]  # Approximately 1 year ago
                
                # Get current and year-ago data
                current_df = df[df['date'] == current_date]
                year_ago_df = df[df['date'] == year_ago_date]
                
                # Merge data to calculate growth
                growth_df = pd.merge(
                    current_df[['town', 'value']].rename(columns={'value': 'current_value'}),
                    year_ago_df[['town', 'value']].rename(columns={'value': 'year_ago_value'}),
                    on='town'
                )
                
                # Calculate growth percentage
                growth_df['growth_pct'] = (growth_df['current_value'] - growth_df['year_ago_value']) / growth_df['year_ago_value'] * 100
                
                # Create growth ranking chart
                fig_growth = px.bar(
                    growth_df.sort_values('growth_pct', ascending=False),
                    x='town',
                    y='growth_pct',
                    title=f"{metric} Year-over-Year Growth by Town",
                    labels={'growth_pct': 'YoY Change (%)', 'town': 'Town'},
                    color='growth_pct',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0
                )
                
                # Update layout
                fig_growth.update_layout(yaxis=dict(ticksuffix='%'))
                
                # Display chart
                st.plotly_chart(fig_growth, use_container_width=True)
            else:
                st.info(f"Insufficient historical data to calculate year-over-year growth for {metric}.")
        else:
            st.info(f"Insufficient historical data to calculate year-over-year growth for {metric}.")

def show_trend_comparison_tab(data_dict: Dict[str, pd.DataFrame],
                             selected_metrics: List[str],
                             selected_towns: List[str],
                             filters: Dict[str, Any]):
    """
    Display the trend comparison tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        selected_towns (list): List of selected towns.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Trend Comparison")
    
    # Create metric selector for trend comparison
    metric = st.selectbox(
        "Select metric for trend comparison",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="trend_metric"
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Filter for selected towns
        df = df[df['town'].isin(selected_towns)]
        
        # Create town trend chart
        fig = px.line(
            df,
            x='date',
            y='value',
            color='town',
            title=f"{metric} Trends by Town",
            labels={'value': metric, 'date': 'Date', 'town': 'Town'},
            line_shape='linear'
        )
        
        # Format y-axis if metric is price or dollar amount
        if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
            fig.update_layout(yaxis=dict(tickprefix='$', tickformat=','))
        
        # Format y-axis if metric is percentage
        if any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
            fig.update_layout(yaxis=dict(ticksuffix='%'))
        
        # Update layout
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add indexed trend comparison option
        st.subheader("Indexed Trend Comparison")
        
        # Create indexed data (all towns start at 100 at the beginning)
        if len(df['date'].unique()) > 1:
            # Group by town and create indexed values
            towns_indexed = []
            
            for town, town_df in df.groupby('town'):
                # Sort by date
                town_df = town_df.sort_values('date')
                
                # Get base value (first period)
                base_value = town_df['value'].iloc[0]
                
                # Calculate indexed values
                town_df['indexed_value'] = town_df['value'] / base_value * 100
                
                towns_indexed.append(town_df)
            
            # Combine all towns
            indexed_df = pd.concat(towns_indexed)
            
            # Create indexed trend chart
            fig_indexed = px.line(
                indexed_df,
                x='date',
                y='indexed_value',
                color='town',
                title=f"{metric} Indexed Trends by Town (Base = 100)",
                labels={'indexed_value': 'Indexed Value (Base = 100)', 'date': 'Date', 'town': 'Town'},
                line_shape='linear'
            )
            
            # Update layout
            fig_indexed.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Display chart
            st.plotly_chart(fig_indexed, use_container_width=True)
        else:
            st.info(f"Insufficient data to create indexed trends for {metric}.")

def show_map_view_tab(data_dict: Dict[str, pd.DataFrame],
                     selected_metrics: List[str],
                     selected_towns: List[str],
                     filters: Dict[str, Any]):
    """
    Display the map view tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        selected_towns (list): List of selected towns.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Geographic Comparison")
    
    # Create metric selector for map view
    metric = st.selectbox(
        "Select metric for map visualization",
        options=selected_metrics,
        index=0 if selected_metrics else None,
        key="map_metric"
    )
    
    if metric and metric in data_dict:
        # Get data for the selected metric
        df = data_dict[metric]
        
        # Filter for selected towns
        df = df[df['town'].isin(selected_towns)]
        
        # In a real implementation, we would use a proper choropleth map with GeoJSON data
        # For this demonstration, we'll use a simplified bubble map
        
        # Get the most recent period
        latest_date = df['date'].max()
        latest_df = df[df['date'] == latest_date]
        
        # Create a bubble map (simulated here with a scatter plot)
        st.info("Interactive map view would be displayed here using GeoJSON data for Fairfield County towns.")
        
        # Get previous period for comparison
        date_list = sorted(df['date'].unique())
        if len(date_list) >= 2:
            prev_date = date_list[-2]
            prev_df = df[df['date'] == prev_date]
            
            # Merge data to calculate change
            change_df = pd.merge(
                latest_df[['town', 'value']].rename(columns={'value': 'current_value'}),
                prev_df[['town', 'value']].rename(columns={'value': 'prev_value'}),
                on='town'
            )
            
            # Calculate change percentage
            change_df['change_pct'] = (change_df['current_value'] - change_df['prev_value']) / change_df['prev_value'] * 100
            
            # Create bubble map simulation
            fig = px.scatter(
                change_df,
                x=range(len(change_df)),  # Placeholder for x-coordinate
                y=range(len(change_df)),  # Placeholder for y-coordinate
                size='current_value',
                color='change_pct',
                hover_name='town',
                text='town',
                title=f"{metric} by Town (Latest: {latest_date.strftime('%Y-%m-%d')})",
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            
            # Update layout to simulate a map
            fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False
            )
            
            # Display simulated map
            st.plotly_chart(fig, use_container_width=True)
            
            # Display explanatory text
            st.write("""
            **Map visualization explanation:**
            - Bubble size represents the current value of the selected metric
            - Color represents the percentage change from the previous period (blue = increase, red = decrease)
            - Hover over bubbles to see town details
            """)
            
            # Display the data in a table for clarity
            st.subheader("Geographic Data Table")
            
            # Format the table data
            table_data = change_df.copy()
            
            # Format values based on metric type
            if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
                table_data['current_value'] = table_data['current_value'].map('${:,.0f}'.format)
                table_data['prev_value'] = table_data['prev_value'].map('${:,.0f}'.format)
            elif any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
                table_data['current_value'] = table_data['current_value'].map('{:.2f}%'.format)
                table_data['prev_value'] = table_data['prev_value'].map('{:.2f}%'.format)
            else:
                table_data['current_value'] = table_data['current_value'].map('{:.2f}'.format)
                table_data['prev_value'] = table_data['prev_value'].map('{:.2f}'.format)
            
            # Format change percentage
            table_data['change_pct'] = table_data['change_pct'].map('{:+.2f}%'.format)
            
            # Rename columns for display
            table_data.columns = ['Town', 'Current Value', 'Previous Value', 'Change']
            
            # Display table
            st.dataframe(table_data, use_container_width=True)
        else:
            st.info(f"Insufficient data to create change comparison for {metric}.")

def show_summary_table_tab(data_dict: Dict[str, pd.DataFrame],
                          selected_metrics: List[str],
                          selected_towns: List[str],
                          filters: Dict[str, Any]):
    """
    Display the summary table tab.
    
    Args:
        data_dict (dict): Dictionary of DataFrames keyed by metric.
        selected_metrics (list): List of selected metrics.
        selected_towns (list): List of selected towns.
        filters (dict): Dictionary containing filter selections.
    """
    st.subheader("Summary Comparison Table")
    
    # Create a summary table for all metrics and towns
    summary_data = []
    
    # Get the most recent period for each metric
    for metric in selected_metrics:
        if metric not in data_dict:
            continue
        
        df = data_dict[metric]
        df = df[df['town'].isin(selected_towns)]
        
        # Get the most recent period
        latest_date = df['date'].max()
        latest_df = df[df['date'] == latest_date]
        
        for _, row in latest_df.iterrows():
            town = row['town']
            value = row['value']
            
            # Format value based on metric type
            if any(term in metric.lower() for term in ['price', 'value', 'dollar', 'income']):
                formatted_value = f"${value:,.0f}"
            elif any(term in metric.lower() for term in ['rate', 'ratio', 'percentage', '%']):
                formatted_value = f"{value:.2f}%"
            else:
                formatted_value = f"{value:.2f}"
            
            # Add to summary data
            summary_data.append({
                'Town': town,
                'Metric': metric,
                'Value': formatted_value,
                'Raw Value': value  # For sorting
            })
    
    if summary_data:
        # Create pivot table
        summary_df = pd.DataFrame(summary_data)
        pivot_df = summary_df.pivot(index='Town', columns='Metric', values='Value')
        
        # Display pivot table
        st.dataframe(pivot_df, use_container_width=True)
        
        # Add download link for the summary table
        if st.button("Download Summary Table"):
            csv = pivot_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="town_comparison_summary.csv",
                mime="text/csv"
            )
        
        # Add heatmap visualization of the summary table
        st.subheader("Metric Heatmap by Town")
        
        # Create heatmap data
        heatmap_df = summary_df.pivot(index='Town', columns='Metric', values='Raw Value')
        
        # Normalize each metric column (0-1 scale)
        normalized_df = pd.DataFrame(index=heatmap_df.index)
        
        for col in heatmap_df.columns:
            min_val = heatmap_df[col].min()
            max_val = heatmap_df[col].max()
            
            if max_val > min_val:
                normalized_df[col] = (heatmap_df[col] - min_val) / (max_val - min_val)
            else:
                normalized_df[col] = 0.5  # Default if all values are the same
        
        # Create heatmap
        fig = px.imshow(
            normalized_df,
            labels=dict(x="Metric", y="Town", color="Normalized Value"),
            x=normalized_df.columns,
            y=normalized_df.index,
            color_continuous_scale="Viridis",
            aspect="auto"
        )
        
        # Update layout
        fig.update_layout(
            title="Normalized Metrics by Town (Higher values are brighter)",
            xaxis=dict(tickangle=45),
            margin=dict(l=50, r=50, t=50, b=100)
        )
        
        # Display heatmap
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for summary table.")

def generate_synthetic_comparison_data(selected_metrics: List[str],
                                      quarters_list: List[str],
                                      towns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic data for town comparison.
    
    Args:
        selected_metrics: List of selected metrics
        quarters_list: List of quarters in date range
        towns: List of towns to generate data for
        
    Returns:
        Dictionary of DataFrames with synthetic comparison data
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
        # Base parameters for each metric
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
        
        # Create dataframe with all towns and dates
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
        
        df = pd.DataFrame(rows)
        data_dict[metric] = df
    
    return data_dict