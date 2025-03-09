import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Callable
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_metrics_row(
    metric_data: Dict[str, Tuple[float, Optional[float], str]],
    columns: int = 3
) -> None:
    """
    Create a row of metric cards with optional percent change indicators
    
    Args:
        metric_data: Dictionary with metric name as key and tuple of 
                     (current value, percent change, description) as value
        columns: Number of columns to arrange metrics in
    """
    logger.info(f"Creating metrics row with {len(metric_data)} metrics")
    
    # Create columns with the specified count
    cols = st.columns(columns)
    
    # Distribute metrics across columns
    for i, (metric_name, (value, pct_change, description)) in enumerate(metric_data.items()):
        col_idx = i % columns
        
        with cols[col_idx]:
            st.metric(
                label=metric_name,
                value=value,
                delta=f"{pct_change:.2f}%" if pct_change is not None else None,
                delta_color="normal",
                help=description
            )

def create_tab_layout(
    tabs: List[str],
    tab_contents: List[Callable[[], None]]
) -> None:
    """
    Create a tabbed interface with content functions for each tab
    
    Args:
        tabs: List of tab names
        tab_contents: List of functions to call for each tab's content
    """
    logger.info(f"Creating tab layout with {len(tabs)} tabs")
    
    if len(tabs) != len(tab_contents):
        logger.error(f"Mismatch between tab names ({len(tabs)}) and content functions ({len(tab_contents)})")
        return
    
    # Create the tabs
    streamlit_tabs = st.tabs(tabs)
    
    # Populate each tab with its content
    for i, tab in enumerate(streamlit_tabs):
        with tab:
            tab_contents[i]()

def create_two_column_chart_layout(
    left_chart: go.Figure,
    right_chart: go.Figure,
    left_title: str = "",
    right_title: str = "",
    left_description: str = "",
    right_description: str = "",
    height: int = 400
) -> None:
    """
    Create a two-column layout with charts side by side
    
    Args:
        left_chart: Plotly figure for left column
        right_chart: Plotly figure for right column
        left_title: Title for left chart section
        right_title: Title for right chart section
        left_description: Description text for left chart
        right_description: Description text for right chart
        height: Height for both charts in pixels
    """
    logger.info(f"Creating two-column chart layout with {left_title} and {right_title}")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Left column
    with col1:
        if left_title:
            st.subheader(left_title)
        if left_description:
            st.write(left_description)
        # Adjust chart height
        left_chart.update_layout(height=height)
        st.plotly_chart(left_chart, use_container_width=True)
    
    # Right column
    with col2:
        if right_title:
            st.subheader(right_title)
        if right_description:
            st.write(right_description)
        # Adjust chart height
        right_chart.update_layout(height=height)
        st.plotly_chart(right_chart, use_container_width=True)

def create_comparison_dashboard(
    charts: List[go.Figure],
    titles: List[str],
    descriptions: Optional[List[str]] = None,
    num_columns: int = 2
) -> None:
    """
    Create a dashboard with multiple charts arranged in a grid
    
    Args:
        charts: List of Plotly figures to display
        titles: List of titles for each chart
        descriptions: Optional list of descriptions for each chart
        num_columns: Number of columns in the grid
    """
    logger.info(f"Creating comparison dashboard with {len(charts)} charts in {num_columns} columns")
    
    if descriptions is None:
        descriptions = [""] * len(charts)
    
    # Calculate number of rows needed
    num_rows = (len(charts) + num_columns - 1) // num_columns
    
    # Create each row
    for row in range(num_rows):
        # Create columns for this row
        cols = st.columns(num_columns)
        
        # Add charts to columns in this row
        for col in range(num_columns):
            chart_idx = row * num_columns + col
            
            # Check if there's a chart for this position
            if chart_idx < len(charts):
                with cols[col]:
                    if titles[chart_idx]:
                        st.subheader(titles[chart_idx])
                    if descriptions[chart_idx]:
                        st.write(descriptions[chart_idx])
                    # Display chart
                    st.plotly_chart(charts[chart_idx], use_container_width=True)

def create_time_series_dashboard(
    time_series_data: pd.DataFrame,
    date_column: str,
    metric_columns: List[str],
    chart_titles: Optional[List[str]] = None,
    chart_descriptions: Optional[List[str]] = None,
    include_percent_change: bool = True,
    num_columns: int = 2
) -> None:
    """
    Create a dashboard with multiple time series charts
    
    Args:
        time_series_data: DataFrame containing time series data
        date_column: Column name for dates
        metric_columns: List of column names for metrics to display
        chart_titles: Optional list of titles for each chart
        chart_descriptions: Optional list of descriptions for each chart
        include_percent_change: Whether to include a percent change chart for each metric
        num_columns: Number of columns in the grid
    """
    logger.info(f"Creating time series dashboard for {len(metric_columns)} metrics")
    
    # Use metric names as titles if not provided
    if chart_titles is None:
        chart_titles = metric_columns
    
    # Use empty descriptions if not provided
    if chart_descriptions is None:
        chart_descriptions = [""] * len(metric_columns)
    
    # Calculate number of rows needed
    num_rows = (len(metric_columns) + num_columns - 1) // num_columns
    
    # Create each row
    for row in range(num_rows):
        # Create columns for this row
        cols = st.columns(num_columns)
        
        # Add charts to columns in this row
        for col in range(num_columns):
            metric_idx = row * num_columns + col
            
            # Check if there's a metric for this position
            if metric_idx < len(metric_columns):
                metric = metric_columns[metric_idx]
                title = chart_titles[metric_idx]
                
                with cols[col]:
                    if title:
                        st.subheader(title)
                    if chart_descriptions[metric_idx]:
                        st.write(chart_descriptions[metric_idx])
                    
                    # Create time series chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=time_series_data[date_column],
                        y=time_series_data[metric],
                        mode='lines',
                        name=metric
                    ))
                    
                    # Enhance layout
                    fig.update_layout(
                        xaxis_title=date_column,
                        yaxis_title=metric,
                        margin=dict(l=40, r=40, t=40, b=40),
                        height=300,
                        template="plotly_white"
                    )
                    
                    # Display chart
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Include percent change if requested
                    if include_percent_change and len(time_series_data) > 1:
                        # Calculate percent change
                        pct_change = time_series_data[metric].pct_change() * 100
                        
                        # Create percent change chart
                        pct_fig = go.Figure()
                        pct_fig.add_trace(go.Scatter(
                            x=time_series_data[date_column][1:],  # Skip first point since it's NaN after pct_change
                            y=pct_change[1:],
                            mode='lines',
                            name=f"{metric} % Change",
                            line=dict(color='rgba(31, 119, 180, 0.7)')
                        ))
                        
                        # Add zero line
                        pct_fig.add_shape(
                            type="line",
                            x0=time_series_data[date_column].iloc[1],
                            y0=0,
                            x1=time_series_data[date_column].iloc[-1],
                            y1=0,
                            line=dict(color="gray", width=1, dash="dash"),
                        )
                        
                        # Enhance layout
                        pct_fig.update_layout(
                            xaxis_title=date_column,
                            yaxis_title="Percent Change (%)",
                            height=150,
                            margin=dict(l=40, r=40, t=10, b=40),
                            showlegend=False,
                            template="plotly_white"
                        )
                        
                        # Display percent change chart
                        st.plotly_chart(pct_fig, use_container_width=True)

def create_market_overview_dashboard(
    inventory_data: pd.DataFrame,
    sales_data: pd.DataFrame,
    price_data: pd.DataFrame,
    key_metrics: Dict[str, Tuple[float, Optional[float], str]],
    date_column: str
) -> None:
    """
    Create a comprehensive market overview dashboard
    
    Args:
        inventory_data: DataFrame with inventory metrics
        sales_data: DataFrame with sales metrics
        price_data: DataFrame with price metrics
        key_metrics: Dictionary with key metrics for the header section
        date_column: Column name for dates
    """
    logger.info("Creating market overview dashboard")
    
    # Create header with key metrics
    st.header("Market Overview")
    create_metrics_row(key_metrics)
    
    # Create inventory and sales section
    st.subheader("Inventory and Sales Trends")
    
    # Create combined chart for inventory and sales
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add inventory trace
    fig.add_trace(
        go.Scatter(
            x=inventory_data[date_column], 
            y=inventory_data["listing_inventory"],
            name="Listing Inventory",
            line=dict(color="blue")
        ),
        secondary_y=False
    )
    
    # Add sales volume trace
    fig.add_trace(
        go.Scatter(
            x=sales_data[date_column], 
            y=sales_data["sales_volume"],
            name="Sales Volume",
            line=dict(color="green")
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title="Inventory and Sales Volume",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Listing Inventory", secondary_y=False)
    fig.update_yaxes(title_text="Sales Volume", secondary_y=True)
    
    # Display chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Create price trends section
    st.subheader("Price Trends")
    
    # Create columns for the price charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create median sale price chart
        median_fig = go.Figure()
        median_fig.add_trace(go.Scatter(
            x=price_data[date_column],
            y=price_data["median_sale_price"],
            mode="lines+markers",
            name="Median Sale Price",
            line=dict(color="red")
        ))
        
        # Update layout
        median_fig.update_layout(
            title="Median Sale Price Trend",
            xaxis_title=date_column,
            yaxis_title="Price ($)",
            template="plotly_white",
            height=350
        )
        
        # Display chart
        st.plotly_chart(median_fig, use_container_width=True)
    
    with col2:
        # Create price-to-income chart
        pti_fig = go.Figure()
        pti_fig.add_trace(go.Scatter(
            x=price_data[date_column],
            y=price_data["price_to_income_ratio"],
            mode="lines+markers",
            name="Price to Income Ratio",
            line=dict(color="purple")
        ))
        
        # Update layout
        pti_fig.update_layout(
            title="Price to Income Ratio Trend",
            xaxis_title=date_column,
            yaxis_title="Ratio",
            template="plotly_white",
            height=350
        )
        
        # Display chart
        st.plotly_chart(pti_fig, use_container_width=True)
    
    # Market health indicators
    st.subheader("Market Health Indicators")
    
    # Create columns for the indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Days on Market
        dom_fig = go.Figure()
        dom_fig.add_trace(go.Scatter(
            x=sales_data[date_column],
            y=sales_data["days_on_market"],
            mode="lines",
            name="Days on Market",
            line=dict(color="orange")
        ))
        
        # Update layout
        dom_fig.update_layout(
            title="Days on Market",
            xaxis_title=date_column,
            yaxis_title="Days",
            template="plotly_white",
            height=250
        )
        
        # Display chart
        st.plotly_chart(dom_fig, use_container_width=True)
    
    with col2:
        # Months of Supply
        mos_fig = go.Figure()
        mos_fig.add_trace(go.Scatter(
            x=inventory_data[date_column],
            y=inventory_data["months_of_supply"],
            mode="lines",
            name="Months of Supply",
            line=dict(color="teal")
        ))
        
        # Update layout
        mos_fig.update_layout(
            title="Months of Supply",
            xaxis_title=date_column,
            yaxis_title="Months",
            template="plotly_white",
            height=250
        )
        
        # Display chart
        st.plotly_chart(mos_fig, use_container_width=True)
    
    with col3:
        # List to Sales Ratio
        lsr_fig = go.Figure()
        lsr_fig.add_trace(go.Scatter(
            x=price_data[date_column],
            y=price_data["list_to_sales_ratio"] * 100,  # Convert to percentage
            mode="lines",
            name="List to Sales Ratio",
            line=dict(color="brown")
        ))
        
        # Update layout
        lsr_fig.update_layout(
            title="List to Sales Ratio",
            xaxis_title=date_column,
            yaxis_title="Percentage (%)",
            template="plotly_white",
            height=250
        )
        
        # Display chart
        st.plotly_chart(lsr_fig, use_container_width=True)

def create_town_comparison_dashboard(
    town_data: pd.DataFrame,
    towns: List[str],
    metrics: List[str],
    metric_labels: Optional[List[str]] = None
) -> None:
    """
    Create a town comparison dashboard
    
    Args:
        town_data: DataFrame with town-level data
        towns: List of towns to compare
        metrics: List of metrics to compare
        metric_labels: Optional friendly labels for metrics
    """
    logger.info(f"Creating town comparison dashboard for {len(towns)} towns and {len(metrics)} metrics")
    
    # Use metric names as labels if not provided
    if metric_labels is None:
        metric_labels = metrics
    
    # Create tabs for different comparison views
    tab_names = ["Bar Charts", "Radar Chart", "Table View"]
    
    with st.container():
        tabs = st.tabs(tab_names)
        
        # Bar charts tab
        with tabs[0]:
            for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
                # Create bar chart for this metric
                fig = go.Figure()
                
                # Add trace for each town
                for town in towns:
                    # Filter data for this town
                    town_value = town_data[town_data["town"] == town][metric].iloc[0] if len(town_data[town_data["town"] == town]) > 0 else 0
                    
                    fig.add_trace(go.Bar(
                        x=[town],
                        y=[town_value],
                        name=town
                    ))
                
                # Update layout
                fig.update_layout(
                    title=f"{label} by Town",
                    xaxis_title="Town",
                    yaxis_title=label,
                    template="plotly_white",
                    height=350,
                    showlegend=False
                )
                
                # Display chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a divider except for the last metric
                if i < len(metrics) - 1:
                    st.divider()
        
        # Radar chart tab
        with tabs[1]:
            # Create a radar chart for town comparison
            fig = go.Figure()
            
            # Find max value for each metric for normalization
            max_values = {metric: town_data[metric].max() for metric in metrics}
            
            # Add trace for each town
            for town in towns:
                # Filter data for this town
                town_row = town_data[town_data["town"] == town]
                
                if len(town_row) > 0:
                    # Normalize values to 0-1 range for better comparison
                    values = [town_row[metric].iloc[0] / max_values[metric] if max_values[metric] > 0 else 0 for metric in metrics]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=metric_labels,
                        fill='toself',
                        name=town
                    ))
            
            # Update layout
            fig.update_layout(
                title="Town Comparison Across All Metrics",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                height=600,
                template="plotly_white"
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
        
        # Table view tab
        with tabs[2]:
            # Create a table view for detailed comparison
            # Filter data for selected towns and metrics
            filtered_data = town_data[town_data["town"].isin(towns)][["town"] + metrics]
            
            # Rename columns to use friendly labels
            renamed_data = filtered_data.copy()
            for i, metric in enumerate(metrics):
                renamed_data = renamed_data.rename(columns={metric: metric_labels[i]})
            
            # Display as a table
            st.dataframe(renamed_data, use_container_width=True)

def create_forecast_dashboard(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    date_column: str,
    metrics: List[str],
    forecast_start_date: pd.Timestamp,
    confidence_intervals: bool = True
) -> None:
    """
    Create a forecast dashboard with historical data and forecasts
    
    Args:
        historical_data: DataFrame with historical data
        forecast_data: DataFrame with forecast data
        date_column: Column name for dates
        metrics: List of metrics to forecast
        forecast_start_date: Date where forecast begins
        confidence_intervals: Whether to show confidence intervals
    """
    logger.info(f"Creating forecast dashboard for {len(metrics)} metrics")
    
    # Create tabs for different metrics
    tabs = st.tabs(metrics)
    
    for i, metric in enumerate(metrics):
        with tabs[i]:
            # Create forecast chart
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=historical_data[date_column],
                y=historical_data[metric],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add forecast
            fig.add_trace(go.Scatter(
                x=forecast_data[date_column],
                y=forecast_data[metric],
                mode='lines',
                name='Forecast',
                line=dict(color='red')
            ))
            
            # Add confidence intervals if requested
            if confidence_intervals and f"{metric}_lower" in forecast_data.columns and f"{metric}_upper" in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data[date_column],
                    y=forecast_data[f"{metric}_upper"],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_data[date_column],
                    y=forecast_data[f"{metric}_lower"],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name='95% Confidence Interval'
                ))
            
            # Add vertical line at forecast start
            fig.add_vline(
                x=forecast_start_date,
                line_dash="dash",
                line_color="gray",
                annotation_text="Forecast Start",
                annotation_position="top right"
            )
            
            # Update layout
            fig.update_layout(
                title=f"{metric} Forecast",
                xaxis_title=date_column,
                yaxis_title=metric,
                template="plotly_white",
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Display chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast metrics
            if len(historical_data) > 0 and len(forecast_data) > 0:
                # Calculate metrics
                last_historical_value = historical_data[metric].iloc[-1]
                first_forecast_value = forecast_data[metric].iloc[0]
                last_forecast_value = forecast_data[metric].iloc[-1]
                
                forecast_change = last_forecast_value - last_historical_value
                forecast_pct_change = (forecast_change / last_historical_value) * 100 if last_historical_value != 0 else 0
                
                # Display metrics in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Current Value",
                        value=f"{last_historical_value:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="Forecasted End Value",
                        value=f"{last_forecast_value:,.2f}"
                    )
                
                with col3:
                    st.metric(
                        label="Forecasted Change",
                        value=f"{forecast_change:,.2f}",
                        delta=f"{forecast_pct_change:.2f}%"
                    )
            
            # Add a section for forecast assumptions and methodology
            with st.expander("Forecast Methodology and Assumptions"):
                st.write("""
                This forecast uses time series analysis techniques to predict future values based on historical patterns.
                
                Key assumptions:
                - Seasonal patterns will continue as observed in historical data
                - No major market disruptions or policy changes
                - Economic conditions follow gradual transition rather than sudden shocks
                
                The model is updated quarterly with new data and recalibrated as needed.
                """)
                
                # If confidence intervals are shown, explain them
                if confidence_intervals:
                    st.write("""
                    **About the confidence intervals:**
                    
                    The shaded area represents the 95% confidence interval for the forecast. This means there is a 95% probability that the actual values will fall within this range, assuming the model assumptions are correct.
                    
                    Wider intervals indicate greater uncertainty in the forecast.
                    """)

def create_economic_indicators_dashboard(
    housing_data: pd.DataFrame,
    economic_data: pd.DataFrame,
    date_column: str,
    housing_metrics: List[str],
    economic_metrics: List[str],
    correlation_matrix: Optional[pd.DataFrame] = None
) -> None:
    """
    Create a dashboard showing relationships between housing and economic indicators
    
    Args:
        housing_data: DataFrame with housing metrics
        economic_data: DataFrame with economic indicators
        date_column: Column name for dates
        housing_metrics: List of housing metrics to display
        economic_metrics: List of economic metrics to display
        correlation_matrix: Optional pre-calculated correlation matrix
    """
    logger.info(f"Creating economic indicators dashboard with {len(housing_metrics)} housing metrics and {len(economic_metrics)} economic metrics")
    
    # Merge datasets on date if needed
    if date_column in housing_data.columns and date_column in economic_data.columns:
        merged_data = pd.merge(housing_data, economic_data, on=date_column, how='inner')
    else:
        st.error("Cannot merge datasets: date column missing from one or both datasets")
        return
    
    # Create tabs for different views
    tab_names = ["Overview", "Time Series Comparison", "Correlation Analysis", "Scatter Plots"]
    tabs = st.tabs(tab_names)
    
    # Overview tab
    with tabs[0]:
        st.subheader("Economic Indicators Overview")
        
        # Display economic indicators in a grid
        indicators_per_row = 2
        for i in range(0, len(economic_metrics), indicators_per_row):
            # Create columns for this row
            cols = st.columns(indicators_per_row)
            
            # Add indicators to columns in this row
            for j in range(indicators_per_row):
                if i + j < len(economic_metrics):
                    metric = economic_metrics[i + j]
                    
                    with cols[j]:
                        # Create time series chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=economic_data[date_column],
                            y=economic_data[metric],
                            mode='lines',
                            name=metric
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=metric,
                            xaxis_title=date_column,
                            yaxis_title=metric,
                            template="plotly_white",
                            height=300
                        )
                        
                        # Display chart
                        st.plotly_chart(fig, use_container_width=True)
    
    # Time Series Comparison tab
    with tabs[1]:
        st.subheader("Housing and Economic Indicators Over Time")
        
        # Create selectors for metrics to compare
        col1, col2 = st.columns(2)
        
        with col1:
            selected_housing_metric = st.selectbox("Select Housing Metric", housing_metrics)
        
        with col2:
            selected_economic_metric = st.selectbox("Select Economic Indicator", economic_metrics)
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add housing metric trace
        fig.add_trace(
            go.Scatter(
                x=merged_data[date_column],
                y=merged_data[selected_housing_metric],
                name=selected_housing_metric,
                line=dict(color="blue")
            ),
            secondary_y=False
        )
        
        # Add economic metric trace
        fig.add_trace(
            go.Scatter(
                x=merged_data[date_column],
                y=merged_data[selected_economic_metric],
                name=selected_economic_metric,
                line=dict(color="red")
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title=f"{selected_housing_metric} vs {selected_economic_metric}",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text=selected_housing_metric, secondary_y=False)
        fig.update_yaxes(title_text=selected_economic_metric, secondary_y=True)
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis tab
    with tabs[2]:
        st.subheader("Correlation Between Housing and Economic Indicators")
        
        # Calculate correlation matrix if not provided
        if correlation_matrix is None and len(merged_data) > 0:
            # Select only numeric columns for correlation
            numeric_columns = merged_data.select_dtypes(include=['number']).columns.tolist()
            # Remove date column if present
            if date_column in numeric_columns:
                numeric_columns.remove(date_column)
            
            correlation_matrix = merged_data[numeric_columns].corr()
        
        if correlation_matrix is not None:
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                zmin=-1, zmax=1,
                colorscale="RdBu_r",
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                showscale=True
            ))
            
            # Update layout
            fig.update_layout(
                title="Correlation Matrix",
                height=800,
                width=800,
                xaxis=dict(tickangle=-45),
                template="plotly_white"
            )
            
            # Display chart
            st.plotly_chart(fig)
            
            # Explain correlation matrix
            with st.expander("Understanding the Correlation Matrix"):
                st.write("""
                The correlation matrix shows the relationship strength between pairs of metrics:
                
                * Values close to 1 indicate a strong positive correlation (both values tend to increase together)
                * Values close to -1 indicate a strong negative correlation (one increases as the other decreases)
                * Values close to 0 indicate little to no correlation
                
                Strong correlations may indicate relationships worth exploring further.
                """)
    
    # Scatter Plots tab
    with tabs[3]:
        st.subheader("Scatter Plot Analysis")
        
        # Create selectors for metrics to plot
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox("X-Axis Metric", housing_metrics + economic_metrics, key="x_metric")
        
        with col2:
            y_metric = st.selectbox("Y-Axis Metric", housing_metrics + economic_metrics, key="y_metric")
        
        # Create options for analysis
        show_regression = st.checkbox("Show Regression Line", value=True)
        show_correlation = st.checkbox("Show Correlation Coefficient", value=True)
        
        # Create scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=merged_data[x_metric],
            y=merged_data[y_metric],
            mode='markers',
            marker=dict(
                size=8,
                color='blue',
                opacity=0.7
            ),
            name='Data Points'
        ))
        
        # Add regression line if requested
        if show_regression and len(merged_data) > 1:
            # Calculate regression line using numpy polyfit
            import numpy as np
            mask = ~(np.isnan(merged_data[x_metric]) | np.isnan(merged_data[y_metric]))
            if sum(mask) > 1:
                x = merged_data[x_metric][mask]
                y = merged_data[y_metric][mask]
                coeffs = np.polyfit(x, y, 1)
                regression_line = coeffs[0] * x + coeffs[1]
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=regression_line,
                    mode='lines',
                    line=dict(color='red'),
                    name=f'Regression Line (y = {coeffs[0]:.3f}x + {coeffs[1]:.3f})'
                ))
        
        # Calculate and show correlation if requested
        if show_correlation and len(merged_data) > 1:
            correlation = merged_data[[x_metric, y_metric]].corr().iloc[0, 1]
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=f"Correlation: {correlation:.3f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        # Update layout
        fig.update_layout(
            title=f"{y_metric} vs {x_metric}",
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            template="plotly_white",
            height=600
        )
        
        # Display chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation based on correlation
        if show_correlation and len(merged_data) > 1:
            correlation = merged_data[[x_metric, y_metric]].corr().iloc[0, 1]
            
            if abs(correlation) > 0.7:
                st.info(f"There is a **strong {'positive' if correlation > 0 else 'negative'}** correlation between {x_metric} and {y_metric}.")
            elif abs(correlation) > 0.3:
                st.info(f"There is a **moderate {'positive' if correlation > 0 else 'negative'}** correlation between {x_metric} and {y_metric}.")
            else:
                st.info(f"There is a **weak** correlation between {x_metric} and {y_metric}.") 