"""
Charts

This module provides chart generation functions for data visualization.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import streamlit as st
import altair as alt
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "",
    color_column: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    height: int = 500,
    width: int = 800,
    show_trendline: bool = False
) -> go.Figure:
    """
    Create a line chart visualization for time series data
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis (typically date)
        y_column: Column name for y-axis
        title: Chart title
        color_column: Column name for color differentiation
        hover_data: Additional columns to show on hover
        height: Chart height in pixels
        width: Chart width in pixels
        show_trendline: Whether to show a trendline
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating line chart for {y_column} vs {x_column}")
    
    if hover_data is None:
        hover_data = []
    
    fig = px.line(
        data,
        x=x_column,
        y=y_column,
        title=title,
        color=color_column,
        hover_data=hover_data,
        height=height,
        width=width
    )
    
    # Add trendline if requested
    if show_trendline and len(data) > 1:
        # Simple linear trendline
        x_numeric = pd.to_numeric(data.index) if x_column == data.index.name else pd.to_numeric(data[x_column])
        y = data[y_column]
        mask = ~np.isnan(y)
        if sum(mask) > 1:  # Need at least 2 points for trendline
            coeffs = np.polyfit(range(len(x_numeric[mask])), y[mask], 1)
            trendline = coeffs[0] * np.array(range(len(x_numeric))) + coeffs[1]
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=trendline,
                mode='lines',
                line=dict(dash='dash', color='rgba(0,0,0,0.5)'),
                name='Trend'
            ))
    
    # Enhance layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "",
    color_column: Optional[str] = None,
    orientation: str = 'v',
    height: int = 500,
    width: int = 800,
    bar_mode: str = 'group'
) -> go.Figure:
    """
    Create a bar chart visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        color_column: Column name for color differentiation
        orientation: 'v' for vertical bars, 'h' for horizontal bars
        height: Chart height in pixels
        width: Chart width in pixels
        bar_mode: 'group', 'stack', 'relative', or 'overlay'
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating bar chart for {y_column} vs {x_column}")
    
    fig = px.bar(
        data,
        x=x_column,
        y=y_column,
        title=title,
        color=color_column,
        orientation=orientation,
        height=height,
        width=width,
        barmode=bar_mode
    )
    
    # Enhance layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "",
    color_column: Optional[str] = None,
    size_column: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    height: int = 500,
    width: int = 800,
    show_regression_line: bool = False,
    show_correlation: bool = False
) -> go.Figure:
    """
    Create a scatter plot visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        title: Chart title
        color_column: Column name for color differentiation
        size_column: Column name for point size variation
        hover_data: Additional columns to show on hover
        height: Chart height in pixels
        width: Chart width in pixels
        show_regression_line: Whether to show regression line
        show_correlation: Whether to show correlation coefficient
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating scatter plot for {y_column} vs {x_column}")
    
    if hover_data is None:
        hover_data = []
    
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=title,
        color=color_column,
        size=size_column,
        hover_data=hover_data,
        height=height,
        width=width
    )
    
    if show_regression_line:
        # Add trendline
        fig.update_layout(showlegend=True)
        if show_regression_line:
            fig_with_reg = px.scatter(
                data,
                x=x_column,
                y=y_column,
                trendline="ols"
            )
            for trace in fig_with_reg.data:
                if trace.mode == "lines":
                    fig.add_trace(trace)
    
    if show_correlation and all(pd.api.types.is_numeric_dtype(data[col]) for col in [x_column, y_column]):
        # Calculate and display correlation
        corr = data[[x_column, y_column]].corr().iloc[0, 1]
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.05,
            text=f"Correlation: {corr:.3f}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="black",
            borderwidth=1
        )
    
    # Enhance layout
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_heatmap(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    z_column: str,
    title: str = "",
    height: int = 500,
    width: int = 800,
    color_scale: str = "Viridis"
) -> go.Figure:
    """
    Create a heatmap visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        z_column: Column name for color intensity values
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        color_scale: Color scale for the heatmap
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating heatmap for {z_column} with {x_column} and {y_column}")
    
    # Pivot the data if it's not already in matrix form
    if not data.index.name == y_column:
        pivot_data = data.pivot_table(index=y_column, columns=x_column, values=z_column)
    else:
        pivot_data = data
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=color_scale
    ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white"
    )
    
    return fig

def create_choropleth_map(
    data: pd.DataFrame,
    location_column: str,
    color_column: str,
    geojson: Dict[str, Any],
    feature_id_property: str,
    title: str = "",
    height: int = 600,
    width: int = 800,
    color_scale: str = "Viridis",
    hover_data: Optional[List[str]] = None
) -> go.Figure:
    """
    Create a choropleth map visualization
    
    Args:
        data: DataFrame containing the data
        location_column: Column name for location identifiers
        color_column: Column name for color values
        geojson: GeoJSON data containing boundaries
        feature_id_property: Property in GeoJSON that matches location_column values
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        color_scale: Color scale for the choropleth
        hover_data: Additional columns to show on hover
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating choropleth map for {color_column} by {location_column}")
    
    if hover_data is None:
        hover_data = []
    
    fig = px.choropleth(
        data,
        geojson=geojson,
        locations=location_column,
        featureidkey=f"properties.{feature_id_property}",
        color=color_column,
        hover_data=hover_data,
        color_continuous_scale=color_scale,
        title=title
    )
    
    # Center the map on Connecticut
    fig.update_geos(
        fitbounds="locations",
        visible=True
    )
    
    # Enhance layout
    fig.update_layout(
        height=height,
        width=width,
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        template="plotly_white"
    )
    
    return fig

def create_bubble_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    size_column: str,
    title: str = "",
    color_column: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    height: int = 600,
    width: int = 800,
    size_max: int = 60
) -> go.Figure:
    """
    Create a bubble chart visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        size_column: Column name for bubble size
        title: Chart title
        color_column: Column name for color differentiation
        hover_data: Additional columns to show on hover
        height: Chart height in pixels
        width: Chart width in pixels
        size_max: Maximum size of bubbles
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating bubble chart for {y_column} vs {x_column} with {size_column} as size")
    
    if hover_data is None:
        hover_data = []
    
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        size=size_column,
        color=color_column,
        hover_data=hover_data,
        title=title,
        size_max=size_max
    )
    
    # Enhance layout
    fig.update_layout(
        height=height,
        width=width,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_box_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "",
    color_column: Optional[str] = None,
    height: int = 500,
    width: int = 800,
    points: str = "outliers"
) -> go.Figure:
    """
    Create a box plot visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis categories
        y_column: Column name for y-axis values
        title: Chart title
        color_column: Column name for color differentiation
        height: Chart height in pixels
        width: Chart width in pixels
        points: 'all', 'outliers', 'suspectedoutliers', or False
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating box plot for {y_column} grouped by {x_column}")
    
    fig = px.box(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        title=title,
        points=points
    )
    
    # Enhance layout
    fig.update_layout(
        height=height,
        width=width,
        xaxis_title=x_column,
        yaxis_title=y_column,
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_stacked_area_chart(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = "",
    height: int = 500,
    width: int = 800,
    normalized: bool = False
) -> go.Figure:
    """
    Create a stacked area chart visualization
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis (typically date)
        y_columns: List of column names for stacked areas
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        normalized: Whether to normalize to 100%
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating stacked area chart for {', '.join(y_columns)} over {x_column}")
    
    fig = go.Figure()
    
    if normalized:
        # Convert to percentages
        sums = data[y_columns].sum(axis=1)
        for col in y_columns:
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[col].div(sums).multiply(100),
                mode='lines',
                stackgroup='one',
                name=col
            ))
    else:
        for col in y_columns:
            fig.add_trace(go.Scatter(
                x=data[x_column],
                y=data[col],
                mode='lines',
                stackgroup='one',
                name=col
            ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis_title=x_column,
        yaxis_title="Percentage" if normalized else "Value",
        legend_title="Categories",
        template="plotly_white"
    )
    
    return fig

def create_histogram(
    data: pd.DataFrame,
    column: str,
    title: str = "",
    color_column: Optional[str] = None,
    height: int = 500,
    width: int = 800,
    nbins: int = 20,
    histnorm: Optional[str] = None
) -> go.Figure:
    """
    Create a histogram visualization
    
    Args:
        data: DataFrame containing the data
        column: Column name for histogram values
        title: Chart title
        color_column: Column name for color differentiation
        height: Chart height in pixels
        width: Chart width in pixels
        nbins: Number of bins
        histnorm: Histogram normalization method ('', 'percent', 'probability', 'density', 'probability density')
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating histogram for {column}")
    
    fig = px.histogram(
        data,
        x=column,
        color=color_column,
        title=title,
        nbins=nbins,
        histnorm=histnorm
    )
    
    # Enhance layout
    fig.update_layout(
        height=height,
        width=width,
        xaxis_title=column,
        yaxis_title=histnorm if histnorm else "Count",
        legend_title=color_column if color_column else None,
        template="plotly_white"
    )
    
    return fig

def create_correlation_matrix(
    data: pd.DataFrame,
    columns: List[str],
    title: str = "Correlation Matrix",
    height: int = 600,
    width: int = 800,
    color_scale: str = "RdBu_r"
) -> go.Figure:
    """
    Create a correlation matrix visualization
    
    Args:
        data: DataFrame containing the data
        columns: List of column names to include in correlation
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        color_scale: Color scale for the correlation matrix
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating correlation matrix for {', '.join(columns)}")
    
    # Calculate correlation matrix
    corr_matrix = data[columns].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        zmin=-1, zmax=1,
        colorscale=color_scale,
        text=corr_matrix.round(3).values,
        texttemplate="%{text}",
        showscale=True
    ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis=dict(tickangle=-45),
        template="plotly_white"
    )
    
    return fig

def create_time_series_comparison(
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    title: str = "",
    height: int = 500,
    width: int = 800,
    include_percent_change: bool = False
) -> Tuple[go.Figure, Optional[go.Figure]]:
    """
    Create a time series comparison visualization with optional percent change chart
    
    Args:
        data: DataFrame containing the data
        x_column: Column name for x-axis (typically date)
        y_columns: List of column names for comparison
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        include_percent_change: Whether to include a percent change chart
        
    Returns:
        Tuple of (main figure, percent change figure or None)
    """
    logger.info(f"Creating time series comparison for {', '.join(y_columns)} over {x_column}")
    
    # Main chart
    fig = go.Figure()
    
    for col in y_columns:
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[col],
            mode='lines',
            name=col
        ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis_title=x_column,
        legend_title="Metrics",
        template="plotly_white"
    )
    
    # Optional percent change chart
    pct_change_fig = None
    if include_percent_change and len(data) > 1:
        pct_change_fig = go.Figure()
        
        for col in y_columns:
            # Calculate percent change
            pct_change = data[col].pct_change() * 100
            
            pct_change_fig.add_trace(go.Scatter(
                x=data[x_column][1:],  # Skip first point since it's NaN after pct_change
                y=pct_change[1:],
                mode='lines',
                name=f"{col} % Change"
            ))
        
        # Enhance layout
        pct_change_fig.update_layout(
            title=f"{title} - Percent Change",
            height=height,
            width=width,
            xaxis_title=x_column,
            yaxis_title="Percent Change (%)",
            legend_title="Metrics",
            template="plotly_white"
        )
    
    return fig, pct_change_fig

def create_forecast_chart(
    historical_data: pd.DataFrame,
    forecast_data: pd.DataFrame,
    x_column: str,
    y_column: str,
    ci_lower_column: Optional[str] = None,
    ci_upper_column: Optional[str] = None,
    title: str = "Forecast Chart",
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create a forecast chart with historical data and forecasts
    
    Args:
        historical_data: DataFrame containing historical data
        forecast_data: DataFrame containing forecast data
        x_column: Column name for x-axis (typically date)
        y_column: Column name for y-axis values
        ci_lower_column: Column name for lower confidence interval
        ci_upper_column: Column name for upper confidence interval
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating forecast chart for {y_column}")
    
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data[x_column],
        y=historical_data[y_column],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_data[x_column],
        y=forecast_data[y_column],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if provided
    if ci_lower_column and ci_upper_column and ci_lower_column in forecast_data.columns and ci_upper_column in forecast_data.columns:
        fig.add_trace(go.Scatter(
            x=forecast_data[x_column],
            y=forecast_data[ci_upper_column],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_data[x_column],
            y=forecast_data[ci_lower_column],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.2)',
            name='95% Confidence Interval'
        ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        height=height,
        width=width,
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white"
    )
    
    return fig

def create_radial_chart(
    data: pd.DataFrame,
    metric_columns: List[str],
    title: str = "Radial Chart",
    height: int = 600,
    width: int = 800,
    scale_to_unit: bool = True
) -> go.Figure:
    """
    Create a radial chart (spider/radar) visualization
    
    Args:
        data: DataFrame containing the data, typically one or few rows
        metric_columns: List of column names for radar chart axes
        title: Chart title
        height: Chart height in pixels
        width: Chart width in pixels
        scale_to_unit: Whether to scale values to 0-1 range
        
    Returns:
        Plotly figure object
    """
    logger.info(f"Creating radial chart for {', '.join(metric_columns)}")
    
    fig = go.Figure()
    
    # Calculate scaling if needed
    scaled_data = data.copy()
    if scale_to_unit:
        for col in metric_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val > min_val:
                scaled_data[col] = (data[col] - min_val) / (max_val - min_val)
    
    # Add traces for each row in data
    for i, row in scaled_data.iterrows():
        row_name = str(i)
        if data.index.name and data.index.name in data.columns:
            row_name = str(data.loc[i, data.index.name])
        
        fig.add_trace(go.Scatterpolar(
            r=[row[col] for col in metric_columns],
            theta=metric_columns,
            fill='toself',
            name=row_name
        ))
    
    # Enhance layout
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if scale_to_unit else None
            )
        ),
        height=height,
        width=width,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def export_chart_to_image(
    fig: go.Figure,
    filename: str,
    format: str = "png",
    width: int = 800,
    height: int = 600,
    scale: int = 2
) -> str:
    """
    Export a Plotly figure to an image file
    
    Args:
        fig: Plotly figure object to export
        filename: Output filename without extension
        format: Output format (png, jpeg, svg, pdf)
        width: Image width in pixels
        height: Image height in pixels
        scale: Image scale factor
        
    Returns:
        Path to saved image file
    """
    logger.info(f"Exporting chart to {filename}.{format}")
    
    full_filename = f"{filename}.{format}"
    fig.write_image(full_filename, width=width, height=height, scale=scale)
    
    return full_filename

def export_chart_data_to_csv(
    fig: go.Figure,
    filename: str
) -> str:
    """
    Export data from a Plotly figure to CSV
    
    Args:
        fig: Plotly figure object with data to export
        filename: Output filename without extension
        
    Returns:
        Path to saved CSV file
    """
    logger.info(f"Exporting chart data to {filename}.csv")
    
    full_filename = f"{filename}.csv"
    
    # Extract data from figure
    data_dfs = []
    for trace in fig.data:
        trace_data = {}
        
        if hasattr(trace, 'x') and trace.x is not None:
            trace_data['x'] = trace.x
        
        if hasattr(trace, 'y') and trace.y is not None:
            trace_data['y'] = trace.y
            
        if trace.name:
            column_name = trace.name
        else:
            column_name = f"Series_{len(data_dfs)}"
            
        if trace_data:
            df = pd.DataFrame(trace_data)
            if 'y' in df.columns:
                df = df.rename(columns={'y': column_name})
            data_dfs.append(df)
    
    if data_dfs:
        # Merge all dataframes on x
        result_df = data_dfs[0]
        for df in data_dfs[1:]:
            if 'x' in df.columns and 'x' in result_df.columns:
                result_df = pd.merge(result_df, df, on='x', how='outer')
            else:
                result_df = pd.concat([result_df, df], axis=1)
        
        result_df.to_csv(full_filename, index=False)
    
    return full_filename

def create_metric_chart(data: pd.DataFrame, 
                       metric: str, 
                       chart_type: str = 'line', 
                       title: Optional[str] = None,
                       x_title: Optional[str] = None, 
                       y_title: Optional[str] = None,
                       color_by: Optional[str] = None,
                       facet_by: Optional[str] = None,
                       height: int = 500,
                       width: int = 800) -> go.Figure:
    """
    Create a chart for the specified metric.
    
    Args:
        data: DataFrame containing the data to plot
        metric: Name of the metric to plot
        chart_type: Type of chart ('line', 'bar', 'scatter', 'area', 'pie')
        title: Chart title
        x_title: X-axis title
        y_title: Y-axis title
        color_by: Column to use for coloring
        facet_by: Column to use for faceting
        height: Chart height in pixels
        width: Chart width in pixels
        
    Returns:
        Plotly figure object
    """
    # Set default title if not provided
    if title is None:
        title = f"{metric} Over Time"
    
    # Set default axis titles if not provided
    if x_title is None:
        x_title = "Date"
    
    if y_title is None:
        y_title = metric
    
    # Create appropriate chart based on type
    if chart_type == 'line':
        fig = px.line(
            data, 
            x='date', 
            y='value', 
            color=color_by,
            facet_col=facet_by,
            title=title,
            labels={'value': y_title, 'date': x_title},
            height=height,
            width=width
        )
    elif chart_type == 'bar':
        fig = px.bar(
            data, 
            x='date', 
            y='value', 
            color=color_by,
            facet_col=facet_by,
            title=title,
            labels={'value': y_title, 'date': x_title},
            height=height,
            width=width
        )
    elif chart_type == 'scatter':
        fig = px.scatter(
            data, 
            x='date', 
            y='value', 
            color=color_by,
            facet_col=facet_by,
            title=title,
            labels={'value': y_title, 'date': x_title},
            height=height,
            width=width
        )
    elif chart_type == 'area':
        fig = px.area(
            data, 
            x='date', 
            y='value', 
            color=color_by,
            facet_col=facet_by,
            title=title,
            labels={'value': y_title, 'date': x_title},
            height=height,
            width=width
        )
    elif chart_type == 'pie':
        # Pie charts need special handling as they don't use date as x-axis
        fig = px.pie(
            data, 
            names=color_by if color_by else 'date', 
            values='value',
            title=title,
            height=height,
            width=width
        )
    else:
        # Default to line chart
        fig = px.line(
            data, 
            x='date', 
            y='value',
            color=color_by,
            facet_col=facet_by,
            title=title,
            labels={'value': y_title, 'date': x_title},
            height=height,
            width=width
        )
    
    # Customize layout
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=50, r=50, t=50, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Format axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )
    
    return fig

def create_comparison_chart(data_dict: Dict[str, pd.DataFrame], 
                          metrics: List[str],
                          entities: List[str],
                          chart_type: str = 'bar', 
                          title: Optional[str] = None) -> go.Figure:
    """
    Create a comparison chart for multiple metrics and entities.
    
    Args:
        data_dict: Dictionary of DataFrames by metric
        metrics: List of metrics to compare
        entities: List of entities to compare (e.g., towns)
        chart_type: Type of chart ('bar', 'radar', 'heatmap')
        title: Chart title
        
    Returns:
        Plotly figure object
    """
    # Implementation placeholder
    # In a real implementation, this would create comparison charts
    fig = go.Figure()
    
    # Return empty figure for now
    return fig

def create_time_series_chart(data: pd.DataFrame,
                            date_column: str = 'date',
                            value_column: str = 'value',
                            group_column: Optional[str] = None,
                            title: Optional[str] = None,
                            show_trend: bool = False) -> go.Figure:
    """
    Create a time series chart with optional trend line.
    
    Args:
        data: DataFrame containing time series data
        date_column: Name of the column containing dates
        value_column: Name of the column containing values
        group_column: Name of the column for grouping
        title: Chart title
        show_trend: Whether to show a trend line
        
    Returns:
        Plotly figure object
    """
    # Implementation placeholder
    # In a real implementation, this would create time series charts
    fig = go.Figure()
    
    # Return empty figure for now
    return fig 