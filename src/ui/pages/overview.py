"""
Overview Page

This module provides the overview dashboard with summary metrics and key trends
for the Fairfield County housing market analysis application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from ...data_collection.data_fetcher import DataFetcher
from ...data_processing.metrics_calculator import MetricsCalculator
from ...data_processing.transformers import DataTransformer
from ..components.filters import DateRangeFilter, LocationFilter, PropertyTypeFilter

# Key metrics to show on overview dashboard
KEY_METRICS = [
    "Median Sale Price",
    "Months of Supply",
    "Days on Market",
    "Absorption Rate", 
    "LP/SP Ratio",
    "Home Price-to-Income Ratio",
    "Total Sales",
    "Active Listings",
    "New Listings"
]

# Trend indicators (icons and colors)
TREND_INDICATORS = {
    "up_good": "📈 <span style='color:green'>▲</span>",
    "up_bad": "📈 <span style='color:red'>▲</span>",
    "down_good": "📉 <span style='color:green'>▼</span>",
    "down_bad": "📉 <span style='color:red'>▼</span>",
    "neutral": "⟷ <span style='color:gray'>◆</span>"
}

def render_overview_page():
    """Render the overview dashboard page."""
    
    st.title("Market Conditions Overview")
    st.write("Statistics and trends for the real estate market")
    
    # Initialize components
    data_fetcher = DataFetcher()
    metrics_calculator = MetricsCalculator()
    data_transformer = DataTransformer()
    
    # Sidebar filters
    with st.sidebar:
        st.header("Filters")
        
        # Date range filter
        date_filter = DateRangeFilter(
            min_date="2015-01-01",
            key_prefix="overview_date"
        )
        start_date, end_date = date_filter.render()
        
        # Location filter
        location_filter = LocationFilter(key_prefix="overview_location")
        selected_location = location_filter.render()
        
        # Property type filter
        property_filter = PropertyTypeFilter(key_prefix="overview_property")
        selected_property_type = property_filter.render()
        
        # Fetch data button
        fetch_data = st.button("Update Dashboard", key="overview_fetch")
    
    # Display loading spinner while fetching data
    with st.spinner("Fetching market data..."):
        # Convert dates to strings for API
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Create a session state key for caching data between runs
        data_state_key = f"overview_data_{start_date_str}_{end_date_str}_{selected_location}_{selected_property_type}"
        
        # Fetch data if button clicked or data not in session state
        if fetch_data or data_state_key not in st.session_state:
            # Fetch data based on filters
            if selected_location == "Fairfield County":
                # County level data
                metrics_data = data_fetcher.fetch_all_metrics(
                    start_date=start_date_str,
                    end_date=end_date_str,
                    frequency="q"
                )
            else:
                # Town level data
                if selected_location == "All Towns":
                    # Handle "All Towns" selection by fetching data for each town
                    # and averaging/aggregating as needed
                    all_towns_metrics = {}
                    
                    for metric in KEY_METRICS:
                        # Skip metrics that don't support town-level breakdown
                        if metric in ["Home Price-to-Income Ratio", "Housing Affordability Index"]:
                            continue
                            
                        # Fetch data for all towns
                        town_data = data_fetcher.fetch_all_towns_data(
                            metric=metric,
                            start_date=start_date_str,
                            end_date=end_date_str
                        )
                        
                        if not town_data.empty:
                            all_towns_metrics[metric] = town_data
                    
                    # Add county-level metrics for those not available at town level
                    county_metrics = data_fetcher.fetch_all_metrics(
                        start_date=start_date_str,
                        end_date=end_date_str,
                        frequency="q"
                    )
                    
                    for metric in KEY_METRICS:
                        if metric not in all_towns_metrics and metric in county_metrics:
                            all_towns_metrics[metric] = county_metrics[metric]
                    
                    metrics_data = all_towns_metrics
                else:
                    # Single town data
                    metrics_data = data_fetcher.fetch_town_metrics(
                        town=selected_location,
                        metrics=KEY_METRICS,
                        start_date=start_date_str,
                        end_date=end_date_str
                    )
            
            # Store in session state
            st.session_state[data_state_key] = metrics_data
        else:
            # Use cached data
            metrics_data = st.session_state[data_state_key]
    
    # Display the dashboard components
    if not metrics_data:
        st.error("No data available for the selected filters. Please try different criteria.")
        return
    
    # Create tabular layout similar to the image
    st.subheader(f"The Overall Real Estate Market in {selected_location}")
    
    # Extract quarters from the data
    quarters = []
    for metric in metrics_data:
        if not metrics_data[metric].empty:
            for idx in metrics_data[metric].index:
                if hasattr(idx, 'quarter') and hasattr(idx, 'year'):
                    quarter_str = f"Q{idx.quarter} {idx.year}"
                    if quarter_str not in quarters:
                        quarters.append(quarter_str)
                elif hasattr(idx, 'month') and hasattr(idx, 'year'):
                    # Calculate quarter from month
                    quarter = (idx.month - 1) // 3 + 1
                    quarter_str = f"Q{quarter} {idx.year}"
                    if quarter_str not in quarters:
                        quarters.append(quarter_str)
    
    # Sort quarters chronologically
    quarters.sort(key=lambda q: (int(q.split()[1]), int(q.split('Q')[1].split()[0])))
    
    # Use the most recent 4 quarters if available
    if len(quarters) > 4:
        quarters = quarters[-4:]  # Get the latest 4 quarters
    
    # Create table layout
    # Structure is: [Metric, Quarter1, Quarter2, Quarter3, Quarter4, % Change Q1-Q2, % Change Q2-Q3, etc.]
    
    # Custom table with CSS
    st.markdown("""
    <style>
    .metric-table {
        width: 100%;
        border-collapse: collapse;
    }
    .metric-table th, .metric-table td {
        padding: 8px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .metric-table th {
        background-color: #f2f2f2;
        font-weight: bold;
    }
    .metric-table tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .metric-row:hover {
        background-color: #e9f7fe;
    }
    .metric-name {
        text-align: left;
        font-weight: bold;
    }
    .positive-change {
        color: green;
    }
    .negative-change {
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create table HTML
    table_html = f"""
    <table class="metric-table">
        <tr>
            <th>Metrics</th>
    """
    
    # Add quarter columns
    for quarter in quarters:
        table_html += f"<th>{quarter}</th>"
    
    # Add percentage change columns
    if len(quarters) >= 2:
        for i in range(len(quarters)-1):
            table_html += f"<th>% Change</th>"
    
    table_html += "</tr>"
    
    # Add metric rows
    for metric in KEY_METRICS:
        if metric not in metrics_data or metrics_data[metric].empty:
            continue
            
        metric_df = metrics_data[metric]
        
        # Format the metric name to match the image
        display_name = metric
        if metric == "Median Sale Price":
            display_name = "Median Price"
        elif metric == "Average Sale Price":
            display_name = "Average Price"
        elif metric == "LP/SP Ratio":
            display_name = "Avg SP/LP Ratio"
        elif metric == "Days on Market":
            display_name = "Avg DOM"
        elif metric == "Home Price-to-Income Ratio":
            display_name = "Price-to-Income Ratio"
        
        # Create row
        table_html += f"""
        <tr class="metric-row">
            <td class="metric-name">{display_name}</td>
        """
        
        # Values for each quarter
        quarter_values = []
        for quarter_str in quarters:
            year = int(quarter_str.split()[1])
            quarter = int(quarter_str.split('Q')[1].split()[0])
            
            # Find the value for this quarter/year
            # First, check if we have quarter data
            quarter_data = pd.DataFrame()
            
            if hasattr(metric_df.index, 'quarter') and hasattr(metric_df.index, 'year'):
                quarter_data = metric_df[
                    (metric_df.index.quarter == quarter) & 
                    (metric_df.index.year == year)
                ]
            # If not, calculate from month data
            elif hasattr(metric_df.index, 'month') and hasattr(metric_df.index, 'year'):
                # Get months for this quarter
                start_month = (quarter - 1) * 3 + 1
                end_month = quarter * 3
                quarter_data = metric_df[
                    (metric_df.index.month >= start_month) & 
                    (metric_df.index.month <= end_month) & 
                    (metric_df.index.year == year)
                ]
                
                # If we found multiple months, average them
                if not quarter_data.empty and len(quarter_data) > 1:
                    quarter_data = pd.DataFrame({
                        "value": [quarter_data["value"].mean()]
                    })
            
            if not quarter_data.empty and "value" in quarter_data.columns:
                value = quarter_data["value"].iloc[0]
                formatted_value = _format_metric_value(metric, value)
                quarter_values.append(value)
                table_html += f"<td>{formatted_value}</td>"
            else:
                quarter_values.append(None)
                table_html += "<td>N/A</td>"
        
        # Calculate percentage changes between quarters
        if len(quarters) >= 2:
            for i in range(len(quarters)-1):
                if quarter_values[i] is not None and quarter_values[i+1] is not None and quarter_values[i] != 0:
                    pct_change = ((quarter_values[i+1] - quarter_values[i]) / quarter_values[i]) * 100
                    
                    # Determine if change is positive or negative for this metric
                    is_good_change = _is_positive_trend(metric, pct_change)
                    
                    # Format the percentage change
                    change_str = f"{pct_change:.1f}%"
                    if pct_change > 0:
                        css_class = "positive-change" if is_good_change else "negative-change"
                        table_html += f'<td class="{css_class}">+{change_str}</td>'
                    elif pct_change < 0:
                        css_class = "positive-change" if not is_good_change else "negative-change"
                        table_html += f'<td class="{css_class}">{change_str}</td>'
                    else:
                        table_html += f"<td>0%</td>"
                else:
                    table_html += "<td>N/A</td>"
        
        table_html += "</tr>"
    
    table_html += "</table>"
    
    # Display the table
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Property Type Distribution chart (pie chart)
    st.subheader(f"Allocation of Total Sales by Property Type in {selected_location}")
    
    # Get property type distribution from the data transformer
    property_distribution = data_transformer.get_property_type_distribution(metrics_data, selected_location)
    property_types = property_distribution["types"]
    property_values = property_distribution["values"]
    
    # Create pie chart
    fig = px.pie(
        values=property_values,
        names=property_types,
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.4
    )
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=0),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create individual trend charts for key metrics
    st.subheader("Trend Charts")
    
    # Create 2x2 grid for trend charts
    col1, col2 = st.columns(2)
    
    # Function to create a trend chart for a specific metric
    def create_trend_chart(metric, container):
        if metric in metrics_data and not metrics_data[metric].empty:
            df = metrics_data[metric].copy()
            
            # Create chart title based on metric
            if metric == "Median Sale Price":
                title = "Trend of Median Price"
            elif metric == "Days on Market":
                title = "Days on Market"
            elif metric == "Active Listings":
                title = "Active Listings"
            elif metric == "New Listings":
                title = "New Listings"
            elif metric == "Total Sales":
                title = "Total Sales"
            else:
                title = f"Trend of {metric}"
            
            # Create line chart
            fig = px.line(
                df,
                x=df.index,
                y="value",
                title=title,
                labels={"value": metric, "index": "Date"}
            )
            
            # Set color based on metric type
            if metric in ["Median Sale Price", "Average Sale Price"]:
                line_color = "blue"
            elif metric in ["Active Listings", "New Listings"]:
                line_color = "orange" if metric == "New Listings" else "red"
            elif metric == "Total Sales":
                line_color = "green"
            elif metric == "Days on Market":
                line_color = "purple"
            else:
                line_color = "black"
            
            fig.update_traces(line_color=line_color)
            
            # Add markers for data points
            fig.update_traces(mode="lines+markers", marker=dict(size=6))
            
            # Format y-axis based on metric type
            if metric in ["Median Sale Price", "Average Sale Price"]:
                fig.update_layout(yaxis_tickprefix="$", yaxis_tickformat=",")
            
            # Highlight quarterly data points
            if hasattr(df.index, 'quarter') and hasattr(df.index, 'year'):
                # Add quarterly annotations
                for idx, row in df.iterrows():
                    fig.add_annotation(
                        x=idx,
                        y=row['value'],
                        text=f"Q{idx.quarter}",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=10)
                    )
            elif hasattr(df.index, 'month') and hasattr(df.index, 'year'):
                # For monthly data, highlight the first month of each quarter
                for idx, row in df.iterrows():
                    if idx.month in [1, 4, 7, 10]:  # First month of each quarter
                        quarter = (idx.month - 1) // 3 + 1
                        fig.add_annotation(
                            x=idx,
                            y=row['value'],
                            text=f"Q{quarter}",
                            showarrow=False,
                            yshift=10,
                            font=dict(size=10)
                        )
            
            # Add quarterly trendlines
            if len(df) >= 4:
                # Add a trendline for the latest 4 quarters
                recent_data = df.iloc[-4:]
                x_trend = np.array(range(len(recent_data)))
                y_trend = recent_data['value'].values
                
                # Simple linear regression
                z = np.polyfit(x_trend, y_trend, 1)
                p = np.poly1d(z)
                
                # Add trendline as a separate trace
                fig.add_trace(
                    go.Scatter(
                        x=recent_data.index,
                        y=p(x_trend),
                        mode='lines',
                        line=dict(color='rgba(0,0,0,0.5)', width=2, dash='dash'),
                        name='Quarterly Trend'
                    )
                )
            
            # Improve layout
            fig.update_layout(
                height=300,
                margin=dict(l=40, r=20, t=40, b=40),
                xaxis_title=None,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            # Add min/max annotations for the latest 4 quarters
            recent_data = df.iloc[-min(4, len(df)):]
            if len(recent_data) > 1:
                min_val = recent_data["value"].min()
                max_val = recent_data["value"].max()
                min_idx = recent_data["value"].idxmin()
                max_idx = recent_data["value"].idxmax()
                
                fig.add_annotation(
                    x=min_idx,
                    y=min_val,
                    text=f"Min: {_format_metric_value(metric, min_val)}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=-15
                )
                
                fig.add_annotation(
                    x=max_idx,
                    y=max_val,
                    text=f"Max: {_format_metric_value(metric, max_val)}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=15
                )
            
            container.plotly_chart(fig, use_container_width=True)
        else:
            container.info(f"No data available for {metric}")
    
    # Create trend charts for key metrics
    create_trend_chart("Median Sale Price", col1)
    create_trend_chart("Active Listings", col2)
    create_trend_chart("Total Sales", col1)
    create_trend_chart("Days on Market", col2)
    
    # Footer with data source information
    st.markdown("---")
    st.caption("Data Sources: FRED API, BLS API, ATTOM Property API | Last Updated: " + 
               datetime.now().strftime("%Y-%m-%d"))


def _is_positive_trend(metric: str, change: float) -> bool:
    """
    Determine if a trend direction is positive for a given metric.
    
    Args:
        metric: Metric name
        change: Percentage change
        
    Returns:
        Boolean indicating if the trend is positive
    """
    # For these metrics, an increase is generally seen as positive
    positive_when_increasing = [
        "Median Sale Price",
        "Average Sale Price",
        "Housing Affordability Index",
        "Absorption Rate",
        "LP/SP Ratio",
        "Total Sales"
    ]
    
    # For these metrics, a decrease is generally seen as positive
    positive_when_decreasing = [
        "Days on Market",
        "Months of Supply",
        "Home Price-to-Income Ratio"
    ]
    
    # For these metrics, the context determines whether an increase is positive
    context_dependent = [
        "Active Listings",
        "New Listings"
    ]
    
    if metric in positive_when_increasing:
        return change > 0
    elif metric in positive_when_decreasing:
        return change < 0
    elif metric in context_dependent:
        # For context-dependent metrics, return neutral (neither positive nor negative)
        return True if change > 0 else False  # Simply return the direction for UI display
    else:
        # Default to neutral
        return True


def _format_metric_value(metric: str, value: float) -> str:
    """
    Format a metric value for display.
    
    Args:
        metric: Metric name
        value: Metric value
        
    Returns:
        Formatted value string
    """
    # Determine formatting based on metric type
    if metric in ["Median Sale Price", "Average Sale Price"]:
        return f"${value:,.0f}"
    elif metric in ["Days on Market"]:
        return f"{value:.0f} days"
    elif metric in ["Months of Supply"]:
        return f"{value:.1f} mo"
    elif metric in ["LP/SP Ratio", "Absorption Rate"]:
        return f"{value:.1f}%"
    elif metric in ["Home Price-to-Income Ratio"]:
        return f"{value:.2f}x"
    elif metric in ["Housing Affordability Index"]:
        return f"{value:.1f}"
    elif metric in ["Total Sales", "Active Listings", "New Listings", "Inventory"]:
        return f"{value:,.0f}"
    else:
        return f"{value:.2f}"


def _generate_market_alerts(metrics_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str]]:
    """
    Generate alert messages for significant market shifts.
    
    Args:
        metrics_data: Dictionary of market metrics data
        
    Returns:
        List of (alert_type, message) tuples
    """
    alerts = []
    
    # Get current and previous quarter strings
    current_quarter = None
    previous_quarter = None
    
    if "Median Sale Price" in metrics_data and not metrics_data["Median Sale Price"].empty:
        df = metrics_data["Median Sale Price"]
        if hasattr(df.index, 'quarter') and hasattr(df.index, 'year') and len(df) > 1:
            current_idx = df.index[-1]
            prev_idx = df.index[-2]
            current_quarter = f"Q{current_idx.quarter} {current_idx.year}"
            previous_quarter = f"Q{prev_idx.quarter} {prev_idx.year}"
        elif hasattr(df.index, 'month') and hasattr(df.index, 'year') and len(df) > 1:
            current_idx = df.index[-1]
            prev_idx = df.index[-2]
            current_q = (current_idx.month - 1) // 3 + 1
            prev_q = (prev_idx.month - 1) // 3 + 1
            current_quarter = f"Q{current_q} {current_idx.year}"
            previous_quarter = f"Q{prev_q} {prev_idx.year}"
    
    # Use generic "current quarter" and "previous quarter" if we couldn't determine specific quarters
    if not current_quarter:
        current_quarter = "current quarter"
        previous_quarter = "previous quarter"
    
    # Check for significant changes in key metrics
    for metric in KEY_METRICS:
        if metric not in metrics_data:
            continue
            
        df = metrics_data[metric]
        
        if df.empty or "value" not in df.columns or len(df) < 2:
            continue
        
        # Calculate percentage change from previous period
        current_val = df["value"].iloc[-1]
        prev_val = df["value"].iloc[-2]
        pct_change = ((current_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
        
        # Set thresholds for alerts
        if metric == "Median Sale Price":
            if pct_change > 10:
                alerts.append(("warning", f"🔔 **Price Alert:** Median sale price increased by {pct_change:.1f}% from {previous_quarter} to {current_quarter}, indicating a rapidly appreciating market."))
            elif pct_change < -5:
                alerts.append(("warning", f"🔔 **Price Alert:** Median sale price decreased by {abs(pct_change):.1f}% from {previous_quarter} to {current_quarter}, suggesting potential market correction."))
        
        elif metric == "Days on Market":
            if pct_change < -20:
                alerts.append(("warning", f"⏱️ **Market Speed Alert:** Properties are selling {abs(pct_change):.1f}% faster in {current_quarter} compared to {previous_quarter}, indicating a hot seller's market."))
            elif pct_change > 30:
                alerts.append(("info", f"⏱️ **Market Speed Alert:** Days on market increased by {pct_change:.1f}% from {previous_quarter} to {current_quarter}, suggesting a shift towards a buyer's market."))
        
        elif metric == "Months of Supply":
            if current_val < 3 and pct_change < -10:
                alerts.append(("warning", f"📊 **Inventory Alert:** Supply decreased by {abs(pct_change):.1f}% to {current_val:.1f} months in {current_quarter}, strengthening the seller's market position."))
            elif current_val > 6 and pct_change > 15:
                alerts.append(("info", f"📊 **Inventory Alert:** Supply increased by {pct_change:.1f}% to {current_val:.1f} months in {current_quarter}, shifting toward a buyer's market."))
        
        elif metric == "Total Sales":
            if pct_change > 20:
                alerts.append(("success", f"💰 **Transaction Alert:** Sales volume jumped {pct_change:.1f}% in {current_quarter} compared to {previous_quarter}, indicating strong market activity."))
            elif pct_change < -20:
                alerts.append(("warning", f"💰 **Transaction Alert:** Sales volume dropped {abs(pct_change):.1f}% in {current_quarter} compared to {previous_quarter}, indicating slowing market activity."))
            
        elif metric == "LP/SP Ratio":
            if current_val > 100 and pct_change > 0:
                alerts.append(("warning", f"💲 **Pricing Power Alert:** Properties in {current_quarter} are selling at {current_val:.1f}% of asking price, indicating strong competition among buyers."))
            elif current_val < 95:
                alerts.append(("info", f"💲 **Pricing Power Alert:** Properties in {current_quarter} are selling at only {current_val:.1f}% of asking price, suggesting increased negotiating power for buyers."))
        
        elif metric == "Active Listings":
            if pct_change > 30:
                alerts.append(("info", f"📋 **Inventory Alert:** Active listings increased by {pct_change:.1f}% in {current_quarter}, providing more options for buyers."))
            elif pct_change < -30:
                alerts.append(("warning", f"📋 **Inventory Alert:** Active listings decreased by {abs(pct_change):.1f}% in {current_quarter}, further limiting buyer options."))
                
    # Check for market balance indicators - compare metrics to identify market direction
    if "Months of Supply" in metrics_data and "Days on Market" in metrics_data and "Total Sales" in metrics_data:
        supply_df = metrics_data["Months of Supply"]
        dom_df = metrics_data["Days on Market"]
        sales_df = metrics_data["Total Sales"]
        
        if (not supply_df.empty and not dom_df.empty and not sales_df.empty and 
            "value" in supply_df.columns and "value" in dom_df.columns and "value" in sales_df.columns):
            
            current_supply = supply_df["value"].iloc[-1]
            current_dom = dom_df["value"].iloc[-1]
            current_sales = sales_df["value"].iloc[-1]
            
            # Check if we have previous period data
            if len(supply_df) > 1 and len(dom_df) > 1 and len(sales_df) > 1:
                prev_supply = supply_df["value"].iloc[-2]
                prev_dom = dom_df["value"].iloc[-2]
                prev_sales = sales_df["value"].iloc[-2]
                
                # Calculate change percentages
                supply_change = ((current_supply - prev_supply) / prev_supply) * 100 if prev_supply != 0 else 0
                dom_change = ((current_dom - prev_dom) / prev_dom) * 100 if prev_dom != 0 else 0
                sales_change = ((current_sales - prev_sales) / prev_sales) * 100 if prev_sales != 0 else 0
                
                # Identify market direction based on combined metrics
                if supply_change < -10 and dom_change < -10 and sales_change > 10:
                    alerts.append(("warning", f"🔥 **Market Direction Alert:** Multiple indicators show strengthening seller's market in {current_quarter} - decreasing supply, faster sales, and increasing transaction volume."))
                elif supply_change > 15 and dom_change > 15 and sales_change < -10:
                    alerts.append(("info", f"❄️ **Market Direction Alert:** Multiple indicators show strengthening buyer's market in {current_quarter} - increasing supply, slower sales, and decreasing transaction volume."))
            
            # Current market state based on absolute values
            if current_supply < 3 and current_dom < 30:
                alerts.append(("warning", f"🏡 **Market State Alert:** Strong seller's market in {current_quarter} with low inventory ({current_supply:.1f} months) and quick sales ({current_dom:.0f} days)."))
            elif current_supply > 6 and current_dom > 60:
                alerts.append(("info", f"🏡 **Market State Alert:** Buyer's market conditions in {current_quarter} with high inventory ({current_supply:.1f} months) and longer selling times ({current_dom:.0f} days)."))
    
    return alerts


def _generate_market_summary(metrics_data: Dict[str, pd.DataFrame], 
                           location: str, 
                           start_date: datetime, 
                           end_date: datetime) -> str:
    """
    Generate a text summary of the market condition.
    
    Args:
        metrics_data: Dictionary of market metrics data
        location: Selected location
        start_date: Start date of analysis
        end_date: End date of analysis
        
    Returns:
        Markdown-formatted summary text
    """
    # Extract key metrics for the summary
    median_price = None
    months_supply = None
    dom = None
    ratio = None
    
    # Get latest quarter data
    current_quarter = None
    previous_quarter = None
    
    # Determine the current and previous quarters
    if "Median Sale Price" in metrics_data and not metrics_data["Median Sale Price"].empty:
        df = metrics_data["Median Sale Price"]
        if hasattr(df.index, 'quarter') and hasattr(df.index, 'year'):
            # If data already has quarter information
            current_idx = df.index[-1]
            current_quarter = f"Q{current_idx.quarter} {current_idx.year}"
            
            if len(df) > 1:
                prev_idx = df.index[-2]
                previous_quarter = f"Q{prev_idx.quarter} {prev_idx.year}"
        elif hasattr(df.index, 'month') and hasattr(df.index, 'year'):
            # Calculate quarter from month for the latest data point
            current_idx = df.index[-1]
            current_q = (current_idx.month - 1) // 3 + 1
            current_quarter = f"Q{current_q} {current_idx.year}"
            
            if len(df) > 1:
                prev_idx = df.index[-2]
                prev_q = (prev_idx.month - 1) // 3 + 1
                previous_quarter = f"Q{prev_q} {prev_idx.year}"
    
    # If we couldn't determine quarters from data, use the date range
    if not current_quarter:
        current_q = (end_date.month - 1) // 3 + 1
        current_quarter = f"Q{current_q} {end_date.year}"
        
        # Calculate previous quarter
        prev_date = end_date - timedelta(days=90)
        prev_q = (prev_date.month - 1) // 3 + 1
        previous_quarter = f"Q{prev_q} {prev_date.year}"
    
    # Get the latest values for key metrics
    if "Median Sale Price" in metrics_data and not metrics_data["Median Sale Price"].empty:
        median_price = metrics_data["Median Sale Price"]["value"].iloc[-1]
        
    if "Months of Supply" in metrics_data and not metrics_data["Months of Supply"].empty:
        months_supply = metrics_data["Months of Supply"]["value"].iloc[-1]
        
    if "Days on Market" in metrics_data and not metrics_data["Days on Market"].empty:
        dom = metrics_data["Days on Market"]["value"].iloc[-1]
        
    if "LP/SP Ratio" in metrics_data and not metrics_data["LP/SP Ratio"].empty:
        ratio = metrics_data["LP/SP Ratio"]["value"].iloc[-1]
    
    # Generate market summary based on available data
    summary_parts = []
    
    # Location and time period with focus on current quarter
    summary_parts.append(f"## Market Summary for {location}: {current_quarter}\n\n")
    
    # Add quarter-over-quarter comparison if available
    if previous_quarter:
        summary_parts.append(f"This analysis compares {current_quarter} with {previous_quarter} to identify key market trends.\n\n")
    
    # Market balance assessment
    if months_supply is not None:
        if months_supply < 3:
            summary_parts.append(f"### Market Balance\nThe market in {current_quarter} shows a **strong seller's market** with only {months_supply:.1f} months of supply. ")
        elif months_supply < 6:
            summary_parts.append(f"### Market Balance\nThe market in {current_quarter} is relatively balanced at {months_supply:.1f} months of supply, with a slight advantage to sellers. ")
        else:
            summary_parts.append(f"### Market Balance\nThe market in {current_quarter} indicates a **buyer's market** with {months_supply:.1f} months of supply. ")
    
    # Price information
    if median_price is not None:
        summary_parts.append(f"\n\n### Pricing Trends\nThe median sale price for {current_quarter} is **${median_price:,.0f}**")
        
        # Add quarter-over-quarter price trend if we have enough data
        if "Median Sale Price" in metrics_data and len(metrics_data["Median Sale Price"]) > 1:
            price_df = metrics_data["Median Sale Price"]
            prev_price = price_df["value"].iloc[-2]
            pct_change = ((median_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            if abs(pct_change) > 1:  # Only mention if change is significant
                if pct_change > 0:
                    summary_parts.append(f", which has **increased {pct_change:.1f}%** since {previous_quarter}. ")
                else:
                    summary_parts.append(f", which has **decreased {abs(pct_change):.1f}%** since {previous_quarter}. ")
            else:
                summary_parts.append(f", which has remained stable since {previous_quarter}. ")
        else:
            summary_parts.append(". ")
    
    # Days on Market
    if dom is not None:
        summary_parts.append(f"\n\n### Market Activity\nProperties in {current_quarter} are selling in an average of **{dom:.0f} days**")
        
        # Add quarter-over-quarter DOM trend if we have enough data
        if "Days on Market" in metrics_data and len(metrics_data["Days on Market"]) > 1:
            dom_df = metrics_data["Days on Market"]
            prev_dom = dom_df["value"].iloc[-2]
            pct_change = ((dom - prev_dom) / prev_dom) * 100 if prev_dom != 0 else 0
            
            if abs(pct_change) > 5:  # Only mention if change is significant
                if pct_change > 0:
                    summary_parts.append(f", a **{pct_change:.1f}% increase** from {previous_quarter}. ")
                else:
                    summary_parts.append(f", a **{abs(pct_change):.1f}% decrease** from {previous_quarter}. ")
            else:
                summary_parts.append(f", similar to {previous_quarter}. ")
        else:
            summary_parts.append(". ")
    
    # List to Sale Price Ratio
    if ratio is not None:
        summary_parts.append("\n\n### Buyer/Seller Dynamics\n")
        if ratio > 100:
            summary_parts.append(f"Buyers in {current_quarter} are paying **{ratio:.1f}%** of asking price on average, indicating significant competition. ")
        elif ratio > 98:
            summary_parts.append(f"Sellers in {current_quarter} are receiving close to asking price at **{ratio:.1f}%** on average. ")
        else:
            summary_parts.append(f"Buyers in {current_quarter} are negotiating prices down to **{ratio:.1f}%** of asking price on average. ")
    
    # Add quarterly sales volume if available
    if "Total Sales" in metrics_data and not metrics_data["Total Sales"].empty:
        sales = metrics_data["Total Sales"]["value"].iloc[-1]
        summary_parts.append(f"\n\n### Transaction Volume\nTotal sales in {current_quarter} reached **{sales:,.0f}** transactions")
        
        if len(metrics_data["Total Sales"]) > 1:
            prev_sales = metrics_data["Total Sales"]["value"].iloc[-2]
            pct_change = ((sales - prev_sales) / prev_sales) * 100 if prev_sales != 0 else 0
            
            if abs(pct_change) > 5:
                if pct_change > 0:
                    summary_parts.append(f", a **{pct_change:.1f}% increase** from {previous_quarter}.")
                else:
                    summary_parts.append(f", a **{abs(pct_change):.1f}% decrease** from {previous_quarter}.")
            else:
                summary_parts.append(f", similar to {previous_quarter}.")
    
    # Combine all parts into a single summary
    summary = "".join(summary_parts)
    
    # Add closing recommendation based on market conditions
    if months_supply is not None and dom is not None:
        summary += "\n\n### Market Outlook\n"
        if months_supply < 3 and dom < 30:
            summary += f"For {current_quarter}, buyers should be prepared to act quickly and potentially offer above asking price. Sellers are in a strong position to maximize returns."
        elif months_supply > 6 and dom > 60:
            summary += f"For {current_quarter}, buyers have negotiating leverage. Sellers should consider pricing competitively and expect longer selling periods."
        else:
            summary += f"For {current_quarter}, the market shows a reasonable balance between buyers and sellers, with opportunities for both sides when positioned correctly."
    
    return summary 