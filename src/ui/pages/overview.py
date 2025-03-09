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
    "List Price to Sales Price Ratio",
    "Home Price-to-Income Ratio"
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
    
    st.title("Fairfield County Housing Market Overview")
    st.write("At-a-glance summary of critical market metrics and trends")
    
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
    
    # 1. Market Health Score
    with st.container():
        st.subheader("Market Health Index")
        
        # Calculate market conditions index
        market_index_df = metrics_calculator.calculate_market_conditions_index(metrics_data)
        
        if not market_index_df.empty and "market_conditions_index" in market_index_df.columns:
            # Get the most recent value
            current_index = market_index_df["market_conditions_index"].iloc[-1]
            current_condition = market_index_df["market_condition"].iloc[-1]
            
            # Previous value (one quarter ago)
            if len(market_index_df) > 1:
                prev_index = market_index_df["market_conditions_index"].iloc[-2]
                index_change = current_index - prev_index
                
                # Determine if change is significant
                significant_change = abs(index_change) > 5
                
                # Display the gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=current_index,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Market Health Score", "font": {"size": 24}},
                    delta={"reference": prev_index, "valueformat": ".1f"},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                        "bar": {"color": "darkblue"},
                        "bgcolor": "white",
                        "borderwidth": 2,
                        "bordercolor": "gray",
                        "steps": [
                            {"range": [0, 30], "color": "red"},
                            {"range": [30, 50], "color": "orange"},
                            {"range": [50, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "green"}
                        ]
                    }
                ))
                
                fig.update_layout(
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Market condition interpretation
                st.markdown(f"**Current Market Condition**: {current_condition}")
                
                if significant_change:
                    if index_change > 0:
                        st.markdown("🔍 **Analysis**: Market showing significant strengthening compared to previous quarter.")
                    else:
                        st.markdown("🔍 **Analysis**: Market showing significant weakening compared to previous quarter.")
            else:
                # Not enough data for comparison
                st.metric("Market Health Score", f"{current_index:.1f}")
                st.markdown(f"**Current Market Condition**: {current_condition}")
        else:
            st.warning("Insufficient data to calculate market health index. Please select a broader date range.")
    
    # 2. Key metrics summary with trend indicators
    st.subheader("Key Market Indicators")
    
    # Create columns for metrics display
    col1, col2, col3 = st.columns(3)
    col_map = {0: col1, 1: col2, 2: col3}
    
    for i, metric in enumerate(KEY_METRICS):
        if metric in metrics_data:
            metric_df = metrics_data[metric]
            
            if not metric_df.empty and "value" in metric_df.columns:
                # Get current value and QoQ change
                if len(metric_df) > 1:
                    current_val = metric_df["value"].iloc[-1]
                    prev_val = metric_df["value"].iloc[-2]
                    pct_change = ((current_val - prev_val) / prev_val) * 100 if prev_val != 0 else 0
                    
                    # Determine if trend is good or bad
                    is_good_trend = _is_positive_trend(metric, pct_change)
                    
                    # Get appropriate trend indicator
                    if pct_change > 1:  # More than 1% change up
                        trend_icon = TREND_INDICATORS["up_good"] if is_good_trend else TREND_INDICATORS["up_bad"]
                    elif pct_change < -1:  # More than 1% change down
                        trend_icon = TREND_INDICATORS["down_good"] if is_good_trend else TREND_INDICATORS["down_bad"]
                    else:
                        trend_icon = TREND_INDICATORS["neutral"]
                    
                    # Format value based on metric type
                    formatted_val = _format_metric_value(metric, current_val)
                    
                    # Display metric with delta and trend indicator
                    with col_map[i % 3]:
                        st.metric(
                            label=metric,
                            value=formatted_val,
                            delta=f"{pct_change:.1f}%"
                        )
                        st.markdown(f"{trend_icon} QoQ Change", unsafe_allow_html=True)
                else:
                    # Only one data point, no trend
                    current_val = metric_df["value"].iloc[-1]
                    formatted_val = _format_metric_value(metric, current_val)
                    
                    with col_map[i % 3]:
                        st.metric(label=metric, value=formatted_val)
            else:
                with col_map[i % 3]:
                    st.metric(label=metric, value="N/A")
        else:
            with col_map[i % 3]:
                st.metric(label=metric, value="N/A")
    
    # 3. Historical trends chart
    st.subheader("Historical Trends")
    
    # Let user select metrics to display
    selected_trend_metrics = st.multiselect(
        "Select metrics to display",
        options=KEY_METRICS,
        default=["Median Sale Price", "Days on Market"],
        key="overview_trend_metrics"
    )
    
    if selected_trend_metrics:
        # Create a comparative dataset with selected metrics
        comparative_df = data_transformer.create_comparative_dataset(
            metrics_data,
            metrics=selected_trend_metrics
        )
        
        if not comparative_df.empty:
            # Normalize data for easier comparison
            normalized_df = pd.DataFrame(index=comparative_df.index)
            
            for col in comparative_df.columns:
                if col in ["year", "quarter", "month", "quarter_year"]:
                    continue
                # Min-max normalization
                series = comparative_df[col]
                min_val = series.min()
                max_val = series.max()
                normalized_df[col] = (series - min_val) / (max_val - min_val) if max_val > min_val else series
            
            # Add date column for plotting
            normalized_df["date"] = normalized_df.index
            
            # Melt dataframe for Plotly
            melted_df = pd.melt(
                normalized_df.reset_index(),
                id_vars=["date"],
                value_vars=[col for col in normalized_df.columns if col != "date"],
                var_name="Metric",
                value_name="Normalized Value"
            )
            
            # Create line chart
            fig = px.line(
                melted_df,
                x="date",
                y="Normalized Value",
                color="Metric",
                title=f"Normalized Trends ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')})",
                labels={"date": "Date", "Normalized Value": "Normalized Value (0-1)"}
            )
            
            # Improve layout
            fig.update_layout(
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=40, t=60, b=40),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of normalization
            with st.expander("About Normalized Values"):
                st.write("""
                The chart shows normalized values (scaled between 0 and 1) to allow comparing metrics with 
                different units and scales on the same graph. The lowest value for each metric becomes 0, 
                and the highest becomes 1, with all other values scaled proportionally.
                """)
        else:
            st.warning("No data available for the selected metrics.")
    else:
        st.info("Please select at least one metric to display historical trends.")
    
    # 4. Alert indicators for significant market shifts
    st.subheader("Market Alerts")
    
    alerts = _generate_market_alerts(metrics_data)
    
    if alerts:
        for alert_type, alert_message in alerts:
            if alert_type == "warning":
                st.warning(alert_message)
            elif alert_type == "info":
                st.info(alert_message)
            elif alert_type == "success":
                st.success(alert_message)
            else:
                st.markdown(f"**{alert_message}**")
    else:
        st.info("No significant market shifts detected in the current time period.")
    
    # 5. Quick market summary
    st.subheader("Market Summary")
    
    # Generate text summary based on available metrics
    summary = _generate_market_summary(
        metrics_data,
        location=selected_location,
        start_date=start_date,
        end_date=end_date
    )
    
    st.markdown(summary)
    
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
        "List Price to Sales Price Ratio"
    ]
    
    # For these metrics, a decrease is generally seen as positive
    positive_when_decreasing = [
        "Days on Market",
        "Months of Supply",
        "Home Price-to-Income Ratio"
    ]
    
    if metric in positive_when_increasing:
        return change > 0
    elif metric in positive_when_decreasing:
        return change < 0
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
    elif metric in ["List Price to Sales Price Ratio", "Absorption Rate"]:
        return f"{value:.1f}%"
    elif metric in ["Home Price-to-Income Ratio"]:
        return f"{value:.2f}x"
    elif metric in ["Housing Affordability Index"]:
        return f"{value:.1f}"
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
                alerts.append(("warning", f"🔔 Median sale price increased by {pct_change:.1f}% since last quarter, indicating a rapidly appreciating market."))
            elif pct_change < -5:
                alerts.append(("warning", f"🔔 Median sale price decreased by {abs(pct_change):.1f}% since last quarter, suggesting potential market correction."))
        
        elif metric == "Days on Market":
            if pct_change < -20:
                alerts.append(("warning", f"⏱️ Properties are selling {abs(pct_change):.1f}% faster than last quarter, indicating a hot seller's market."))
            elif pct_change > 30:
                alerts.append(("info", f"⏱️ Days on market increased by {pct_change:.1f}%, suggesting a shift towards a buyer's market."))
        
        elif metric == "Months of Supply":
            if current_val < 3 and pct_change < 0:
                alerts.append(("warning", f"📊 Inventory remains very low at {current_val:.1f} months of supply, creating a strong seller's market."))
            elif current_val > 6 and pct_change > 0:
                alerts.append(("info", f"📊 Supply has increased to {current_val:.1f} months, shifting toward a buyer's market."))
            
        elif metric == "List Price to Sales Price Ratio":
            if current_val > 100 and pct_change > 0:
                alerts.append(("warning", f"💰 Properties are selling above asking price ({current_val:.1f}%), indicating strong competition among buyers."))
            elif current_val < 95:
                alerts.append(("info", f"💰 Properties are selling at {current_val:.1f}% of asking price, suggesting increased negotiating power for buyers."))
                
    # Check for market balance indicators
    if "Months of Supply" in metrics_data and "Days on Market" in metrics_data:
        supply_df = metrics_data["Months of Supply"]
        dom_df = metrics_data["Days on Market"]
        
        if not supply_df.empty and not dom_df.empty and "value" in supply_df.columns and "value" in dom_df.columns:
            current_supply = supply_df["value"].iloc[-1]
            current_dom = dom_df["value"].iloc[-1]
            
            if current_supply < 3 and current_dom < 30:
                alerts.append(("warning", "🔥 Strong seller's market with low inventory and quick sales."))
            elif current_supply > 6 and current_dom > 60:
                alerts.append(("info", "❄️ Market has shifted in favor of buyers with high inventory and longer selling times."))
    
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
    
    if "Median Sale Price" in metrics_data and not metrics_data["Median Sale Price"].empty:
        median_price = metrics_data["Median Sale Price"]["value"].iloc[-1]
        
    if "Months of Supply" in metrics_data and not metrics_data["Months of Supply"].empty:
        months_supply = metrics_data["Months of Supply"]["value"].iloc[-1]
        
    if "Days on Market" in metrics_data and not metrics_data["Days on Market"].empty:
        dom = metrics_data["Days on Market"]["value"].iloc[-1]
        
    if "List Price to Sales Price Ratio" in metrics_data and not metrics_data["List Price to Sales Price Ratio"].empty:
        ratio = metrics_data["List Price to Sales Price Ratio"]["value"].iloc[-1]
    
    # Generate market summary based on available data
    summary_parts = []
    
    # Location and time period
    summary_parts.append(f"The housing market in **{location}** from {start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')} ")
    
    # Market balance assessment
    if months_supply is not None:
        if months_supply < 3:
            summary_parts.append(f"shows a **strong seller's market** with only {months_supply:.1f} months of supply. ")
        elif months_supply < 6:
            summary_parts.append(f"is relatively balanced at {months_supply:.1f} months of supply, with a slight advantage to sellers. ")
        else:
            summary_parts.append(f"indicates a **buyer's market** with {months_supply:.1f} months of supply. ")
    
    # Price information
    if median_price is not None:
        summary_parts.append(f"The median sale price is **${median_price:,.0f}**")
        
        # Add price trend if we have enough data
        if "Median Sale Price" in metrics_data and len(metrics_data["Median Sale Price"]) > 1:
            price_df = metrics_data["Median Sale Price"]
            prev_price = price_df["value"].iloc[-2]
            pct_change = ((median_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
            
            if abs(pct_change) > 1:  # Only mention if change is significant
                if pct_change > 0:
                    summary_parts.append(f", which has **increased {pct_change:.1f}%** since the previous quarter. ")
                else:
                    summary_parts.append(f", which has **decreased {abs(pct_change):.1f}%** since the previous quarter. ")
            else:
                summary_parts.append(", which has remained stable since the previous quarter. ")
        else:
            summary_parts.append(". ")
    
    # Days on Market
    if dom is not None:
        summary_parts.append(f"Properties are selling in an average of **{dom:.0f} days**")
        
        # Add DOM trend if we have enough data
        if "Days on Market" in metrics_data and len(metrics_data["Days on Market"]) > 1:
            dom_df = metrics_data["Days on Market"]
            prev_dom = dom_df["value"].iloc[-2]
            pct_change = ((dom - prev_dom) / prev_dom) * 100 if prev_dom != 0 else 0
            
            if abs(pct_change) > 5:  # Only mention if change is significant
                if pct_change > 0:
                    summary_parts.append(f", a **{pct_change:.1f}% increase** from the previous quarter. ")
                else:
                    summary_parts.append(f", a **{abs(pct_change):.1f}% decrease** from the previous quarter. ")
            else:
                summary_parts.append(", similar to the previous quarter. ")
        else:
            summary_parts.append(". ")
    
    # List to Sale Price Ratio
    if ratio is not None:
        if ratio > 100:
            summary_parts.append(f"Buyers are paying **{ratio:.1f}%** of asking price on average, indicating significant competition. ")
        elif ratio > 98:
            summary_parts.append(f"Sellers are receiving close to asking price at **{ratio:.1f}%** on average. ")
        else:
            summary_parts.append(f"Buyers are negotiating prices down to **{ratio:.1f}%** of asking price on average. ")
    
    # Combine all parts into a single summary
    summary = "".join(summary_parts)
    
    # Add closing recommendation based on market conditions
    if months_supply is not None and dom is not None:
        if months_supply < 3 and dom < 30:
            summary += "\n\n**Recommendation**: Buyers should be prepared to act quickly and potentially offer above asking price. Sellers are in a strong position to maximize returns."
        elif months_supply > 6 and dom > 60:
            summary += "\n\n**Recommendation**: Buyers have negotiating leverage in the current market. Sellers should consider pricing competitively and expect longer selling periods."
        else:
            summary += "\n\n**Recommendation**: The market shows a reasonable balance between buyers and sellers, with opportunities for both sides when positioned correctly."
    
    return summary 