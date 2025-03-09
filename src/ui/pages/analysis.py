import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
import statsmodels.api as sm

# Import custom modules
from src.data_processing.metrics_calculator import calculate_correlation_matrix
from src.data_processing.time_series import align_time_series, resample_time_series
from src.visualization.charts import create_line_chart, create_scatter_plot, create_heatmap
from src.ui.components.filters import create_date_filter, create_location_filter, create_metric_filter
from src.utils.logger import get_logger
from src.utils.config import get_config

# Initialize logger
logger = get_logger(__name__)

def render_analysis_page():
    """
    Renders the data analysis page with interactive time series overlays,
    correlation analysis, and statistical tools.
    """
    st.title("Data Analysis")
    
    with st.container():
        st.markdown("""
        This page provides advanced tools for analyzing housing market data.
        Explore correlations between different metrics, overlay time series,
        and perform statistical analysis to gain deeper insights.
        """)
    
    # Sidebar filters
    with st.sidebar:
        st.header("Analysis Filters")
        
        # Date range filter
        start_date, end_date = create_date_filter()
        
        # Location filter
        selected_locations = create_location_filter(multi=True)
        
        # Metrics filter
        primary_metrics = create_metric_filter(
            label="Primary Metrics",
            key="primary_metrics",
            default=["Median Sale Price"]
        )
        
        secondary_metrics = create_metric_filter(
            label="Secondary Metrics",
            key="secondary_metrics",
            default=["Mortgage Rates"]
        )
        
        # Analysis type selector
        analysis_type = st.selectbox(
            "Analysis Type",
            options=[
                "Time Series Overlay",
                "Correlation Analysis",
                "Scatter Matrix",
                "Statistical Tests",
                "Custom Metric Combination"
            ],
            index=0
        )
    
    # Load data based on filters
    try:
        data = load_data_for_analysis(
            locations=selected_locations,
            metrics=primary_metrics + secondary_metrics,
            start_date=start_date,
            end_date=end_date
        )
        
        if data.empty:
            st.warning("No data available for the selected filters. Please adjust your selections.")
            return
            
        # Render the appropriate analysis view based on selection
        if analysis_type == "Time Series Overlay":
            render_time_series_overlay(data, primary_metrics, secondary_metrics, selected_locations)
        
        elif analysis_type == "Correlation Analysis":
            render_correlation_analysis(data, primary_metrics + secondary_metrics, selected_locations)
        
        elif analysis_type == "Scatter Matrix":
            render_scatter_matrix(data, primary_metrics + secondary_metrics, selected_locations)
        
        elif analysis_type == "Statistical Tests":
            render_statistical_tests(data, primary_metrics, secondary_metrics, selected_locations)
        
        elif analysis_type == "Custom Metric Combination":
            render_custom_metric_combination(data, primary_metrics, secondary_metrics, selected_locations)
            
    except Exception as e:
        logger.error(f"Error rendering analysis page: {str(e)}")
        st.error(f"An error occurred while rendering the analysis: {str(e)}")
        st.exception(e)


def load_data_for_analysis(locations, metrics, start_date, end_date):
    """
    Loads and prepares data for analysis based on selected filters.
    
    Parameters:
    -----------
    locations : list
        List of selected locations
    metrics : list
        List of selected metrics
    start_date : datetime
        Start date for filtering
    end_date : datetime
        End date for filtering
        
    Returns:
    --------
    pandas.DataFrame
        Processed data for analysis
    """
    try:
        # In a real implementation, this would load from processed data files
        # For now, we'll use a placeholder function that could be implemented in data_processing
        from src.data_processing.transformers import load_processed_data
        
        data = load_processed_data(
            metrics=metrics,
            locations=locations,
            start_date=start_date,
            end_date=end_date
        )
        
        # Ensure data is properly formatted for time series analysis
        if not data.empty:
            # Make sure date column is datetime type and set as index
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            
            # Fill missing values with forward fill method
            data = data.fillna(method='ffill')
            
        return data
        
    except Exception as e:
        logger.error(f"Error loading data for analysis: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error


def render_time_series_overlay(data, primary_metrics, secondary_metrics, locations):
    """
    Renders time series overlay visualization with dual y-axes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    primary_metrics : list
        Metrics to plot on primary y-axis
    secondary_metrics : list
        Metrics to plot on secondary y-axis
    locations : list
        Selected locations
    """
    st.subheader("Time Series Overlay Analysis")
    
    st.markdown("""
    Compare multiple metrics over time using overlay charts. Primary metrics are plotted
    on the left y-axis, while secondary metrics are plotted on the right y-axis.
    """)
    
    # Options for normalization and smoothing
    col1, col2 = st.columns(2)
    
    with col1:
        normalize = st.checkbox("Normalize Values (0-100%)", value=False)
        if normalize:
            normalization_method = st.radio(
                "Normalization Method",
                options=["Min-Max", "Z-Score", "Percentage Change"],
                index=0
            )
    
    with col2:
        apply_smoothing = st.checkbox("Apply Smoothing", value=False)
        if apply_smoothing:
            window_size = st.slider("Smoothing Window Size", min_value=2, max_value=12, value=3)
            smoothing_method = st.radio(
                "Smoothing Method",
                options=["Moving Average", "Exponential", "Savitzky-Golay"],
                index=0
            )
    
    # Process data based on options
    plot_data = data.copy()
    
    # Apply normalization if selected
    if normalize:
        for location in locations:
            location_data = plot_data[plot_data['location'] == location] if 'location' in plot_data.columns else plot_data
            
            for metric in primary_metrics + secondary_metrics:
                if metric in location_data.columns:
                    series = location_data[metric]
                    
                    if normalization_method == "Min-Max":
                        min_val = series.min()
                        max_val = series.max()
                        if max_val > min_val:  # Avoid division by zero
                            normalized = (series - min_val) / (max_val - min_val) * 100
                            plot_data.loc[location_data.index, metric] = normalized
                    
                    elif normalization_method == "Z-Score":
                        mean = series.mean()
                        std = series.std()
                        if std > 0:  # Avoid division by zero
                            normalized = ((series - mean) / std) * 25 + 50  # Center around 50
                            plot_data.loc[location_data.index, metric] = normalized
                    
                    elif normalization_method == "Percentage Change":
                        normalized = series.pct_change() * 100
                        plot_data.loc[location_data.index, metric] = normalized
    
    # Apply smoothing if selected
    if apply_smoothing:
        for location in locations:
            location_data = plot_data[plot_data['location'] == location] if 'location' in plot_data.columns else plot_data
            
            for metric in primary_metrics + secondary_metrics:
                if metric in location_data.columns:
                    series = location_data[metric]
                    
                    if smoothing_method == "Moving Average":
                        smoothed = series.rolling(window=window_size, min_periods=1).mean()
                        plot_data.loc[location_data.index, metric] = smoothed
                    
                    elif smoothing_method == "Exponential":
                        smoothed = series.ewm(span=window_size).mean()
                        plot_data.loc[location_data.index, metric] = smoothed
                    
                    elif smoothing_method == "Savitzky-Golay":
                        try:
                            from scipy.signal import savgol_filter
                            # Need to have enough data points for the filter
                            if len(series) > window_size:
                                smoothed = savgol_filter(series, window_size, 3)
                                plot_data.loc[location_data.index, metric] = smoothed
                        except Exception as e:
                            logger.warning(f"Could not apply Savitzky-Golay filter: {str(e)}")
    
    # Create the overlay chart for each location
    for location in locations:
        st.subheader(f"{location} - Metric Overlay")
        
        location_data = plot_data[plot_data['location'] == location] if 'location' in plot_data.columns else plot_data
        
        if location_data.empty:
            st.warning(f"No data available for {location}.")
            continue
            
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add primary metrics
        for i, metric in enumerate(primary_metrics):
            if metric in location_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=location_data.index,
                        y=location_data[metric],
                        name=f"{metric} - {location}",
                        line=dict(width=2),
                        mode='lines'
                    ),
                    secondary_y=False
                )
        
        # Add secondary metrics
        for i, metric in enumerate(secondary_metrics):
            if metric in location_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=location_data.index,
                        y=location_data[metric],
                        name=f"{metric} - {location}",
                        line=dict(width=2, dash='dash'),
                        mode='lines'
                    ),
                    secondary_y=True
                )
        
        # Set axis titles
        primary_title = "Normalized Values (%)" if normalize else " / ".join(primary_metrics)
        secondary_title = "Normalized Values (%)" if normalize else " / ".join(secondary_metrics)
        
        fig.update_layout(
            title=f"{location} - Metric Comparison over Time",
            xaxis_title="Date",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            template="plotly_white"
        )
        
        fig.update_yaxes(title_text=primary_title, secondary_y=False)
        fig.update_yaxes(title_text=secondary_title, secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation value between primary and secondary metrics
        if len(primary_metrics) == 1 and len(secondary_metrics) == 1:
            primary = location_data[primary_metrics[0]].dropna()
            secondary = location_data[secondary_metrics[0]].dropna()
            
            # Make sure indexes align
            common_index = primary.index.intersection(secondary.index)
            primary = primary.loc[common_index]
            secondary = secondary.loc[common_index]
            
            if len(primary) > 1:
                correlation, p_value = stats.pearsonr(primary, secondary)
                
                corr_col1, corr_col2 = st.columns(2)
                corr_col1.metric(
                    "Correlation Coefficient", 
                    f"{correlation:.3f}",
                    help="Pearson correlation coefficient between the primary and secondary metrics (-1 to 1)."
                )
                corr_col2.metric(
                    "P-Value", 
                    f"{p_value:.4f}",
                    help="P-value of the correlation. Values below 0.05 suggest statistically significant correlation."
                )


def render_correlation_analysis(data, metrics, locations):
    """
    Renders correlation analysis for selected metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    metrics : list
        List of metrics to analyze
    locations : list
        Selected locations
    """
    st.subheader("Correlation Analysis")
    
    st.markdown("""
    Explore correlations between different housing market metrics.
    The heatmap shows Pearson correlation coefficients between pairs of metrics.
    """)
    
    # Filter options
    correlation_period = st.radio(
        "Correlation Period",
        options=["Entire Period", "Quarterly", "Yearly"],
        index=0,
        horizontal=True
    )
    
    # Create an empty correlation dataframe to store results
    corr_results = pd.DataFrame()
    
    # Process for each location
    tabs = st.tabs(locations)
    
    for i, tab in enumerate(tabs):
        location = locations[i]
        
        with tab:
            # Filter data for the current location
            location_data = data[data['location'] == location] if 'location' in data.columns else data
            
            if location_data.empty:
                st.warning(f"No data available for {location}.")
                continue
                
            # Keep only the columns in metrics list that are in the data
            available_metrics = [metric for metric in metrics if metric in location_data.columns]
            
            if len(available_metrics) < 2:
                st.warning("Need at least two metrics for correlation analysis.")
                continue
                
            # Create filtered dataset with only the selected metrics
            metric_data = location_data[available_metrics]
            
            # Calculate correlations based on selected period
            if correlation_period == "Entire Period":
                corr_matrix = metric_data.corr()
                period_label = "entire period"
            
            elif correlation_period == "Quarterly":
                # Resample to quarterly and calculate correlation
                quarterly_data = metric_data.resample('Q').mean()
                corr_matrix = quarterly_data.corr()
                period_label = "quarterly averages"
            
            elif correlation_period == "Yearly":
                # Resample to yearly and calculate correlation
                yearly_data = metric_data.resample('Y').mean()
                corr_matrix = yearly_data.corr()
                period_label = "yearly averages"
            
            # Create heatmap
            fig = create_heatmap(
                corr_matrix,
                title=f"{location} - Metric Correlations ({period_label})",
                colorscale="RdBu_r",
                zmin=-1,
                zmax=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation for interpretation
            with st.expander("How to interpret correlation values"):
                st.markdown("""
                - **1.0**: Perfect positive correlation
                - **0.7 to 0.9**: Strong positive correlation
                - **0.4 to 0.6**: Moderate positive correlation
                - **0.1 to 0.3**: Weak positive correlation
                - **0**: No correlation
                - **-0.1 to -0.3**: Weak negative correlation
                - **-0.4 to -0.6**: Moderate negative correlation
                - **-0.7 to -0.9**: Strong negative correlation
                - **-1.0**: Perfect negative correlation
                
                **Note**: Correlation does not imply causation. High correlation values indicate that the metrics tend to move together, but don't necessarily mean that one causes the other.
                """)
            
            # Display the strongest correlations
            st.subheader("Strongest Correlations")
            
            # Prepare correlation data
            corr_pairs = []
            for i in range(len(available_metrics)):
                for j in range(i+1, len(available_metrics)):
                    metric1 = available_metrics[i]
                    metric2 = available_metrics[j]
                    correlation = corr_matrix.iloc[i, j]
                    corr_pairs.append({
                        "Metric 1": metric1,
                        "Metric 2": metric2,
                        "Correlation": correlation,
                        "Abs. Correlation": abs(correlation)
                    })
            
            # Convert to DataFrame and sort
            corr_df = pd.DataFrame(corr_pairs)
            if not corr_df.empty:
                corr_df = corr_df.sort_values("Abs. Correlation", ascending=False)
                
                # Display top correlations
                st.dataframe(
                    corr_df[["Metric 1", "Metric 2", "Correlation"]].head(10),
                    use_container_width=True,
                    column_config={
                        "Correlation": st.column_config.NumberColumn(
                            format="%.3f",
                            help="Pearson correlation coefficient (-1 to 1)"
                        )
                    }
                )


def render_scatter_matrix(data, metrics, locations):
    """
    Renders a scatter plot matrix for selected metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    metrics : list
        List of metrics to include in scatter matrix
    locations : list
        Selected locations
    """
    st.subheader("Scatter Plot Matrix")
    
    st.markdown("""
    Visualize relationships between multiple metrics using a scatter plot matrix.
    Each plot shows the relationship between two metrics, with a fit line to indicate trend.
    """)
    
    # Limit the number of metrics for scatter matrix to prevent overloading
    max_metrics = 5
    if len(metrics) > max_metrics:
        selected_metrics = st.multiselect(
            f"Select up to {max_metrics} metrics to include in the scatter matrix:",
            options=metrics,
            default=metrics[:min(3, len(metrics))]
        )
    else:
        selected_metrics = metrics
    
    if len(selected_metrics) < 2:
        st.warning("Please select at least two metrics for the scatter matrix.")
        return
    
    # Create tabs for each location
    tabs = st.tabs(locations)
    
    for i, tab in enumerate(tabs):
        location = locations[i]
        
        with tab:
            # Filter data for the current location
            location_data = data[data['location'] == location] if 'location' in data.columns else data
            
            if location_data.empty:
                st.warning(f"No data available for {location}.")
                continue
                
            # Keep only the columns in metrics list that are in the data
            available_metrics = [metric for metric in selected_metrics if metric in location_data.columns]
            
            if len(available_metrics) < 2:
                st.warning("Need at least two metrics for scatter matrix.")
                continue
                
            # Create filtered dataset with only the selected metrics
            metric_data = location_data[available_metrics].copy()
            
            # Create scatter matrix
            try:
                fig = px.scatter_matrix(
                    metric_data,
                    dimensions=available_metrics,
                    title=f"{location} - Scatter Plot Matrix",
                    opacity=0.7
                )
                
                # Update layout
                fig.update_layout(
                    height=150 * len(available_metrics),
                    width=150 * len(available_metrics)
                )
                
                fig.update_traces(
                    diagonal_visible=False,
                    showupperhalf=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                logger.error(f"Error creating scatter matrix: {str(e)}")
                st.error(f"Error creating scatter matrix: {str(e)}")


def render_statistical_tests(data, primary_metrics, secondary_metrics, locations):
    """
    Renders statistical tests for analyzing relationships between metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    primary_metrics : list
        Primary metrics for analysis
    secondary_metrics : list
        Secondary metrics for analysis
    locations : list
        Selected locations
    """
    st.subheader("Statistical Analysis")
    
    st.markdown("""
    Apply statistical tests to understand relationships between metrics
    and analyze time series properties.
    """)
    
    # Select test type
    test_type = st.selectbox(
        "Select Statistical Test",
        options=[
            "Granger Causality",
            "Augmented Dickey-Fuller Test",
            "Mann-Kendall Trend Test",
            "Linear Regression Analysis"
        ],
        index=0
    )
    
    # Create tabs for each location
    tabs = st.tabs(locations)
    
    for i, tab in enumerate(tabs):
        location = locations[i]
        
        with tab:
            # Filter data for the current location
            location_data = data[data['location'] == location] if 'location' in data.columns else data
            
            if location_data.empty:
                st.warning(f"No data available for {location}.")
                continue
            
            # Run the selected statistical test
            if test_type == "Granger Causality":
                render_granger_causality_test(location_data, primary_metrics, secondary_metrics)
            
            elif test_type == "Augmented Dickey-Fuller Test":
                render_adf_test(location_data, primary_metrics + secondary_metrics)
            
            elif test_type == "Mann-Kendall Trend Test":
                render_mann_kendall_test(location_data, primary_metrics + secondary_metrics)
            
            elif test_type == "Linear Regression Analysis":
                render_linear_regression(location_data, primary_metrics, secondary_metrics)


def render_granger_causality_test(data, primary_metrics, secondary_metrics):
    """
    Renders Granger causality test between metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    primary_metrics : list
        Primary metrics for analysis (potential cause)
    secondary_metrics : list
        Secondary metrics for analysis (potential effect)
    """
    st.markdown("""
    **Granger Causality Test**
    
    This test evaluates if past values of one metric (X) help predict future values 
    of another metric (Y) beyond what can be predicted by past values of Y alone.
    """)
    
    # Set test parameters
    max_lag = st.slider(
        "Maximum Lag (periods)",
        min_value=1,
        max_value=12,
        value=4,
        help="Maximum number of time periods to test for causality."
    )
    
    # Run tests for each pair
    results = []
    
    for cause_metric in primary_metrics:
        if cause_metric not in data.columns:
            continue
            
        for effect_metric in secondary_metrics:
            if effect_metric not in data.columns or cause_metric == effect_metric:
                continue
                
            # Get the two series
            cause_series = data[cause_metric].dropna()
            effect_series = data[effect_metric].dropna()
            
            # Align series on the same index
            common_idx = cause_series.index.intersection(effect_series.index)
            if len(common_idx) < max_lag + 2:
                # Not enough data points for the test
                continue
                
            cause_series = cause_series.loc[common_idx]
            effect_series = effect_series.loc[common_idx]
            
            # Run the test
            try:
                from statsmodels.tsa.stattools import grangercausalitytests
                
                test_results = grangercausalitytests(
                    pd.concat([effect_series, cause_series], axis=1),
                    maxlag=max_lag,
                    verbose=False
                )
                
                # Extract p-values for each lag
                for lag in range(1, max_lag + 1):
                    # Get p-value from F-test
                    p_value = test_results[lag][0]['ssr_ftest'][1]
                    results.append({
                        "Potential Cause": cause_metric,
                        "Potential Effect": effect_metric,
                        "Lag": lag,
                        "P-Value": p_value,
                        "Significant": p_value < 0.05
                    })
            
            except Exception as e:
                logger.error(f"Error in Granger causality test: {str(e)}")
                st.warning(f"Could not run Granger causality test for {cause_metric} -> {effect_metric}: {str(e)}")
    
    if results:
        # Convert to DataFrame and display
        results_df = pd.DataFrame(results)
        
        # Filter to significant results
        significant_results = results_df[results_df["Significant"]]
        
        if not significant_results.empty:
            st.subheader("Significant Granger Causality Relationships")
            st.dataframe(
                significant_results,
                use_container_width=True,
                column_config={
                    "P-Value": st.column_config.NumberColumn(
                        format="%.4f",
                        help="P-value of the test. Values below 0.05 suggest potential causality."
                    )
                }
            )
            
            st.markdown("""
            **Interpretation**: A significant result (p < 0.05) suggests that changes in the "Potential Cause" 
            metric may precede and help predict changes in the "Potential Effect" metric. The "Lag" indicates 
            the time delay of this relationship in periods.
            """)
        else:
            st.info("No significant Granger causality relationships found.")
    else:
        st.warning("Could not run Granger causality tests. Ensure you have sufficient time series data.")


def render_adf_test(data, metrics):
    """
    Renders Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    metrics : list
        Metrics to test for stationarity
    """
    st.markdown("""
    **Augmented Dickey-Fuller Test for Stationarity**
    
    This test evaluates whether a time series is stationary (i.e., its statistical properties
    such as mean, variance, and autocorrelation are constant over time).
    """)
    
    # Run the test for each metric
    results = []
    
    for metric in metrics:
        if metric not in data.columns:
            continue
            
        series = data[metric].dropna()
        
        if len(series) < 10:  # Need enough data points
            continue
            
        try:
            from statsmodels.tsa.stattools import adfuller
            
            test_result = adfuller(series, autolag='AIC')
            
            results.append({
                "Metric": metric,
                "Test Statistic": test_result[0],
                "P-Value": test_result[1],
                "Critical Value (1%)": test_result[4]['1%'],
                "Critical Value (5%)": test_result[4]['5%'],
                "Critical Value (10%)": test_result[4]['10%'],
                "Is Stationary": test_result[1] < 0.05
            })
            
        except Exception as e:
            logger.error(f"Error in ADF test: {str(e)}")
            st.warning(f"Could not run ADF test for {metric}: {str(e)}")
    
    if results:
        # Convert to DataFrame and display
        results_df = pd.DataFrame(results)
        
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "Test Statistic": st.column_config.NumberColumn(format="%.3f"),
                "P-Value": st.column_config.NumberColumn(format="%.4f"),
                "Critical Value (1%)": st.column_config.NumberColumn(format="%.3f"),
                "Critical Value (5%)": st.column_config.NumberColumn(format="%.3f"),
                "Critical Value (10%)": st.column_config.NumberColumn(format="%.3f"),
                "Is Stationary": st.column_config.CheckboxColumn(help="Stationary if p-value < 0.05")
            }
        )
        
        with st.expander("How to interpret the ADF test"):
            st.markdown("""
            - **P-Value**: If less than 0.05, we can reject the null hypothesis that the series has a unit root,
              meaning the series is stationary.
            - **Test Statistic**: More negative values indicate stronger rejection of the null hypothesis.
              If the test statistic is less than the critical value, the series is stationary.
            - **Stationary time series** are easier to predict and model, as their properties don't change over time.
            - **Non-stationary time series** may exhibit trends or seasonality and often need differencing
              before they can be effectively modeled.
            """)
        
        # Offer differencing for non-stationary series
        non_stationary = results_df[~results_df["Is Stationary"]]["Metric"].tolist()
        
        if non_stationary:
            st.subheader("Differencing Non-Stationary Series")
            
            st.markdown("""
            The following series appear to be non-stationary. Differencing can help make them stationary
            by removing trends and seasonality.
            """)
            
            selected_metric = st.selectbox(
                "Select metric to difference",
                options=non_stationary
            )
            
            diff_order = st.radio(
                "Differencing Order",
                options=["First Difference", "Second Difference", "Seasonal Difference (12 periods)"],
                index=0,
                horizontal=True
            )
            
            series = data[selected_metric].dropna()
            
            if diff_order == "First Difference":
                diff_series = series.diff().dropna()
                diff_label = "First Difference"
            elif diff_order == "Second Difference":
                diff_series = series.diff().diff().dropna()
                diff_label = "Second Difference"
            else:  # Seasonal
                diff_series = series.diff(12).dropna()
                diff_label = "Seasonal Difference (12 periods)"
            
            # Create figure with original and differenced series
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            
            fig.add_trace(
                go.Scatter(x=series.index, y=series, name="Original Series"),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=diff_series.index, y=diff_series, name=diff_label),
                row=2, col=1
            )
            
            fig.update_layout(
                height=600,
                title=f"{selected_metric} - Original vs. Differenced Series",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Run ADF test on differenced series
            try:
                from statsmodels.tsa.stattools import adfuller
                diff_result = adfuller(diff_series.dropna(), autolag='AIC')
                
                col1, col2 = st.columns(2)
                col1.metric(
                    "Differenced Series P-Value",
                    f"{diff_result[1]:.4f}",
                    help="P-value of the ADF test for the differenced series."
                )
                
                is_stationary = diff_result[1] < 0.05
                col2.metric(
                    "Is Stationary After Differencing?",
                    "Yes" if is_stationary else "No",
                    help="Stationary if p-value < 0.05"
                )
            except Exception as e:
                st.warning(f"Could not run ADF test on differenced series: {str(e)}")
    else:
        st.warning("Could not run ADF test. Ensure you have sufficient time series data.")


def render_mann_kendall_test(data, metrics):
    """
    Renders Mann-Kendall trend test for time series.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    metrics : list
        Metrics to test for trends
    """
    st.markdown("""
    **Mann-Kendall Trend Test**
    
    This test evaluates whether there is a monotonic upward or downward trend in a time series.
    It's a non-parametric test that doesn't assume a specific distribution of the data.
    """)
    
    # Run the test for each metric
    results = []
    
    for metric in metrics:
        if metric not in data.columns:
            continue
            
        series = data[metric].dropna()
        
        if len(series) < 10:  # Need enough data points
            continue
            
        try:
            from scipy import stats
            import pymannkendall as mk
            
            # Simple linear regression for slope
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Mann-Kendall test
            result = mk.original_test(series)
            
            results.append({
                "Metric": metric,
                "Trend": result.trend,
                "MK Trend Strength": result.s,
                "MK P-Value": result.p,
                "Slope (per period)": slope,
                "Regression P-Value": p_value,
                "R-squared": r_value**2
            })
            
        except Exception as e:
            try:
                # Alternative if pymannkendall is not available
                from scipy import stats
                
                # Simple linear regression for slope
                x = np.arange(len(series))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
                
                # Use linear regression as fallback
                trend = "increasing" if slope > 0 and p_value < 0.05 else "decreasing" if slope < 0 and p_value < 0.05 else "no trend"
                
                results.append({
                    "Metric": metric,
                    "Trend": trend,
                    "MK Trend Strength": "N/A",
                    "MK P-Value": "N/A",
                    "Slope (per period)": slope,
                    "Regression P-Value": p_value,
                    "R-squared": r_value**2
                })
                
            except Exception as e2:
                logger.error(f"Error in trend test: {str(e)} / {str(e2)}")
                st.warning(f"Could not run trend test for {metric}")
    
    if results:
        # Convert to DataFrame and display
        results_df = pd.DataFrame(results)
        
        # Format the display
        display_df = results_df.copy()
        for col in ["MK P-Value", "Regression P-Value", "R-squared"]:
            if col in display_df.columns and display_df[col].dtype != object:
                display_df[col] = display_df[col].round(4)
        
        st.dataframe(
            display_df,
            use_container_width=True
        )
        
        with st.expander("How to interpret trend tests"):
            st.markdown("""
            - **Trend**: Direction of the trend (increasing, decreasing, or no trend)
            - **MK Trend Strength**: The Mann-Kendall statistic (S) indicates the strength and direction of the trend.
              Positive values indicate an increasing trend, negative values indicate a decreasing trend.
            - **MK P-Value**: If less than 0.05, there is a statistically significant trend.
            - **Slope**: The change in the metric per time period based on linear regression.
            - **Regression P-Value**: Statistical significance of the linear regression.
            - **R-squared**: Proportion of variance explained by the linear trend (0-1).
            """)
        
        # Visualize trends for a selected metric
        st.subheader("Trend Visualization")
        
        selected_metric = st.selectbox(
            "Select metric to visualize trend",
            options=results_df["Metric"].tolist()
        )
        
        series = data[selected_metric].dropna()
        
        # Prepare trend line
        x = np.arange(len(series))
        x_dates = series.index
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        trend_line = intercept + slope * x
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=series,
                mode='markers+lines',
                name=selected_metric,
                line=dict(width=1)
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_dates,
                y=trend_line,
                mode='lines',
                name='Linear Trend',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title=f"{selected_metric} - Trend Analysis",
            xaxis_title="Date",
            yaxis_title=selected_metric,
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show trend statistics
        metric_result = results_df[results_df["Metric"] == selected_metric].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "Trend Direction",
            metric_result["Trend"].capitalize(),
            help="Direction of the trend based on statistical tests."
        )
        
        col2.metric(
            "Change per Period",
            f"{metric_result['Slope (per period)']:.4f}",
            help="The average change in the metric per time period."
        )
        
        col3.metric(
            "R-squared",
            f"{metric_result['R-squared']:.4f}",
            help="Proportion of variance explained by the trend (0-1)."
        )
        
    else:
        st.warning("Could not run trend tests. Ensure you have sufficient time series data.")


def render_linear_regression(data, x_metrics, y_metrics):
    """
    Renders linear regression analysis between metrics.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    x_metrics : list
        Metrics to use as independent variables
    y_metrics : list
        Metrics to use as dependent variables
    """
    st.markdown("""
    **Linear Regression Analysis**
    
    This analysis examines the linear relationship between two metrics,
    quantifying how one metric (independent variable) may relate to another 
    (dependent variable).
    """)
    
    # Select one x and one y
    if not x_metrics or not y_metrics:
        st.warning("Please select at least one primary and one secondary metric.")
        return
        
    x_metric = st.selectbox("Select independent variable (X)", options=x_metrics)
    y_metric = st.selectbox("Select dependent variable (Y)", options=y_metrics)
    
    if x_metric not in data.columns or y_metric not in data.columns:
        st.warning("Selected metrics not available in the data.")
        return
    
    # Get the data
    x_data = data[x_metric].dropna()
    y_data = data[y_metric].dropna()
    
    # Align on common index
    common_idx = x_data.index.intersection(y_data.index)
    if len(common_idx) < 10:
        st.warning("Not enough data points for regression analysis.")
        return
        
    x_aligned = x_data.loc[common_idx].values
    y_aligned = y_data.loc[common_idx].values
    dates = data.index[data.index.isin(common_idx)]
    
    # Run regression
    try:
        # Simple linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_aligned, y_aligned)
        
        # More detailed regression with statsmodels
        X = sm.add_constant(x_aligned)  # Add constant for intercept
        model = sm.OLS(y_aligned, X)
        results = model.fit()
        
        # Create scatter plot with regression line
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=x_aligned,
                y=y_aligned,
                mode='markers',
                name='Data Points',
                marker=dict(
                    size=8,
                    color=np.arange(len(dates)),  # Color by time
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Time")
                ),
                text=dates.strftime('%Y-%m-%d')  # Show dates on hover
            )
        )
        
        # Add regression line
        x_range = np.linspace(min(x_aligned), max(x_aligned), 100)
        y_pred = intercept + slope * x_range
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name='Regression Line',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            title=f"Linear Regression: {y_metric} vs. {x_metric}",
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display regression statistics
        st.subheader("Regression Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        col1.metric(
            "R-squared",
            f"{r_value**2:.4f}",
            help="Proportion of variance explained by the model (0-1)."
        )
        
        col2.metric(
            "P-value",
            f"{p_value:.4f}",
            help="Statistical significance of the relationship. Values below 0.05 suggest a significant relationship."
        )
        
        col3.metric(
            "Slope",
            f"{slope:.4f}",
            help=f"Change in {y_metric} for a 1-unit change in {x_metric}."
        )
        
        # Display regression equation
        st.markdown(f"""
        **Regression Equation**:  
        {y_metric} = {intercept:.4f} + {slope:.4f} × {x_metric}
        """)
        
        # Display detailed regression results
        with st.expander("Detailed Regression Results"):
            st.text(results.summary().as_text())
        
        # Perform analysis of residuals
        st.subheader("Residual Analysis")
        
        # Calculate residuals
        y_pred_full = intercept + slope * x_aligned
        residuals = y_aligned - y_pred_full
        
        # Create residual plots
        fig = make_subplots(rows=1, cols=2)
        
        # Residuals vs. Fitted values
        fig.add_trace(
            go.Scatter(
                x=y_pred_full,
                y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(size=8)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred_full), max(y_pred_full)],
                y=[0, 0],
                mode='lines',
                name='Zero Line',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        # QQ plot for normality of residuals
        from scipy import stats
        
        # Calculate quantiles for QQ plot
        residuals_sorted = np.sort(residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals_sorted)))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=residuals_sorted,
                mode='markers',
                name='QQ Plot',
                marker=dict(size=8)
            ),
            row=1, col=2
        )
        
        # Add reference line
        slope_qq = np.std(residuals_sorted)
        intercept_qq = np.mean(residuals_sorted)
        ref_line = intercept_qq + slope_qq * theoretical_quantiles
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=ref_line,
                mode='lines',
                name='Reference Line',
                line=dict(color='red', width=1, dash='dash')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            template="plotly_white"
        )
        
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation of residual analysis
        with st.expander("How to interpret residual plots"):
            st.markdown("""
            **Residuals vs. Fitted Values**:
            - Residuals should be randomly scattered around the zero line.
            - Patterns (like a curved shape) suggest the relationship might not be linear.
            - Residuals with increasing/decreasing spread suggest non-constant variance (heteroscedasticity).
            
            **QQ Plot (Quantile-Quantile Plot)**:
            - Points following the reference line suggest residuals are normally distributed.
            - Systematic deviations from the line suggest non-normality.
            - Outliers appear as points far from the reference line.
            """)
        
    except Exception as e:
        logger.error(f"Error in regression analysis: {str(e)}")
        st.error(f"Error in regression analysis: {str(e)}")


def render_custom_metric_combination(data, primary_metrics, secondary_metrics, locations):
    """
    Renders custom metric combination analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for analysis
    primary_metrics : list
        Primary metrics for combination
    secondary_metrics : list
        Secondary metrics for combination
    locations : list
        Selected locations
    """
    st.subheader("Custom Metric Combination")
    
    st.markdown("""
    Create custom combinations of metrics to analyze relationships and 
    derive new insights. This tool allows you to combine metrics using
    mathematical operations and visualize the results.
    """)
    
    # Available metrics
    all_metrics = primary_metrics + secondary_metrics
    if not all_metrics:
        st.warning("Please select metrics in the sidebar to create combinations.")
        return
        
    # Create tabs for each location
    tabs = st.tabs(locations)
    
    for i, tab in enumerate(tabs):
        location = locations[i]
        
        with tab:
            # Filter data for the current location
            location_data = data[data['location'] == location] if 'location' in data.columns else data
            
            if location_data.empty:
                st.warning(f"No data available for {location}.")
                continue
            
            # Keep only the selected metrics
            available_metrics = [m for m in all_metrics if m in location_data.columns]
            
            if not available_metrics:
                st.warning("None of the selected metrics are available in the data.")
                continue
                
            # Create custom combinations
            st.subheader(f"Create Custom Metric for {location}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                custom_name = st.text_input(
                    "Name for custom metric",
                    value="Custom Metric",
                    key=f"custom_name_{location}"
                )
                
            with col2:
                show_formula = st.checkbox(
                    "Show formula on chart",
                    value=True,
                    key=f"show_formula_{location}"
                )
            
            # Create formula
            st.markdown("### Define Formula")
            
            formula_components = []
            operations = ['+', '-', '*', '/', 'None']
            
            # First metric (always required)
            first_metric = st.selectbox(
                "First metric",
                options=available_metrics,
                key=f"first_metric_{location}"
            )
            
            formula_components.append(first_metric)
            
            # Option to apply transformation to the first metric
            first_transform = st.selectbox(
                "Transform first metric",
                options=["None", "log", "sqrt", "square", "percent_change", "rolling_avg"],
                index=0,
                key=f"first_transform_{location}"
            )
            
            if first_transform != "None":
                if first_transform == "rolling_avg":
                    window = st.slider(
                        "Rolling window size",
                        min_value=2,
                        max_value=12,
                        value=3,
                        key=f"window_{location}"
                    )
                    formula_components[-1] = f"Rolling Avg({formula_components[-1]}, {window})"
                else:
                    formula_components[-1] = f"{first_transform}({formula_components[-1]})"
            
            # Add more components
            num_components = st.slider(
                "Number of additional components",
                min_value=0,
                max_value=5,
                value=1,
                key=f"num_components_{location}"
            )
            
            for j in range(num_components):
                col1, col2 = st.columns(2)
                
                with col1:
                    operation = st.selectbox(
                        f"Operation {j+1}",
                        options=operations,
                        index=0,
                        key=f"operation_{j}_{location}"
                    )
                
                with col2:
                    if operation != "None":
                        metric = st.selectbox(
                            f"Metric {j+1}",
                            options=available_metrics,
                            index=min(j+1, len(available_metrics)-1),
                            key=f"metric_{j}_{location}"
                        )
                        
                        # Option to apply transformation
                        transform = st.selectbox(
                            f"Transform metric {j+1}",
                            options=["None", "log", "sqrt", "square", "percent_change", "rolling_avg"],
                            index=0,
                            key=f"transform_{j}_{location}"
                        )
                        
                        if transform != "None":
                            if transform == "rolling_avg":
                                window = st.slider(
                                    f"Rolling window size {j+1}",
                                    min_value=2,
                                    max_value=12,
                                    value=3,
                                    key=f"window_{j}_{location}"
                                )
                                metric_str = f"Rolling Avg({metric}, {window})"
                            else:
                                metric_str = f"{transform}({metric})"
                        else:
                            metric_str = metric
                            
                        formula_components.append(operation)
                        formula_components.append(metric_str)
            
            # Create the formula string for display
            formula_str = " ".join(formula_components)
            
            # Display the formula
            st.subheader("Formula Preview")
            st.code(formula_str)
            
            # Calculate the custom metric
            try:
                result_series = calculate_custom_metric(
                    location_data,
                    formula_components
                )
                
                if result_series is not None:
                    # Plot the custom metric
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=result_series.index,
                            y=result_series,
                            mode='lines+markers',
                            name=custom_name,
                            line=dict(width=2)
                        )
                    )
                    
                    # Add the formula as an annotation if requested
                    if show_formula:
                        fig.add_annotation(
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=1.05,
                            text=formula_str,
                            showarrow=False,
                            font=dict(size=12),
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            bgcolor="white",
                            opacity=0.8
                        )
                    
                    fig.update_layout(
                        title=f"{custom_name} for {location}",
                        xaxis_title="Date",
                        yaxis_title=custom_name,
                        template="plotly_white",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate basic statistics
                    if len(result_series) > 0:
                        stats_data = {
                            "Statistic": [
                                "Mean", "Median", "Standard Deviation",
                                "Minimum", "Maximum", "Current Value",
                                "Change from Previous", "YoY Change"
                            ],
                            "Value": [
                                result_series.mean(),
                                result_series.median(),
                                result_series.std(),
                                result_series.min(),
                                result_series.max(),
                                result_series.iloc[-1],
                                result_series.iloc[-1] - result_series.iloc[-2] if len(result_series) > 1 else None,
                                result_series.iloc[-1] - result_series.iloc[-12] if len(result_series) > 12 else None
                            ]
                        }
                        
                        stats_df = pd.DataFrame(stats_data)
                        st.dataframe(stats_df, use_container_width=True)
                
            except Exception as e:
                logger.error(f"Error calculating custom metric: {str(e)}")
                st.error(f"Error calculating custom metric: {str(e)}")


def calculate_custom_metric(data, formula_components):
    """
    Calculates a custom metric based on a formula.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data for calculation
    formula_components : list
        List of formula components including metrics and operations
    
    Returns:
    --------
    pandas.Series
        Calculated custom metric
    """
    try:
        # Process the first component
        first_comp = formula_components[0]
        
        # Extract the base metric and any transformations
        if "(" in first_comp:
            transform, metric = extract_transform_and_metric(first_comp)
            result = apply_transform(data[metric], transform)
        else:
            result = data[first_comp].copy()
        
        # Process the remaining components
        for i in range(1, len(formula_components), 2):
            if i+1 < len(formula_components):
                operation = formula_components[i]
                component = formula_components[i+1]
                
                if operation == "None":
                    continue
                
                # Extract the base metric and any transformations
                if "(" in component:
                    transform, metric = extract_transform_and_metric(component)
                    value = apply_transform(data[metric], transform)
                else:
                    value = data[component]
                
                # Apply the operation
                if operation == "+":
                    result = result + value
                elif operation == "-":
                    result = result - value
                elif operation == "*":
                    result = result * value
                elif operation == "/":
                    # Avoid division by zero
                    result = result / value.replace(0, np.nan)
        
        return result
    
    except Exception as e:
        logger.error(f"Error in calculate_custom_metric: {str(e)}")
        raise


def extract_transform_and_metric(component):
    """
    Extracts transform type and metric from a component string.
    
    Parameters:
    -----------
    component : str
        Component string, e.g., "log(Median Sale Price)" or "Rolling Avg(Mortgage Rates, 3)"
    
    Returns:
    --------
    tuple
        (transform_type, metric_name)
    """
    if "Rolling Avg" in component:
        # Extract the metric and window size
        parts = component.strip(")")
        window_parts = parts.split(", ")
        metric = window_parts[0].split("(")[1]
        window = int(window_parts[1])
        return ("rolling_avg", metric, window)
    else:
        transform = component.split("(")[0]
        metric = component.split("(")[1].strip(")")
        return (transform, metric)


def apply_transform(series, transform):
    """
    Applies a transformation to a time series.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series to transform
    transform : str or tuple
        Transformation to apply
    
    Returns:
    --------
    pandas.Series
        Transformed series
    """
    if isinstance(transform, tuple):
        if transform[0] == "rolling_avg":
            return series.rolling(window=transform[2], min_periods=1).mean()
    elif transform == "log":
        # Ensure all values are positive
        min_val = series.min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            return np.log(series + offset)
        else:
            return np.log(series)
    elif transform == "sqrt":
        # Ensure all values are positive
        min_val = series.min()
        if min_val < 0:
            offset = abs(min_val) + 1
            return np.sqrt(series + offset)
        else:
            return np.sqrt(series)
    elif transform == "square":
        return np.square(series)
    elif transform == "percent_change":
        return series.pct_change() * 100
    
    # Default: return original series
    return series
