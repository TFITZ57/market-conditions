"""
UI Pages Module

This module contains the different pages of the Streamlit application:
- overview: Dashboard with summary metrics and key trends
- housing_metrics: Detailed housing market metrics visualizations
- economic: Economic indicators analysis
- comparison: Comparative analysis between locations and time periods
- forecast: Time series forecasting of market metrics
- analysis: Advanced data analysis tools
- reports: Generated report management
"""

from .overview import render_overview_page
from .housing_metrics import render_housing_metrics_page
from .economic import render_economic_page
from .comparison import render_comparison_page
from .forecast import render_forecast_page
from .analysis import render_analysis_page
from .reports import render_reports_page

__all__ = [
    'render_overview_page',
    'render_housing_metrics_page',
    'render_economic_page',
    'render_comparison_page',
    'render_forecast_page',
    'render_analysis_page',
    'render_reports_page'
]
