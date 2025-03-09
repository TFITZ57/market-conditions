"""
Reports Page

This module renders the Reports page of the application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
import os
import json
from typing import Dict, Any

from src.utils.config import load_config
from src.ai_analysis.report_generator import generate_report, generate_comparative_report
from src.ai_analysis.report_formatter import format_report
from src.visualization.charts import create_metric_chart
from src.visualization.exporters import export_visualizations
from src.ui.components.report_viewer import (
    display_report, 
    display_report_list, 
    display_report_editor,
    manage_report_repository,
    save_report_to_repository,
    delete_report_from_repository
)

def show_reports_page(filters: Dict[str, Any], report_settings: Dict[str, Any]):
    """
    Render the Reports page.
    
    Args:
        filters: Dictionary containing filtering options
        report_settings: Dictionary containing report generation settings
    """
    st.header("Generated Reports")
    
    # Placeholder for actual implementation
    st.info("Report generation and management tools will be displayed here.")
    
    # Display the filters being used (for debugging)
    with st.expander("Current Filters"):
        st.write(filters)
        
    # Display report settings
    with st.expander("Report Settings"):
        st.write(report_settings)

def show_report_generation(filters, report_settings, config):
    """
    Display the report generation section.
    
    Args:
        filters (dict): Dictionary containing filter selections.
        report_settings (dict): Dictionary containing report generation settings.
        config (dict): Application configuration.
    """
    st.subheader("Generate New Report")
    
    # Display current filter selections
    st.write(f"**Selected Time Period:** {filters['date_range']['start_quarter']} to {filters['date_range']['end_quarter']}")
    if filters['location']['towns'] != config['locations']['towns']:
        st.write(f"**Selected Towns:** {', '.join(filters['location']['towns'])}")
    if filters['property_types'] != config['property_types']:
        st.write(f"**Selected Property Types:** {', '.join(filters['property_types'])}")
    
    # Display report settings if provided
    if report_settings:
        st.write(f"**LLM Provider:** {report_settings['llm']['provider']}")
        st.write(f"**Model:** {report_settings['llm']['model']}")
        st.write(f"**Report Type:** {report_settings['report']['type']}")
        st.write(f"**Tone:** {report_settings['report']['tone']}")
    else:
        st.warning("Report settings not configured. Please configure settings in the sidebar.")
        report_settings = {
            'llm': {'provider': 'openai', 'model': 'gpt-4o'},
            'report': {
                'type': 'General Analysis',
                'tone': 'Informative',
                'length': 'Standard',
                'include_charts': True,
                'include_recommendations': True,
                'include_data_tables': False
            }
        }
    
    # Select metrics to include in the report
    selected_metrics = st.multiselect(
        "Select metrics to include in the report",
        options=config["metrics"]["housing"] + config["metrics"]["economic"],
        default=["Median Sale Price", "Sales Volume", "Days on Market", "Listing Inventory", "Unemployment Rate"]
    )
    
    if not selected_metrics:
        st.warning("Please select at least one metric to include in the report.")
    
    # Add button to generate report
    generate_button = st.button("Generate Report", type="primary", disabled=not selected_metrics)
    
    if generate_button:
        with st.spinner("Generating report... This may take a minute."):
            # In a real implementation, we would load actual data from the processed data directory
            # For this demonstration, we'll generate synthetic data
            data_dict = generate_synthetic_data(
                selected_metrics, 
                filters['date_range']['quarters_list'],
                filters['location']['towns']
            )
            
            # Create parameters for report generation
            report_params = {
                'provider': report_settings['llm']['provider'],
                'model': report_settings['llm']['model'],
                'tone': report_settings['report']['tone'],
                'length': report_settings['report']['length'],
                'include_charts': report_settings['report']['include_charts'],
                'include_recommendations': report_settings['report']['include_recommendations'],
                'include_data_tables': report_settings['report']['include_data_tables'],
                'selected_metrics': selected_metrics,
                'filters': filters
            }
            
            # Generate report
            if report_settings['report']['type'] == 'Comparative Analysis':
                # For comparative analysis, we need to specify locations to compare
                report = generate_comparative_report(
                    data_dict, 
                    filters['location']['towns'],
                    selected_metrics,
                    report_params
                )
            else:
                report = generate_report(
                    data_dict, 
                    report_settings['report']['type'].lower(),
                    report_params
                )
            
            # Save report to session state
            st.session_state['current_report'] = report
            
            # Create a flag to indicate report generation is complete
            st.session_state['report_generated'] = True
    
    # Display generated report if available
    if 'report_generated' in st.session_state and st.session_state['report_generated']:
        report = st.session_state.get('current_report')
        
        if report:
            st.divider()
            st.subheader("Generated Report")
            
            display_report(report)
            
            # Add button to save report
            if st.button("Save Report to Repository"):
                saved_path = save_report_to_repository(report)
                
                if saved_path:
                    st.success(f"Report saved to repository")
                    # Reset report generation flag
                    st.session_state['report_generated'] = False
                    st.session_state['current_report'] = None

def show_saved_reports():
    """Display the saved reports section."""
    st.subheader("Saved Reports")
    
    # Load reports from repository
    reports = manage_report_repository()
    
    if not reports:
        st.info("No saved reports found. Generate and save reports to see them here.")
        return
    
    # Display report list
    def handle_report_selection(report):
        st.session_state['selected_report'] = report
    
    display_report_list(reports, handle_report_selection)
    
    # Display selected report if available
    if 'selected_report' in st.session_state and st.session_state['selected_report']:
        st.divider()
        st.subheader("Selected Report")
        
        selected_report = st.session_state['selected_report']
        display_report(selected_report)
        
        # Add button to clear selection
        if st.button("Close Report"):
            st.session_state['selected_report'] = None
            st.experimental_rerun()
    
    # Display report editor if editing
    if 'editing_report' in st.session_state and st.session_state['editing_report']:
        st.divider()
        
        def handle_save(updated_report):
            # Save updated report
            reports[st.session_state['editing_report_index']] = updated_report
            saved_path = save_report_to_repository(updated_report)
            
            if saved_path:
                st.success(f"Report updated and saved to repository")
                st.session_state['editing_report'] = None
                st.experimental_rerun()
        
        updated_report = display_report_editor(st.session_state['editing_report'], handle_save)
        
        if updated_report:
            # Report was updated
            reports[st.session_state['editing_report_index']] = updated_report
            st.session_state['editing_report'] = None
            st.experimental_rerun()

def generate_synthetic_data(metrics, quarters, towns):
    """
    Generate synthetic data for report generation.
    
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
        "List Price to Sales Price Ratio": {"base": 98, "trend": -0.1, "volatility": 1, "seasonality": [0.01, 0.02, -0.01, -0.02]},
        "Home Price-to-Income Ratio": {"base": 5.8, "trend": 0.05, "volatility": 0.2, "seasonality": [0.01, 0.02, -0.01, -0.02]},
        "Mortgage Rates": {"base": 6.5, "trend": 0.05, "volatility": 0.2, "seasonality": [-0.05, 0.1, 0.05, -0.1]},
        "Housing Affordability Index": {"base": 95, "trend": -0.3, "volatility": 3, "seasonality": [0.02, -0.02, -0.01, 0.01]},
        "Vacancy Rates": {"base": 3.8, "trend": -0.02, "volatility": 0.3, "seasonality": [0.05, -0.05, -0.05, 0.05]},
        "Seller Concessions": {"base": 1.2, "trend": 0.03, "volatility": 0.2, "seasonality": [0.1, -0.05, -0.1, 0.05]},
        # Economic metrics
        "Local Job Growth": {"base": 1.8, "trend": 0.05, "volatility": 0.3, "seasonality": [0.2, 0.1, -0.1, -0.2]},
        "Employment Trends": {"base": 95.5, "trend": 0.1, "volatility": 0.5, "seasonality": [0.1, 0.2, -0.1, -0.2]},
        "Unemployment Rate": {"base": 4.2, "trend": -0.05, "volatility": 0.3, "seasonality": [0.1, -0.2, -0.1, 0.2]},
        "Median Household Income": {"base": 95000, "trend": 1000, "volatility": 2000, "seasonality": [0, 0, 0, 0]},
        "GDP Growth": {"base": 2.5, "trend": 0.1, "volatility": 0.5, "seasonality": [0.3, 0.1, -0.1, -0.3]},
        "Population Growth": {"base": 0.8, "trend": -0.01, "volatility": 0.1, "seasonality": [0, 0, 0, 0]},
        "Consumer Price Index": {"base": 295, "trend": 3, "volatility": 2, "seasonality": [0.5, 0.3, -0.3, -0.5]},
        "Interest Rates": {"base": 5.8, "trend": 0.1, "volatility": 0.2, "seasonality": [-0.1, 0.1, 0.1, -0.1]}
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
        
        # Check if this is a town-level metric
        is_town_level = metric not in ["Mortgage Rates", "Housing Affordability Index", "GDP Growth", "Population Growth", "Consumer Price Index", "Interest Rates"]
        
        if is_town_level:
            # Generate data for each town
            for town in towns:
                town_factor = town_factors.get(town, 1.0)
                
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
                    value = base * town_factor