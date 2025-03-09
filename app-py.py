import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import application modules
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data_collection.data_fetcher import DataFetcher
from src.data_processing.metrics_calculator import MetricsCalculator
from src.ui.pages.housing_metrics import show_housing_metrics_page
from src.ui.pages.economic import show_economic_page
from src.ui.pages.comparison import show_comparison_page
from src.ui.pages.forecast import show_forecast_page
from src.ui.pages.reports import show_reports_page
from src.utils.security import get_api_key

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)

# Load configuration
config = load_config()

def check_api_keys_available():
    """
    Check if all required API keys are available.
    
    Returns:
        bool: True if all API keys are available, False otherwise.
    """
    required_apis = ["fred", "bls", "attom"]
    missing_keys = []
    
    for api in required_apis:
        if not get_api_key(api):
            missing_keys.append(api)
    
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        return False
    
    return True

def check_data_available():
    """
    Check if data is available in the raw or processed directories.
    
    Returns:
        bool: True if data is available, False otherwise.
    """
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "api_data" / "raw"
    processed_dir = base_dir / "api_data" / "processed"
    
    # Check if directories exist
    if not raw_dir.exists() and not processed_dir.exists():
        logger.warning("Data directories do not exist")
        return False
    
    # Check if directories have files
    raw_files = list(raw_dir.glob("*")) if raw_dir.exists() else []
    processed_files = list(processed_dir.glob("*")) if processed_dir.exists() else []
    
    if not raw_files and not processed_files:
        logger.warning("No data files found in raw or processed directories")
        return False
    
    return True

def generate_synthetic_data():
    """
    Generate synthetic data using the generate_synthetic_data.py script.
    
    Returns:
        bool: True if successful, False otherwise.
    """
    logger.info("Generating synthetic data...")
    
    try:
        script_path = Path(__file__).resolve().parent / "scripts" / "generate_synthetic_data.py"
        
        if not script_path.exists():
            logger.error(f"Synthetic data generation script not found at: {script_path}")
            return False
        
        # Make the script executable if it's not already
        os.chmod(script_path, 0o755)
        
        # Run the script to generate all datasets
        result = subprocess.run([str(script_path), "--dataset", "all"], 
                               capture_output=True, text=True, check=False)
        
        if result.returncode != 0:
            logger.error(f"Error generating synthetic data: {result.stderr}")
            return False
        
        logger.info("Synthetic data generated successfully")
        logger.debug(f"Script output: {result.stdout}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        return False

def main():
    """Main application entry point."""
    # Set page configuration
    st.set_page_config(
        page_title="Fairfield County Housing Market Analysis",
        page_icon="🏡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check for API keys and data availability
    api_keys_available = check_api_keys_available()
    data_available = check_data_available()
    
    # If API keys are missing or no data is available, generate synthetic data
    if not api_keys_available or not data_available:
        if st.session_state.get('synthetic_data_generated') is None:
            st.session_state['synthetic_data_generated'] = False
            
            # Show a message about missing data
            data_source_msg = ""
            if not api_keys_available:
                data_source_msg += "API keys are missing. "
            if not data_available:
                data_source_msg += "No data is available. "
                
            with st.spinner(f"{data_source_msg}Generating synthetic data for demonstration..."):
                success = generate_synthetic_data()
                if success:
                    st.session_state['synthetic_data_generated'] = True
                    st.success("Synthetic data generated successfully. Using synthetic data for demonstration.")
                else:
                    st.error("Failed to generate synthetic data. Please check logs for details.")
    
    # Set up application header
    with st.container():
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Try to display logo if available
            logo_path = os.path.join("assets", "fairfield_county_logo.png")
            if os.path.exists(logo_path):
                st.image(logo_path, width=100)
            else:
                st.title("FC")
        
        with col2:
            st.title("Fairfield County Housing Market Analysis")
            
            # Display data source indicator
            if st.session_state.get('synthetic_data_generated', False):
                st.caption("⚠️ Using synthetic data for demonstration. API keys are required for real data.")
    
    # Create sidebar with filters
    with st.sidebar:
        st.header("Data Filters")
        
        # Date range selection
        st.subheader("Time Period")
        
        # Create list of available quarters from 2015 to present
        current_year = datetime.now().year
        current_quarter = (datetime.now().month - 1) // 3 + 1
        
        available_quarters = []
        
        for year in range(2015, current_year + 1):
            for quarter in range(1, 5):
                if year == current_year and quarter > current_quarter:
                    # Skip future quarters
                    continue
                
                available_quarters.append(f"{year} Q{quarter}")
        
        # Create date range slider
        start_quarter_idx, end_quarter_idx = st.select_slider(
            "Select date range",
            options=list(range(len(available_quarters))),
            value=(0, len(available_quarters) - 1),
            format_func=lambda x: available_quarters[x]
        )
        
        start_quarter = available_quarters[start_quarter_idx]
        end_quarter = available_quarters[end_quarter_idx]
        
        # Create list of all quarters in the selected range
        quarters_list = available_quarters[start_quarter_idx:(end_quarter_idx + 1)]
        
        # Create date range dictionary
        date_range = {
            "start_quarter": start_quarter,
            "end_quarter": end_quarter,
            "quarters_list": quarters_list
        }
        
        # Location selection
        st.subheader("Location")
        
        # County level selection
        county_selected = st.checkbox("Fairfield County", value=True)
        
        # Town selection
        st.write("Towns:")
        all_towns_selected = st.checkbox("Select All Towns", value=True)
        
        # Get list of towns
        towns = config["locations"]["towns"]
        
        if all_towns_selected:
            town_selection = towns
        else:
            town_selection = st.multiselect(
                "Select towns",
                options=towns,
                default=towns[:5]  # Default to first 5 towns
            )
        
        # Create location dictionary
        location = {
            "county": ["Fairfield"] if county_selected else [],
            "towns": town_selection
        }
        
        # Property type selection
        st.subheader("Property Type")
        
        all_property_types = st.checkbox("All Property Types", value=True)
        
        # Get list of property types
        property_types = config["property_types"]
        
        if all_property_types:
            property_type_selection = property_types
        else:
            property_type_selection = st.multiselect(
                "Select property types",
                options=property_types,
                default=property_types
            )
        
        # Create filters dictionary
        filters = {
            "date_range": date_range,
            "location": location,
            "property_types": property_type_selection
        }
        
        # Add report generation settings
        st.header("Report Settings")
        
        # LLM provider selection
        st.subheader("LLM Provider")
        
        llm_provider = st.selectbox(
            "Select provider",
            options=["OpenAI", "Anthropic"]
        )
        
        # Model selection based on provider
        available_models = (
            ["gpt-4o", "gpt-o1", "gpt-o3-mini"] 
            if llm_provider == "OpenAI" 
            else ["claude-3.5-sonnet-20241022", "claude-3.7-sonnet"]
        )
        
        llm_model = st.selectbox(
            "Select model",
            options=available_models
        )
        
        # Report type selection
        st.subheader("Report Type")
        
        report_type = st.selectbox(
            "Select report type",
            options=[
                "General Analysis",
                "Investment Analysis",
                "Market Forecast",
                "Comparative Analysis",
                "Selling Recommendations",
                "Buying Recommendations"
            ]
        )
        
        # Report tone selection
        report_tone = st.selectbox(
            "Select tone",
            options=[
                "Analytical",
                "Informative",
                "Casual",
                "Skeptical",
                "Academic"
            ],
            index=1  # Default to Informative
        )
        
        # Report length selection
        report_length = st.select_slider(
            "Report length",
            options=["Brief", "Standard", "Comprehensive"],
            value="Standard"
        )
        
        # Add additional report options
        include_charts = st.checkbox("Include Chart References", value=True)
        include_recommendations = st.checkbox("Include Recommendations", value=True)
        include_data_tables = st.checkbox("Include Data Tables", value=False)
        
        # Create report settings dictionary
        report_settings = {
            "llm": {
                "provider": llm_provider.lower(),
                "model": llm_model
            },
            "report": {
                "type": report_type,
                "tone": report_tone,
                "length": report_length,
                "include_charts": include_charts,
                "include_recommendations": include_recommendations,
                "include_data_tables": include_data_tables
            }
        }
    
    # Create tabs for main content pages
    tabs = st.tabs([
        "Housing Metrics", 
        "Economic Indicators", 
        "Comparison", 
        "Forecast", 
        "Reports"
    ])
    
    # Show Housing Metrics page
    with tabs[0]:
        show_housing_metrics_page(filters)
    
    # Show Economic Indicators page
    with tabs[1]:
        show_economic_page(filters)
    
    # Show Comparison page
    with tabs[2]:
        show_comparison_page(filters)
    
    # Show Forecast page
    with tabs[3]:
        show_forecast_page(filters)
    
    # Show Reports page
    with tabs[4]:
        show_reports_page(filters, report_settings)
    
    # Add footer
    st.markdown("---")
    st.caption("© 2025 Fairfield County Housing Market Analysis | Data sources: FRED, BLS, ATTOM")

if __name__ == "__main__":
    main()
