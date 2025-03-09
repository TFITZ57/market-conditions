#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update data from APIs for the Fairfield County Housing Market Analysis application.
This script fetches the latest data from FRED, BLS, and ATTOM Property APIs
and updates the local data storage. It is designed to be run quarterly as a cron job.
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory for data storage
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "api_data" / "raw"
PROCESSED_DIR = BASE_DIR / "api_data" / "processed"

# Ensure directories exist
for directory in [RAW_DIR, PROCESSED_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "data_update.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("data_updater")

# Create logs directory if it doesn't exist
(BASE_DIR / "logs").mkdir(exist_ok=True)

# API Keys (from environment variables)
FRED_API_KEY = os.getenv("FRED_API_KEY")
BLS_API_KEY = os.getenv("BLS_API_KEY")
ATTOM_API_KEY = os.getenv("ATTOM_API_KEY")

# Fairfield County towns
FAIRFIELD_TOWNS = [
    "Bridgeport", "Danbury", "Darien", "Easton", "Fairfield", 
    "Greenwich", "Monroe", "New Canaan", "New Fairfield", "Newtown",
    "Norwalk", "Redding", "Ridgefield", "Shelton", "Sherman",
    "Stamford", "Stratford", "Trumbull", "Weston", "Westport", "Wilton"
]

# FIPS code for Fairfield County, CT
FAIRFIELD_FIPS = "09001"

# API Endpoints
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
BLS_BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
ATTOM_BASE_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/salestrend/snapshot"


def update_fred_data():
    """
    Update economic indicator data from FRED API.
    """
    if not FRED_API_KEY:
        logger.error("FRED API key not found in environment variables")
        return False
    
    logger.info("Updating data from FRED API...")
    
    # FRED series to fetch
    series_ids = {
        "MORTGAGE30US": "30-Year Fixed Rate Mortgage Average",
        "MEDLISPRI_CT": "Median Listing Price of Homes in Connecticut",
        "CTYHOUVACRT_CT": "Housing Vacancies in Connecticut",
        "CTGSP": "Connecticut Gross State Product",
        "CTHUR": "Connecticut Housing Price Index (All-Transactions)"
    }
    
    # Set date range (from 2015 to present)
    observation_start = "2015-01-01"
    observation_end = datetime.now().strftime("%Y-%m-%d")
    
    success_count = 0
    for series_id, series_name in series_ids.items():
        try:
            logger.info(f"Fetching {series_name} (Series ID: {series_id})")
            
            # Construct API request URL
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "observation_start": observation_start,
                "observation_end": observation_end
            }
            
            response = requests.get(FRED_BASE_URL, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Save raw response to file
                output_file = RAW_DIR / f"fred_{series_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"Successfully updated {series_id} data with {len(data.get('observations', []))} observations")
                success_count += 1
            else:
                logger.error(f"Failed to fetch {series_id} data - Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
            
            # Sleep to avoid hitting API rate limits
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Error fetching {series_id} data: {str(e)}")
    
    logger.info(f"FRED data update complete. Updated {success_count}/{len(series_ids)} series.")
    return success_count == len(series_ids)


def update_bls_data():
    """
    Update employment and economic data from BLS API.
    """
    if not BLS_API_KEY:
        logger.error("BLS API key not found in environment variables")
        return False
    
    logger.info("Updating data from BLS API...")
    
    # Define BLS series to fetch
    # Format: Area code + Measure code (see BLS documentation for codes)
    series_list = [
        # Local Area Unemployment Statistics for Fairfield County
        f"LAUS{FAIRFIELD_FIPS}03",  # Unemployment rate
        f"LAUS{FAIRFIELD_FIPS}04",  # Unemployment
        f"LAUS{FAIRFIELD_FIPS}05",  # Employment
        f"LAUS{FAIRFIELD_FIPS}06",  # Labor force
        
        # Quarterly Census of Employment and Wages - Construction
        f"ENU{FAIRFIELD_FIPS}05236",  # Construction employment in Fairfield County
        
        # Quarterly Census of Employment and Wages - Real Estate
        f"ENU{FAIRFIELD_FIPS}05238",  # Real Estate employment in Fairfield County
    ]
    
    # Current year and last year
    current_year = datetime.now().year
    start_year = 2015
    
    try:
        # Prepare request JSON
        headers = {'Content-type': 'application/json'}
        data = {
            "seriesid": series_list,
            "startyear": str(start_year),
            "endyear": str(current_year),
            "registrationkey": BLS_API_KEY
        }
        
        response = requests.post(BLS_BASE_URL, json=data, headers=headers)
        
        if response.status_code == 200:
            bls_data = response.json()
            
            # Check for response errors
            if bls_data.get("status") != "REQUEST_SUCCEEDED":
                logger.error(f"BLS API request failed: {bls_data.get('message', 'Unknown error')}")
                return False
            
            # Process each series
            for series in bls_data.get("Results", {}).get("series", []):
                series_id = series.get("seriesID")
                if series_id:
                    # Save individual series data
                    output_file = RAW_DIR / f"bls_{series_id}.json"
                    
                    # Format as a standalone response for easier processing later
                    series_response = {
                        "status": bls_data.get("status"),
                        "responseTime": bls_data.get("responseTime"),
                        "Results": {
                            "series": [series]
                        }
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(series_response, f, indent=2)
                    
                    data_count = len(series.get("data", []))
                    logger.info(f"Successfully updated {series_id} data with {data_count} observations")
            
            # Save complete response for reference
            complete_output_file = RAW_DIR / "bls_complete_response.json"
            with open(complete_output_file, 'w') as f:
                json.dump(bls_data, f, indent=2)
            
            logger.info(f"BLS data update complete. Updated {len(bls_data.get('Results', {}).get('series', []))}/{len(series_list)} series.")
            return True
        else:
            logger.error(f"Failed to fetch BLS data - Status code: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating BLS data: {str(e)}")
        return False


def update_attom_data():
    """
    Update property data from ATTOM Property API.
    """
    if not ATTOM_API_KEY:
        logger.error("ATTOM API key not found in environment variables")
        return False
    
    logger.info("Updating data from ATTOM Property API...")
    
    # Property types
    property_types = ["SingleFamily", "Condo", "MultiFamily"]  # API specific categories
    
    # Metrics to request
    metrics = [
        "SalesTrend", "MedianSalePrice", "AverageSalePrice", "MedianDaysOnMarket",
        "SalesToListRatio", "PricePerSquareFoot", "InventoryCount"
    ]
    
    # Headers for ATTOM API
    headers = {
        "apikey": ATTOM_API_KEY,
        "accept": "application/json"
    }
    
    success_count = 0
    total_requests = len(FAIRFIELD_TOWNS) * len(property_types)
    
    for town in FAIRFIELD_TOWNS:
        town_data = {}
        
        for property_type in property_types:
            try:
                logger.info(f"Fetching {property_type} data for {town}")
                
                # Construct API request parameters
                params = {
                    "geoid": "",  # Use postal code instead
                    "postalcode": "",  # Will be populated with ZIP code
                    "propertytype": property_type,
                    "startmonth": "01",
                    "startyear": "2015",
                    "endmonth": datetime.now().strftime("%m"),
                    "endyear": datetime.now().strftime("%Y")
                }
                
                # Get ZIP code for the town (in a real implementation, you'd use a geocoding service or lookup table)
                # This is a simplified approach - in reality, towns may have multiple ZIP codes
                # For this example, we'll use an environment variable with town-to-ZIP mapping
                zip_code_map = json.loads(os.getenv("TOWN_ZIP_MAP", "{}"))
                zip_code = zip_code_map.get(town)
                
                if not zip_code:
                    logger.warning(f"No ZIP code found for {town}, skipping")
                    continue
                
                params["postalcode"] = zip_code
                
                # Make API request
                response = requests.get(ATTOM_BASE_URL, headers=headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Save raw response
                    property_key = "SingleFamily" if property_type == "SingleFamily" else "MultiUnit" if property_type == "MultiFamily" else "CondoTownhouse"
                    town_data[property_key] = data
                    
                    logger.info(f"Successfully updated {property_type} data for {town}")
                    success_count += 1
                else:
                    logger.error(f"Failed to fetch {property_type} data for {town} - Status code: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                
                # Sleep to avoid hitting API rate limits
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching {property_type} data for {town}: {str(e)}")
        
        # Save town data
        if town_data:
            output_file = RAW_DIR / f"attom_{town.lower().replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(town_data, f, indent=2)
    
    logger.info(f"ATTOM data update complete. Updated {success_count}/{total_requests} requests.")
    return success_count > 0  # Consider successful if at least one request succeeded


def update_derived_metrics():
    """
    Generate derived metrics from raw data.
    This function would need to be implemented based on specific calculation requirements.
    """
    logger.info("Generating derived metrics from raw data...")
    
    try:
        # Example placeholder for calculating housing affordability index
        # In a real implementation, this would use the raw data to calculate metrics
        
        # Calculate affordability index (median home price / median income)
        # This is just a placeholder - actual implementation would load and process the raw data
        affordability_data = {}
        
        # Save derived metrics
        output_file = PROCESSED_DIR / "affordability_index.json"
        with open(output_file, 'w') as f:
            json.dump(affordability_data, f, indent=2)
        
        logger.info("Successfully updated derived metrics")
        return True
        
    except Exception as e:
        logger.error(f"Error generating derived metrics: {str(e)}")
        return False


def update_all_data():
    """
    Update all data sources.
    """
    logger.info("Starting complete data update...")
    
    fred_success = update_fred_data()
    bls_success = update_bls_data()
    attom_success = update_attom_data()
    derived_success = update_derived_metrics()
    
    # Count successful updates
    success_count = sum([fred_success, bls_success, attom_success, derived_success])
    total_sources = 4
    
    # Log overall update result
    if success_count == total_sources:
        logger.info("All data sources updated successfully")
    else:
        logger.warning(f"{success_count}/{total_sources} data sources updated successfully")
    
    # Create update status file
    status = {
        "last_update": datetime.now().isoformat(),
        "successful_updates": success_count,
        "total_sources": total_sources,
        "source_status": {
            "fred": fred_success,
            "bls": bls_success,
            "attom": attom_success,
            "derived_metrics": derived_success
        }
    }
    
    with open(BASE_DIR / "api_data" / "update_status.json", 'w') as f:
        json.dump(status, f, indent=2)
    
    return success_count == total_sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update data from API sources for the housing market analysis application")
    parser.add_argument("--source", choices=["fred", "bls", "attom", "derived", "all"], 
                        default="all", help="Specify which data source to update (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.source == "all":
        success = update_all_data()
    elif args.source == "fred":
        success = update_fred_data()
    elif args.source == "bls":
        success = update_bls_data()
    elif args.source == "attom":
        success = update_attom_data()
    elif args.source == "derived":
        success = update_derived_metrics()
    
    sys.exit(0 if success else 1) 