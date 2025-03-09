"""
FRED API Client

This module provides a client for fetching data from the Federal Reserve Economic Data API.
Focus on housing market indicators for Fairfield County, CT.
"""

import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from ..utils.logger import get_logger
from ..utils.caching import cache_response

logger = get_logger(__name__)

class FredApiClient:
    """Client for the Federal Reserve Economic Data (FRED) API."""
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FRED API client.
        
        Args:
            api_key: FRED API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self.api_key:
            logger.warning("FRED API key not provided. Set FRED_API_KEY environment variable.")
        
        # FRED Series IDs relevant to housing market analysis
        self.series_ids = {
            "mortgage_rates": "MORTGAGE30US",  # 30-Year Fixed Rate Mortgage Average
            "housing_starts": "HOUST",         # Housing Starts
            "vacancy_rate": "RRVRUSQ156N",     # Rental Vacancy Rate
            "housing_inventory": "MSACSR",     # Monthly Supply of Houses
            "housing_price_index": "CSUSHPINSA", # Case-Shiller Home Price Index
            "affordability_index": "FIXHAI",   # Housing Affordability Index
            "new_home_sales": "HSN1F",         # New Home Sales
            "pending_home_sales": "PENDINGHOMESALESINDEX", # Pending Home Sales Index
            "median_home_price": "MSPUS",      # Median Sales Price of Houses Sold
        }
        
        # Map from our internal metric names to FRED series IDs
        self.metric_to_series = {
            "Months of Supply": "MSACSR", 
            "Housing Starts": "HOUST",
            "Mortgage Rates": "MORTGAGE30US",
            "Housing Affordability Index": "FIXHAI",
            "Vacancy Rates": "RRVRUSQ156N",
            "Median Sale Price": "MSPUS",
            "Pending Home Sales": "PENDINGHOMESALESINDEX"
        }
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """
        Make a request to the FRED API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params["api_key"] = self.api_key
        params["file_type"] = "json"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to FRED API: {e}")
            raise
    
    @cache_response
    def get_series_data(self, series_id: str, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        frequency: str = "q") -> pd.DataFrame:
        """
        Get time series data for a specific FRED series.
        
        Args:
            series_id: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('d'=daily, 'w'=weekly, 'm'=monthly, 'q'=quarterly, 'a'=annual)
            
        Returns:
            DataFrame with date index and series values
        """
        if not start_date:
            start_date = "2015-01-01"  # Default to 2015 as per requirements
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        params = {
            "series_id": series_id,
            "observation_start": start_date,
            "observation_end": end_date,
            "frequency": frequency,
            "units": "lin"  # Linear units (no transformation)
        }
        
        logger.info(f"Fetching FRED series {series_id} from {start_date} to {end_date}")
        
        try:
            data = self._make_request("series/observations", params)
            
            if "observations" not in data:
                logger.warning(f"No observations found for series {series_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data["observations"])
            # Convert date string to datetime and set as index
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            # Convert value column to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Get series information for metadata
            series_info = self.get_series_info(series_id)
            df.attrs["title"] = series_info.get("title", "")
            df.attrs["units"] = series_info.get("units", "")
            df.attrs["frequency"] = series_info.get("frequency", "")
            df.attrs["series_id"] = series_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching series data for {series_id}: {e}")
            raise
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get metadata about a specific FRED series.
        
        Args:
            series_id: FRED series ID
            
        Returns:
            Series metadata as a dictionary
        """
        params = {"series_id": series_id}
        
        try:
            data = self._make_request("series", params)
            return data.get("seriess", [{}])[0]
        except Exception as e:
            logger.error(f"Error fetching series info for {series_id}: {e}")
            return {}
    
    def get_metric_data(self, metric_name: str, 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        frequency: str = "q") -> pd.DataFrame:
        """
        Get data for a specific housing market metric.
        
        Args:
            metric_name: The name of the metric (from our standard set)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('d'=daily, 'w'=weekly, 'm'=monthly, 'q'=quarterly, 'a'=annual)
            
        Returns:
            DataFrame with date index and metric values
        """
        if metric_name not in self.metric_to_series:
            logger.error(f"Metric {metric_name} not available from FRED API")
            raise ValueError(f"Metric {metric_name} not available from FRED API")
        
        series_id = self.metric_to_series[metric_name]
        df = self.get_series_data(series_id, start_date, end_date, frequency)
        
        # Add metadata about the metric
        df.attrs["metric_name"] = metric_name
        
        return df
    
    def get_all_housing_metrics(self, 
                              start_date: Optional[str] = None, 
                              end_date: Optional[str] = None,
                              frequency: str = "q") -> Dict[str, pd.DataFrame]:
        """
        Get data for all available housing market metrics.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('d'=daily, 'w'=weekly, 'm'=monthly, 'q'=quarterly, 'a'=annual)
            
        Returns:
            Dictionary mapping metric names to DataFrames with their data
        """
        results = {}
        
        for metric_name in self.metric_to_series:
            try:
                df = self.get_metric_data(metric_name, start_date, end_date, frequency)
                results[metric_name] = df
            except Exception as e:
                logger.error(f"Error fetching metric {metric_name}: {e}")
                # Continue with other metrics even if one fails
        
        return results