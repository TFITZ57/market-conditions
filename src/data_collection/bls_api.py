"""
BLS API Client

This module provides a client for fetching data from the Bureau of Labor Statistics API.
Focus on employment data relevant to housing market analysis for Fairfield County, CT.
"""

import os
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from ..utils.logger import get_logger
from ..utils.caching import cache_response

logger = get_logger(__name__)

class BlsApiClient:
    """Client for the Bureau of Labor Statistics (BLS) API."""
    
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the BLS API client.
        
        Args:
            api_key: BLS API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.environ.get("BLS_API_KEY")
        if not self.api_key:
            logger.warning("BLS API key not provided. Set BLS_API_KEY environment variable.")
        
        # BLS Series IDs relevant to housing market analysis
        # These series IDs focus on employment data for Fairfield County, CT
        # Format: 
        # LAUS = Local Area Unemployment Statistics
        # SM = State and Metro Area Employment, Hours, & Earnings
        self.series_ids = {
            # Fairfield County Unemployment Rate
            "unemployment_rate": "LAUCN09001000000003",
            # Bridgeport-Stamford-Norwalk, CT MSA Total Nonfarm Employment
            "total_employment": "SMU09716800000000001",
            # Bridgeport-Stamford-Norwalk, CT MSA Construction Employment
            "construction_employment": "SMU09716802000000001",
            # Bridgeport-Stamford-Norwalk, CT MSA Financial Activities Employment
            "financial_employment": "SMU09716805500000001",
            # Bridgeport-Stamford-Norwalk, CT MSA Real Estate Employment
            "real_estate_employment": "SMU09716805300000001",
        }
        
        # Map from our internal metric names to BLS series IDs
        self.metric_to_series = {
            "Local Job Growth": "SMU09716800000000001",  # Total employment
            "Employment Trends": "LAUCN09001000000003",  # Unemployment rate
        }
    
    def _make_request(self, series_ids: List[str], start_year: str, end_year: str) -> Dict:
        """
        Make a request to the BLS API.
        
        Args:
            series_ids: List of BLS series IDs
            start_year: Start year (YYYY)
            end_year: End year (YYYY)
            
        Returns:
            API response as a dictionary
        """
        headers = {"Content-Type": "application/json"}
        
        # BLS API allows up to 50 series per request
        if len(series_ids) > 50:
            logger.warning("BLS API allows max 50 series per request. Truncating list.")
            series_ids = series_ids[:50]
        
        payload = {
            "seriesid": series_ids,
            "startyear": start_year,
            "endyear": end_year,
            "registrationkey": self.api_key
        }
        
        try:
            response = requests.post(self.BASE_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to BLS API: {e}")
            raise
    
    @cache_response
    def get_series_data(self, series_id: str, 
                        start_year: Optional[str] = None, 
                        end_year: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for a specific BLS series.
        
        Args:
            series_id: BLS series ID
            start_year: Start year in YYYY format
            end_year: End year in YYYY format
            
        Returns:
            DataFrame with date index and series values
        """
        if not start_year:
            start_year = "2015"  # Default to 2015 as per requirements
        
        if not end_year:
            end_year = datetime.now().strftime("%Y")
        
        logger.info(f"Fetching BLS series {series_id} from {start_year} to {end_year}")
        
        try:
            data = self._make_request([series_id], start_year, end_year)
            
            if (not data.get("Results") or 
                not data["Results"].get("series") or 
                len(data["Results"]["series"]) == 0):
                logger.warning(f"No data found for series {series_id}")
                return pd.DataFrame()
            
            series_data = data["Results"]["series"][0]
            
            if not series_data.get("data"):
                logger.warning(f"No data points found for series {series_id}")
                return pd.DataFrame()
            
            # Create dataframe from the data
            records = []
            for item in series_data["data"]:
                year = item["year"]
                period = item["period"]
                
                # Convert BLS period format (e.g., M01, M02, ...) to month number
                if period.startswith("M"):
                    month = period[1:].zfill(2)
                    # Create date string in YYYY-MM-01 format
                    date_str = f"{year}-{month}-01"
                    
                    records.append({
                        "date": date_str,
                        "value": item["value"],
                        "footnotes": item.get("footnotes", [])
                    })
                elif period.startswith("Q"):
                    # Handle quarterly data
                    quarter = int(period[1:])
                    # Approximate the quarter to the middle month
                    month = str(quarter * 3 - 1).zfill(2)
                    date_str = f"{year}-{month}-01"
                    
                    records.append({
                        "date": date_str,
                        "value": item["value"],
                        "footnotes": item.get("footnotes", []),
                        "quarter": f"Q{quarter}"
                    })
                elif period == "A01":
                    # Handle annual data, set to mid-year
                    date_str = f"{year}-07-01"
                    
                    records.append({
                        "date": date_str,
                        "value": item["value"],
                        "footnotes": item.get("footnotes", []),
                        "annual": True
                    })
            
            if not records:
                logger.warning(f"No usable data records for series {series_id}")
                return pd.DataFrame()
            
            df = pd.DataFrame(records)
            # Convert date string to datetime and set as index
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            # Convert value column to numeric
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            
            # Sort by date
            df = df.sort_index()
            
            # Add series metadata
            df.attrs["series_id"] = series_id
            df.attrs["title"] = series_data.get("seriesName", "")
            
            return df
        except Exception as e:
            logger.error(f"Error fetching series data for {series_id}: {e}")
            raise
    
    def get_metric_data(self, metric_name: str, 
                        start_year: Optional[str] = None, 
                        end_year: Optional[str] = None) -> pd.DataFrame:
        """
        Get data for a specific employment metric.
        
        Args:
            metric_name: The name of the metric (from our standard set)
            start_year: Start year in YYYY format
            end_year: End year in YYYY format
            
        Returns:
            DataFrame with date index and metric values
        """
        if metric_name not in self.metric_to_series:
            logger.error(f"Metric {metric_name} not available from BLS API")
            raise ValueError(f"Metric {metric_name} not available from BLS API")
        
        series_id = self.metric_to_series[metric_name]
        df = self.get_series_data(series_id, start_year, end_year)
        
        # Add metadata about the metric
        df.attrs["metric_name"] = metric_name
        
        return df
    
    def get_employment_metrics(self, 
                               start_year: Optional[str] = None, 
                               end_year: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all available employment metrics.
        
        Args:
            start_year: Start year in YYYY format
            end_year: End year in YYYY format
            
        Returns:
            Dictionary mapping metric names to DataFrames with their data
        """
        results = {}
        
        for metric_name in self.metric_to_series:
            try:
                df = self.get_metric_data(metric_name, start_year, end_year)
                results[metric_name] = df
            except Exception as e:
                logger.error(f"Error fetching metric {metric_name}: {e}")
                # Continue with other metrics even if one fails
        
        return results
    
    def get_employment_growth(self, 
                             region: str = "Bridgeport-Stamford-Norwalk", 
                             start_year: Optional[str] = None, 
                             end_year: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate job growth rates for a specific region.
        
        Args:
            region: Region name (default is Bridgeport-Stamford-Norwalk MSA)
            start_year: Start year in YYYY format
            end_year: End year in YYYY format
            
        Returns:
            DataFrame with date index and job growth rates
        """
        # Use total employment series for the region
        series_id = self.series_ids["total_employment"]
        
        # Get the employment data
        df = self.get_series_data(series_id, start_year, end_year)
        
        if df.empty:
            logger.warning(f"No employment data found for {region}")
            return pd.DataFrame()
        
        # Calculate year-over-year percentage change
        df["pct_change_yoy"] = df["value"].pct_change(periods=12) * 100
        
        # Calculate quarter-over-quarter percentage change
        df["pct_change_qoq"] = df["value"].pct_change(periods=3) * 100
        
        # Add metadata
        df.attrs["metric_name"] = "Job Growth"
        df.attrs["region"] = region
        
        return df