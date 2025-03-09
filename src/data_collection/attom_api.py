"""
ATTOM API Client

This module provides a client for fetching data from the ATTOM Property API.
Focus on housing market data for Fairfield County, CT at town/zip code level.
"""

import os
import requests
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from ..utils.logger import get_logger
from ..utils.caching import cache_response

logger = get_logger(__name__)

class AttomApiClient:
    """Client for the ATTOM Property Data API."""
    
    BASE_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ATTOM API client.
        
        Args:
            api_key: ATTOM API key. If None, will try to get from environment variable.
        """
        self.api_key = api_key or os.environ.get("ATTOM_API_KEY")
        if not self.api_key:
            logger.warning("ATTOM API key not provided. Set ATTOM_API_KEY environment variable.")
        
        # Fairfield County, CT FIPS code: 09001
        self.fairfield_fips = "09001"
        
        # Dictionary of town names to FIPS codes in Fairfield County
        self.towns_fips = {
            "Bridgeport": "0908000",
            "Danbury": "0918430",
            "Darien": "0918850",
            "Fairfield": "0926620",
            "Greenwich": "0933620",
            "New Canaan": "0949460",
            "Norwalk": "0955990",
            "Stamford": "0973000",
            "Stratford": "0973560",
            "Westport": "0982800",
            "Weston": "0982940",
            "Wilton": "0985810",
            "Bethel": "0904720",
            "Brookfield": "0908490",
            "Easton": "0923890",
            "Monroe": "0947360",
            "Newtown": "0952980",
            "Redding": "0963480",
            "Ridgefield": "0963970",
            "Shelton": "0969640",
            "Trumbull": "0976570"
        }
        
        # Dictionary of town names to ZIP codes in Fairfield County
        self.towns_zips = {
            "Bridgeport": ["06604", "06605", "06606", "06607", "06608", "06610"],
            "Danbury": ["06810", "06811"],
            "Darien": ["06820"],
            "Fairfield": ["06824", "06825"],
            "Greenwich": ["06830", "06831", "06836"],
            "New Canaan": ["06840"],
            "Norwalk": ["06850", "06851", "06854", "06855", "06856"],
            "Stamford": ["06901", "06902", "06903", "06905", "06906", "06907"],
            "Stratford": ["06614", "06615"],
            "Westport": ["06880", "06881"],
            "Weston": ["06883"],
            "Wilton": ["06897"],
            "Bethel": ["06801"],
            "Brookfield": ["06804"],
            "Easton": ["06612"],
            "Monroe": ["06468"],
            "Newtown": ["06470"],
            "Redding": ["06896"],
            "Ridgefield": ["06877", "06879"],
            "Shelton": ["06484"],
            "Trumbull": ["06611"]
        }
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make a request to the ATTOM API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            "accept": "application/json",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to ATTOM API: {e}")
            # If rate limit error, pause and retry
            if hasattr(e.response, 'status_code') and e.response.status_code == 429:
                logger.warning("Rate limit hit, pausing before retry")
                time.sleep(2)  # Pause for 2 seconds
                return self._make_request(endpoint, params)
            raise
    
    @cache_response
    def get_market_stats(self, 
                       area_type: str = "county", 
                       area_id: str = None, 
                       start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get market statistics for a specific area.
        
        Args:
            area_type: Type of area ('county', 'citystate', 'zipcode')
            area_id: ID of the area (FIPS code for county, ZIP code, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with market statistics
        """
        if not area_id:
            # Default to Fairfield County
            area_type = "county"
            area_id = self.fairfield_fips
        
        if not start_date:
            start_date = "2015-01-01"  # Default to 2015 as per requirements
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Convert dates to ATTOM format (MM/DD/YYYY)
        start_date_attom = datetime.strptime(start_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        end_date_attom = datetime.strptime(end_date, "%Y-%m-%d").strftime("%m/%d/%Y")
        
        endpoint = "salestrend/snapshot"
        
        params = {
            f"{area_type}id": area_id,
            "startmonth": "01",
            "startyear": start_date.split("-")[0],
            "endmonth": "12",
            "endyear": end_date.split("-")[0]
        }
        
        logger.info(f"Fetching ATTOM market stats for {area_type} {area_id} from {start_date} to {end_date}")
        
        try:
            data = self._make_request(endpoint, params)
            
            if (not data.get("status") or 
                data["status"].get("code") != 0 or 
                not data.get("salestrends")):
                logger.warning(f"No market stats found for {area_type} {area_id}")
                return pd.DataFrame()
            
            # Extract sales trends data
            sales_trends = data["salestrends"]
            
            # Convert to DataFrame
            df = pd.DataFrame(sales_trends)
            
            # Convert date strings to datetime
            if "month" in df.columns and "year" in df.columns:
                df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01")
                df = df.set_index("date")
            
            # Add metadata
            df.attrs["area_type"] = area_type
            df.attrs["area_id"] = area_id
            
            return df
        except Exception as e:
            logger.error(f"Error fetching market stats for {area_type} {area_id}: {e}")
            raise
    
    @cache_response
    def get_property_details(self, 
                           address: str = None, 
                           zip_code: str = None) -> Dict:
        """
        Get property details for a specific address.
        
        Args:
            address: Property address
            zip_code: ZIP code
            
        Returns:
            Dictionary of property details
        """
        if not address:
            logger.error("Property address is required")
            raise ValueError("Property address is required")
        
        endpoint = "property/detail"
        
        params = {
            "address": address
        }
        
        if zip_code:
            params["postalcode"] = zip_code
        
        logger.info(f"Fetching property details for {address}")
        
        try:
            data = self._make_request(endpoint, params)
            
            if (not data.get("status") or 
                data["status"].get("code") != 0 or 
                not data.get("property")):
                logger.warning(f"No property details found for {address}")
                return {}
            
            return data["property"][0]
        except Exception as e:
            logger.error(f"Error fetching property details for {address}: {e}")
            raise
    
    def get_inventory_by_town(self, 
                             town: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get housing inventory metrics for a specific town.
        
        Args:
            town: Town name in Fairfield County
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with inventory metrics
        """
        if town not in self.towns_zips:
            logger.error(f"Town '{town}' not recognized in Fairfield County")
            raise ValueError(f"Town '{town}' not recognized in Fairfield County")
        
        town_zips = self.towns_zips[town]
        
        # Collect data for each ZIP code in the town
        dfs = []
        for zip_code in town_zips:
            try:
                df = self.get_market_stats(area_type="zipcode", area_id=zip_code, 
                                          start_date=start_date, end_date=end_date)
                df["zip_code"] = zip_code
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error getting data for {town} ZIP {zip_code}: {e}")
                # Continue with other ZIP codes
        
        if not dfs:
            logger.warning(f"No inventory data found for any ZIP codes in {town}")
            return pd.DataFrame()
        
        # Combine data from all ZIP codes
        town_df = pd.concat(dfs)
        
        # Add town name
        town_df["town"] = town
        
        # Add metadata
        town_df.attrs["town"] = town
        town_df.attrs["zip_codes"] = town_zips
        
        return town_df
    
    def get_all_towns_metrics(self, 
                             metric: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get a specific metric for all towns in Fairfield County.
        
        Args:
            metric: The metric to collect ('inventory', 'sales', 'price', etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with the metric for all towns
        """
        all_towns_data = []
        
        for town in self.towns_zips.keys():
            try:
                if metric == "inventory":
                    df = self.get_inventory_by_town(town, start_date, end_date)
                elif metric == "sales":
                    # Get sales data using sales endpoint
                    df = self.get_sales_by_town(town, start_date, end_date)
                else:
                    # Default to getting market stats
                    town_zips = self.towns_zips[town]
                    dfs = []
                    for zip_code in town_zips:
                        zip_df = self.get_market_stats(area_type="zipcode", area_id=zip_code, 
                                                     start_date=start_date, end_date=end_date)
                        zip_df["zip_code"] = zip_code
                        dfs.append(zip_df)
                    df = pd.concat(dfs) if dfs else pd.DataFrame()
                
                if not df.empty:
                    df["town"] = town
                    all_towns_data.append(df)
            except Exception as e:
                logger.error(f"Error getting {metric} for {town}: {e}")
                # Continue with other towns
        
        if not all_towns_data:
            logger.warning(f"No {metric} data found for any towns")
            return pd.DataFrame()
        
        # Combine data from all towns
        all_towns_df = pd.concat(all_towns_data)
        
        # Add metadata
        all_towns_df.attrs["metric"] = metric
        
        return all_towns_df
    
    def get_sales_by_town(self, 
                         town: str, 
                         start_date: Optional[str] = None, 
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get real estate sales data for a specific town.
        
        Args:
            town: Town name in Fairfield County
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with sales data
        """
        if town not in self.towns_zips:
            logger.error(f"Town '{town}' not recognized in Fairfield County")
            raise ValueError(f"Town '{town}' not recognized in Fairfield County")
        
        # Use town FIPS code if available, otherwise use ZIP codes
        if town in self.towns_fips:
            area_type = "citystate"  # ATTOM uses citystate for city-level queries
            area_id = self.towns_fips[town]
            
            df = self.get_market_stats(area_type=area_type, area_id=area_id, 
                                      start_date=start_date, end_date=end_date)
            
            if not df.empty:
                df["town"] = town
                return df
        
        # Fallback to ZIP code approach if city lookup failed
        town_zips = self.towns_zips[town]
        
        # Collect data for each ZIP code in the town
        dfs = []
        for zip_code in town_zips:
            try:
                df = self.get_market_stats(area_type="zipcode", area_id=zip_code, 
                                          start_date=start_date, end_date=end_date)
                if not df.empty:
                    df["zip_code"] = zip_code
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error getting sales data for {town} ZIP {zip_code}: {e}")
                # Continue with other ZIP codes
        
        if not dfs:
            logger.warning(f"No sales data found for any ZIP codes in {town}")
            return pd.DataFrame()
        
        # Combine data from all ZIP codes
        town_df = pd.concat(dfs)
        
        # Add town name
        town_df["town"] = town
        
        # Add metadata
        town_df.attrs["town"] = town
        town_df.attrs["zip_codes"] = town_zips
        
        return town_df
    
    def get_market_metrics(self, 
                          metric_type: str, 
                          area_type: str = "county", 
                          area_id: str = None, 
                          start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get specific market metrics for an area.
        
        Args:
            metric_type: Type of metric ('price', 'inventory', 'dom', 'sales_volume', etc.)
            area_type: Type of area ('county', 'citystate', 'zipcode')
            area_id: ID of the area (FIPS code for county, ZIP code, etc.)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with the requested metric
        """
        # Get full market stats
        df = self.get_market_stats(area_type, area_id, start_date, end_date)
        
        if df.empty:
            return df
        
        # Map metric_type to relevant columns
        metric_columns = {
            "price": ["median_sale_price", "average_sale_price"],
            "inventory": ["listing_inventory", "new_listings"],
            "dom": ["days_on_market", "median_days_on_market"],
            "sales_volume": ["sales_count", "sales_volume"],
            "supply": ["months_of_supply", "new_listings_vs_inventory_ratio"],
            "list_to_sale": ["list_price_to_sale_price_ratio", "median_list_price"],
            "absorption": ["absorption_rate"]
        }
        
        if metric_type in metric_columns:
            # Filter to relevant columns
            columns = metric_columns[metric_type].copy()
            # Always include date columns and geographic identifiers
            extra_cols = [col for col in df.columns if col in ['month', 'year', 'quarter', 'town', 'zip_code']]
            columns.extend(extra_cols)
            
            # Filter to just the columns we need
            filtered_df = df[columns] if all(col in df.columns for col in columns) else df
            
            # Add metadata
            filtered_df.attrs["metric_type"] = metric_type
            filtered_df.attrs["area_type"] = area_type
            filtered_df.attrs["area_id"] = area_id
            
            return filtered_df
        else:
            # Return full dataset if metric_type not recognized
            logger.warning(f"Metric type '{metric_type}' not recognized, returning all metrics")
            return df