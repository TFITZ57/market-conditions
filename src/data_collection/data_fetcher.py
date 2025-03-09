"""
Data Fetcher

This module provides a unified interface for fetching data from multiple sources.
It orchestrates data collection from FRED, BLS, and ATTOM APIs.
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
from .fred_api import FredApiClient
from .bls_api import BlsApiClient
from .attom_api import AttomApiClient
from ..utils.logger import get_logger
from ..utils.security import get_api_key

logger = get_logger(__name__)

class DataFetcher:
    """
    Orchestrates data collection from multiple sources for housing market analysis.
    """
    
    def __init__(self, raw_data_dir: Optional[str] = None, processed_data_dir: Optional[str] = None, synthetic_data_dir: Optional[str] = None):
        """
        Initialize the DataFetcher.
        
        Args:
            raw_data_dir: Directory to store raw API responses
            processed_data_dir: Directory to store processed data
            synthetic_data_dir: Directory to store synthetic data
        """
        # Setup data directories
        base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.raw_data_dir = Path(raw_data_dir) if raw_data_dir else base_dir / "api_data" / "raw"
        self.processed_data_dir = Path(processed_data_dir) if processed_data_dir else base_dir / "api_data" / "processed"
        self.synthetic_data_dir = Path(synthetic_data_dir) if synthetic_data_dir else base_dir / "api_data" / "synthetic"
        
        # Ensure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.synthetic_data_dir, exist_ok=True)
        
        # Initialize API clients
        self.fred_client = FredApiClient()
        self.bls_client = BlsApiClient()
        self.attom_client = AttomApiClient()
        
        # Flag to indicate if we're using synthetic data
        self.using_synthetic_data = not (get_api_key('fred') and get_api_key('bls') and get_api_key('attom'))
        
        # Map metrics to data sources
        self.metric_sources = {
            "Listing Inventory": "attom",
            "Months of Supply": "fred",
            "New Listings": "attom",
            "Housing Starts": "fred",
            "Sales Volume": "attom",
            "Days on Market": "attom",
            "Absorption Rate": "attom",
            "Pending Home Sales": "fred",
            "Median Sale Price": "attom", 
            "Average Sale Price": "attom",
            "LP/SP Ratio": "attom",
            "HPIncome Ratio": "calculated",  # Requires data from multiple sources
            "Mortgage Rates": "fred",
            "Housing Affordability Index": "fred",
            "Local Job Growth": "bls",
            "Employment Trends": "bls",
            "Vacancy Rates": "fred",
            "Seller Concessions": "attom",
            "Total Sales": "attom",
            "Active Listings": "attom"
        }
    
    def load_synthetic_data(self, data_type: str, identifier: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load synthetic data when API keys are not available.
        
        Args:
            data_type: Type of data to load ('fred', 'bls', 'attom', 'housing_starts', 'affordability')
            identifier: Specific identifier (e.g., series ID for FRED, town name for ATTOM)
            
        Returns:
            Dictionary of pandas DataFrames with synthetic data
        """
        logger.info(f"Loading synthetic data for {data_type}")
        results = {}
        
        try:
            if data_type == 'fred':
                # Load synthetic FRED data
                filename_pattern = "fred_*.json" if not identifier else f"fred_{identifier}.json"
                for file_path in self.synthetic_data_dir.glob(filename_pattern):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    series_id = data.get('series_id', file_path.stem.replace('fred_', ''))
                    
                    # Convert to DataFrame
                    observations = data.get('observations', [])
                    if observations:
                        df = pd.DataFrame(observations)
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        df = df.set_index('date')
                        
                        # Map series ID to metric name
                        metric_name = next((k for k, v in self.fred_client.metric_to_series.items() 
                                            if v == series_id), series_id)
                        results[metric_name] = df
            
            elif data_type == 'bls':
                # Load synthetic BLS data
                filename_pattern = "bls_*.json" if not identifier else f"bls_{identifier}.json"
                for file_path in self.synthetic_data_dir.glob(filename_pattern):
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract series data
                    series_list = data.get('Results', {}).get('series', [])
                    if series_list:
                        series = series_list[0]
                        series_id = series.get('seriesID', file_path.stem.replace('bls_', ''))
                        
                        # Convert to DataFrame
                        observations = series.get('data', [])
                        if observations:
                            rows = []
                            for obs in observations:
                                year = obs.get('year')
                                period = obs.get('period')
                                if year and period and period.startswith('M'):
                                    month = int(period[1:])
                                    date = f"{year}-{month:02d}-01"
                                    rows.append({
                                        'date': date,
                                        'value': float(obs.get('value', 0))
                                    })
                            
                            df = pd.DataFrame(rows)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            
                            # Map series ID to metric name
                            metric_name = f"BLS_{series_id}"
                            if 'Unemployment' in series_id:
                                metric_name = 'Unemployment Rate'
                            elif 'Employment' in series_id:
                                metric_name = 'Employment Trends'
                            
                            results[metric_name] = df
            
            elif data_type == 'attom':
                # Load synthetic ATTOM data
                town_pattern = identifier.lower().replace(' ', '_') if identifier else '*'
                filename_pattern = f"attom_{town_pattern}.json"
                
                for file_path in self.synthetic_data_dir.glob(filename_pattern):
                    with open(file_path, 'r') as f:
                        town_data = json.load(f)
                    
                    town_name = file_path.stem.replace('attom_', '').replace('_', ' ').title()
                    
                    # Process data for each property type and metric
                    for property_type, metrics in town_data.items():
                        for metric, data_points in metrics.items():
                            # Convert to DataFrame
                            df = pd.DataFrame(data_points)
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            
                            # Create metric name
                            metric_name = f"{metric}_{town_name}_{property_type}"
                            results[metric_name] = df
            
            elif data_type == 'housing_starts':
                # Load synthetic housing starts data
                file_path = self.synthetic_data_dir / "housing_starts.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        housing_starts_data = json.load(f)
                    
                    # If a specific town is requested
                    towns = [identifier] if identifier else housing_starts_data.keys()
                    
                    for town in towns:
                        if town in housing_starts_data:
                            town_data = housing_starts_data[town]
                            
                            for category, data_points in town_data.items():
                                # Convert to DataFrame
                                df = pd.DataFrame(data_points)
                                df['date'] = pd.to_datetime(df['date'])
                                df = df.set_index('date')
                                
                                # Create metric name
                                metric_name = f"HousingStarts_{town}_{category}"
                                results[metric_name] = df
            
            elif data_type == 'affordability':
                # Load synthetic affordability index data
                file_path = self.synthetic_data_dir / "affordability_index.json"
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        affordability_data = json.load(f)
                    
                    # If a specific town is requested
                    towns = [identifier] if identifier else affordability_data.keys()
                    
                    for town in towns:
                        if town in affordability_data:
                            # Convert to DataFrame
                            df = pd.DataFrame(affordability_data[town])
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.set_index('date')
                            
                            # Create metric name
                            metric_name = f"AffordabilityIndex_{town}"
                            results[metric_name] = df
            
        except Exception as e:
            logger.error(f"Error loading synthetic data for {data_type}: {str(e)}")
        
        return results
    
    def fetch_all_metrics(self,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         frequency: str = "q",
                         save_raw: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch all housing market metrics from all data sources.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('q'=quarterly, 'm'=monthly, 'a'=annual)
            save_raw: Whether to save raw API responses
            
        Returns:
            Dictionary mapping metric names to DataFrames
        """
        if not start_date:
            start_date = "2015-01-01"  # Default to 2015 as per requirements
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Fetching all metrics from {start_date} to {end_date}")
        
        results = {}
        
        # If using synthetic data, load it instead of making API calls
        if self.using_synthetic_data:
            logger.info("Using synthetic data (API keys not available)")
            
            # Load synthetic data for each data source
            fred_metrics = self.load_synthetic_data('fred')
            results.update(fred_metrics)
            
            bls_metrics = self.load_synthetic_data('bls')
            results.update(bls_metrics)
            
            # Load synthetic data for each town
            for town in self.attom_client.towns_zips.keys():
                town_metrics = self.load_synthetic_data('attom', town)
                results.update(town_metrics)
            
            # Load housing starts data
            housing_starts = self.load_synthetic_data('housing_starts')
            results.update(housing_starts)
            
            # Load affordability index data
            affordability = self.load_synthetic_data('affordability')
            results.update(affordability)
            
            # Calculate derived metrics
            try:
                derived_metrics = self._calculate_derived_metrics(results)
                results.update(derived_metrics)
            except Exception as e:
                logger.error(f"Error calculating derived metrics with synthetic data: {str(e)}")
            
            return results
        
        # Otherwise, proceed with normal API calls
        # Convert dates for different APIs
        start_year = start_date.split("-")[0]
        end_year = end_date.split("-")[0]
        
        # Fetch FRED data
        try:
            fred_metrics = self.fred_client.get_all_housing_metrics(
                start_date=start_date, 
                end_date=end_date, 
                frequency=frequency
            )
            results.update(fred_metrics)
            if save_raw:
                self._save_raw_data("fred", fred_metrics, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching FRED metrics: {e}")
            # Try to use synthetic data for this source
            fred_metrics = self.load_synthetic_data('fred')
            results.update(fred_metrics)
        
        # Fetch BLS data
        try:
            bls_metrics = self.bls_client.get_employment_metrics(
                start_year=start_year,
                end_year=end_year
            )
            results.update(bls_metrics)
            if save_raw:
                self._save_raw_data("bls", bls_metrics, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching BLS metrics: {e}")
            # Try to use synthetic data for this source
            bls_metrics = self.load_synthetic_data('bls')
            results.update(bls_metrics)
        
        # Fetch ATTOM county-level data
        try:
            county_metrics = self._fetch_attom_county_metrics(start_date, end_date)
            results.update(county_metrics)
            if save_raw:
                self._save_raw_data("attom_county", county_metrics, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching ATTOM county metrics: {e}")
            # Try to use synthetic data for this source
            for town in self.attom_client.towns_zips.keys():
                town_metrics = self.load_synthetic_data('attom', town)
                results.update(town_metrics)
        
        # Calculate derived metrics
        try:
            derived_metrics = self._calculate_derived_metrics(results)
            results.update(derived_metrics)
        except Exception as e:
            logger.error(f"Error calculating derived metrics: {e}")
            # Try to load synthetic derived metrics
            derived_metrics = self.load_synthetic_data('affordability')
            results.update(derived_metrics)
        
        return results
    
    def fetch_metric(self, 
                    metric_name: str, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    frequency: str = "q",
                    area_type: str = "county",
                    area_id: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch a specific housing market metric.
        
        Args:
            metric_name: Name of the metric to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            frequency: Data frequency ('q'=quarterly, 'm'=monthly, 'a'=annual)
            area_type: Type of area ('county', 'town')
            area_id: ID of the area (county FIPS, town name)
            
        Returns:
            DataFrame with the requested metric
        """
        if metric_name not in self.metric_sources:
            logger.error(f"Metric {metric_name} not recognized")
            raise ValueError(f"Metric {metric_name} not recognized")
        
        if not start_date:
            start_date = "2015-01-01"  # Default to 2015 as per requirements
        
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        source = self.metric_sources[metric_name]
        
        if source == "fred":
            return self.fred_client.get_metric_data(
                metric_name=metric_name,
                start_date=start_date,
                end_date=end_date,
                frequency=frequency
            )
        elif source == "bls":
            return self.bls_client.get_metric_data(
                metric_name=metric_name,
                start_year=start_date.split("-")[0],
                end_year=end_date.split("-")[0]
            )
        elif source == "attom":
            if area_type == "county":
                return self.attom_client.get_market_metrics(
                    metric_type=self._get_attom_metric_type(metric_name),
                    area_type="county",
                    area_id=area_id or self.attom_client.fairfield_fips,
                    start_date=start_date,
                    end_date=end_date
                )
            elif area_type == "town":
                if not area_id:
                    logger.error("Town name is required for town-level metrics")
                    raise ValueError("Town name is required for town-level metrics")
                
                return self.attom_client.get_market_metrics(
                    metric_type=self._get_attom_metric_type(metric_name),
                    area_type="citystate",
                    area_id=self.attom_client.towns_fips.get(area_id),
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                logger.error(f"Area type {area_type} not supported")
                raise ValueError(f"Area type {area_type} not supported")
        elif source == "calculated":
            # For metrics that require calculation from multiple sources
            if metric_name == "Home Price-to-Income Ratio":
                return self._calculate_price_to_income_ratio(
                    start_date=start_date,
                    end_date=end_date,
                    area_type=area_type,
                    area_id=area_id
                )
            else:
                logger.error(f"Calculation not implemented for {metric_name}")
                raise NotImplementedError(f"Calculation not implemented for {metric_name}")
        else:
            logger.error(f"Source {source} not supported")
            raise ValueError(f"Source {source} not supported")
    
    def fetch_town_metrics(self, 
                          town: str, 
                          metrics: List[str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch specified metrics for a specific town.
        
        Args:
            town: Town name
            metrics: List of metric names to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping metric names to DataFrames
        """
        if town not in self.attom_client.towns_zips:
            logger.error(f"Town {town} not recognized")
            raise ValueError(f"Town {town} not recognized")
        
        results = {}
        
        for metric in metrics:
            try:
                df = self.fetch_metric(
                    metric_name=metric,
                    start_date=start_date,
                    end_date=end_date,
                    area_type="town",
                    area_id=town
                )
                results[metric] = df
            except Exception as e:
                logger.error(f"Error fetching {metric} for {town}: {e}")
                # Continue with other metrics
        
        return results
    
    def fetch_all_towns_data(self, 
                           metric: str, 
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch a specific metric for all towns in Fairfield County.
        
        Args:
            metric: Metric name to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with data for all towns
        """
        if self.metric_sources.get(metric) != "attom":
            logger.error(f"Metric {metric} not available at town level")
            raise ValueError(f"Metric {metric} not available at town level")
        
        metric_type = self._get_attom_metric_type(metric)
        
        return self.attom_client.get_all_towns_metrics(
            metric=metric_type,
            start_date=start_date,
            end_date=end_date
        )
    
    def _fetch_attom_county_metrics(self, 
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch all ATTOM metrics for Fairfield County.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary mapping metric names to DataFrames
        """
        # Get full market stats for Fairfield County
        county_stats = self.attom_client.get_market_stats(
            area_type="county",
            area_id=self.attom_client.fairfield_fips,
            start_date=start_date,
            end_date=end_date
        )
        
        if county_stats.empty:
            logger.warning("No county-level data found from ATTOM API")
            return {}
        
        # Extract different metrics into separate DataFrames
        results = {}
        
        # Map from metric names to column names in ATTOM data
        attom_columns = {
            "Listing Inventory": "listing_inventory",
            "New Listings": "new_listings",
            "Sales Volume": "sales_volume",
            "Days on Market": "days_on_market",
            "Absorption Rate": "absorption_rate",
            "Median Sale Price": "median_sale_price",
            "Average Sale Price": "average_sale_price",
            "List Price to Sales Price Ratio": "list_price_to_sale_price_ratio",
            "Seller Concessions": "avg_seller_concession",
            "Total Sales": "sales_count",
            "Active Listings": "listing_inventory"
        }
        
        for metric, column in attom_columns.items():
            if column in county_stats.columns:
                # Create a DataFrame with just this metric
                metric_df = pd.DataFrame(county_stats[column])
                metric_df.columns = ["value"]  # Rename to match FRED/BLS format
                
                # Add metadata
                metric_df.attrs["metric_name"] = metric
                metric_df.attrs["source"] = "attom"
                
                results[metric] = metric_df
        
        return results
    
    def _calculate_derived_metrics(self, 
                                  metrics_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate derived metrics from fetched data.
        
        Args:
            metrics_data: Dictionary mapping metric names to DataFrames
            
        Returns:
            Dictionary mapping derived metric names to DataFrames
        """
        derived = {}
        
        # Calculate Home Price-to-Income Ratio if we have both price and income data
        if "Median Sale Price" in metrics_data and "Employment Trends" in metrics_data:
            try:
                price_df = metrics_data["Median Sale Price"].copy()
                # BLS employment data may be monthly while price data could be quarterly
                # Need to align the frequencies
                
                # For simplicity, we'll use a fixed income value
                # In a real implementation, we'd get median income data from Census API or similar
                median_income = 84000  # Example median household income for Fairfield County
                
                # Calculate price-to-income ratio
                price_df["price_to_income"] = price_df["value"] / median_income
                
                # Create new DataFrame for the derived metric
                ratio_df = pd.DataFrame(price_df["price_to_income"])
                ratio_df.columns = ["value"]
                
                # Add metadata
                ratio_df.attrs["metric_name"] = "Home Price-to-Income Ratio"
                ratio_df.attrs["source"] = "calculated"
                
                derived["Home Price-to-Income Ratio"] = ratio_df
            except Exception as e:
                logger.error(f"Error calculating Home Price-to-Income Ratio: {e}")
        
        return derived
    
    def _calculate_price_to_income_ratio(self,
                                       start_date: Optional[str] = None,
                                       end_date: Optional[str] = None,
                                       area_type: str = "county",
                                       area_id: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate Home Price-to-Income Ratio for a specific area.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            area_type: Type of area ('county', 'town')
            area_id: ID of the area (county FIPS, town name)
            
        Returns:
            DataFrame with Home Price-to-Income Ratio
        """
        # Fetch median sale price data
        price_df = self.fetch_metric(
            metric_name="Median Sale Price",
            start_date=start_date,
            end_date=end_date,
            area_type=area_type,
            area_id=area_id
        )
        
        if price_df.empty:
            logger.warning("No price data available for price-to-income calculation")
            return pd.DataFrame()
        
        # For real implementation, fetch actual income data from Census API
        # Here we'll use a fixed income value that varies slightly by year
        years = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='A')
        incomes = {}
        
        # Generate sample income data (in reality, would come from Census API)
        base_income = 84000  # Starting median household income
        for year in years:
            year_str = year.strftime("%Y")
            # Assume 2% annual income growth
            year_income = base_income * (1.02 ** (int(year_str) - 2015))
            incomes[year_str] = year_income
        
        # Calculate price-to-income ratio
        price_df["year"] = price_df.index.year.astype(str)
        price_df["median_income"] = price_df["year"].map(incomes)
        price_df["value"] = price_df["value"] / price_df["median_income"]
        
        # Clean up the DataFrame
        result_df = price_df[["value"]].copy()
        
        # Add metadata
        result_df.attrs["metric_name"] = "Home Price-to-Income Ratio"
        result_df.attrs["source"] = "calculated"
        
        return result_df
    
    def _get_attom_metric_type(self, metric_name: str) -> str:
        """
        Map metric names to ATTOM metric types.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            ATTOM metric type
        """
        attom_metric_types = {
            "Listing Inventory": "inventory",
            "New Listings": "inventory",
            "Sales Volume": "sales_volume",
            "Days on Market": "dom",
            "Absorption Rate": "absorption",
            "Median Sale Price": "price",
            "Average Sale Price": "price",
            "List Price to Sales Price Ratio": "list_to_sale",
            "Seller Concessions": "price",
            "Total Sales": "sales_volume",
            "Active Listings": "inventory"
        }
        
        return attom_metric_types.get(metric_name, "")
    
    def _save_raw_data(self, 
                     source: str, 
                     data: Dict[str, pd.DataFrame], 
                     start_date: str, 
                     end_date: str) -> None:
        """
        Save raw API response data to disk.
        
        Args:
            source: Data source name
            data: Dictionary of DataFrames
            start_date: Start date of data
            end_date: End date of data
        """
        # Create directory for this source if it doesn't exist
        source_dir = self.raw_data_dir / source
        os.makedirs(source_dir, exist_ok=True)
        
        # Create a date-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        period = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}"
        filename = f"{source}_{period}_{timestamp}.json"
        
        # Convert DataFrames to JSON-serializable format
        serializable_data = {}
        for metric, df in data.items():
            # Convert DataFrame to dictionary with ISO date strings as keys
            df_dict = {}
            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d")
                df_dict[date_str] = row.to_dict()
            
            # Include any metadata from DataFrame attrs
            metadata = {k: v for k, v in df.attrs.items() if isinstance(v, (str, int, float, bool))}
            
            serializable_data[metric] = {
                "data": df_dict,
                "metadata": metadata
            }
        
        # Save to file
        file_path = source_dir / filename
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved raw {source} data to {file_path}")
    
    def save_processed_data(self, 
                          data: Dict[str, pd.DataFrame], 
                          prefix: str = "metrics") -> str:
        """
        Save processed data to disk.
        
        Args:
            data: Dictionary mapping metric names to DataFrames
            prefix: Prefix for the output file
            
        Returns:
            Path to the saved file
        """
        # Create a date-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.parquet"
        
        # Combine all metrics into a single DataFrame with a metric column
        combined_data = []
        
        for metric, df in data.items():
            # Add metric name as a column
            df_copy = df.copy()
            df_copy["metric"] = metric
            
            # Move index to a column for easier combination
            df_copy = df_copy.reset_index()
            
            combined_data.append(df_copy)
        
        if not combined_data:
            logger.warning("No data to save")
            return ""
        
        # Combine all metrics
        all_metrics_df = pd.concat(combined_data, ignore_index=True)
        
        # Save to parquet file
        file_path = self.processed_data_dir / filename
        all_metrics_df.to_parquet(file_path, index=False)
        
        logger.info(f"Saved processed data to {file_path}")
        
        return str(file_path)
    
    def load_processed_data(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from disk.
        
        Args:
            file_path: Path to the processed data file
            
        Returns:
            Dictionary mapping metric names to DataFrames
        """
        try:
            # Load the combined DataFrame
            all_metrics_df = pd.read_parquet(file_path)
            
            # Split back into separate DataFrames by metric
            metrics = all_metrics_df["metric"].unique()
            result = {}
            
            for metric in metrics:
                metric_df = all_metrics_df[all_metrics_df["metric"] == metric].copy()
                # Drop the metric column
                metric_df = metric_df.drop(columns=["metric"])
                # Set the date as index
                if "date" in metric_df.columns:
                    metric_df = metric_df.set_index("date")
                
                result[metric] = metric_df
            
            return result
        except Exception as e:
            logger.error(f"Error loading processed data from {file_path}: {e}")
            raise 