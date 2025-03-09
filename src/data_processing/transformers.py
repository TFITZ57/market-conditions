"""
Data Transformers

This module provides utilities for transforming data into analysis-ready formats.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataTransformer:
    """Utilities for transforming housing market data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataTransformer.
        
        Args:
            config: Optional configuration dictionary with transformation parameters
        """
        self.config = config or {}

    def normalize_columns(self, df: pd.DataFrame, date_col: str = 'date', 
                        value_col: str = 'value', location_col: str = 'town', 
                        metric_col: str = 'metric') -> pd.DataFrame:
        """
        Normalize DataFrame columns to a standard format.
        
        Args:
            df: DataFrame to normalize
            date_col: Name of the date column
            value_col: Name of the value column
            location_col: Name of the location column
            metric_col: Name of the metric column
            
        Returns:
            Normalized DataFrame
        """
        try:
            # Make a copy to avoid modifying the original
            df_norm = df.copy()
            
            # Normalize date column
            if date_col in df_norm.columns:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df_norm[date_col]):
                    df_norm[date_col] = pd.to_datetime(df_norm[date_col])
                
                # Standardize column name
                if date_col != 'date':
                    df_norm = df_norm.rename(columns={date_col: 'date'})
            
            # Normalize value column
            if value_col in df_norm.columns and value_col != 'value':
                df_norm = df_norm.rename(columns={value_col: 'value'})
            
            # Normalize location column
            if location_col in df_norm.columns and location_col != 'town':
                df_norm = df_norm.rename(columns={location_col: 'town'})
            
            # Normalize metric column
            if metric_col in df_norm.columns and metric_col != 'metric':
                df_norm = df_norm.rename(columns={metric_col: 'metric'})
            
            return df_norm
        
        except Exception as e:
            logger.error(f"Error normalizing columns: {e}")
            return df
    
    def pivot_data(self, df: pd.DataFrame, index_col: str = 'date', 
                 columns_col: str = 'town', values_col: str = 'value') -> pd.DataFrame:
        """
        Pivot data to create a wide format DataFrame.
        
        Args:
            df: DataFrame to pivot
            index_col: Column to use as index
            columns_col: Column to use as columns
            values_col: Column to use as values
            
        Returns:
            Pivoted DataFrame
        """
        try:
            # Check if required columns exist
            required_cols = [index_col, columns_col, values_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for pivot: {missing_cols}")
                return df
            
            # Pivot the data
            pivot_df = df.pivot(index=index_col, columns=columns_col, values=values_col)
            
            # Reset column names to single level if they're multi-level
            if isinstance(pivot_df.columns, pd.MultiIndex):
                pivot_df.columns = pivot_df.columns.get_level_values(-1)
            
            return pivot_df
        
        except Exception as e:
            logger.error(f"Error pivoting data: {e}")
            return df
    
    def unpivot_data(self, df: pd.DataFrame, id_vars: Optional[List[str]] = None,
                   var_name: str = 'town', value_name: str = 'value') -> pd.DataFrame:
        """
        Unpivot (melt) data to create a long format DataFrame.
        
        Args:
            df: DataFrame to unpivot
            id_vars: Columns to use as identifier variables
            var_name: Name for the variable column
            value_name: Name for the value column
            
        Returns:
            Unpivoted DataFrame
        """
        try:
            # If id_vars is None, use the index as id_vars
            if id_vars is None:
                # Reset index to make it a column
                df_reset = df.reset_index()
                id_vars = [df_reset.columns[0]]
                df_to_melt = df_reset
            else:
                # Make sure id_vars columns exist
                missing_cols = [col for col in id_vars if col not in df.columns]
                
                if missing_cols:
                    logger.error(f"Missing columns for unpivot: {missing_cols}")
                    return df
                
                df_to_melt = df
            
            # Unpivot the data
            melted_df = pd.melt(df_to_melt, id_vars=id_vars, var_name=var_name, value_name=value_name)
            
            return melted_df
        
        except Exception as e:
            logger.error(f"Error unpivoting data: {e}")
            return df
    
    def resample_time_series(self, df: pd.DataFrame, date_col: str = 'date', 
                           freq: str = 'Q', agg_func: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data to a different frequency.
        
        Args:
            df: DataFrame to resample
            date_col: Name of the date column
            freq: Frequency for resampling ('D'=daily, 'W'=weekly, 'M'=monthly, 'Q'=quarterly, 'A'=annual)
            agg_func: Aggregation function ('mean', 'median', 'sum', 'min', 'max')
            
        Returns:
            Resampled DataFrame
        """
        try:
            # Check if date column exists
            if date_col not in df.columns:
                logger.error(f"Date column '{date_col}' not found for resampling")
                return df
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
                df = df.copy()
                df[date_col] = pd.to_datetime(df[date_col])
            
            # Set date as index for resampling
            df_indexed = df.set_index(date_col)
            
            # Get numeric columns for resampling
            numeric_cols = df_indexed.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                logger.warning("No numeric columns found for resampling")
                return df
            
            # Map aggregation function name to actual function
            agg_func_map = {
                'mean': np.mean,
                'median': np.median,
                'sum': np.sum,
                'min': np.min,
                'max': np.max
            }
            
            agg_function = agg_func_map.get(agg_func, np.mean)
            
            # Create a dictionary mapping each numeric column to the aggregation function
            agg_dict = {col: agg_function for col in numeric_cols}
            
            # For categorical columns, use first value as aggregation
            cat_cols = [col for col in df_indexed.columns if col not in numeric_cols]
            if cat_cols:
                for col in cat_cols:
                    agg_dict[col] = 'first'
            
            # Resample the data
            resampled = df_indexed.resample(freq).agg(agg_dict)
            
            # Reset index to make date a column again
            resampled = resampled.reset_index()
            
            return resampled
        
        except Exception as e:
            logger.error(f"Error resampling time series: {e}")
            return df
    
    def calculate_year_over_year_change(self, df: pd.DataFrame, date_col: str = 'date', 
                                       value_col: str = 'value') -> pd.DataFrame:
        """
        Calculate year-over-year change for a time series.
        
        Args:
            df: DataFrame with time series data
            date_col: Name of the date column
            value_col: Name of the value column
            
        Returns:
            DataFrame with year-over-year change added
        """
        try:
            # Check if required columns exist
            required_cols = [date_col, value_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for YoY calculation: {missing_cols}")
                return df
            
            # Make a copy of the input DataFrame
            result_df = df.copy()
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Sort by date
            result_df = result_df.sort_values(date_col)
            
            # Set date as index for time-based operations
            result_df = result_df.set_index(date_col)
            
            # Calculate year-over-year absolute change
            result_df[f'{value_col}_yoy_abs'] = result_df[value_col].diff(periods=4)  # 4 quarters = 1 year
            
            # Calculate year-over-year percentage change
            result_df[f'{value_col}_yoy'] = result_df[value_col].pct_change(periods=4) * 100
            
            # Reset index to make date a column again
            result_df = result_df.reset_index()
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating year-over-year change: {e}")
            return df
    
    def calculate_quarter_over_quarter_change(self, df: pd.DataFrame, date_col: str = 'date', 
                                            value_col: str = 'value') -> pd.DataFrame:
        """
        Calculate quarter-over-quarter change for a time series.
        
        Args:
            df: DataFrame with time series data
            date_col: Name of the date column
            value_col: Name of the value column
            
        Returns:
            DataFrame with quarter-over-quarter change added
        """
        try:
            # Check if required columns exist
            required_cols = [date_col, value_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for QoQ calculation: {missing_cols}")
                return df
            
            # Make a copy of the input DataFrame
            result_df = df.copy()
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Sort by date
            result_df = result_df.sort_values(date_col)
            
            # Set date as index for time-based operations
            result_df = result_df.set_index(date_col)
            
            # Calculate quarter-over-quarter absolute change
            result_df[f'{value_col}_qoq_abs'] = result_df[value_col].diff(periods=1)
            
            # Calculate quarter-over-quarter percentage change
            result_df[f'{value_col}_qoq'] = result_df[value_col].pct_change(periods=1) * 100
            
            # Reset index to make date a column again
            result_df = result_df.reset_index()
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error calculating quarter-over-quarter change: {e}")
            return df
    
    def normalize_by_baseline(self, df: pd.DataFrame, date_col: str = 'date', 
                            value_col: str = 'value', baseline_period: Optional[str] = None, 
                            groups: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalize values relative to a baseline period.
        
        Args:
            df: DataFrame to normalize
            date_col: Name of the date column
            value_col: Name of the value column
            baseline_period: Date string for baseline period (e.g., '2015-01-01')
                            If None, the first period is used as baseline
            groups: List of columns to group by before normalization
                  If None, no grouping is applied
            
        Returns:
            DataFrame with normalized values added
        """
        try:
            # Check if required columns exist
            required_cols = [date_col, value_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Missing columns for normalization: {missing_cols}")
                return df
            
            # Make a copy of the input DataFrame
            result_df = df.copy()
            
            # Convert date column to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                result_df[date_col] = pd.to_datetime(result_df[date_col])
            
            # Create a new column for normalized values
            norm_col = f'{value_col}_normalized'
            
            # Apply normalization for each group if groups are specified
            if groups:
                # Check if group columns exist
                missing_group_cols = [col for col in groups if col not in result_df.columns]
                
                if missing_group_cols:
                    logger.error(f"Missing group columns for normalization: {missing_group_cols}")
                    return df
                
                # Process each group separately
                for name, group in result_df.groupby(groups):
                    # Sort group by date
                    group = group.sort_values(date_col)
                    
                    # Get baseline value
                    if baseline_period:
                        # Try to find the baseline period
                        baseline_df = group[group[date_col] == pd.to_datetime(baseline_period)]
                        
                        if baseline_df.empty:
                            logger.warning(f"Baseline period '{baseline_period}' not found for group {name}, using first period")
                            baseline_value = group[value_col].iloc[0]
                        else:
                            baseline_value = baseline_df[value_col].iloc[0]
                    else:
                        # Use first period as baseline
                        baseline_value = group[value_col].iloc[0]
                    
                    # Avoid division by zero
                    if baseline_value == 0:
                        logger.warning(f"Baseline value is 0 for group {name}, setting to 1 to avoid division by zero")
                        baseline_value = 1
                    
                    # Calculate normalized values
                    group_indices = group.index
                    result_df.loc[group_indices, norm_col] = result_df.loc[group_indices, value_col] / baseline_value * 100
            else:
                # No grouping, normalize the entire dataset
                # Sort by date
                result_df = result_df.sort_values(date_col)
                
                # Get baseline value
                if baseline_period:
                    # Try to find the baseline period
                    baseline_df = result_df[result_df[date_col] == pd.to_datetime(baseline_period)]
                    
                    if baseline_df.empty:
                        logger.warning(f"Baseline period '{baseline_period}' not found, using first period")
                        baseline_value = result_df[value_col].iloc[0]
                    else:
                        baseline_value = baseline_df[value_col].iloc[0]
                else:
                    # Use first period as baseline
                    baseline_value = result_df[value_col].iloc[0]
                
                # Avoid division by zero
                if baseline_value == 0:
                    logger.warning("Baseline value is 0, setting to 1 to avoid division by zero")
                    baseline_value = 1
                
                # Calculate normalized values
                result_df[norm_col] = result_df[value_col] / baseline_value * 100
            
            return result_df
        
        except Exception as e:
            logger.error(f"Error normalizing by baseline: {e}")
            return df
    
    def transform_metrics_data(self, data: Dict[str, pd.DataFrame], 
                             transformations: List[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Apply multiple transformations to metrics data.
        
        Args:
            data: Dictionary mapping metric names to DataFrames
            transformations: List of transformation specifications
                           Each specification is a dict with:
                           - 'type': Transformation type ('normalize_columns', 'resample_time_series', etc.)
                           - 'params': Parameters for the transformation
                           - 'metrics': List of metrics to apply to (None for all)
            
        Returns:
            Dictionary of transformed DataFrames
        """
        try:
            transformed_data = {}
            
            # Copy input data to avoid modifying the original
            for metric, df in data.items():
                transformed_data[metric] = df.copy()
            
            # Apply each transformation
            for transform_spec in transformations:
                transform_type = transform_spec.get('type')
                params = transform_spec.get('params', {})
                metrics = transform_spec.get('metrics')
                
                # Determine which metrics to transform
                metrics_to_transform = metrics if metrics else list(transformed_data.keys())
                
                # Apply the transformation to each metric
                for metric in metrics_to_transform:
                    if metric not in transformed_data:
                        logger.warning(f"Metric '{metric}' not found for transformation '{transform_type}'")
                        continue
                    
                    df = transformed_data[metric]
                    
                    # Apply transformation based on type
                    if transform_type == 'normalize_columns':
                        transformed_data[metric] = self.normalize_columns(df, **params)
                    
                    elif transform_type == 'pivot_data':
                        transformed_data[metric] = self.pivot_data(df, **params)
                    
                    elif transform_type == 'unpivot_data':
                        transformed_data[metric] = self.unpivot_data(df, **params)
                    
                    elif transform_type == 'resample_time_series':
                        transformed_data[metric] = self.resample_time_series(df, **params)
                    
                    elif transform_type == 'calculate_year_over_year_change':
                        transformed_data[metric] = self.calculate_year_over_year_change(df, **params)
                    
                    elif transform_type == 'calculate_quarter_over_quarter_change':
                        transformed_data[metric] = self.calculate_quarter_over_quarter_change(df, **params)
                    
                    elif transform_type == 'normalize_by_baseline':
                        transformed_data[metric] = self.normalize_by_baseline(df, **params)
                    
                    else:
                        logger.warning(f"Unknown transformation type: {transform_type}")
            
            return transformed_data
        
        except Exception as e:
            logger.error(f"Error applying transformations: {e}")
            return data

    def get_property_type_distribution(self, metrics_data, location=None):
        """
        Generate property type distribution data for pie charts.
        
        Args:
            metrics_data: Dictionary of market metrics data
            location: Selected location
            
        Returns:
            Dictionary with property types and their percentages
        """
        # In a real implementation, this would use actual data from the metrics_data
        # For now, return sample data in the format expected by the pie chart
        property_types = {
            "Single-Family Homes": 62,
            "Condos/Apartments": 23,
            "Multi-Family": 8,
            "Townhouses": 7
        }
        
        # Return property types and their percentages
        return {
            "types": list(property_types.keys()),
            "values": list(property_types.values())
        }
