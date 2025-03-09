"""
Data Cleaners

This module provides utilities for cleaning and validating housing market data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
from ..utils.logger import get_logger

logger = get_logger(__name__)

class DataCleaner:
    """Utilities for cleaning and validating housing market data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataCleaner.
        
        Args:
            config: Optional configuration dictionary with cleaning parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "remove_outliers": True,
            "outlier_std_threshold": 3.0,  # Number of standard deviations for outlier detection
            "fill_missing": True,
            "interpolation_method": "linear",  # Method for interpolating missing values
            "min_date": "2015-01-01",  # Minimum date to keep in the data
            "max_date": datetime.now().strftime("%Y-%m-%d"),  # Maximum date to keep
            "validate_values": True,  # Whether to validate values (e.g., prices must be positive)
            "value_constraints": {
                "price": {"min": 0},  # Minimum price must be non-negative
                "inventory": {"min": 0},  # Inventory must be non-negative
                "days": {"min": 0},  # Days must be non-negative
                "ratio": {"min": 0, "max": 2}  # Ratios typically between 0 and 2
            }
        }
        
        # Update default config with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
    
    def clean_dataframe(self, df: pd.DataFrame, metric_type: Optional[str] = None) -> pd.DataFrame:
        """
        Clean a DataFrame containing housing market data.
        
        Args:
            df: DataFrame to clean
            metric_type: Type of metric (e.g., 'price', 'inventory', 'days', 'ratio')
                         Used for applying appropriate value constraints
                         
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for cleaning")
            return df
        
        logger.info(f"Cleaning DataFrame with {len(df)} rows")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Remove rows with all NaN values
        df_clean = df_clean.dropna(how='all')
        
        # Filter by date range
        if df_clean.index.name == 'date' or isinstance(df_clean.index, pd.DatetimeIndex):
            min_date = pd.to_datetime(self.config["min_date"])
            max_date = pd.to_datetime(self.config["max_date"])
            df_clean = df_clean[(df_clean.index >= min_date) & (df_clean.index <= max_date)]
        
        # Handle numeric columns
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        
        for col in numeric_cols:
            # Remove outliers if configured
            if self.config["remove_outliers"]:
                df_clean = self._remove_outliers(df_clean, col)
            
            # Apply value constraints if configured and metric_type is provided
            if self.config["validate_values"] and metric_type:
                df_clean = self._apply_value_constraints(df_clean, col, metric_type)
        
        # Fill missing values if configured
        if self.config["fill_missing"] and len(df_clean) > 1:
            df_clean = self._fill_missing_values(df_clean)
        
        logger.info(f"Cleaned DataFrame now has {len(df_clean)} rows")
        
        return df_clean
    
    def clean_metrics_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean a dictionary of DataFrames containing housing market metrics.
        
        Args:
            data: Dictionary mapping metric names to DataFrames
                         
        Returns:
            Dictionary of cleaned DataFrames
        """
        cleaned_data = {}
        
        for metric, df in data.items():
            try:
                # Determine metric type from the metric name
                metric_type = self._infer_metric_type(metric)
                
                # Clean the DataFrame
                cleaned_df = self.clean_dataframe(df, metric_type)
                
                cleaned_data[metric] = cleaned_df
            except Exception as e:
                logger.error(f"Error cleaning data for metric {metric}: {e}")
                # Include the original data if cleaning fails
                cleaned_data[metric] = df
        
        return cleaned_data
    
    def validate_data_consistency(self, 
                              data: Dict[str, pd.DataFrame]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Check data for consistency and potential issues.
        
        Args:
            data: Dictionary mapping metric names to DataFrames
                         
        Returns:
            Tuple of (is_valid, issues_dict) where:
            - is_valid: Boolean indicating if the data is valid overall
            - issues_dict: Dictionary mapping metric names to lists of issue descriptions
        """
        issues = {}
        overall_valid = True
        
        for metric, df in data.items():
            metric_issues = []
            
            # Check for empty DataFrame
            if df.empty:
                metric_issues.append("Empty dataset")
                overall_valid = False
            
            # Check for insufficient data points
            elif len(df) < 4:  # Arbitrary threshold for minimum data points
                metric_issues.append(f"Insufficient data points: {len(df)}")
                overall_valid = False
            
            # Check for large gaps in time series
            if not df.empty and isinstance(df.index, pd.DatetimeIndex):
                gaps = self._find_time_series_gaps(df)
                if gaps:
                    gap_descriptions = [f"{gap[0]} to {gap[1]}" for gap in gaps]
                    metric_issues.append(f"Time series gaps: {', '.join(gap_descriptions)}")
            
            # Check for extreme values
            if not df.empty:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                for col in numeric_cols:
                    # Check for zeros where they might be unexpected
                    if "price" in metric.lower() and (df[col] == 0).any():
                        metric_issues.append(f"Zero values in price data column {col}")
                    
                    # Check for negative values where they shouldn't exist
                    if (df[col] < 0).any() and self._should_be_positive(metric):
                        metric_issues.append(f"Negative values in column {col}")
            
            if metric_issues:
                issues[metric] = metric_issues
        
        return overall_valid, issues
    
    def _remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers from a DataFrame column using standard deviation method.
        
        Args:
            df: DataFrame to process
            column: Column to remove outliers from
                         
        Returns:
            DataFrame with outliers removed
        """
        # Skip if column doesn't exist or has no variation
        if column not in df.columns or df[column].std() == 0:
            return df
        
        # Calculate mean and standard deviation
        mean = df[column].mean()
        std = df[column].std()
        threshold = self.config["outlier_std_threshold"]
        
        # Create mask for non-outlier values
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        
        # Count outliers for logging
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers.empty:
            logger.info(f"Removing {len(outliers)} outliers from column {column}")
        
        # Filter out outliers
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    def _apply_value_constraints(self, 
                               df: pd.DataFrame, 
                               column: str, 
                               metric_type: str) -> pd.DataFrame:
        """
        Apply value constraints to a DataFrame column.
        
        Args:
            df: DataFrame to process
            column: Column to apply constraints to
            metric_type: Type of metric to determine which constraints to apply
                         
        Returns:
            DataFrame with constraints applied
        """
        if metric_type not in self.config["value_constraints"]:
            return df
        
        constraints = self.config["value_constraints"][metric_type]
        
        # Apply minimum constraint if defined
        if "min" in constraints and constraints["min"] is not None:
            min_val = constraints["min"]
            invalid_count = (df[column] < min_val).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} values below minimum {min_val} in column {column}")
                df.loc[df[column] < min_val, column] = np.nan
        
        # Apply maximum constraint if defined
        if "max" in constraints and constraints["max"] is not None:
            max_val = constraints["max"]
            invalid_count = (df[column] > max_val).sum()
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} values above maximum {max_val} in column {column}")
                df.loc[df[column] > max_val, column] = np.nan
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing values in a DataFrame using interpolation.
        
        Args:
            df: DataFrame to process
                         
        Returns:
            DataFrame with missing values filled
        """
        # Count missing values for logging
        na_count_before = df.isna().sum().sum()
        
        if na_count_before == 0:
            return df
        
        # Choose interpolation method
        method = self.config["interpolation_method"]
        
        # Only interpolate numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        if not numeric_cols:
            return df
        
        # Interpolate missing values
        df_interpolated = df.copy()
        df_interpolated[numeric_cols] = df_interpolated[numeric_cols].interpolate(
            method=method, 
            axis=0,  # Interpolate along index (usually time)
            limit_direction='both'  # Fill NaNs at beginning and end
        )
        
        # Count remaining missing values
        na_count_after = df_interpolated.isna().sum().sum()
        
        logger.info(f"Filled {na_count_before - na_count_after} missing values using {method} interpolation")
        
        return df_interpolated
    
    def _find_time_series_gaps(self, df: pd.DataFrame, max_gap_days: int = 120) -> List[Tuple[str, str]]:
        """
        Find gaps in a time series DataFrame.
        
        Args:
            df: DataFrame with datetime index
            max_gap_days: Maximum acceptable gap in days
                         
        Returns:
            List of (start_date, end_date) tuples representing gaps
        """
        if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 2:
            return []
        
        # Sort index to ensure correct gap detection
        sorted_idx = df.index.sort_values()
        
        # Calculate differences between consecutive dates
        diffs = sorted_idx[1:] - sorted_idx[:-1]
        
        # Find large gaps
        gaps = []
        for i, diff in enumerate(diffs):
            if diff.days > max_gap_days:
                gap_start = sorted_idx[i].strftime("%Y-%m-%d")
                gap_end = sorted_idx[i+1].strftime("%Y-%m-%d")
                gaps.append((gap_start, gap_end))
        
        return gaps
    
    def _infer_metric_type(self, metric_name: str) -> str:
        """
        Infer the metric type from a metric name for applying appropriate constraints.
        
        Args:
            metric_name: Name of the metric
                         
        Returns:
            Inferred metric type ('price', 'inventory', 'days', 'ratio')
        """
        metric_name_lower = metric_name.lower()
        
        if any(term in metric_name_lower for term in ["price", "cost", "value", "sale", "income"]):
            return "price"
        elif any(term in metric_name_lower for term in ["inventory", "supply", "listing"]):
            return "inventory"
        elif any(term in metric_name_lower for term in ["days", "dom", "time"]):
            return "days"
        elif any(term in metric_name_lower for term in ["ratio", "rate", "percent", "index"]):
            return "ratio"
        else:
            return "other"
    
    def _should_be_positive(self, metric_name: str) -> bool:
        """
        Determine if a metric should always have positive values.
        
        Args:
            metric_name: Name of the metric
                         
        Returns:
            Boolean indicating if the metric should be positive
        """
        # Most housing metrics should be positive
        negative_allowed = ["change", "growth", "delta", "diff", "index"]
        
        # Check if any of the negative-allowed terms are in the metric name
        for term in negative_allowed:
            if term in metric_name.lower():
                return False
        
        return True 