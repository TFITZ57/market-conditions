"""
Time Series Processor

This module provides utilities for time series operations and alignment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import scipy.stats as stats
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TimeSeriesProcessor:
    """Utilities for time series operations and alignment."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the TimeSeriesProcessor.
        
        Args:
            config: Optional configuration dictionary with time series parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "seasonal_period": 4,  # Default seasonal period for quarterly data
            "detrend_method": "multiplicative",  # Method for seasonal decomposition
            "smoothing_method": "multiplicative",  # Method for exponential smoothing
            "correlation_method": "pearson",  # Method for correlation calculation
            "forecast_periods": 4,  # Number of periods to forecast
            "moving_avg_windows": [4, 8],  # Windows for moving averages (e.g., 1-year, 2-year for quarterly data)
            "detect_outliers": True,  # Whether to detect and flag outliers
            "outlier_threshold": 2.0,  # Z-score threshold for outlier detection
        }
        
        # Update default config with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
    
    def decompose_time_series(self, 
                           series: pd.Series, 
                           period: Optional[int] = None,
                           model: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Decompose a time series into trend, seasonal, and residual components.
        
        Args:
            series: Time series to decompose
            period: Seasonal period (default from config if None)
            model: Decomposition model ('additive' or 'multiplicative')
                         
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        if len(series) < 2 * (period or self.config["seasonal_period"]):
            logger.warning(f"Time series too short for decomposition: {len(series)} points")
            return {
                "trend": series,
                "seasonal": pd.Series(1, index=series.index),
                "residual": pd.Series(0, index=series.index)
            }
        
        # Use defaults from config if not provided
        period = period or self.config["seasonal_period"]
        model = model or self.config["detrend_method"]
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                series, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            result = {
                "trend": decomposition.trend,
                "seasonal": decomposition.seasonal,
                "residual": decomposition.resid
            }
            
            return result
        except Exception as e:
            logger.error(f"Error decomposing time series: {e}")
            # Return original series if decomposition fails
            return {
                "trend": series,
                "seasonal": pd.Series(1 if model == 'multiplicative' else 0, index=series.index),
                "residual": pd.Series(0, index=series.index)
            }
    
    def smooth_time_series(self, 
                         series: pd.Series, 
                         method: Optional[str] = None) -> pd.Series:
        """
        Smooth a time series using exponential smoothing.
        
        Args:
            series: Time series to smooth
            method: Smoothing method ('additive' or 'multiplicative')
                         
        Returns:
            Smoothed time series
        """
        if len(series) < 4:
            logger.warning(f"Time series too short for smoothing: {len(series)} points")
            return series
        
        # Use default from config if not provided
        method = method or self.config["smoothing_method"]
        
        try:
            # Perform exponential smoothing with trend and seasonal components
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=method,
                seasonal_periods=self.config["seasonal_period"]
            )
            
            # Fit the model
            fit = model.fit()
            
            # Get the smoothed values
            smoothed = fit.fittedvalues
            
            return smoothed
        except Exception as e:
            logger.error(f"Error smoothing time series: {e}")
            # Return original series if smoothing fails
            return series
    
    def calculate_moving_averages(self, 
                               series: pd.Series,
                               windows: Optional[List[int]] = None) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for a time series.
        
        Args:
            series: Time series to process
            windows: List of window sizes for moving averages
                         
        Returns:
            Dictionary mapping window sizes to moving average series
        """
        if len(series) < 2:
            logger.warning(f"Time series too short for moving averages: {len(series)} points")
            return {}
        
        # Use default from config if not provided
        windows = windows or self.config["moving_avg_windows"]
        
        result = {}
        
        for window in windows:
            try:
                # Calculate moving average
                ma = series.rolling(window=window, min_periods=1).mean()
                
                # Add to result
                result[f"MA_{window}"] = ma
            except Exception as e:
                logger.error(f"Error calculating {window}-period moving average: {e}")
        
        return result
    
    def calculate_correlation(self, 
                           series1: pd.Series, 
                           series2: pd.Series,
                           method: Optional[str] = None) -> Tuple[float, float]:
        """
        Calculate correlation between two time series.
        
        Args:
            series1: First time series
            series2: Second time series
            method: Correlation method ('pearson', 'spearman', 'kendall')
                         
        Returns:
            Tuple of (correlation coefficient, p-value)
        """
        # Align the series to the same dates
        series1, series2 = series1.align(series2, join='inner')
        
        if len(series1) < 3:
            logger.warning(f"Time series too short for correlation: {len(series1)} points")
            return 0.0, 1.0
        
        # Use default from config if not provided
        method = method or self.config["correlation_method"]
        
        try:
            # Calculate correlation
            if method == 'pearson':
                r, p = stats.pearsonr(series1, series2)
            elif method == 'spearman':
                r, p = stats.spearmanr(series1, series2)
            elif method == 'kendall':
                r, p = stats.kendalltau(series1, series2)
            else:
                # Default to Pearson
                r, p = stats.pearsonr(series1, series2)
            
            return r, p
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0, 1.0
    
    def forecast_time_series(self, 
                          series: pd.Series, 
                          periods: Optional[int] = None,
                          return_conf_int: bool = True) -> Dict[str, pd.Series]:
        """
        Forecast a time series using exponential smoothing.
        
        Args:
            series: Time series to forecast
            periods: Number of periods to forecast
            return_conf_int: Whether to return confidence intervals
                         
        Returns:
            Dictionary with forecast and optional confidence intervals
        """
        if len(series) < 2 * self.config["seasonal_period"]:
            logger.warning(f"Time series too short for forecasting: {len(series)} points")
            return {"forecast": pd.Series(dtype=float)}
        
        # Use default from config if not provided
        periods = periods or self.config["forecast_periods"]
        
        try:
            # Create model for exponential smoothing
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=self.config["smoothing_method"],
                seasonal_periods=self.config["seasonal_period"]
            )
            
            # Fit the model
            fit = model.fit()
            
            # Generate forecast
            forecast = fit.forecast(periods)
            
            result = {"forecast": forecast}
            
            # Calculate confidence intervals if requested
            if return_conf_int:
                forecast_ci = fit.get_prediction(start=len(series), end=len(series) + periods - 1)
                result["lower"] = forecast_ci.conf_int().iloc[:, 0]
                result["upper"] = forecast_ci.conf_int().iloc[:, 1]
            
            return result
        except Exception as e:
            logger.error(f"Error forecasting time series: {e}")
            return {"forecast": pd.Series(dtype=float)}
    
    def detect_outliers(self, 
                     series: pd.Series,
                     threshold: Optional[float] = None) -> pd.Series:
        """
        Detect outliers in a time series using Z-scores.
        
        Args:
            series: Time series to analyze
            threshold: Z-score threshold for outlier detection
                         
        Returns:
            Boolean Series indicating outliers (True) and non-outliers (False)
        """
        if len(series) < 3:
            logger.warning(f"Time series too short for outlier detection: {len(series)} points")
            return pd.Series(False, index=series.index)
        
        # Use default from config if not provided
        threshold = threshold or self.config["outlier_threshold"]
        
        try:
            # Calculate Z-scores
            z_scores = (series - series.mean()) / series.std()
            
            # Identify outliers
            outliers = abs(z_scores) > threshold
            
            # Log number of outliers
            num_outliers = outliers.sum()
            if num_outliers > 0:
                logger.info(f"Detected {num_outliers} outliers in time series")
            
            return outliers
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
            return pd.Series(False, index=series.index)
    
    def interpolate_missing_values(self, 
                                series: pd.Series,
                                method: str = 'linear') -> pd.Series:
        """
        Interpolate missing values in a time series.
        
        Args:
            series: Time series with missing values
            method: Interpolation method
                         
        Returns:
            Time series with interpolated values
        """
        if series.isna().sum() == 0:
            # No missing values
            return series
        
        try:
            # Interpolate missing values
            interpolated = series.interpolate(method=method, limit_direction='both')
            
            # Log number of interpolated values
            num_interpolated = series.isna().sum()
            logger.info(f"Interpolated {num_interpolated} missing values in time series")
            
            return interpolated
        except Exception as e:
            logger.error(f"Error interpolating missing values: {e}")
            return series
    
    def calculate_year_over_year_change(self, 
                                     series: pd.Series,
                                     periods_per_year: int = 4) -> pd.Series:
        """
        Calculate year-over-year percentage change.
        
        Args:
            series: Time series to process
            periods_per_year: Number of periods per year
                         
        Returns:
            Series with year-over-year percentage change
        """
        if len(series) <= periods_per_year:
            logger.warning(f"Time series too short for YoY change: {len(series)} points")
            return pd.Series(dtype=float, index=series.index)
        
        try:
            # Calculate percentage change
            yoy_change = series.pct_change(periods=periods_per_year) * 100
            
            return yoy_change
        except Exception as e:
            logger.error(f"Error calculating year-over-year change: {e}")
            return pd.Series(dtype=float, index=series.index)
    
    def align_multiple_series(self, 
                           series_dict: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Align multiple time series to the same date range.
        
        Args:
            series_dict: Dictionary mapping names to time series
                         
        Returns:
            Dictionary of aligned time series
        """
        if not series_dict:
            return {}
        
        # Find common date range
        common_dates = None
        
        for series in series_dict.values():
            if common_dates is None:
                common_dates = series.index
            else:
                common_dates = common_dates.intersection(series.index)
        
        if common_dates is None or len(common_dates) == 0:
            logger.warning("No common dates found for time series alignment")
            return {}
        
        # Align series to common dates
        aligned_dict = {}
        
        for name, series in series_dict.items():
            aligned_dict[name] = series.loc[common_dates]
        
        return aligned_dict
    
    def calculate_trend_strength(self, series: pd.Series) -> float:
        """
        Calculate the strength of the trend component in a time series.
        
        Args:
            series: Time series to analyze
                         
        Returns:
            Trend strength (0-1, higher values indicate stronger trend)
        """
        if len(series) < 2 * self.config["seasonal_period"]:
            logger.warning(f"Time series too short for trend analysis: {len(series)} points")
            return 0.0
        
        try:
            # Decompose the series
            components = self.decompose_time_series(series)
            
            # Extract components
            trend = components["trend"]
            residual = components["residual"]
            
            # Calculate variance of residuals
            var_resid = residual.var()
            
            if var_resid == 0:
                return 1.0
            
            # Calculate variance of detrended series
            detrended = series - trend
            var_detrended = detrended.var()
            
            if var_detrended == 0:
                return 0.0
            
            # Calculate trend strength
            trend_strength = 1 - (var_resid / var_detrended)
            
            # Ensure result is between 0 and 1
            trend_strength = max(0, min(1, trend_strength))
            
            return trend_strength
        except Exception as e:
            logger.error(f"Error calculating trend strength: {e}")
            return 0.0 