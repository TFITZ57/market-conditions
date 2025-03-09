"""
Metrics Calculator

This module provides utilities for calculating derived housing market metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """Utilities for calculating derived housing market metrics."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the MetricsCalculator.
        
        Args:
            config: Optional configuration dictionary with calculation parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "median_income": 84000,  # Default median income for Fairfield County, CT
            "mortgage_rate": 0.07,  # Default mortgage rate if not available from data
            "mortgage_term_years": 30,  # Default mortgage term in years
            "down_payment_percent": 0.20,  # Default down payment percentage
            "property_tax_rate": 0.0175,  # Default property tax rate for Fairfield County
            "insurance_rate": 0.0035,  # Default homeowner's insurance rate
            "affordability_threshold": 0.28,  # Housing cost to income ratio threshold
            "absorption_window": 3,  # Number of months to use for absorption rate calculation
            "price_tier_percentiles": [0.25, 0.5, 0.75],  # Percentiles for price tier calculation
        }
        
        # Update default config with provided config
        self.default_config.update(self.config)
        self.config = self.default_config
    
    def calculate_affordability_index(self, 
                                   median_price: float, 
                                   median_income: Optional[float] = None,
                                   mortgage_rate: Optional[float] = None) -> float:
        """
        Calculate housing affordability index.
        
        A value of 100 means the median household income is exactly sufficient to purchase
        the median-priced home. Values above 100 indicate greater affordability.
        
        Args:
            median_price: Median home price
            median_income: Median household income (default from config if None)
            mortgage_rate: Mortgage interest rate (default from config if None)
                         
        Returns:
            Affordability index value
        """
        # Use defaults from config if not provided
        median_income = median_income or self.config["median_income"]
        mortgage_rate = mortgage_rate or self.config["mortgage_rate"]
        
        # Calculate down payment
        down_payment = median_price * self.config["down_payment_percent"]
        
        # Calculate loan amount
        loan_amount = median_price - down_payment
        
        # Convert annual rate to monthly
        monthly_rate = mortgage_rate / 12
        
        # Calculate number of payments
        num_payments = self.config["mortgage_term_years"] * 12
        
        # Calculate monthly mortgage payment
        if monthly_rate == 0:
            # Handle edge case of zero interest rate
            monthly_mortgage_payment = loan_amount / num_payments
        else:
            monthly_mortgage_payment = loan_amount * (
                monthly_rate * (1 + monthly_rate) ** num_payments
            ) / ((1 + monthly_rate) ** num_payments - 1)
        
        # Calculate monthly property tax and insurance
        monthly_property_tax = (median_price * self.config["property_tax_rate"]) / 12
        monthly_insurance = (median_price * self.config["insurance_rate"]) / 12
        
        # Calculate total monthly housing cost
        total_monthly_cost = monthly_mortgage_payment + monthly_property_tax + monthly_insurance
        
        # Calculate monthly income
        monthly_income = median_income / 12
        
        # Calculate maximum affordable monthly payment based on threshold
        max_affordable_payment = monthly_income * self.config["affordability_threshold"]
        
        # Calculate affordability index
        affordability_index = (max_affordable_payment / total_monthly_cost) * 100
        
        return affordability_index
    
    def calculate_price_to_income_ratio(self, 
                                     median_price: float, 
                                     median_income: Optional[float] = None) -> float:
        """
        Calculate home price to income ratio.
        
        This ratio indicates how many years of income are needed to buy a home.
        Higher values indicate lower affordability.
        
        Args:
            median_price: Median home price
            median_income: Median household income (default from config if None)
                         
        Returns:
            Price to income ratio
        """
        # Use defaults from config if not provided
        median_income = median_income or self.config["median_income"]
        
        # Calculate ratio
        ratio = median_price / median_income
        
        return ratio
    
    def calculate_months_of_supply(self, 
                                inventory: float, 
                                sales_volume: float) -> float:
        """
        Calculate months of supply (inventory divided by monthly sales).
        
        Args:
            inventory: Number of homes for sale
            sales_volume: Number of homes sold in the past month
                         
        Returns:
            Months of supply
        """
        if sales_volume == 0:
            logger.warning("Sales volume is zero, cannot calculate months of supply")
            return float('inf')
        
        months_supply = inventory / sales_volume
        
        return months_supply
    
    def calculate_absorption_rate(self, 
                               sales_df: pd.DataFrame,
                               inventory_df: pd.DataFrame,
                               window: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate absorption rate (percentage of inventory sold per period).
        
        Args:
            sales_df: DataFrame with sales data
            inventory_df: DataFrame with inventory data
            window: Number of periods to use for calculation (default from config if None)
                         
        Returns:
            DataFrame with absorption rate
        """
        if sales_df.empty or inventory_df.empty:
            logger.warning("Empty DataFrames provided, cannot calculate absorption rate")
            return pd.DataFrame()
        
        # Use default from config if not provided
        window = window or self.config["absorption_window"]
        
        # Ensure DataFrames have datetime indices
        if not isinstance(sales_df.index, pd.DatetimeIndex) or not isinstance(inventory_df.index, pd.DatetimeIndex):
            logger.warning("DataFrames don't have DatetimeIndex, cannot calculate absorption rate")
            return pd.DataFrame()
        
        # Align the DataFrames to the same dates
        common_dates = sales_df.index.intersection(inventory_df.index)
        
        if len(common_dates) == 0:
            logger.warning("No common dates between sales and inventory data")
            return pd.DataFrame()
        
        sales_aligned = sales_df.loc[common_dates]
        inventory_aligned = inventory_df.loc[common_dates]
        
        # Calculate rolling sum of sales
        if 'value' in sales_aligned.columns:
            rolling_sales = sales_aligned['value'].rolling(window=window, min_periods=1).sum()
        else:
            logger.warning("Sales DataFrame doesn't have a 'value' column")
            return pd.DataFrame()
        
        # Get inventory values
        if 'value' in inventory_aligned.columns:
            inventory = inventory_aligned['value']
        else:
            logger.warning("Inventory DataFrame doesn't have a 'value' column")
            return pd.DataFrame()
        
        # Calculate absorption rate
        absorption_rate = (rolling_sales / inventory) * 100
        
        # Create result DataFrame
        result = pd.DataFrame({
            'absorption_rate': absorption_rate
        })
        
        return result
    
    def calculate_market_conditions_index(self, metrics_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate a composite market conditions index based on multiple metrics.
        
        Args:
            metrics_data: Dictionary mapping metric names to DataFrames
                         
        Returns:
            DataFrame with market conditions index
        """
        # Required metrics for the index
        required_metrics = [
            "Months of Supply",
            "Days on Market",
            "Median Sale Price",
            "List Price to Sales Price Ratio",
            "Absorption Rate"
        ]
        
        # Check if all required metrics are available
        missing_metrics = [m for m in required_metrics if m not in metrics_data]
        
        if missing_metrics:
            logger.warning(f"Missing required metrics for market conditions index: {missing_metrics}")
            return pd.DataFrame()
        
        # Extract values from each metric DataFrame
        metric_values = {}
        
        for metric in required_metrics:
            df = metrics_data[metric]
            
            if df.empty or 'value' not in df.columns:
                logger.warning(f"Metric {metric} has no 'value' column")
                return pd.DataFrame()
            
            metric_values[metric] = df['value']
        
        # Align all metrics to the same dates
        values_df = pd.DataFrame(metric_values)
        
        # Handle any missing values
        values_df = values_df.dropna()
        
        if values_df.empty:
            logger.warning("No complete data points for all metrics")
            return pd.DataFrame()
        
        # Normalize each metric to a 0-1 scale
        normalized_df = pd.DataFrame(index=values_df.index)
        
        for metric in required_metrics:
            series = values_df[metric]
            
            # Determine direction: higher is better (positive) or worse (negative)
            positive_direction = metric in ["Absorption Rate", "List Price to Sales Price Ratio", "Median Sale Price"]
            
            # Normalize
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                # Avoid division by zero
                normalized_df[f"{metric}_norm"] = 0.5
            else:
                if positive_direction:
                    # Higher values are better
                    normalized_df[f"{metric}_norm"] = (series - min_val) / (max_val - min_val)
                else:
                    # Lower values are better
                    normalized_df[f"{metric}_norm"] = 1 - ((series - min_val) / (max_val - min_val))
        
        # Calculate the composite index as the average of normalized metrics
        normalized_columns = [col for col in normalized_df.columns if col.endswith('_norm')]
        normalized_df['market_conditions_index'] = normalized_df[normalized_columns].mean(axis=1) * 100
        
        # Add interpretation column
        normalized_df['market_condition'] = normalized_df['market_conditions_index'].apply(self._interpret_market_condition)
        
        return normalized_df
    
    def calculate_price_tiers(self, 
                           price_data: pd.DataFrame,
                           percentiles: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Calculate price tiers based on percentiles.
        
        Args:
            price_data: DataFrame with price data
            percentiles: List of percentiles to use (default from config if None)
                         
        Returns:
            DataFrame with price tiers
        """
        if price_data.empty:
            logger.warning("Empty DataFrame provided, cannot calculate price tiers")
            return pd.DataFrame()
        
        # Use defaults from config if not provided
        percentiles = percentiles or self.config["price_tier_percentiles"]
        
        # Ensure price_data has a 'value' column
        if 'value' not in price_data.columns:
            logger.warning("Price DataFrame doesn't have a 'value' column")
            return pd.DataFrame()
        
        # Create a result DataFrame
        result = pd.DataFrame(index=price_data.index)
        
        # Add original price
        result['price'] = price_data['value']
        
        # Calculate price tiers for each date
        # Group by year and quarter to capture typical time periods
        if not isinstance(price_data.index, pd.DatetimeIndex):
            logger.warning("Price DataFrame doesn't have a DatetimeIndex")
            return result
        
        price_data = price_data.copy()
        price_data['year'] = price_data.index.year
        price_data['quarter'] = price_data.index.quarter
        
        for perc in percentiles:
            perc_name = int(perc * 100)
            result[f'tier_{perc_name}'] = float('nan')
        
        # Calculate percentiles for each year-quarter group
        for (year, quarter), group in price_data.groupby(['year', 'quarter']):
            for perc in percentiles:
                perc_name = int(perc * 100)
                perc_value = group['value'].quantile(perc)
                
                # Assign to the result DataFrame
                for idx in group.index:
                    result.loc[idx, f'tier_{perc_name}'] = perc_value
        
        return result
    
    def calculate_buyer_advantage_index(self, 
                                     list_price_ratio_df: pd.DataFrame,
                                     seller_concessions_df: pd.DataFrame,
                                     days_on_market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate a buyer advantage index based on multiple metrics.
        
        Higher values indicate more favorable conditions for buyers.
        
        Args:
            list_price_ratio_df: DataFrame with list price to sales price ratio
            seller_concessions_df: DataFrame with seller concessions data
            days_on_market_df: DataFrame with days on market data
                         
        Returns:
            DataFrame with buyer advantage index
        """
        # Check for empty DataFrames
        if list_price_ratio_df.empty or seller_concessions_df.empty or days_on_market_df.empty:
            logger.warning("Empty DataFrames provided, cannot calculate buyer advantage index")
            return pd.DataFrame()
        
        # Ensure all DataFrames have 'value' column
        for name, df in [
            ("list_price_ratio", list_price_ratio_df),
            ("seller_concessions", seller_concessions_df),
            ("days_on_market", days_on_market_df)
        ]:
            if 'value' not in df.columns:
                logger.warning(f"{name} DataFrame doesn't have a 'value' column")
                return pd.DataFrame()
        
        # Align all DataFrames to the same dates
        common_dates = list_price_ratio_df.index.intersection(
            seller_concessions_df.index.intersection(days_on_market_df.index)
        )
        
        if len(common_dates) == 0:
            logger.warning("No common dates between metrics")
            return pd.DataFrame()
        
        # Create a DataFrame with all metrics
        aligned_df = pd.DataFrame(index=common_dates)
        aligned_df['list_price_ratio'] = list_price_ratio_df.loc[common_dates, 'value']
        aligned_df['seller_concessions'] = seller_concessions_df.loc[common_dates, 'value']
        aligned_df['days_on_market'] = days_on_market_df.loc[common_dates, 'value']
        
        # Normalize each metric to a 0-1 scale
        normalized_df = pd.DataFrame(index=aligned_df.index)
        
        # Normalize list price ratio (lower ratio = better for buyers)
        series = aligned_df['list_price_ratio']
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            normalized_df['list_price_ratio_norm'] = 0.5
        else:
            normalized_df['list_price_ratio_norm'] = 1 - ((series - min_val) / (max_val - min_val))
        
        # Normalize seller concessions (higher concessions = better for buyers)
        series = aligned_df['seller_concessions']
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            normalized_df['seller_concessions_norm'] = 0.5
        else:
            normalized_df['seller_concessions_norm'] = (series - min_val) / (max_val - min_val)
        
        # Normalize days on market (higher days = better for buyers)
        series = aligned_df['days_on_market']
        min_val = series.min()
        max_val = series.max()
        
        if max_val == min_val:
            normalized_df['days_on_market_norm'] = 0.5
        else:
            normalized_df['days_on_market_norm'] = (series - min_val) / (max_val - min_val)
        
        # Calculate the buyer advantage index as the average of normalized metrics
        normalized_columns = [col for col in normalized_df.columns if col.endswith('_norm')]
        normalized_df['buyer_advantage_index'] = normalized_df[normalized_columns].mean(axis=1) * 100
        
        # Add interpretation column
        normalized_df['market_advantage'] = normalized_df['buyer_advantage_index'].apply(
            lambda x: "Strong Buyer's Market" if x >= 75 else
                     "Buyer's Market" if x >= 60 else
                     "Slight Buyer Advantage" if x >= 55 else
                     "Balanced Market" if x >= 45 else
                     "Slight Seller Advantage" if x >= 40 else
                     "Seller's Market" if x >= 25 else
                     "Strong Seller's Market"
        )
        
        return normalized_df
    
    def _interpret_market_condition(self, index_value: float) -> str:
        """
        Interpret a market condition index value.
        
        Args:
            index_value: Market condition index value
                         
        Returns:
            String interpretation of the market condition
        """
        if index_value >= 80:
            return "Very Strong"
        elif index_value >= 70:
            return "Strong"
        elif index_value >= 60:
            return "Moderately Strong"
        elif index_value >= 50:
            return "Balanced"
        elif index_value >= 40:
            return "Moderately Weak"
        elif index_value >= 30:
            return "Weak"
        else:
            return "Very Weak" 