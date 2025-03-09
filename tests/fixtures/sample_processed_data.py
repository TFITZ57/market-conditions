"""
This module contains sample processed data for testing.
These fixtures represent processed data after raw API responses have been transformed.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json

def get_processed_mortgage_rates():
    """Get processed mortgage rate data for testing"""
    dates = pd.date_range(start='2015-01-01', end='2023-01-01', freq='QS')
    
    rates_30yr = [3.87, 3.67, 4.05, 3.95, 3.85, 3.97, 4.20, 4.30,
                  4.55, 4.28, 4.08, 3.72, 3.69, 3.59, 3.67, 2.88,
                  2.85, 2.95, 3.10, 5.55, 6.90, 6.36]
    
    rates_15yr = [3.09, 2.94, 3.20, 3.15, 3.05, 3.17, 3.60, 3.70,
                  3.85, 3.55, 3.38, 3.16, 3.15, 3.05, 3.08, 2.40,
                  2.25, 2.35, 2.50, 4.75, 6.12, 5.65]
    
    # Pad the lists with NaN if necessary
    if len(rates_30yr) < len(dates):
        rates_30yr += [np.nan] * (len(dates) - len(rates_30yr))
    if len(rates_15yr) < len(dates):
        rates_15yr += [np.nan] * (len(dates) - len(rates_15yr))
    
    df = pd.DataFrame({
        'date': dates,
        '30yr_fixed': rates_30yr[:len(dates)],
        '15yr_fixed': rates_15yr[:len(dates)]
    })
    
    return df

def get_processed_housing_metrics(town=None):
    """
    Get processed housing metrics data for testing
    
    Parameters:
    -----------
    town : str or None
        If provided, return data for specific town
        If None, return data for all towns
    
    Returns:
    --------
    DataFrame : processed housing metrics
    """
    
    towns = [
        "Bethel", "Bridgeport", "Brookfield", "Danbury", "Darien", 
        "Easton", "Fairfield", "Greenwich", "Monroe", "New Canaan", 
        "New Fairfield", "Newtown", "Norwalk", "Redding", "Ridgefield", 
        "Shelton", "Sherman", "Stamford", "Stratford", "Trumbull", 
        "Weston", "Westport", "Wilton"
    ]
    
    if town is not None and town not in towns:
        raise ValueError(f"Town {town} not found in sample data")
    
    # If a specific town is requested, filter towns list
    if town is not None:
        towns = [town]
    
    # Create date range
    dates = pd.date_range(start='2015-01-01', end='2023-01-01', freq='QS')
    
    # Create empty dataframe
    data = []
    
    # Generate data for each town
    for t in towns:
        # Set seed based on town name for reproducibility
        np.random.seed(hash(t) % 10000)
        
        # Base values - different for each town
        base_median_price = 500000 + (hash(t) % 1000000)
        base_average_price = base_median_price * 1.2
        base_sales = 50 + (hash(t) % 100)
        base_dom = 30 + (hash(t) % 30)
        base_inventory = 100 + (hash(t) % 200)
        
        # Generate trending data with noise for each quarter
        for i, date in enumerate(dates):
            # Apply trends
            trend_factor = 1.01 ** i  # 1% quarterly growth
            
            median_price = base_median_price * trend_factor * np.random.normal(1, 0.05)
            average_price = base_average_price * trend_factor * np.random.normal(1, 0.05)
            
            # Sales volume - seasonal pattern
            season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (i % 4) / 4)
            sales = base_sales * trend_factor * season_factor * np.random.normal(1, 0.2)
            
            # Days on market - slight downward trend
            dom_trend = 0.995 ** i  # 0.5% quarterly decline
            dom = base_dom * dom_trend * np.random.normal(1, 0.15)
            
            # Inventory - downward trend
            inv_trend = 0.99 ** i  # 1% quarterly decline
            inventory = base_inventory * inv_trend * np.random.normal(1, 0.1)
            
            # List-to-sale ratio - slight upward trend
            ratio_trend = 1.001 ** i  # 0.1% quarterly growth
            ratio = 0.95 * ratio_trend * np.random.normal(1, 0.02)
            ratio = min(ratio, 1.05)  # Cap at 105%
            
            # Months of supply
            months_supply = inventory / (sales / 3)
            
            # Home price-to-income ratio
            median_income = 95000 + (hash(t) % 80000)
            price_to_income = median_price / median_income
            
            # Affordability index (inverted price-to-income ratio, scaled)
            affordability_index = (100 / price_to_income) * 3
            
            # New listings
            new_listings = sales * np.random.uniform(1.1, 1.5)
            
            # Absorption rate
            absorption_rate = sales / inventory if inventory > 0 else np.nan
            
            data.append({
                'date': date,
                'town': t,
                'median_sale_price': median_price,
                'average_sale_price': average_price,
                'sales_volume': sales,
                'days_on_market': dom,
                'inventory': inventory,
                'list_to_sale_ratio': ratio,
                'months_supply': months_supply,
                'price_to_income_ratio': price_to_income,
                'affordability_index': affordability_index,
                'new_listings': new_listings,
                'absorption_rate': absorption_rate
            })
    
    return pd.DataFrame(data)

def get_processed_economic_indicators():
    """Get processed economic indicators for testing"""
    dates = pd.date_range(start='2015-01-01', end='2023-01-01', freq='QS')
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Base values
    base_unemployment = 5.2
    base_job_growth = 1.5
    base_gdp_growth = 2.0
    base_inflation = 1.8
    
    data = []
    
    for i, date in enumerate(dates):
        # Unemployment rate - general downward trend, with COVID spike
        if date >= pd.Timestamp('2020-04-01') and date <= pd.Timestamp('2020-10-01'):
            unemployment = base_unemployment * 2.5 * np.random.normal(1, 0.05)
        else:
            unemployment_trend = 0.98 ** i  # 2% quarterly decline
            unemployment = base_unemployment * unemployment_trend * np.random.normal(1, 0.03)
        
        # Job growth - positive with COVID dip
        if date >= pd.Timestamp('2020-04-01') and date <= pd.Timestamp('2020-10-01'):
            job_growth = base_job_growth * -3 * np.random.normal(1, 0.2)
        else:
            job_growth = base_job_growth * np.random.normal(1, 0.2)
        
        # GDP growth - positive with COVID dip
        if date >= pd.Timestamp('2020-04-01') and date <= pd.Timestamp('2020-07-01'):
            gdp_growth = base_gdp_growth * -4 * np.random.normal(1, 0.3)
        else:
            gdp_growth = base_gdp_growth * np.random.normal(1, 0.3)
        
        # Inflation - low then high after 2021
        if date >= pd.Timestamp('2021-07-01'):
            inflation = base_inflation * 3 * np.random.normal(1, 0.2)
        else:
            inflation = base_inflation * np.random.normal(1, 0.1)
        
        data.append({
            'date': date,
            'unemployment_rate': unemployment,
            'job_growth': job_growth,
            'gdp_growth': gdp_growth,
            'inflation': inflation
        })
    
    return pd.DataFrame(data)

def get_forecasted_data():
    """Get sample forecasted data for testing"""
    # Start with actual data
    housing_df = get_processed_housing_metrics(town="Stamford")
    
    # Filter to the most recent data
    actual_data = housing_df[housing_df['date'] <= '2023-01-01'].copy()
    
    # Generate forecast data
    forecast_dates = pd.date_range(start='2023-04-01', end='2024-01-01', freq='QS')
    
    # For each metric, create a simple forecast with increasing uncertainty
    forecast_data = []
    
    last_row = actual_data.iloc[-1]
    metrics = ['median_sale_price', 'average_sale_price', 'sales_volume', 
               'days_on_market', 'inventory', 'list_to_sale_ratio']
    
    for i, date in enumerate(forecast_dates):
        row = {'date': date, 'town': 'Stamford', 'forecast': True}
        
        # Increasing uncertainty bounds
        uncertainty = 0.05 * (i + 1)
        
        for metric in metrics:
            # Simple trend forecast (last value + small trend)
            last_value = last_row[metric]
            trend = 0.02  # 2% quarterly increase
            forecasted_value = last_value * (1 + trend) ** (i + 1)
            
            # Add to the data
            row[metric] = forecasted_value
            row[f'{metric}_lower'] = forecasted_value * (1 - uncertainty)
            row[f'{metric}_upper'] = forecasted_value * (1 + uncertainty)
        
        forecast_data.append(row)
    
    # Combine actual and forecast
    forecast_df = pd.DataFrame(forecast_data)
    actual_data['forecast'] = False
    
    # Return combined data
    return pd.concat([actual_data, forecast_df], ignore_index=True)

def get_llm_analysis_samples():
    """Get sample LLM analysis results for testing"""
    return {
        "market_overview": """
        # Fairfield County Housing Market Overview: Q4 2022
        
        ## Market Summary
        
        The Fairfield County housing market showed signs of cooling in Q4 2022, with median sale prices declining 2.3% from the previous quarter but remaining 5.7% higher year-over-year. This cooling trend follows the national pattern as higher mortgage rates impact buyer demand.
        
        ## Key Metrics:
        
        - Median Sale Price: $750,000 (-2.3% QoQ, +5.7% YoY)
        - Average Days on Market: 35 days (+40% QoQ)
        - Inventory Levels: 850 listings (+15% QoQ)
        - Months of Supply: 3.2 (+33% QoQ)
        
        ## Geographic Trends:
        
        Higher-priced coastal towns (Greenwich, Darien, Westport) have shown greater resilience, with smaller price adjustments compared to inland communities. Stamford continues to demonstrate strength in the condominium market segment.
        
        ## Outlook:
        
        The market is expected to stabilize in early 2023 as buyers adjust to the higher interest rate environment. While price growth will likely moderate, limited inventory should prevent significant price declines in desirable communities.
        """,
        
        "investment_analysis": """
        # Investment Opportunities in Fairfield County: Q1 2023
        
        ## Investment Climate
        
        The current market presents selective opportunities for investors as the market transitions from the highly competitive seller's market of 2020-2021. Rising interest rates have reduced competition from first-time homebuyers, creating potential entry points for cash buyers and long-term investors.
        
        ## Promising Areas:
        
        1. **Norwalk**: Showing value with median prices 20% below county average while benefiting from strong rental demand and ongoing development.
        
        2. **Stratford**: Entry-level price points with improving infrastructure and proximity to major employers.
        
        3. **Eastern Stamford**: Continued redevelopment and strong rental yields for multi-family properties.
        
        ## Investment Strategies:
        
        - Multi-family properties remain attractive with rental demand increasing as purchasing power decreases.
        - Value-add opportunities are emerging as marketing times lengthen.
        - Long-term appreciation potential remains strongest in towns with excellent school districts and commuting access.
        
        ## Risk Factors:
        
        Property tax reassessments and rising insurance costs require careful consideration in cash flow projections.
        """,
        
        "comparative_analysis": """
        # Comparative Analysis: Greenwich vs. Stamford Markets
        
        ## Market Positioning
        
        Greenwich and Stamford represent distinct market segments within Fairfield County, with Greenwich positioned as a luxury market and Stamford offering greater diversity in housing stock and price points.
        
        ## Price Metrics (Q4 2022):
        
        | Metric | Greenwich | Stamford | Difference |
        |--------|-----------|----------|------------|
        | Median Price | $2,150,000 | $675,000 | 219% higher |
        | Price/Sq Ft | $745 | $375 | 99% higher |
        | Days on Market | 95 | 40 | 138% longer |
        
        ## Market Dynamics:
        
        Greenwich has experienced more significant price adjustments in the $3M+ segment, with a 12% increase in inventory. Stamford has maintained better absorption rates, particularly in properties under $1M.
        
        ## Investment Considerations:
        
        Stamford offers superior rental yields (4.5% vs 2.8% in Greenwich) but Greenwich has demonstrated superior long-term appreciation (+45% vs +35% over 5 years).
        
        ## Outlook:
        
        Both markets will likely stabilize by mid-2023, with Greenwich potentially experiencing more significant price adjustments in the luxury segment while Stamford's diverse housing stock provides more consistent performance across market cycles.
        """
    }

def get_chart_data_samples():
    """Get sample chart data for testing visualization components"""
    
    # Time series data
    time_series = {
        "dates": pd.date_range(start='2015-01-01', end='2023-01-01', freq='QS'),
        "median_prices": {
            "Greenwich": [1500000, 1525000, 1575000, 1600000, 1625000, 1650000, 
                          1675000, 1700000, 1750000, 1775000, 1800000, 1850000,
                          1900000, 1950000, 2000000, 2100000, 2175000, 2200000,
                          2150000, 2175000, 2225000, 2250000, 2275000, 2300000,
                          2350000, 2400000, 2450000, 2475000, 2500000, 2525000,
                          2550000, 2150000],
            "Stamford": [550000, 555000, 560000, 565000, 570000, 575000,
                         580000, 585000, 590000, 595000, 600000, 605000,
                         610000, 615000, 620000, 625000, 635000, 645000,
                         655000, 665000, 675000, 685000, 695000, 700000,
                         705000, 710000, 715000, 710000, 705000, 695000,
                         685000, 675000],
            "Norwalk": [425000, 430000, 435000, 440000, 445000, 450000,
                        455000, 460000, 465000, 470000, 475000, 480000,
                        485000, 490000, 495000, 500000, 510000, 520000,
                        530000, 540000, 550000, 560000, 570000, 575000,
                        580000, 585000, 590000, 585000, 580000, 575000,
                        570000, 565000]
        }
    }
    
    # Correlation matrix
    metrics = ['median_sale_price', 'sales_volume', 'days_on_market', 
               'inventory', 'list_to_sale_ratio', 'months_supply']
    
    np.random.seed(42)  # For reproducibility
    corr_matrix = np.random.uniform(-1, 1, size=(len(metrics), len(metrics)))
    
    # Ensure it's symmetric and has 1s on diagonal
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(corr_matrix, 1)  # Set diagonal to 1
    
    # Clamp values to [-1, 1]
    corr_matrix = np.clip(corr_matrix, -1, 1)
    
    correlation_data = {
        'metrics': metrics,
        'matrix': corr_matrix
    }
    
    # Geographic data (choropleth)
    geo_data = {
        'town': fairfield_county_towns,
        'median_price': [np.random.uniform(400000, 2500000) for _ in range(len(fairfield_county_towns))],
        'price_change': [np.random.uniform(-0.1, 0.2) for _ in range(len(fairfield_county_towns))]
    }
    
    return {
        "time_series": time_series,
        "correlation": correlation_data,
        "geographic": geo_data
    }

# Helper to get fairfield county towns list
fairfield_county_towns = [
    "Bethel", "Bridgeport", "Brookfield", "Danbury", "Darien", 
    "Easton", "Fairfield", "Greenwich", "Monroe", "New Canaan", 
    "New Fairfield", "Newtown", "Norwalk", "Redding", "Ridgefield", 
    "Shelton", "Sherman", "Stamford", "Stratford", "Trumbull", 
    "Weston", "Westport", "Wilton"
] 