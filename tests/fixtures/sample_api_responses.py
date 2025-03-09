"""
This module contains sample API responses for testing.
These fixtures mimic the actual API responses to enable reliable and consistent testing.
"""
import json
import datetime

# FRED API Sample Responses
fred_sample_response = {
    "realtime_start": "2023-01-01",
    "realtime_end": "2023-01-01",
    "observation_start": "2015-01-01",
    "observation_end": "2023-01-01",
    "units": "lin",
    "output_type": 1,
    "file_type": "json",
    "order_by": "observation_date",
    "sort_order": "asc",
    "count": 24,
    "offset": 0,
    "limit": 100,
    "observations": [
        {
            "realtime_start": "2023-01-01",
            "realtime_end": "2023-01-01",
            "date": "2015-01-01",
            "value": "4.95"
        },
        {
            "realtime_start": "2023-01-01",
            "realtime_end": "2023-01-01",
            "date": "2015-04-01",
            "value": "4.85"
        },
        {
            "realtime_start": "2023-01-01",
            "realtime_end": "2023-01-01",
            "date": "2022-10-01",
            "value": "7.10"
        }
    ]
}

# BLS API Sample Responses
bls_sample_response = {
    "status": "REQUEST_SUCCEEDED",
    "responseTime": 120,
    "message": [],
    "Results": {
        "series": [
            {
                "seriesID": "LAUCN09001000000003",
                "data": [
                    {
                        "year": "2015",
                        "period": "M01",
                        "periodName": "January",
                        "value": "5.2",
                        "footnotes": [{}]
                    },
                    {
                        "year": "2015",
                        "period": "M02",
                        "periodName": "February",
                        "value": "5.0",
                        "footnotes": [{}]
                    },
                    {
                        "year": "2022",
                        "period": "M12",
                        "periodName": "December",
                        "value": "3.2",
                        "footnotes": [{}]
                    }
                ]
            }
        ]
    }
}

# ATTOM Property API Sample Responses
attom_sample_response = {
    "status": {
        "version": "1.0.0",
        "code": 0,
        "msg": "SuccessWithResult",
        "total": 42,
        "page": 1,
        "pagesize": 10,
        "transactionID": "3b5b7874-b5bf-4ad0-a95b-b70e123456"
    },
    "property": [
        {
            "identifier": {
                "Id": "1234567",
                "fips": "09001",
                "apn": "1234-5678-90",
                "attomId": 12345678
            },
            "lot": {
                "depth": 125,
                "width": 60,
                "lotsize1": 0.25,
                "lotsize2": 10890
            },
            "area": {
                "totalvalue": 750000,
                "bldgsize": 2200,
                "universalsize": 2200,
                "livingsize": 1950,
                "groundfloorsize": 1100,
                "grosssize": 2200,
                "bathfixtures": 8,
                "bathstotal": 2.5,
                "bedrooms": 4
            },
            "address": {
                "country": "US",
                "countrySubd": "CT",
                "line1": "123 Main St",
                "line2": "",
                "locality": "Stamford",
                "matchCode": "ExaStr",
                "oneLine": "123 Main St, Stamford, CT 06901",
                "postal1": "06901",
                "postal2": "",
                "postal3": ""
            },
            "vintage": {
                "lastModified": "2022-12-15",
                "pubDate": "2022-12-16"
            },
            "sale": {
                "saleTransDate": "2022-05-15",
                "saleDocNum": "12345",
                "saleTransType": "Resale",
                "saleRecDate": "2022-05-20",
                "saleAmt": 750000,
                "saleDisclosureType": "FULL"
            }
        }
    ]
}

# Sample mortgage rates data
mortgage_rates_sample = {
    "30yr_fixed": [
        {"date": "2015-01-01", "rate": 3.87},
        {"date": "2015-04-01", "rate": 3.67},
        {"date": "2015-07-01", "rate": 4.05},
        {"date": "2022-10-01", "rate": 6.90},
        {"date": "2022-12-01", "rate": 6.36}
    ],
    "15yr_fixed": [
        {"date": "2015-01-01", "rate": 3.09},
        {"date": "2015-04-01", "rate": 2.94},
        {"date": "2015-07-01", "rate": 3.20},
        {"date": "2022-10-01", "rate": 6.12},
        {"date": "2022-12-01", "rate": 5.65}
    ]
}

# Generate mock time series data for various housing metrics
def generate_mock_time_series(start_date='2015-01-01', end_date='2023-01-01', 
                              freq='QS', metric_name='median_sale_price',
                              start_value=500000, trend=0.01, noise=0.05):
    """
    Generate synthetic time series data with trend and noise
    
    Parameters:
    -----------
    start_date : str, starting date
    end_date : str, ending date
    freq : str, frequency ('QS' for quarter start)
    metric_name : str, name of the metric
    start_value : float, starting value of the metric
    trend : float, quarterly percentage trend
    noise : float, noise level (standard deviation)
    
    Returns:
    --------
    dict : dictionary with dates and values
    """
    import pandas as pd
    import numpy as np
    
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    periods = len(date_range)
    
    # Generate trending series with noise
    trend_factor = np.power(1 + trend, np.arange(periods))
    base_values = start_value * trend_factor
    noise_factor = np.random.normal(1, noise, periods)
    values = base_values * noise_factor
    
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in date_range],
        "values": values.tolist(),
        "metric_name": metric_name
    }

# Generate sample data for each town in Fairfield County
fairfield_county_towns = [
    "Bethel", "Bridgeport", "Brookfield", "Danbury", "Darien", 
    "Easton", "Fairfield", "Greenwich", "Monroe", "New Canaan", 
    "New Fairfield", "Newtown", "Norwalk", "Redding", "Ridgefield", 
    "Shelton", "Sherman", "Stamford", "Stratford", "Trumbull", 
    "Weston", "Westport", "Wilton"
]

town_sample_data = {}
for town in fairfield_county_towns:
    # Adjust starting values and trends to make each town unique
    base_value = 500000 + (hash(town) % 1000000) 
    trend = 0.01 + (hash(town) % 10) / 1000
    
    town_sample_data[town] = {
        "median_sale_price": generate_mock_time_series(
            metric_name="median_sale_price", 
            start_value=base_value,
            trend=trend
        ),
        "average_sale_price": generate_mock_time_series(
            metric_name="average_sale_price",
            start_value=base_value * 1.2,
            trend=trend
        ),
        "sales_volume": generate_mock_time_series(
            metric_name="sales_volume",
            start_value=50 + (hash(town) % 100),
            trend=0.005,
            noise=0.2
        ),
        "days_on_market": generate_mock_time_series(
            metric_name="days_on_market",
            start_value=30 + (hash(town) % 30),
            trend=-0.005,
            noise=0.15
        ),
        "inventory": generate_mock_time_series(
            metric_name="inventory",
            start_value=100 + (hash(town) % 200),
            trend=-0.01,
            noise=0.1
        ),
        "list_to_sale_ratio": generate_mock_time_series(
            metric_name="list_to_sale_ratio",
            start_value=0.95,
            trend=0.001,
            noise=0.02
        )
    }

# Generate raw API response files (simulated JSON)
def get_raw_api_responses():
    """Get dictionary of raw API responses for testing"""
    return {
        "fred_mortgage_rates.json": fred_sample_response,
        "bls_employment.json": bls_sample_response,
        "attom_property_data.json": attom_sample_response
    } 