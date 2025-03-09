#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic data for testing the Fairfield County Housing Market Analysis application.
This script creates realistic mock data matching API formats from FRED, BLS, and ATTOM Property API.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path

# Base directory for data storage
BASE_DIR = Path(__file__).resolve().parent.parent / "api_data"
RAW_DIR = BASE_DIR / "raw"
SYNTHETIC_DIR = BASE_DIR / "synthetic"

# Ensure directories exist
for directory in [RAW_DIR, SYNTHETIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# List of Fairfield County towns
FAIRFIELD_TOWNS = [
    "Bridgeport", "Danbury", "Darien", "Easton", "Fairfield", 
    "Greenwich", "Monroe", "New Canaan", "New Fairfield", "Newtown",
    "Norwalk", "Redding", "Ridgefield", "Shelton", "Sherman",
    "Stamford", "Stratford", "Trumbull", "Weston", "Westport", "Wilton"
]

# Property types
PROPERTY_TYPES = ["Single-family", "Multi-unit", "Condo/Townhouse"]

# Time range
START_DATE = datetime(2015, 1, 1)
END_DATE = datetime.now()


def generate_time_series(start_date, end_date, frequency='Q', trend=0, seasonality=0, noise=1.0, base_value=100):
    """Generate a synthetic time series with trend, seasonality, and noise components."""
    date_range = pd.date_range(start=start_date, end=end_date, freq=frequency)
    n = len(date_range)
    
    # Generate trend component
    trend_component = np.linspace(0, n * trend, n)
    
    # Generate seasonality component (using sine wave for simplicity)
    if frequency == 'Q':
        periods_per_year = 4
    elif frequency == 'M':
        periods_per_year = 12
    else:
        periods_per_year = 1
    
    t = np.arange(n)
    seasonality_component = seasonality * np.sin(2 * np.pi * t / periods_per_year)
    
    # Generate noise component
    noise_component = np.random.normal(0, noise, n)
    
    # Combine components and add base value
    values = base_value + trend_component + seasonality_component + noise_component
    
    # Ensure no negative values for certain metrics
    values = np.maximum(values, 0)
    
    return pd.Series(values, index=date_range)


def generate_fred_data():
    """Generate synthetic FRED API data for economic indicators."""
    print("Generating FRED API synthetic data...")
    
    fred_indicators = {
        "MORTGAGE30US": {"name": "30-Year Fixed Rate Mortgage Average", "units": "Percent", "base": 4.0, "trend": 0.05, "seasonality": 0.1, "noise": 0.2},
        "MEDLISPRI_CT": {"name": "Median Listing Price of Homes in Connecticut", "units": "USD", "base": 400000, "trend": 1000, "seasonality": 5000, "noise": 10000},
        "CTYHOUVACRT_CT": {"name": "Housing Vacancies in Connecticut", "units": "Percent", "base": 5.0, "trend": -0.01, "seasonality": 0.5, "noise": 0.3},
        "CTGSP": {"name": "Connecticut Gross State Product", "units": "USD Million", "base": 200000, "trend": 2000, "seasonality": 1000, "noise": 3000},
        "CTHUR": {"name": "Connecticut Housing Price Index (All-Transactions)", "units": "Index 1980:Q1=100", "base": 300, "trend": 2, "seasonality": 5, "noise": 3},
    }
    
    for series_id, params in fred_indicators.items():
        # Generate time series data
        data = generate_time_series(
            START_DATE, END_DATE, 'Q', 
            trend=params["trend"], 
            seasonality=params["seasonality"], 
            noise=params["noise"],
            base_value=params["base"]
        )
        
        # Format data as FRED API response
        observations = []
        for date, value in data.items():
            observations.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": str(round(value, 2))
            })
        
        fred_response = {
            "series_id": series_id,
            "name": params["name"],
            "units": params["units"],
            "observation_start": START_DATE.strftime("%Y-%m-%d"),
            "observation_end": END_DATE.strftime("%Y-%m-%d"),
            "observations": observations
        }
        
        # Save to file
        filename = SYNTHETIC_DIR / f"fred_{series_id}.json"
        with open(filename, 'w') as f:
            json.dump(fred_response, f, indent=2)
        
        print(f"Generated {filename}")


def generate_bls_data():
    """Generate synthetic BLS API data for employment and economic indicators."""
    print("Generating BLS API synthetic data...")
    
    # Generate employment data for Fairfield County
    employment_data = {}
    
    # Generate time series for different employment metrics
    metrics = {
        "Total_Employment": {"base": 500000, "trend": 1000, "seasonality": 5000, "noise": 2000},
        "Unemployment_Rate": {"base": 4.5, "trend": -0.02, "seasonality": 0.5, "noise": 0.3},
        "Labor_Force_Participation": {"base": 65.0, "trend": -0.01, "seasonality": 0.2, "noise": 0.4},
        "Construction_Employment": {"base": 25000, "trend": 100, "seasonality": 500, "noise": 300},
        "Real_Estate_Employment": {"base": 15000, "trend": 50, "seasonality": 200, "noise": 150}
    }
    
    for metric, params in metrics.items():
        # Generate county-wide data
        county_data = generate_time_series(
            START_DATE, END_DATE, 'M', 
            trend=params["trend"], 
            seasonality=params["seasonality"], 
            noise=params["noise"],
            base_value=params["base"]
        )
        
        # Format as a BLS-like response
        series_id = f"LAUCN0900100000000{hash(metric) % 10000:04d}"
        
        observations = []
        for date, value in county_data.items():
            observations.append({
                "year": date.year,
                "period": f"M{date.month:02d}",
                "periodName": date.strftime("%B"),
                "value": str(round(value, 2)),
                "footnotes": []
            })
        
        # Structure similar to BLS API response
        bls_response = {
            "status": "REQUEST_SUCCEEDED",
            "responseTime": 352,
            "Results": {
                "series": [{
                    "seriesID": series_id,
                    "data": observations
                }]
            }
        }
        
        # Save to file
        filename = SYNTHETIC_DIR / f"bls_{metric}.json"
        with open(filename, 'w') as f:
            json.dump(bls_response, f, indent=2)
        
        print(f"Generated {filename}")


def generate_attom_data():
    """Generate synthetic ATTOM Property API data for housing metrics by town."""
    print("Generating ATTOM Property API synthetic data...")
    
    # Housing metrics to generate
    metrics = {
        "MedianSalePrice": {"base": 500000, "trend": 2000, "seasonality": 10000, "noise": 20000},
        "AverageSalePrice": {"base": 600000, "trend": 2500, "seasonality": 15000, "noise": 25000},
        "ListingInventory": {"base": 500, "trend": -2, "seasonality": 50, "noise": 30},
        "MonthsOfSupply": {"base": 3.5, "trend": -0.02, "seasonality": 0.5, "noise": 0.2},
        "NewListings": {"base": 200, "trend": -0.5, "seasonality": 40, "noise": 15},
        "SalesVolume": {"base": 150, "trend": -0.3, "seasonality": 30, "noise": 20},
        "DaysOnMarket": {"base": 45, "trend": -0.1, "seasonality": 10, "noise": 5},
        "AbsorptionRate": {"base": 25, "trend": 0.1, "seasonality": 5, "noise": 3},
        "PendingHomeSales": {"base": 100, "trend": -0.2, "seasonality": 20, "noise": 10},
        "ListToSaleRatio": {"base": 98, "trend": 0.01, "seasonality": 1, "noise": 2},
        "PriceToIncomeRatio": {"base": 4.5, "trend": 0.02, "seasonality": 0.1, "noise": 0.2},
        "SellerConcessions": {"base": 5000, "trend": 10, "seasonality": 1000, "noise": 2000}
    }
    
    # Generate data for each town and property type
    for town in FAIRFIELD_TOWNS:
        town_data = {}
        
        for property_type in PROPERTY_TYPES:
            property_data = {}
            
            for metric, params in metrics.items():
                # Adjust base values slightly for each town and property type
                town_factor = 0.7 + (hash(town) % 10) / 10.0  # Random factor between 0.7-1.6
                property_factor = 0.8 + (hash(property_type) % 5) / 10.0  # Random factor between 0.8-1.3
                
                adjusted_base = params["base"] * town_factor * property_factor
                
                # Generate time series
                data = generate_time_series(
                    START_DATE, END_DATE, 'Q', 
                    trend=params["trend"], 
                    seasonality=params["seasonality"], 
                    noise=params["noise"],
                    base_value=adjusted_base
                )
                
                # Format data points
                data_points = []
                for date, value in data.items():
                    data_points.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "value": round(value, 2)
                    })
                
                property_data[metric] = data_points
            
            town_data[property_type] = property_data
        
        # Save town data
        filename = SYNTHETIC_DIR / f"attom_{town.lower().replace(' ', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(town_data, f, indent=2)
        
        print(f"Generated {filename}")


def generate_housing_starts_data():
    """Generate synthetic housing starts/residential construction data."""
    print("Generating housing starts data...")
    
    housing_starts = {}
    
    for town in FAIRFIELD_TOWNS:
        # Base values vary by town size
        town_size_factor = 0.5 + (hash(town) % 15) / 10.0  # Random factor between 0.5-2.0
        
        # Generate time series for single-family, multi-unit, and total
        town_data = {}
        
        for category in ["Single-family", "Multi-unit", "Total"]:
            if category == "Single-family":
                base = int(25 * town_size_factor)
                trend = -0.05
                noise = 5
            elif category == "Multi-unit":
                base = int(10 * town_size_factor)
                trend = 0.1
                noise = 3
            else:  # Total
                continue  # We'll calculate this after
            
            # Generate time series
            data = generate_time_series(
                START_DATE, END_DATE, 'Q', 
                trend=trend, 
                seasonality=2, 
                noise=noise,
                base_value=base
            )
            
            # Round to integers (housing units)
            data = data.round().astype(int)
            
            # Format data points
            data_points = []
            for date, value in data.items():
                data_points.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "value": int(value)
                })
            
            town_data[category] = data_points
        
        # Calculate total as sum of single-family and multi-unit
        total_data_points = []
        for i in range(len(town_data["Single-family"])):
            date = town_data["Single-family"][i]["date"]
            single_value = town_data["Single-family"][i]["value"]
            multi_value = town_data["Multi-unit"][i]["value"]
            
            total_data_points.append({
                "date": date,
                "value": single_value + multi_value
            })
        
        town_data["Total"] = total_data_points
        
        housing_starts[town] = town_data
    
    # Save housing starts data
    filename = SYNTHETIC_DIR / "housing_starts.json"
    with open(filename, 'w') as f:
        json.dump(housing_starts, f, indent=2)
    
    print(f"Generated {filename}")


def generate_affordability_index_data():
    """Generate synthetic housing affordability index data."""
    print("Generating affordability index data...")
    
    affordability_data = {}
    
    for town in FAIRFIELD_TOWNS:
        # Town affordability factor - higher values mean less affordable
        town_factor = 0.8 + (hash(town) % 15) / 10.0  # Random factor between 0.8-2.3
        
        # Generate time series
        data = generate_time_series(
            START_DATE, END_DATE, 'Q', 
            trend=-0.5,  # Declining affordability over time
            seasonality=2,
            noise=1,
            base_value=100 / town_factor  # Higher value = more affordable
        )
        
        # Format data points
        data_points = []
        for date, value in data.items():
            data_points.append({
                "date": date.strftime("%Y-%m-%d"),
                "value": round(value, 2)
            })
        
        affordability_data[town] = data_points
    
    # Save affordability index data
    filename = SYNTHETIC_DIR / "affordability_index.json"
    with open(filename, 'w') as f:
        json.dump(affordability_data, f, indent=2)
    
    print(f"Generated {filename}")


def generate_all_data():
    """Generate all synthetic datasets."""
    generate_fred_data()
    generate_bls_data()
    generate_attom_data()
    generate_housing_starts_data()
    generate_affordability_index_data()
    
    print("\nSynthetic data generation complete. Data saved to:", SYNTHETIC_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for the housing market analysis application")
    parser.add_argument("--dataset", choices=["fred", "bls", "attom", "housing-starts", "affordability", "all"], 
                        default="all", help="Specify which dataset to generate (default: all)")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        generate_all_data()
    elif args.dataset == "fred":
        generate_fred_data()
    elif args.dataset == "bls":
        generate_bls_data()
    elif args.dataset == "attom":
        generate_attom_data()
    elif args.dataset == "housing-starts":
        generate_housing_starts_data()
    elif args.dataset == "affordability":
        generate_affordability_index_data() 