# preload_data.py
import os
import sys
from datetime import datetime
import pandas as pd

# Add the project root to sys.path if needed
sys.path.append('/Users/TFitz/Documents_.s/Cypress Docs/market_conditions_app/')

# Import the necessary components
from src.data_collection.data_fetcher import DataFetcher
from src.data_processing.metrics_calculator import MetricsCalculator
from src.data_processing.transformers import DataTransformer

def preload_data():
    # Initialize the components
    data_fetcher = DataFetcher()
    
    # Set parameters for data fetching
    start_date = "2015-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    location = "Fairfield County"  # or specific town
    
    # Fetch data
    print(f"Fetching data for {location} from {start_date} to {end_date}")
    metrics_data = data_fetcher.fetch_all_metrics(
        start_date=start_date,
        end_date=end_date,
        frequency="q"
    )
    
    # Print some basic stats about the fetched data
    for metric, df in metrics_data.items():
        if not df.empty:
            print(f"Fetched {len(df)} data points for {metric}")
    
    # Save the data if needed
    cache_file = data_fetcher.save_processed_data(metrics_data, prefix="preloaded")
    print(f"Data saved to {cache_file}")
    
    return metrics_data

def generate_synthetic_data():
    # Initialize with paths to store synthetic data
    data_fetcher = DataFetcher(
        synthetic_data_dir="./api_data/synthetic"
    )
    
    # Force using synthetic data
    data_fetcher.using_synthetic_data = True
    
    # Generate synthetic metrics if they don't exist
    # Check if the synthetic directory exists and has files
    if not os.path.exists(data_fetcher.synthetic_data_dir) or \
       len(os.listdir(data_fetcher.synthetic_data_dir)) == 0:
        print("Generating synthetic data...")
        
        # Generate some example data for each metric
        for metric in [
            "Median Sale Price", "Days on Market", "Months of Supply", 
            "Total Sales", "Active Listings", "New Listings"
        ]:
            # Create synthetic time series
            dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="Q")
            values = [generate_random_value_for_metric(metric, d) for d in dates]
            
            # Create DataFrame
            df = pd.DataFrame({"value": values}, index=dates)
            
            # Save to synthetic data directory
            os.makedirs(data_fetcher.synthetic_data_dir, exist_ok=True)
            file_path = os.path.join(data_fetcher.synthetic_data_dir, f"synthetic_{metric.lower().replace(' ', '_')}.csv")
            df.to_csv(file_path)
            print(f"Generated synthetic data for {metric} at {file_path}")
    else:
        print(f"Synthetic data directory already exists at {data_fetcher.synthetic_data_dir}")
        
    # Try loading the synthetic data
    metrics_data = data_fetcher.load_synthetic_data("attom")
    print(f"Loaded {len(metrics_data)} synthetic metrics")
    
    return metrics_data

def generate_random_value_for_metric(metric, date):
    """Generate a reasonable random value for a given metric"""
    import random
    import math
    
    # Base value + seasonal component + trend component + random noise
    month = date.month
    year = date.year
    base_year = 2020
    
    if metric == "Median Sale Price":
        base = 450000
        seasonal = 20000 * math.sin((month / 12) * 2 * math.pi)
        trend = 15000 * (year - base_year) + 5000 * ((year - base_year) ** 2)
        noise = random.uniform(-10000, 10000)
        return max(200000, base + seasonal + trend + noise)
    
    elif metric == "Days on Market":
        base = 45
        seasonal = 15 * math.sin((month / 12) * 2 * math.pi)
        trend = -3 * (year - base_year)
        noise = random.uniform(-5, 5)
        return max(10, base + seasonal + trend + noise)
    
    elif metric == "Total Sales":
        base = 1500
        seasonal = 500 * math.sin((month / 12) * 2 * math.pi)
        trend = 50 * (year - base_year)
        noise = random.uniform(-100, 100)
        return max(500, base + seasonal + trend + noise)
    
    # Add more metrics as needed
    else:
        return random.uniform(100, 1000)

if __name__ == "__main__":
    # Choose which function to run
    import argparse
    
    parser = argparse.ArgumentParser(description='Preload data for the market conditions app')
    parser.add_argument('--synthetic', action='store_true', help='Generate synthetic data instead of fetching real data')
    
    args = parser.parse_args()
    
    if args.synthetic:
        print("Generating synthetic data...")
        metrics_data = generate_synthetic_data()
    else:
        print("Fetching real data...")
        metrics_data = preload_data()
    
    print("Done!")