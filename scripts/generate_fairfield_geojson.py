#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate or download GeoJSON data for Fairfield County, CT towns.

This script obtains town boundary data for Fairfield County towns and 
saves it as a GeoJSON file for use in the application's map visualizations.
"""

import os
import json
import logging
import requests
from pathlib import Path
import geopandas as gpd
import pandas as pd
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("generate_fairfield_geojson")

# Towns in Fairfield County, CT
FAIRFIELD_TOWNS = [
    "Bridgeport", "Danbury", "Darien", "Easton", "Fairfield", 
    "Greenwich", "Monroe", "New Canaan", "New Fairfield", "Newtown",
    "Norwalk", "Redding", "Ridgefield", "Shelton", "Sherman",
    "Stamford", "Stratford", "Trumbull", "Weston", "Westport", "Wilton"
]

def download_ct_town_boundaries():
    """
    Download Connecticut town boundaries from CT Open Data.
    
    Returns:
        GeoDataFrame: GeoPandas DataFrame with town boundaries
    """
    # URL for Connecticut town boundary data from CT Open Data
    ct_town_url = "https://data.ct.gov/api/geospatial/88g8-ppbq?method=export&format=GeoJSON"
    
    try:
        logger.info("Downloading Connecticut town boundaries...")
        response = requests.get(ct_town_url)
        response.raise_for_status()
        
        # Load GeoJSON into a GeoPandas DataFrame
        gdf = gpd.read_file(response.text)
        logger.info(f"Downloaded {len(gdf)} town boundaries")
        return gdf
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download town boundaries: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing town boundaries: {e}")
        return None

def generate_simplified_geojson():
    """
    Generate a simplified GeoJSON for Fairfield County towns.
    Used as a fallback when real data cannot be obtained.
    
    Returns:
        dict: GeoJSON dictionary
    """
    logger.info("Generating simplified town boundaries...")
    
    features = []
    for i, town in enumerate(FAIRFIELD_TOWNS):
        # Create a simple square for each town as a placeholder
        x_base = (i % 5) * 0.1 - 73.8  # Base longitude near Fairfield County
        y_base = (i // 5) * 0.1 + 41.0  # Base latitude near Fairfield County
        
        coordinates = [[[
            [x_base, y_base],
            [x_base + 0.09, y_base],
            [x_base + 0.09, y_base + 0.09],
            [x_base, y_base + 0.09],
            [x_base, y_base]
        ]]]
        
        feature = {
            "type": "Feature",
            "properties": {
                "town": town,
                "name": town
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": coordinates
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return geojson

def filter_fairfield_towns(gdf):
    """
    Filter GeoDataFrame to only include Fairfield County towns.
    
    Args:
        gdf: GeoPandas DataFrame with town boundaries
        
    Returns:
        GeoDataFrame: Filtered GeoPandas DataFrame
    """
    # Filter to just Fairfield County towns
    # The column name might vary depending on the data source
    town_col = None
    for col in ['town', 'TOWN', 'NAME', 'name']:
        if col in gdf.columns:
            town_col = col
            break
    
    if town_col is None:
        logger.error("Could not find town name column in GeoJSON data")
        return None
    
    # Filter by town name
    fairfield_towns_df = gdf[gdf[town_col].isin(FAIRFIELD_TOWNS)]
    
    # If no towns were found, try case-insensitive matching
    if len(fairfield_towns_df) == 0:
        fairfield_towns_df = gdf[gdf[town_col].str.upper().isin([t.upper() for t in FAIRFIELD_TOWNS])]
    
    # Check if we found any towns
    if len(fairfield_towns_df) == 0:
        logger.warning("No Fairfield County towns found in the data")
        return None
    
    logger.info(f"Found {len(fairfield_towns_df)} Fairfield County towns")
    return fairfield_towns_df

def save_geojson(geojson_data, output_path):
    """
    Save GeoJSON data to a file.
    
    Args:
        geojson_data: GeoJSON data to save
        output_path: Path to save GeoJSON file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save GeoJSON file
        with open(output_path, 'w') as f:
            json.dump(geojson_data, f)
        
        logger.info(f"Saved GeoJSON to {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving GeoJSON: {e}")
        return False

def main():
    """Main function to generate or download GeoJSON for Fairfield County towns."""
    parser = argparse.ArgumentParser(description='Generate GeoJSON data for Fairfield County towns.')
    parser.add_argument('--output', default='assets/fairfield_towns.geojson', 
                        help='Output path for GeoJSON file')
    parser.add_argument('--force', action='store_true', 
                        help='Force regeneration of GeoJSON even if file exists')
    args = parser.parse_args()
    
    output_path = args.output
    
    # Determine absolute path
    if not os.path.isabs(output_path):
        # If relative, make it relative to project root
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / output_path
    
    # Check if output file already exists
    if os.path.exists(output_path) and not args.force:
        logger.info(f"GeoJSON file already exists at {output_path}. Use --force to regenerate.")
        return 0
    
    # Try to download real data
    gdf = download_ct_town_boundaries()
    
    if gdf is not None:
        # Filter to just Fairfield County towns
        fairfield_towns_df = filter_fairfield_towns(gdf)
        
        if fairfield_towns_df is not None:
            # Convert to GeoJSON
            geojson_data = json.loads(fairfield_towns_df.to_json())
            
            # Save GeoJSON
            if save_geojson(geojson_data, output_path):
                logger.info("Successfully created Fairfield County towns GeoJSON")
                return 0
    
    # If we get here, either download failed or filtering failed
    # Fall back to simplified GeoJSON
    logger.warning("Falling back to simplified GeoJSON")
    simplified_geojson = generate_simplified_geojson()
    
    if save_geojson(simplified_geojson, output_path):
        logger.info("Successfully created simplified Fairfield County towns GeoJSON")
        return 0
    else:
        logger.error("Failed to create GeoJSON file")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 