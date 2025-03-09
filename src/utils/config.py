"""
Configuration Management

This module provides utilities for loading and managing application configuration.
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "api": {
        "fred": {
            "base_url": "https://api.stlouisfed.org/fred",
            "default_frequency": "q",
            "cache_ttl": 86400
        },
        "bls": {
            "base_url": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            "cache_ttl": 86400
        },
        "attom": {
            "base_url": "https://api.gateway.attomdata.com/propertyapi/v1.0.0",
            "cache_ttl": 86400
        },
        "openai": {
            "default_model": "gpt-4o"
        },
        "anthropic": {
            "default_model": "claude-3.7-sonnet"
        }
    },
    "data": {
        "raw_dir": "api_data/raw",
        "processed_dir": "api_data/processed",
        "synthetic_dir": "api_data/synthetic",
        "cache_dir": ".cache"
    },
    "locations": {
        "county": "Fairfield",
        "state": "CT",
        "towns": [
            "Bridgeport",
            "Danbury",
            "Darien",
            "Easton",
            "Fairfield",
            "Greenwich",
            "Monroe",
            "New Canaan",
            "New Fairfield", 
            "Newtown",
            "Norwalk",
            "Redding",
            "Ridgefield",
            "Shelton",
            "Stamford",
            "Stratford",
            "Trumbull",
            "Weston",
            "Westport",
            "Wilton"
        ]
    },
    "property_types": [
        "Single-family",
        "Multi-unit",
        "Condo/Townhouse"
    ],
    "metrics": {
        "housing": [
            "Median Sale Price",
            "Average Sale Price",
            "Sales Volume",
            "Days on Market",
            "Listing Inventory",
            "Months of Supply",
            "New Listings",
            "Absorption Rate",
            "Pending Home Sales",
            "List Price to Sales Price Ratio",
            "Home Price-to-Income Ratio",
            "Mortgage Rates",
            "Housing Affordability Index",
            "Vacancy Rates",
            "Seller Concessions"
        ],
        "economic": [
            "Local Job Growth",
            "Employment Trends",
            "Unemployment Rate",
            "Median Household Income",
            "GDP Growth",
            "Population Growth",
            "Consumer Price Index",
            "Interest Rates"
        ]
    },
    "ui": {
        "theme": {
            "primaryColor": "#4682b4",
            "backgroundColor": "#ffffff",
            "secondaryBackgroundColor": "#f5f8fa",
            "textColor": "#262730",
            "font": "sans-serif"
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "log_dir": "logs"
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use default configuration.
    
    Args:
        config_path: Path to configuration file. If None, default config is used.
        
    Returns:
        Dictionary containing configuration.
    """
    config = DEFAULT_CONFIG.copy()
    
    # Load configuration from file if provided
    if config_path is not None and os.path.exists(config_path):
        try:
            # Determine file type from extension
            ext = os.path.splitext(config_path)[1].lower()
            
            if ext == '.json':
                # Load JSON config
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
            
            elif ext in ['.yaml', '.yml']:
                # Load YAML config
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
            
            else:
                logger.warning(f"Unsupported config file format: {ext}")
                file_config = {}
            
            # Merge file config with default config
            _deep_update(config, file_config)
            
            logger.info(f"Loaded configuration from {config_path}")
        
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")
    
    return config

def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using a dot-separated path.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the configuration value (e.g., "api.fred.base_url").
        default: Default value to return if key is not found.
        
    Returns:
        Configuration value or default.
    """
    keys = key_path.split('.')
    
    # Navigate through the keys
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def _deep_update(target: Dict, source: Dict) -> Dict:
    """
    Deep update a nested dictionary with another dictionary.
    
    Args:
        target: Dictionary to update.
        source: Dictionary with new values.
        
    Returns:
        Updated dictionary.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    
    return target

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to a file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save the configuration.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Determine file type from extension
        ext = os.path.splitext(config_path)[1].lower()
        
        if ext == '.json':
            # Save as JSON
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        elif ext in ['.yaml', '.yml']:
            # Save as YAML
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        else:
            logger.error(f"Unsupported config file format: {ext}")
            return False
        
        logger.info(f"Saved configuration to {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False
