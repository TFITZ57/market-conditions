"""
Data Collection Module

This module contains API clients for collecting housing market data from various sources:
- FRED API: Federal Reserve Economic Data
- BLS API: Bureau of Labor Statistics
- ATTOM API: Property data
"""

from .fred_api import FredApiClient
from .bls_api import BlsApiClient
from .attom_api import AttomApiClient
from .data_fetcher import DataFetcher

__all__ = ['FredApiClient', 'BlsApiClient', 'AttomApiClient', 'DataFetcher']
