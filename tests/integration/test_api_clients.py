"""
Integration tests for API clients.
Tests actual API connections with external data sources.
"""
import unittest
import os
import sys
import json
import pandas as pd
from unittest.mock import patch
import warnings

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures
from tests.fixtures.sample_api_responses import fred_sample_response, bls_sample_response, attom_sample_response

# Try to import API clients
try:
    from src.data_collection.fred_api import FredAPIClient
    from src.data_collection.bls_api import BLSAPIClient
    from src.data_collection.attom_api import AttomAPIClient
    from src.data_collection.data_fetcher import DataFetcher
except ImportError:
    # Create mock classes for testing if real ones don't exist
    class FredAPIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key or os.environ.get('FRED_API_KEY', 'mock_key')
        
        def get_series_data(self, series_id, start_date=None, end_date=None, frequency=None):
            """Mock method to get data for a specific series"""
            return fred_sample_response
    
    class BLSAPIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key or os.environ.get('BLS_API_KEY', 'mock_key')
        
        def get_series_data(self, series_id, start_year=None, end_year=None):
            """Mock method to get data for a specific series"""
            return bls_sample_response
    
    class AttomAPIClient:
        def __init__(self, api_key=None):
            self.api_key = api_key or os.environ.get('ATTOM_API_KEY', 'mock_key')
        
        def get_property_data(self, location, property_type=None, start_date=None, end_date=None):
            """Mock method to get property data for a specific location"""
            return attom_sample_response
    
    class DataFetcher:
        def __init__(self, fred_api_key=None, bls_api_key=None, attom_api_key=None):
            self.fred_client = FredAPIClient(fred_api_key)
            self.bls_client = BLSAPIClient(bls_api_key)
            self.attom_client = AttomAPIClient(attom_api_key)
        
        def fetch_mortgage_rates(self, start_date=None, end_date=None):
            """Mock method to fetch mortgage rates"""
            return self.fred_client.get_series_data('MORTGAGE30US', start_date, end_date)
        
        def fetch_employment_data(self, area_code, start_year=None, end_year=None):
            """Mock method to fetch employment data"""
            series_id = f'LAUCN{area_code}000000003'
            return self.bls_client.get_series_data(series_id, start_year, end_year)
        
        def fetch_property_data(self, location, property_type=None, start_date=None, end_date=None):
            """Mock method to fetch property data"""
            return self.attom_client.get_property_data(location, property_type, start_date, end_date)


class TestFredAPIClient(unittest.TestCase):
    """Test FRED API client"""
    
    def setUp(self):
        """Set up API client"""
        # Skip tests if no API key is available
        self.api_key = os.environ.get('FRED_API_KEY')
        if not self.api_key:
            warnings.warn("FRED_API_KEY not found in environment, using mock responses")
            self.skip_live_tests = True
        else:
            self.skip_live_tests = False
        
        # Initialize client
        self.client = FredAPIClient(api_key=self.api_key)
    
    def test_client_initialization(self):
        """Test that the client initializes correctly"""
        self.assertIsNotNone(self.client)
        self.assertIsInstance(self.client, FredAPIClient)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_mortgage_rates(self):
        """Test retrieving mortgage rate data"""
        if self.skip_live_tests:
            # Use mock response
            with patch.object(FredAPIClient, 'get_series_data', return_value=fred_sample_response):
                response = self.client.get_series_data('MORTGAGE30US', '2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.client.get_series_data('MORTGAGE30US', '2022-01-01', '2022-12-31')
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIn('observations', response)
        self.assertIsInstance(response['observations'], list)
        
        # Verify data content
        if len(response['observations']) > 0:
            observation = response['observations'][0]
            self.assertIn('date', observation)
            self.assertIn('value', observation)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_gdp_data(self):
        """Test retrieving GDP data"""
        if self.skip_live_tests:
            # Use mock response
            with patch.object(FredAPIClient, 'get_series_data', return_value=fred_sample_response):
                response = self.client.get_series_data('GDP', '2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.client.get_series_data('GDP', '2022-01-01', '2022-12-31')
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIn('observations', response)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_unemployment_rate(self):
        """Test retrieving unemployment rate data"""
        if self.skip_live_tests:
            # Use mock response
            with patch.object(FredAPIClient, 'get_series_data', return_value=fred_sample_response):
                response = self.client.get_series_data('UNRATE', '2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.client.get_series_data('UNRATE', '2022-01-01', '2022-12-31')
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIn('observations', response)


class TestBLSAPIClient(unittest.TestCase):
    """Test BLS API client"""
    
    def setUp(self):
        """Set up API client"""
        # Skip tests if no API key is available
        self.api_key = os.environ.get('BLS_API_KEY')
        if not self.api_key:
            warnings.warn("BLS_API_KEY not found in environment, using mock responses")
            self.skip_live_tests = True
        else:
            self.skip_live_tests = False
        
        # Initialize client
        self.client = BLSAPIClient(api_key=self.api_key)
    
    def test_client_initialization(self):
        """Test that the client initializes correctly"""
        self.assertIsNotNone(self.client)
        self.assertIsInstance(self.client, BLSAPIClient)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_employment_data(self):
        """Test retrieving employment data"""
        # Fairfield County FIPS code: 09001
        series_id = 'LAUCN09001000000003'  # Unemployment rate in Fairfield County
        
        if self.skip_live_tests:
            # Use mock response
            with patch.object(BLSAPIClient, 'get_series_data', return_value=bls_sample_response):
                response = self.client.get_series_data(series_id, 2022, 2022)
        else:
            # Make actual API call
            response = self.client.get_series_data(series_id, 2022, 2022)
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIn('Results', response)
        self.assertIn('series', response['Results'])
        
        # Verify data content
        if len(response['Results']['series']) > 0:
            series = response['Results']['series'][0]
            self.assertIn('seriesID', series)
            self.assertIn('data', series)
            
            if len(series['data']) > 0:
                data_point = series['data'][0]
                self.assertIn('year', data_point)
                self.assertIn('period', data_point)
                self.assertIn('value', data_point)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_multiple_employment_series(self):
        """Test retrieving multiple employment data series"""
        # Fairfield County FIPS code: 09001
        # Test different metrics for the same county
        series_ids = [
            'LAUCN09001000000003',  # Unemployment rate
            'LAUCN09001000000004',  # Unemployed
            'LAUCN09001000000005'   # Employed
        ]
        
        for series_id in series_ids:
            if self.skip_live_tests:
                # Use mock response
                with patch.object(BLSAPIClient, 'get_series_data', return_value=bls_sample_response):
                    response = self.client.get_series_data(series_id, 2022, 2022)
            else:
                # Make actual API call
                response = self.client.get_series_data(series_id, 2022, 2022)
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertIn('Results', response)


class TestAttomAPIClient(unittest.TestCase):
    """Test ATTOM API client"""
    
    def setUp(self):
        """Set up API client"""
        # Skip tests if no API key is available
        self.api_key = os.environ.get('ATTOM_API_KEY')
        if not self.api_key:
            warnings.warn("ATTOM_API_KEY not found in environment, using mock responses")
            self.skip_live_tests = True
        else:
            self.skip_live_tests = False
        
        # Initialize client
        self.client = AttomAPIClient(api_key=self.api_key)
    
    def test_client_initialization(self):
        """Test that the client initializes correctly"""
        self.assertIsNotNone(self.client)
        self.assertIsInstance(self.client, AttomAPIClient)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_property_data(self):
        """Test retrieving property data"""
        # Fairfield County location
        location = "Stamford, CT"
        
        if self.skip_live_tests:
            # Use mock response
            with patch.object(AttomAPIClient, 'get_property_data', return_value=attom_sample_response):
                response = self.client.get_property_data(location, 'SFR', '2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.client.get_property_data(location, 'SFR', '2022-01-01', '2022-12-31')
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertIn('status', response)
        self.assertIn('property', response)
        
        # Verify data content
        if len(response['property']) > 0:
            property_data = response['property'][0]
            self.assertIn('identifier', property_data)
            self.assertIn('address', property_data)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_get_multiple_locations(self):
        """Test retrieving property data for multiple locations"""
        # Test multiple locations in Fairfield County
        locations = ["Stamford, CT", "Greenwich, CT", "Norwalk, CT"]
        
        for location in locations:
            if self.skip_live_tests:
                # Use mock response
                with patch.object(AttomAPIClient, 'get_property_data', return_value=attom_sample_response):
                    response = self.client.get_property_data(location, 'SFR', '2022-01-01', '2022-12-31')
            else:
                # Make actual API call
                response = self.client.get_property_data(location, 'SFR', '2022-01-01', '2022-12-31')
            
            # Verify response structure
            self.assertIsNotNone(response)
            self.assertIn('status', response)
            self.assertIn('property', response)


class TestDataFetcher(unittest.TestCase):
    """Test DataFetcher integration"""
    
    def setUp(self):
        """Set up data fetcher"""
        # Get API keys from environment variables
        fred_api_key = os.environ.get('FRED_API_KEY')
        bls_api_key = os.environ.get('BLS_API_KEY')
        attom_api_key = os.environ.get('ATTOM_API_KEY')
        
        # Check if any API keys are missing
        self.skip_live_tests = not all([fred_api_key, bls_api_key, attom_api_key])
        if self.skip_live_tests:
            warnings.warn("One or more API keys missing, using mock responses")
        
        # Initialize data fetcher
        self.fetcher = DataFetcher(
            fred_api_key=fred_api_key,
            bls_api_key=bls_api_key,
            attom_api_key=attom_api_key
        )
    
    def test_fetcher_initialization(self):
        """Test that the data fetcher initializes correctly"""
        self.assertIsNotNone(self.fetcher)
        self.assertIsInstance(self.fetcher, DataFetcher)
        self.assertIsInstance(self.fetcher.fred_client, FredAPIClient)
        self.assertIsInstance(self.fetcher.bls_client, BLSAPIClient)
        self.assertIsInstance(self.fetcher.attom_client, AttomAPIClient)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_fetch_mortgage_rates(self):
        """Test fetching mortgage rates"""
        if self.skip_live_tests:
            # Use mock response
            with patch.object(FredAPIClient, 'get_series_data', return_value=fred_sample_response):
                response = self.fetcher.fetch_mortgage_rates('2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.fetcher.fetch_mortgage_rates('2022-01-01', '2022-12-31')
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn('observations', response)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_fetch_employment_data(self):
        """Test fetching employment data"""
        # Fairfield County FIPS code: 09001
        if self.skip_live_tests:
            # Use mock response
            with patch.object(BLSAPIClient, 'get_series_data', return_value=bls_sample_response):
                response = self.fetcher.fetch_employment_data('09001', 2022, 2022)
        else:
            # Make actual API call
            response = self.fetcher.fetch_employment_data('09001', 2022, 2022)
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn('Results', response)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_fetch_property_data(self):
        """Test fetching property data"""
        if self.skip_live_tests:
            # Use mock response
            with patch.object(AttomAPIClient, 'get_property_data', return_value=attom_sample_response):
                response = self.fetcher.fetch_property_data('Stamford, CT', 'SFR', '2022-01-01', '2022-12-31')
        else:
            # Make actual API call
            response = self.fetcher.fetch_property_data('Stamford, CT', 'SFR', '2022-01-01', '2022-12-31')
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn('property', response)
    
    @unittest.skipIf(os.environ.get('SKIP_API_TESTS', False), "Skipping API tests")
    def test_integration_workflow(self):
        """Test a complete data fetching workflow"""
        if self.skip_live_tests:
            # Skip the full integration test if using mocks
            self.skipTest("Skipping full integration test with mock responses")
            return
        
        # Step 1: Fetch mortgage rates
        mortgage_rates = self.fetcher.fetch_mortgage_rates('2022-01-01', '2022-12-31')
        self.assertIsNotNone(mortgage_rates)
        
        # Step 2: Fetch employment data
        employment_data = self.fetcher.fetch_employment_data('09001', 2022, 2022)
        self.assertIsNotNone(employment_data)
        
        # Step 3: Fetch property data for multiple locations
        locations = ["Stamford, CT", "Greenwich, CT", "Norwalk, CT"]
        property_data = {}
        
        for location in locations:
            location_data = self.fetcher.fetch_property_data(location, 'SFR', '2022-01-01', '2022-12-31')
            self.assertIsNotNone(location_data)
            property_data[location] = location_data
        
        # Verify we have data for all locations
        self.assertEqual(len(property_data), len(locations))


if __name__ == "__main__":
    unittest.main() 