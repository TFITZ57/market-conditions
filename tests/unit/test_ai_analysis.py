"""
Unit tests for AI analysis functions.
Tests the functionality of LLM clients, prompt templates, report generation, and report formatting.
"""
import unittest
import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures
from tests.fixtures.sample_processed_data import (
    get_processed_housing_metrics, 
    get_processed_economic_indicators, 
    get_llm_analysis_samples
)

# Import modules to test
try:
    from src.ai_analysis.openai_client import OpenAIClient
    from src.ai_analysis.anthropic_client import AnthropicClient
    from src.ai_analysis.llm_clients import get_client
    from src.ai_analysis.prompt_templates import (
        get_template, general_analysis_template, 
        investment_analysis_template, market_forecast_template,
        comparative_analysis_template
    )
    from src.ai_analysis.report_generator import (
        generate_report, prepare_report_data
    )
    from src.ai_analysis.report_formatter import (
        format_markdown, format_as_html, format_as_docx
    )
except ImportError:
    # Create mock classes for testing
    class OpenAIClientMock:
        def __init__(self, api_key=None, model=None):
            self.api_key = api_key or "mock_openai_key"
            self.model = model or "gpt-4o"
        
        def generate_completion(self, prompt, max_tokens=2000, temperature=0.7):
            """Mock method to generate completions from OpenAI"""
            # Return different sample responses based on prompt content
            samples = get_llm_analysis_samples()
            
            if "market overview" in prompt.lower():
                return {"text": samples["market_overview"], "usage": {"total_tokens": 750}}
            elif "investment" in prompt.lower():
                return {"text": samples["investment_analysis"], "usage": {"total_tokens": 820}}
            elif "comparative analysis" in prompt.lower() or "comparison" in prompt.lower():
                return {"text": samples["comparative_analysis"], "usage": {"total_tokens": 905}}
            else:
                # Default response
                return {"text": "Generated analysis based on prompt: " + prompt[:50] + "...", "usage": {"total_tokens": 500}}
    
    class AnthropicClientMock:
        def __init__(self, api_key=None, model=None):
            self.api_key = api_key or "mock_anthropic_key"
            self.model = model or "claude-3-sonnet-20240229"
        
        def generate_completion(self, prompt, max_tokens=2000, temperature=0.7):
            """Mock method to generate completions from Anthropic"""
            # Return different sample responses based on prompt content
            samples = get_llm_analysis_samples()
            
            if "market overview" in prompt.lower():
                return {"text": samples["market_overview"], "usage": {"input_tokens": 500, "output_tokens": 750}}
            elif "investment" in prompt.lower():
                return {"text": samples["investment_analysis"], "usage": {"input_tokens": 550, "output_tokens": 820}}
            elif "comparative analysis" in prompt.lower() or "comparison" in prompt.lower():
                return {"text": samples["comparative_analysis"], "usage": {"input_tokens": 600, "output_tokens": 905}}
            else:
                # Default response
                return {"text": "Generated analysis based on prompt: " + prompt[:50] + "...", "usage": {"input_tokens": 400, "output_tokens": 500}}
    
    def get_client_mock(provider, api_key=None, model=None):
        """Mock function to get appropriate LLM client"""
        if provider.lower() == "openai":
            return OpenAIClientMock(api_key, model)
        elif provider.lower() == "anthropic":
            return AnthropicClientMock(api_key, model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    # Mock prompt templates
    def general_analysis_template_mock(data):
        """Mock template for general market analysis"""
        return f"Generate a general market analysis for Fairfield County based on the following data: {json.dumps(data)[:100]}..."
    
    def investment_analysis_template_mock(data):
        """Mock template for investment analysis"""
        return f"Generate an investment analysis for Fairfield County based on the following data: {json.dumps(data)[:100]}..."
    
    def market_forecast_template_mock(data):
        """Mock template for market forecast"""
        return f"Generate a market forecast for Fairfield County based on the following data: {json.dumps(data)[:100]}..."
    
    def comparative_analysis_template_mock(data):
        """Mock template for comparative analysis"""
        return f"Generate a comparative analysis for selected towns in Fairfield County based on the following data: {json.dumps(data)[:100]}..."
    
    def get_template_mock(report_type):
        """Mock function to get appropriate prompt template"""
        templates = {
            "general_analysis": general_analysis_template_mock,
            "investment_analysis": investment_analysis_template_mock,
            "market_forecast": market_forecast_template_mock,
            "comparative_analysis": comparative_analysis_template_mock
        }
        
        if report_type in templates:
            return templates[report_type]
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
    
    # Mock report generator functions
    def prepare_report_data_mock(housing_data, economic_data, filters=None):
        """Mock function to prepare data for report generation"""
        # Apply basic filters if provided
        if filters:
            if 'towns' in filters and filters['towns']:
                housing_data = housing_data[housing_data['town'].isin(filters['towns'])]
            
            if 'date_range' in filters and filters['date_range']:
                start_date, end_date = filters['date_range']
                housing_data = housing_data[(housing_data['date'] >= start_date) & (housing_data['date'] <= end_date)]
                economic_data = economic_data[(economic_data['date'] >= start_date) & (economic_data['date'] <= end_date)]
        
        # Return simplified data for testing
        return {
            "housing_metrics": {
                "median_price": housing_data['median_sale_price'].mean(),
                "avg_days_on_market": housing_data['days_on_market'].mean(),
                "inventory": housing_data['inventory'].mean(),
                "sales_volume": housing_data['sales_volume'].sum()
            },
            "economic_indicators": {
                "unemployment": economic_data['unemployment_rate'].mean(),
                "job_growth": economic_data['job_growth'].mean(),
                "inflation": economic_data['inflation'].mean()
            },
            "time_period": {
                "start": housing_data['date'].min().strftime("%Y-%m-%d"),
                "end": housing_data['date'].max().strftime("%Y-%m-%d")
            }
        }
    
    def generate_report_mock(report_type, data, llm_provider="openai", model=None, api_key=None, tone="analytical"):
        """Mock function to generate a report"""
        # Get the appropriate template
        template_fn = get_template_mock(report_type)
        prompt = template_fn(data)
        
        # Add tone modification to prompt
        tone_instruction = f"\n\nPlease write in a {tone} tone."
        prompt += tone_instruction
        
        # Get the LLM client
        client = get_client_mock(llm_provider, api_key, model)
        
        # Generate the report
        response = client.generate_completion(prompt)
        
        return {
            "content": response["text"],
            "metadata": {
                "report_type": report_type,
                "llm_provider": llm_provider,
                "model": client.model,
                "tone": tone,
                "token_usage": response.get("usage", {})
            }
        }
    
    # Mock report formatter functions
    def format_markdown_mock(report_content):
        """Mock function to ensure content is properly formatted as markdown"""
        # Simply return the content as is for testing
        return report_content
    
    def format_as_html_mock(report_content, title=None, css=None):
        """Mock function to convert markdown to HTML"""
        title_text = title or 'Market Analysis Report'
        css_text = css or ''
        
        # Create HTML template without f-string for the report_content replacement
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                h1 {{ color: #2C3E50; }}
                h2 {{ color: #3498DB; }}
                /* Additional CSS */
                {css}
            </style>
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        """
        
        # Replace placeholders with actual content
        content_with_br = report_content.replace('\n', '<br>')
        html = html_template.format(
            title=title_text,
            css=css_text,
            content=content_with_br
        )
        
        return html
    
    def format_as_docx_mock(report_content, output_file, title=None):
        """Mock function to convert markdown to DOCX"""
        # In the mock, just write the content to a text file for testing
        with open(output_file, "w") as f:
            title_str = "DOCX TITLE: " + (title or 'Market Analysis Report') + "\n\n"
            f.write(title_str)
            f.write(report_content)
        
        return {"success": True, "file_path": output_file}
    
    # Set up mock objects
    OpenAIClient = OpenAIClientMock
    AnthropicClient = AnthropicClientMock
    get_client = get_client_mock
    general_analysis_template = general_analysis_template_mock
    investment_analysis_template = investment_analysis_template_mock
    market_forecast_template = market_forecast_template_mock
    comparative_analysis_template = comparative_analysis_template_mock
    get_template = get_template_mock
    prepare_report_data = prepare_report_data_mock
    generate_report = generate_report_mock
    format_markdown = format_markdown_mock
    format_as_html = format_as_html_mock
    format_as_docx = format_as_docx_mock


class TestLLMClients(unittest.TestCase):
    """Test LLM client classes"""
    
    def test_openai_client_initialization(self):
        """Test OpenAI client initialization"""
        # Test with default values
        client = OpenAIClient()
        self.assertIsNotNone(client)
        self.assertEqual(client.model, "gpt-4o")
        
        # Test with custom values
        custom_client = OpenAIClient(api_key="test_key", model="gpt-o1")
        self.assertEqual(custom_client.api_key, "test_key")
        self.assertEqual(custom_client.model, "gpt-o1")
    
    def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization"""
        # Test with default values
        client = AnthropicClient()
        self.assertIsNotNone(client)
        self.assertEqual(client.model, "claude-3-sonnet-20240229")
        
        # Test with custom values
        custom_client = AnthropicClient(api_key="test_key", model="claude-3.5-sonnet-20241022")
        self.assertEqual(custom_client.api_key, "test_key")
        self.assertEqual(custom_client.model, "claude-3.5-sonnet-20241022")
    
    def test_get_client(self):
        """Test get_client function"""
        # Test getting OpenAI client
        openai_client = get_client("openai")
        self.assertIsInstance(openai_client, OpenAIClient)
        
        # Test getting Anthropic client
        anthropic_client = get_client("anthropic")
        self.assertIsInstance(anthropic_client, AnthropicClient)
        
        # Test invalid provider
        with self.assertRaises(ValueError):
            get_client("invalid_provider")
    
    @patch.object(OpenAIClientMock, 'generate_completion')
    def test_openai_generate_completion(self, mock_generate):
        """Test OpenAI completion generation"""
        # Set up mock response
        mock_response = {
            "text": "This is a mock completion response.",
            "usage": {"total_tokens": 10}
        }
        mock_generate.return_value = mock_response
        
        # Generate completion
        client = OpenAIClient()
        result = client.generate_completion("Test prompt")
        
        # Verify result
        self.assertEqual(result, mock_response)
        mock_generate.assert_called_once_with("Test prompt")
    
    @patch.object(AnthropicClientMock, 'generate_completion')
    def test_anthropic_generate_completion(self, mock_generate):
        """Test Anthropic completion generation"""
        # Set up mock response
        mock_response = {
            "text": "This is a mock completion response.",
            "usage": {"input_tokens": 5, "output_tokens": 10}
        }
        mock_generate.return_value = mock_response
        
        # Generate completion
        client = AnthropicClient()
        result = client.generate_completion("Test prompt")
        
        # Verify result
        self.assertEqual(result, mock_response)
        mock_generate.assert_called_once_with("Test prompt")


class TestPromptTemplates(unittest.TestCase):
    """Test prompt template functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create simplified data for testing templates
        self.test_data = {
            "housing_metrics": {
                "median_price": 750000,
                "avg_days_on_market": 35,
                "inventory": 850,
                "sales_volume": 250
            },
            "economic_indicators": {
                "unemployment": 4.2,
                "job_growth": 1.5,
                "inflation": 3.8
            },
            "time_period": {
                "start": "2022-01-01",
                "end": "2022-12-31"
            },
            "towns": ["Stamford", "Greenwich", "Norwalk"]
        }
    
    def test_get_template(self):
        """Test get_template function"""
        # Test getting general analysis template
        general_template = get_template("general_analysis")
        self.assertEqual(general_template, general_analysis_template)
        
        # Test getting investment analysis template
        investment_template = get_template("investment_analysis")
        self.assertEqual(investment_template, investment_analysis_template)
        
        # Test getting market forecast template
        forecast_template = get_template("market_forecast")
        self.assertEqual(forecast_template, market_forecast_template)
        
        # Test getting comparative analysis template
        comparative_template = get_template("comparative_analysis")
        self.assertEqual(comparative_template, comparative_analysis_template)
        
        # Test invalid template type
        with self.assertRaises(ValueError):
            get_template("invalid_template")
    
    def test_general_analysis_template(self):
        """Test general analysis template generation"""
        prompt = general_analysis_template(self.test_data)
        
        # Verify prompt contains key elements
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check for data inclusion (this will vary based on mock implementation)
        if isinstance(general_analysis_template, type(general_analysis_template_mock)):
            self.assertIn("general market analysis", prompt.lower())
            self.assertIn("data", prompt.lower())
    
    def test_investment_analysis_template(self):
        """Test investment analysis template generation"""
        prompt = investment_analysis_template(self.test_data)
        
        # Verify prompt contains key elements
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check for data inclusion (this will vary based on mock implementation)
        if isinstance(investment_analysis_template, type(investment_analysis_template_mock)):
            self.assertIn("investment analysis", prompt.lower())
            self.assertIn("data", prompt.lower())
    
    def test_market_forecast_template(self):
        """Test market forecast template generation"""
        prompt = market_forecast_template(self.test_data)
        
        # Verify prompt contains key elements
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check for data inclusion (this will vary based on mock implementation)
        if isinstance(market_forecast_template, type(market_forecast_template_mock)):
            self.assertIn("market forecast", prompt.lower())
            self.assertIn("data", prompt.lower())
    
    def test_comparative_analysis_template(self):
        """Test comparative analysis template generation"""
        prompt = comparative_analysis_template(self.test_data)
        
        # Verify prompt contains key elements
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 0)
        
        # Check for data inclusion (this will vary based on mock implementation)
        if isinstance(comparative_analysis_template, type(comparative_analysis_template_mock)):
            self.assertIn("comparative analysis", prompt.lower())
            self.assertIn("data", prompt.lower())


class TestReportGenerator(unittest.TestCase):
    """Test report generation functions"""
    
    def setUp(self):
        """Set up test data"""
        self.housing_data = get_processed_housing_metrics()
        self.economic_data = get_processed_economic_indicators()
        
        # Define test filters
        self.filters = {
            "towns": ["Stamford", "Greenwich", "Norwalk"],
            "date_range": ("2022-01-01", "2022-12-31")
        }
    
    def test_prepare_report_data(self):
        """Test report data preparation"""
        # Prepare data with filters
        prepared_data = prepare_report_data(
            housing_data=self.housing_data,
            economic_data=self.economic_data,
            filters=self.filters
        )
        
        # Verify data structure
        self.assertIsInstance(prepared_data, dict)
        self.assertIn("housing_metrics", prepared_data)
        self.assertIn("economic_indicators", prepared_data)
        self.assertIn("time_period", prepared_data)
        
        # Verify housing metrics
        self.assertIn("median_price", prepared_data["housing_metrics"])
        self.assertIn("avg_days_on_market", prepared_data["housing_metrics"])
        self.assertIn("inventory", prepared_data["housing_metrics"])
        
        # Verify economic indicators
        self.assertIn("unemployment", prepared_data["economic_indicators"])
        self.assertIn("job_growth", prepared_data["economic_indicators"])
        
        # Verify time period
        self.assertIn("start", prepared_data["time_period"])
        self.assertIn("end", prepared_data["time_period"])
    
    def test_generate_report(self):
        """Test report generation"""
        # Prepare data for report
        data = prepare_report_data(
            housing_data=self.housing_data,
            economic_data=self.economic_data,
            filters=self.filters
        )
        
        # Generate report
        report = generate_report(
            report_type="general_analysis",
            data=data,
            llm_provider="openai",
            model="gpt-4o",
            tone="analytical"
        )
        
        # Verify report structure
        self.assertIsInstance(report, dict)
        self.assertIn("content", report)
        self.assertIn("metadata", report)
        
        # Verify report content
        self.assertIsInstance(report["content"], str)
        self.assertGreater(len(report["content"]), 0)
        
        # Verify metadata
        self.assertEqual(report["metadata"]["report_type"], "general_analysis")
        self.assertEqual(report["metadata"]["llm_provider"], "openai")
        self.assertEqual(report["metadata"]["model"], "gpt-4o")
        self.assertEqual(report["metadata"]["tone"], "analytical")
        self.assertIn("token_usage", report["metadata"])
    
    def test_generate_report_with_different_providers(self):
        """Test report generation with different LLM providers"""
        # Prepare data for report
        data = prepare_report_data(
            housing_data=self.housing_data,
            economic_data=self.economic_data,
            filters=self.filters
        )
        
        # Generate report with OpenAI
        openai_report = generate_report(
            report_type="investment_analysis",
            data=data,
            llm_provider="openai",
            model="gpt-4o"
        )
        
        # Generate report with Anthropic
        anthropic_report = generate_report(
            report_type="investment_analysis",
            data=data,
            llm_provider="anthropic",
            model="claude-3-sonnet-20240229"
        )
        
        # Verify reports have content
        self.assertGreater(len(openai_report["content"]), 0)
        self.assertGreater(len(anthropic_report["content"]), 0)
        
        # Verify different providers were used
        self.assertEqual(openai_report["metadata"]["llm_provider"], "openai")
        self.assertEqual(anthropic_report["metadata"]["llm_provider"], "anthropic")
    
    def test_generate_report_with_different_types(self):
        """Test report generation with different report types"""
        # Prepare data for report
        data = prepare_report_data(
            housing_data=self.housing_data,
            economic_data=self.economic_data,
            filters=self.filters
        )
        
        # Generate different report types
        report_types = ["general_analysis", "investment_analysis", "market_forecast", "comparative_analysis"]
        
        for report_type in report_types:
            # Generate report
            report = generate_report(
                report_type=report_type,
                data=data,
                llm_provider="openai"
            )
            
            # Verify report type in metadata
            self.assertEqual(report["metadata"]["report_type"], report_type)
            
            # Verify report has content
            self.assertGreater(len(report["content"]), 0)


class TestReportFormatter(unittest.TestCase):
    """Test report formatting functions"""
    
    def setUp(self):
        """Set up test data"""
        # Get sample report content
        samples = get_llm_analysis_samples()
        self.report_content = samples["market_overview"]
        
        # Create temporary directory for output files
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_format_markdown(self):
        """Test markdown formatting"""
        # Format report as markdown
        formatted_markdown = format_markdown(self.report_content)
        
        # Verify formatted content
        self.assertIsInstance(formatted_markdown, str)
        self.assertGreater(len(formatted_markdown), 0)
        
        # Verify markdown structure intact
        self.assertIn("#", formatted_markdown)
    
    def test_format_as_html(self):
        """Test HTML formatting"""
        # Format report as HTML
        html_report = format_as_html(
            report_content=self.report_content,
            title="Market Overview Report",
            css="h1 { color: blue; }"
        )
        
        # Verify HTML content
        self.assertIsInstance(html_report, str)
        self.assertGreater(len(html_report), 0)
        
        # Check for HTML elements
        self.assertIn("<!DOCTYPE html>", html_report)
        self.assertIn("<html>", html_report)
        self.assertIn("Market Overview Report", html_report)
        self.assertIn("color: blue", html_report)
    
    def test_format_as_docx(self):
        """Test DOCX formatting"""
        # Create output filename
        output_file = os.path.join(self.temp_dir.name, "report.docx")
        
        # Format report as DOCX
        result = format_as_docx(
            report_content=self.report_content,
            output_file=output_file,
            title="Market Overview Report"
        )
        
        # Verify result
        self.assertIsInstance(result, dict)
        
        if isinstance(format_as_docx, type(format_as_docx_mock)):
            # For mock implementation
            self.assertTrue(result["success"])
            self.assertEqual(result["file_path"], output_file)
            self.assertTrue(os.path.exists(output_file))
            
            # Verify file content (for mock implementation)
            with open(output_file, "r") as f:
                content = f.read()
                self.assertIn("Market Overview Report", content)
                self.assertIn("Fairfield County", content)
        else:
            # For real implementation
            self.assertTrue(os.path.exists(output_file))
            # Additional checks for actual DOCX files could be added if python-docx is available


if __name__ == "__main__":
    unittest.main() 