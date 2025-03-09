# Fairfield County Housing Market Analysis Application

## Project Overview

This Streamlit application analyzes and visualizes housing market conditions in Fairfield County, CT, with data granularity down to specific towns. The application processes data from 2015 to present, enabling users to generate customizable visualizations and AI-powered reports for market insights.

## Application Architecture

### High-Level Architecture

```bash
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Data Collection│     │  Data Processing│     │  Visualization  │
│     Layer       │────▶│     Layer       │────▶│     Layer       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │   AI Analysis   │
                                               │     Layer       │
                                               └─────────────────┘
```

### Data Flow

1. **Data Collection Layer**: Fetches data from APIs and stores raw responses
2. **Data Processing Layer**: Cleans, normalizes, and transforms raw data
3. **Visualization Layer**: Renders interactive charts and graphs via Streamlit UI
4. **AI Analysis Layer**: Generates insights using LLMs and produces reports

## Project Structure

```bash
fairfield_housing_analysis/
├── .streamlit/
│   └── config.toml                # Streamlit configuration and theming
├── api_data/
│   ├── raw/                       # Raw API response storage
│   ├── processed/                 # Cleaned and transformed data
│   └── synthetic/                 # Synthetic data for testing
├── src/
│   ├── data_collection/           # API client implementations
│   │   ├── __init__.py
│   │   ├── fred_api.py            # Federal Reserve Economic Data API client
│   │   ├── bls_api.py             # Bureau of Labor Statistics API client
│   │   ├── attom_api.py           # ATTOM Property API client
│   │   └── data_fetcher.py        # Unified data collection orchestrator
│   ├── data_processing/           # Data transformation and analysis
│   │   ├── __init__.py
│   │   ├── cleaners.py            # Data cleaning utilities
│   │   ├── transformers.py        # Data transformation utilities
│   │   ├── metrics_calculator.py  # Derived metrics calculation
│   │   └── time_series.py         # Time series alignment utilities
│   ├── visualization/             # Chart and graph components
│   │   ├── __init__.py
│   │   ├── charts.py              # Chart generation functions
│   │   ├── maps.py                # Geographic visualization components
│   │   ├── dashboards.py          # Dashboard layout components
│   │   └── exporters.py           # Visualization export utilities
│   ├── ai_analysis/               # LLM integration for report generation
│   │   ├── __init__.py
│   │   ├── llm_clients.py         # API clients for LLM providers
│   │   ├── prompt_templates.py    # Templates for different report types
│   │   ├── report_generator.py    # Report generation pipeline
│   │   └── report_formatter.py    # Format reports for presentation/export
│   ├── utils/                     # Shared utilities
│   │   ├── __init__.py
│   │   ├── caching.py             # API response caching utilities
│   │   ├── config.py              # Configuration management
│   │   ├── logger.py              # Logging utilities
│   │   └── security.py            # API key management
│   └── ui/                        # UI components and pages
│       ├── __init__.py
│       ├── components/            # Reusable UI components
│       │   ├── __init__.py
│       │   ├── filters.py         # Filter components
│       │   ├── selectors.py       # Selection components
│       │   └── report_viewer.py   # Report viewing components
│       └── pages/                 # Application pages
│           ├── __init__.py
│           ├── overview.py        # Overview dashboard
│           ├── housing_metrics.py # Housing metrics page
│           ├── economic.py        # Economic indicators page
│           ├── comparison.py      # Comparison analysis page
│           ├── forecast.py        # Forecast models page
│           ├── analysis.py        # Data analysis page
│           └── reports.py         # Generated reports page
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── unit/                      # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data_processing.py
│   │   ├── test_visualization.py
│   │   └── test_ai_analysis.py
│   ├── integration/               # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_clients.py
│   │   └── test_report_generation.py
│   └── fixtures/                  # Test fixtures
│       ├── __init__.py
│       ├── sample_api_responses.py
│       └── sample_processed_data.py
├── scripts/                       # Utility scripts
│   ├── generate_synthetic_data.py # Script to generate test data
│   ├── setup_project.sh           # Project setup shell script
│   └── update_data.py             # Script for scheduled data updates
├── docs/                          # Documentation
│   ├── api_references.md          # API documentation
│   ├── user_guide.md              # Application user guide
│   └── development_guide.md       # Development documentation
├── .env.example                   # Example environment variables file
├── .gitignore                     # Git ignore file
├── app.py                         # Main Streamlit application entry point
├── app.yaml                       # Google App Engine configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata and build configuration
└── README.md                      # Project overview and setup instructions
```

## File Descriptions and Purposes

### Configuration Files

#### `.streamlit/config.toml`

- **Purpose**: Configures Streamlit application settings and theme
- **Contents**: Theme settings (colors, fonts), page configuration, caching settings
- **Output**: Applies styling and configuration to the Streamlit application

#### `.env.example`

- **Purpose**: Template for environment variables needed by the application
- **Contents**: API keys placeholders, configuration options
- **Output**: Serves as a guide for setting up the actual `.env` file

#### `app.yaml`

- **Purpose**: Google App Engine deployment configuration
- **Contents**: Runtime specifications, scaling settings, environment variables
- **Output**: Used by Google Cloud Platform for application deployment

#### `requirements.txt`

- **Purpose**: Lists Python package dependencies
- **Contents**: Package names and version specifications
- **Output**: Used by pip to install required dependencies

#### `pyproject.toml`

- **Purpose**: Project metadata and build configuration
- **Contents**: Project information, build settings, development dependencies
- **Output**: Used by modern Python packaging tools

### Application Core

#### `app.py`

- **Purpose**: Main application entry point
- **Contents**: Streamlit app initialization, page routing, sidebar configuration
- **Output**: Renders the main application interface and handles page navigation

### Data Collection Layer

#### `src/data_collection/fred_api.py`

- **Purpose**: Interacts with Federal Reserve Economic Data API
- **Contents**: API client class, methods for fetching economic indicators
- **Output**: Raw economic data for processing

#### `src/data_collection/bls_api.py`

- **Purpose**: Interacts with Bureau of Labor Statistics API
- **Contents**: API client class, methods for fetching employment and economic data
- **Output**: Raw employment and economic data for processing

#### `src/data_collection/attom_api.py`

- **Purpose**: Interacts with ATTOM Property API
- **Contents**: API client class, methods for fetching property data
- **Output**: Raw property data for processing

#### `src/data_collection/data_fetcher.py`

- **Purpose**: Orchestrates data collection from multiple sources
- **Contents**: Methods to coordinate API requests, caching, and data storage
- **Output**: Saves raw data to the api_data/raw/ directory

### Data Processing Layer

#### `src/data_processing/cleaners.py`

- **Purpose**: Cleans and validates raw data
- **Contents**: Functions for handling missing values, removing outliers, data validation
- **Output**: Cleaned data ready for transformation

#### `src/data_processing/transformers.py`

- **Purpose**: Transforms data into analysis-ready formats
- **Contents**: Functions for data reshaping, normalization, encoding
- **Output**: Transformed data suitable for analysis

#### `src/data_processing/metrics_calculator.py`

- **Purpose**: Calculates derived housing market metrics
- **Contents**: Functions for computing metrics like affordability index, price-to-income ratio
- **Output**: Computed metrics for visualization and analysis

#### `src/data_processing/time_series.py`

- **Purpose**: Handles time series operations
- **Contents**: Functions for resampling, alignment, seasonal adjustment
- **Output**: Time-aligned data for trend analysis and visualization

### Visualization Layer

#### `src/visualization/charts.py`

- **Purpose**: Creates various chart types for data visualization
- **Contents**: Functions for line charts, bar charts, scatter plots, etc.
- **Output**: Plotly/Matplotlib/Altair visualization objects

#### `src/visualization/maps.py`

- **Purpose**: Creates geographic visualizations
- **Contents**: Functions for chloropleth maps, geo-spatial visualizations
- **Output**: Map visualization objects

#### `src/visualization/dashboards.py`

- **Purpose**: Arranges visualizations into dashboards
- **Contents**: Functions to create multi-visualization layouts
- **Output**: Organized dashboard components

#### `src/visualization/exporters.py`

- **Purpose**: Exports visualizations to various formats
- **Contents**: Functions for saving charts as images, CSV, Excel
- **Output**: Downloadable visualization files

### AI Analysis Layer

#### `src/ai_analysis/llm_clients.py`

- **Purpose**: Interacts with LLM providers
- **Contents**: API clients for OpenAI, Anthropic
- **Output**: Raw LLM responses

#### `src/ai_analysis/prompt_templates.py`

- **Purpose**: Defines templates for different report types
- **Contents**: Template strings with placeholders for data
- **Output**: Formatted prompts for LLM requests

#### `src/ai_analysis/report_generator.py`

- **Purpose**: Generates reports using LLM responses
- **Contents**: Pipeline for collecting data, generating prompts, processing responses
- **Output**: Generated report content

#### `src/ai_analysis/report_formatter.py`

- **Purpose**: Formats reports for presentation and export
- **Contents**: Functions for formatting as HTML, DOCX, PDF
- **Output**: Formatted reports for display and download

### Utilities

#### `src/utils/caching.py`

- **Purpose**: Manages caching of API responses and processed data
- **Contents**: Caching mechanisms to minimize API calls
- **Output**: Cached data for improved performance

#### `src/utils/config.py`

- **Purpose**: Manages application configuration
- **Contents**: Configuration loading, validation, access functions
- **Output**: Validated configuration values

#### `src/utils/logger.py`

- **Purpose**: Provides application logging
- **Contents**: Logger configuration, custom logging functions
- **Output**: Structured logs for monitoring and debugging

#### `src/utils/security.py`

- **Purpose**: Handles security-sensitive operations
- **Contents**: API key management, secure storage
- **Output**: Secure access to protected resources

### UI Components

#### `src/ui/components/filters.py`

- **Purpose**: Provides UI filter components
- **Contents**: Date range selectors, location selectors, property type filters
- **Output**: Streamlit UI components for filtering data

#### `src/ui/components/selectors.py`

- **Purpose**: Provides UI selection components
- **Contents**: Metric selectors, report type selectors, LLM provider selectors
- **Output**: Streamlit UI components for selection

#### `src/ui/components/report_viewer.py`

- **Purpose**: Displays and manages reports
- **Contents**: Report viewing, editing, download components
- **Output**: Interactive report display UI

### UI Pages

#### `src/ui/pages/overview.py`

- **Purpose**: Renders overview dashboard
- **Contents**: Summary metrics, key trends, alert indicators
- **Output**: At-a-glance market summary

#### `src/ui/pages/housing_metrics.py`

- **Purpose**: Displays detailed housing metrics
- **Contents**: Visualizations for selected housing metrics, historical trends
- **Output**: Detailed housing market visualizations

#### `src/ui/pages/economic.py`

- **Purpose**: Shows economic indicators
- **Contents**: Economic data visualizations, correlation analysis
- **Output**: Economic health indicators and visualizations

#### `src/ui/pages/comparison.py`

- **Purpose**: Enables comparison between locations/metrics
- **Contents**: Side-by-side analysis tools, comparison charts
- **Output**: Multi-location, multi-metric comparisons

#### `src/ui/pages/forecast.py`

- **Purpose**: Provides forecasting tools
- **Contents**: Time series forecasting models, scenario analysis
- **Output**: Forecast visualizations with confidence intervals

#### `src/ui/pages/analysis.py`

- **Purpose**: Offers advanced data analysis tools
- **Contents**: Time series overlays, correlation tools, statistical analysis
- **Output**: Interactive analysis components

#### `src/ui/pages/reports.py`

- **Purpose**: Manages generated reports
- **Contents**: Report listing, management, generation UI
- **Output**: Report repository and management interface

### Scripts

#### `scripts/generate_synthetic_data.py`

- **Purpose**: Generates synthetic test data
- **Contents**: Functions to create realistic mock data matching API formats
- **Output**: Synthetic data files in api_data/synthetic/

#### `scripts/setup_project.sh`

- **Purpose**: Sets up the project structure
- **Contents**: Shell commands to create directories, install dependencies
- **Output**: Initialized project structure

#### `scripts/update_data.py`

- **Purpose**: Updates data from APIs
- **Contents**: Data fetching routines for scheduled updates
- **Output**: Updated data files in api_data/raw/

### Tests

#### `tests/unit/test_data_processing.py`

- **Purpose**: Tests data processing functions
- **Contents**: Unit tests for cleaners, transformers, metrics calculation
- **Output**: Test results for data processing

#### `tests/unit/test_visualization.py`

- **Purpose**: Tests visualization components
- **Contents**: Unit tests for chart generation, formatting
- **Output**: Test results for visualization

#### `tests/unit/test_ai_analysis.py`

- **Purpose**: Tests AI analysis components
- **Contents**: Unit tests for prompt generation, report formatting
- **Output**: Test results for AI analysis

#### `tests/integration/test_api_clients.py`

- **Purpose**: Tests API client integration
- **Contents**: Integration tests for API clients
- **Output**: Test results for API integration

#### `tests/integration/test_report_generation.py`

- **Purpose**: Tests end-to-end report generation
- **Contents**: Integration tests for the report generation pipeline
- **Output**: Test results for report generation

#### `tests/fixtures/sample_api_responses.py`

- **Purpose**: Provides sample API responses for testing
- **Contents**: Mock API response data
- **Output**: Test fixtures for API testing

#### `tests/fixtures/sample_processed_data.py`

- **Purpose**: Provides sample processed data for testing
- **Contents**: Mock processed data structures
- **Output**: Test fixtures for data processing testing

## Implementation Details

### Data Processing Pipeline

1. **Collection**: API clients fetch raw data and store in `api_data/raw/`
2. **Cleaning**: Raw data is cleaned using functions in `cleaners.py`
3. **Transformation**: Cleaned data is transformed using functions in `transformers.py`
4. **Metric Calculation**: Derived metrics are calculated using `metrics_calculator.py`
5. **Time Series Alignment**: Time series are aligned using `time_series.py`
6. **Storage**: Processed data is stored in `api_data/processed/`

### Visualization Generation

1. **Data Loading**: Processed data is loaded from `api_data/processed/`
2. **Filter Application**: UI filters are applied to the data
3. **Chart Generation**: Charts are generated using functions in `charts.py`
4. **Dashboard Assembly**: Charts are assembled into dashboards using `dashboards.py`
5. **Rendering**: Dashboards are rendered in the Streamlit UI

### Report Generation Pipeline

1. **Data Collection**: Relevant data is collected based on user selections
2. **Prompt Generation**: Prompts are generated using templates from `prompt_templates.py`
3. **LLM Querying**: Prompts are sent to LLMs using clients in `llm_clients.py`
4. **Response Processing**: LLM responses are processed by `report_generator.py`
5. **Formatting**: Reports are formatted by `report_formatter.py`
6. **Presentation**: Reports are presented in the UI using `report_viewer.py`

## Deployment Process

1. **Local Development**: Run with `streamlit run app.py`
2. **Testing**: Run tests with pytest
3. **Build**: Package application
4. **Deploy**: Deploy to Google App Engine using `gcloud app deploy app.yaml`

## Data Update Schedule

The application is configured to update its data quarterly using the `scripts/update_data.py` script, which can be scheduled as a cron job or cloud function.

## Security Considerations

- API keys are stored securely using environment variables
- Data is cached locally to minimize API usage
- User inputs are validated to prevent injection attacks
- Access controls are implemented for sensitive operations

## Performance Optimization

- Data is preprocessed and cached for improved UI responsiveness
- Visualizations are optimized for rendering performance
- LLM requests are batched where possible to minimize API calls
- Resource-intensive operations are performed asynchronously
