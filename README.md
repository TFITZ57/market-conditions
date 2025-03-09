# Fairfield County Housing Market Analysis

An interactive Streamlit application that analyzes and visualizes housing market conditions in Fairfield County, CT, with granular data down to specific towns.

## Features

- Comprehensive data analysis of 17 key housing market indicators
- Granular data visualization down to specific towns
- Time series analysis from 2015 to present
- AI-powered report generation and insights with export to PDF and DOCX
- Customizable visualizations and comparison tools
- Market forecasting and trend analysis
- Geographic visualizations with town-level data

## Getting Started

### Prerequisites

- Python 3.9 or later
- API keys for:
  - FRED (Federal Reserve Economic Data)
  - BLS (Bureau of Labor Statistics)
  - ATTOM Property Data
  - OpenAI and/or Anthropic (for AI-powered reports)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/fairfield-housing-analysis.git
   cd fairfield-housing-analysis
   ```

2. Run the setup script to initialize the project:
   ```
   ./scripts/setup_project.sh
   ```
   
   This script will:
   - Create all necessary directories
   - Set up a virtual environment
   - Install dependencies
   - Generate synthetic data for testing
   - Download GeoJSON data for Fairfield County towns

   Alternatively, follow these manual steps:

3. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your API keys
   ```

6. Generate synthetic data for testing:
   ```
   python scripts/generate_synthetic_data.py --dataset all
   ```

7. Generate GeoJSON data for map visualizations:
   ```
   python scripts/generate_fairfield_geojson.py
   ```

### Running the Application

```
streamlit run app.py
```

## Project Structure

- `app.py`: Main application entry point
- `src/`: Source code
  - `data_collection/`: API client implementations
  - `data_processing/`: Data transformation and analysis
  - `visualization/`: Chart and visualization components
  - `ai_analysis/`: LLM integration for insights
  - `utils/`: Shared utilities
  - `ui/`: Streamlit UI components and pages
- `api_data/`: Data storage
  - `raw/`: Raw API data
  - `processed/`: Processed data
  - `synthetic/`: Synthetic data for development
- `scripts/`: Utility scripts
  - `setup_project.sh`: Project initialization script
  - `generate_synthetic_data.py`: Creates test data for development
  - `generate_fairfield_geojson.py`: Downloads or creates GeoJSON data for maps
  - `update_data.py`: Updates data from APIs (can be scheduled)
- `assets/`: Static assets
  - `fairfield_towns.geojson`: Geographic data for Fairfield County towns
- `tests/`: Test suite
- `docs/`: Documentation

## Data Sources

- FRED API (Federal Reserve Economic Data)
- BLS API (Bureau of Labor Statistics)
- ATTOM Property API
- CT Open Data (for town boundary data)
- Additional local/regional data sources for Fairfield County specifics

## Application Features

### Overview Dashboard
At-a-glance summary of critical market metrics with alerts for significant market shifts.

### Housing Metrics
Detailed visualizations for selected housing market metrics with historical trends.

### Economic Indicators
Visualizations of economic data affecting housing markets with correlation analysis.

### Comparison
Side-by-side analysis of multiple locations with multi-metric comparison.

### Forecast
Time series forecasting models for key metrics with scenario analysis.

### Data Analysis
Interactive time series overlays, correlation tools, and statistical analysis.

### Reports
AI-generated market analysis reports with export to PDF and DOCX formats.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# market-conditions
