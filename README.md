# Fairfield County Housing Market Analysis

An interactive Streamlit application that analyzes and visualizes housing market conditions in Fairfield County, CT, with granular data down to specific towns.

## Features

- Comprehensive data analysis of 17 key housing market indicators
- Granular data visualization down to specific towns
- Time series analysis from 2015 to present
- AI-powered report generation and insights
- Customizable visualizations and comparison tools
- Market forecasting and trend analysis

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

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit .env with your API keys
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
- `tests/`: Test suite
- `docs/`: Documentation

## Data Sources

- FRED API (Federal Reserve Economic Data)
- BLS API (Bureau of Labor Statistics)
- ATTOM Property API
- Additional local/regional data sources for Fairfield County specifics

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# market-conditions
