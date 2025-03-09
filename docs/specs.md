# Market Conditions App Specifications

## Data Requirements

### Key Housing Market Indicators

1. Listing Inventory
2. Months of Supply
3. New Listings
4. Housing Starts/Residential Construction
5. Sales Volume
6. Days on Market (DOM)
7. Absorption Rate
8. Pending Home Sales
9. Median Sale Price
10. Average Sale Price
11. List Price to Sales Price Ratio
12. Home Price-to-Income Ratio
13. Mortgage Rates
14. Housing Affordability Index
15. Local Job Growth/Employment Trends
16. Vacancy Rates
17. Seller Concessions

### Data Sources

* FRED API (Federal Reserve Economic Data)
* BLS API (Bureau of Labor Statistics)
* ATTOM Property API
* Additional local/regional data sources for Fairfield County specifics

## Application Architecture

### Data Flow

1. **Data Collection Layer**: Scripts to fetch and store API data
    * Store raw API responses in api-data/raw/ directory
    * Implement API request caching to minimize calls
    * Schedule quarterly data updates
2. **Data Processing Layer**: Transform raw data into analysis-ready formats
    * Data cleaning and normalization
    * Calculation of derived metrics
    * Time series alignment
3. **Visualization Layer**: Streamlit UI components for data presentation
    * Interactive charts and graphs
    * Comparative analysis tools
    * Downloadable visualizations
4. **AI Analysis Layer**: LLM integration for insights generation
    * Report creation pipeline
    * Analysis templates for different report types
    * Custom prompt engineering for housing market analysis

### Data Storage Strategy

* Store raw API data locally to minimize API calls
* Implement a synthetic data generator for development and testing
* Structure data storage to optimize query performance for UI responsiveness

## User Interface

### Pages/Tabs

1. **Overview Dashboard**
    * At-a-glance summary of critical market metrics
    * Key trend indicators with quarter-over-quarter changes
    * Alert indicators for significant market shifts
2. **Housing Metrics**
    * Detailed visualizations for selected housing market metrics
    * Historical trend analysis with customizable time periods
    * Drill-down capabilities for geographic areas
3. **Economic Indicators**
    * Visualizations of economic data affecting housing markets
    * Correlation analysis with housing metrics
    * Regional economic health indicators
4. **Comparison**
    * Side-by-side analysis of multiple locations
    * Multi-metric comparison across selected areas
    * Exportable comparison charts and data tables
5. **Forecast**
    * Time series forecasting models for key metrics
    * Scenario analysis with adjustable parameters
    * Confidence intervals and forecast accuracy metrics
6. **Data Analysis**
    * Interactive time series overlays for correlation analysis
    * Custom metric combinations for trend identification
    * Statistical analysis tools for market insights
7. **Generated Reports**
    * Repository of user-generated reports
    * Report management (view, download, delete, edit)
    * Collaborative AI editing interface

### Sidebar Filters and Controls

#### Data Filters

* Date range selector (by quarter, 2015-present)
* Location hierarchy (County → Town)
* Property type selector (Single-family, Multi-unit, Condo/Townhouse)
* Market metrics selector (multi-select from the 17 key indicators)
* Economic metrics selector

#### Report Settings

* **LLM Provider selection**:
* OpenAI: gpt-4o, gpt-o1, gpt-o3-mini
* Anthropic: claude-3.5-sonnet-20241022, claude-3.7-sonnet
* **Report type selection**:
* General Analysis
* Investment Analysis
* Market Forecast
* Comparative Analysis
* Selling Recommendations
* Buying Recommendations
* **Tone selection** (Analytical, Informative, Casual, Skeptical, Academic, etc.)

### UI/UX Design Requirements

* Implement streamlit-shadcn-ui package for modern component styling
* Utilize streamlit-extras Card component for report and visualization containers
* Configure native theming in .streamlit/config.toml
* Ensure responsive layout for various screen sizes
* Maintain consistent typography and color scheme
* Implement proper spacing and hierarchy for readability
* Design intuitive navigation between related data views
* Ensure all charts have appropriate sizing for legibility
* Implement loading states and progress indicators for data processing

## AI Report Generation

### Report Generation Pipeline

1. Collect all selected parameters and visualized data
2. Format data into structured input for the selected LLM
3. Generate appropriate prompts based on report type
4. Process LLM response into formatted report
5. Enable user review and collaborative editing
6. Support export to DOCX and PDF formats

### Report Types

* **General Analysis**: Overall market health and trends
* **Investment Analysis**: Opportunities and risks for investors
* **Market Forecast**: Projected market movements with confidence levels
* **Comparative Analysis**: Cross-location or cross-time period analysis
* **Selling Recommendations**: Optimal timing and pricing strategies
* **Buying Recommendations**: Value opportunities and negotiation guidance

## Implementation Requirements

### Codebase Structure

* Modular design with separation of concerns:
* Data collection modules
* Data processing utilities
* Visualization components
* LLM integration services
* Report generation pipeline
* UI layout and state management

### Development Requirements

* Synthetic data generation that faithfully mimics API response formats
* Comprehensive unit and integration tests
* Error handling for API failures and data inconsistencies
* Performance optimization for data processing and visualization
* Security handling for API keys and credentials
* Detailed documentation and code comments

### Deliverables

1. Complete source code with modular organization
2. README with detailed setup and usage instructions
3. Requirements file(s) (requirements.txt and/or pyproject.toml)
4. Configuration files (.streamlit/config.toml)
5. Deployment instructions and configuration for Google App Engine
6. Unit and integration tests
7. Synthetic data generation scripts
8. API data handling scripts
9. Shell script to generate the entire project structure

## Technical Specifications

### Dependencies

* Streamlit for UI framework
* Pandas and NumPy for data processing
* Plotly/Matplotlib/Altair for visualizations
* Scikit-learn for basic forecasting models
* Prophet or statsmodels for advanced time series analysis
* OpenAI/Anthropic API clients for LLM integration
* FPDF or similar for PDF generation
* python-docx for DOCX generation
* Streamlit-extras and streamlit-shadcn-ui for enhanced components

### Testing Strategy

* Unit tests for data processing functions
* Integration tests for API clients
* UI component tests
* End-to-end tests for critical user journeys
* Performance benchmarks for data processing operations

### Deployment Configuration

* Google App Engine configuration (app.yaml)
* Environment variable management
* API key security handling
* Resource allocation recommendations
