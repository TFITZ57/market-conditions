#!/bin/bash
# This script sets up the entire project structure for the Fairfield County Housing Market Analysis application

# Define color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print section header
print_header() {
    echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Print success message
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Print error message
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Print info message
print_info() {
    echo -e "${YELLOW}i $1${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main project directory (current directory)
PROJECT_DIR="$(pwd)"
print_header "Setting up Fairfield County Housing Market Analysis in ${PROJECT_DIR}"

# Create directory structure
print_header "Creating Directory Structure"

# Define directories to create
directories=(
    ".streamlit"
    "api_data/raw/fred"
    "api_data/raw/bls"
    "api_data/raw/attom"
    "api_data/processed"
    "api_data/synthetic"
    "src/data_collection"
    "src/data_processing"
    "src/visualization"
    "src/ai_analysis"
    "src/utils"
    "src/ui/components"
    "src/ui/pages"
    "tests/unit"
    "tests/integration"
    "tests/fixtures"
    "scripts"
    "docs"
    "logs"
    "assets"
)

# Create each directory
for dir in "${directories[@]}"; do
    mkdir -p "${PROJECT_DIR}/${dir}"
    if [ $? -eq 0 ]; then
        print_success "Created directory: ${dir}"
    else
        print_error "Failed to create directory: ${dir}"
    fi
done

# Create empty __init__.py files in Python directories to make them proper packages
print_header "Creating Python Package Structure"

# Define directories that need __init__.py
init_dirs=(
    "src"
    "src/data_collection"
    "src/data_processing"
    "src/visualization"
    "src/ai_analysis"
    "src/utils"
    "src/ui"
    "src/ui/components"
    "src/ui/pages"
    "tests"
    "tests/unit"
    "tests/integration"
    "tests/fixtures"
)

# Create __init__.py files
for dir in "${init_dirs[@]}"; do
    touch "${PROJECT_DIR}/${dir}/__init__.py"
    if [ $? -eq 0 ]; then
        print_success "Created __init__.py in ${dir}"
    else
        print_error "Failed to create __init__.py in ${dir}"
    fi
done

# Create environment file
print_header "Creating Environment File"
cat > "${PROJECT_DIR}/.env.example" << 'EOL'
# API Keys
FRED_API_KEY=your_fred_api_key
BLS_API_KEY=your_bls_api_key
ATTOM_API_KEY=your_attom_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key

# Application Settings
DEBUG=False
LOG_LEVEL=INFO
CACHE_TTL=86400
EOL
print_success "Created .env.example file"

# Create .gitignore
print_header "Creating .gitignore"
cat > "${PROJECT_DIR}/.gitignore" << 'EOL'
# Python files
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
ENV/
env/

# Environment variables
.env

# Cache files
.cache/
.pytest_cache/

# OS specific files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# Application specific
logs/
api_data/raw/
api_data/processed/

# Keep synthetic data
!api_data/synthetic/
EOL
print_success "Created .gitignore file"

# Create requirements.txt
print_header "Creating requirements.txt"
cat > "${PROJECT_DIR}/requirements.txt" << 'EOL'
# Core dependencies
streamlit==1.31.0
pandas==2.1.3
numpy==1.26.2
plotly==5.18.0
matplotlib==3.8.2
altair==5.2.0

# API clients
requests==2.31.0
python-dotenv==1.0.0

# Data processing
scikit-learn==1.3.2
statsmodels==0.14.0
prophet==1.1.4

# LLM integration
openai==1.3.5
anthropic==0.8.1

# Document generation
fpdf==1.7.2
python-docx==1.0.1

# Enhanced Streamlit components
streamlit-extras==0.3.5
streamlit-shadcn-ui==0.1.7
st-pages==0.4.5

# Testing
pytest==7.4.3
pytest-cov==4.1.0

# Utilities
pyyaml==6.0.1
tabulate==0.9.0
tqdm==4.66.1
EOL
print_success "Created requirements.txt file"

# Create pyproject.toml
print_header "Creating pyproject.toml"
cat > "${PROJECT_DIR}/pyproject.toml" << 'EOL'
[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "fairfield_housing_analysis"
version = "0.1.0"
description = "Fairfield County Housing Market Analysis Application"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "streamlit>=1.31.0",
    "pandas>=2.1.3",
    "numpy>=1.26.2",
    "plotly>=5.18.0",
    "matplotlib>=3.8.2",
    "altair>=5.2.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "scikit-learn>=1.3.2",
    "statsmodels>=0.14.0",
    "prophet>=1.1.4",
    "openai>=1.3.5",
    "anthropic>=0.8.1",
    "fpdf>=1.7.2",
    "python-docx>=1.0.1",
    "streamlit-extras>=0.3.5",
    "streamlit-shadcn-ui>=0.1.7",
    "st-pages>=0.4.5",
    "pyyaml>=6.0.1",
    "tabulate>=0.9.0",
    "tqdm>=4.66.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.3",
    "pytest-cov>=4.1.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "flake8>=6.1.0",
    "pre-commit>=3.5.0",
]

[tool.setuptools]
packages = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
python_classes = "Test*"

[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
EOL
print_success "Created pyproject.toml file"

# Create README.md
print_header "Creating README.md"
cat > "${PROJECT_DIR}/README.md" << 'EOL'
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
EOL
print_success "Created README.md file"

# Create Streamlit config
print_header "Creating Streamlit Config"
cat > "${PROJECT_DIR}/.streamlit/config.toml" << 'EOL'
[theme]
primaryColor = "#4682b4"  # Steel blue
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f8fa"
textColor = "#262730"
font = "sans-serif"

[server]
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 50
maxMessageSize = 100

[browser]
gatherUsageStats = false

[runner]
# Enable memory profiling to optimize performance
magicEnabled = true
installTracer = false
fixMatplotlib = true

[logger]
level = "info"
messageFormat = "%(asctime)s %(levelname)s: %(message)s"
EOL
print_success "Created Streamlit config file"

# Create Google App Engine configuration
print_header "Creating App Engine Configuration"
cat > "${PROJECT_DIR}/app.yaml" << 'EOL'
runtime: python39

instance_class: F2

env_variables:
  STREAMLIT_SERVER_PORT: 8080
  STREAMLIT_SERVER_HEADLESS: true
  STREAMLIT_SERVER_ENABLE_CORS: true
  STREAMLIT_BROWSER_GATHER_USAGE_STATS: false
  STREAMLIT_THEME_PRIMARY_COLOR: "#4682b4"

handlers:
- url: /.*
  script: auto
  secure: always

automatic_scaling:
  min_instances: 0
  max_instances: 2
  min_idle_instances: 0
  max_idle_instances: 1
  min_pending_latency: automatic
  max_pending_latency: automatic
  target_cpu_utilization: 0.65
  target_throughput_utilization: 0.65

inbound_services:
- warmup
EOL
print_success "Created app.yaml file"

# Install requirements if pip is available
print_header "Installing Dependencies"
if command_exists pip; then
    echo "Installing Python dependencies..."
    pip install -r "${PROJECT_DIR}/requirements.txt"
    if [ $? -eq 0 ]; then
        print_success "Installed dependencies"
    else
        print_error "Failed to install dependencies"
    fi
else
    print_info "pip not found, skipping dependency installation"
    print_info "You can install dependencies manually with: pip install -r requirements.txt"
fi

# Create Fairfield County logo placeholder
print_header "Creating Asset Placeholders"
mkdir -p "${PROJECT_DIR}/assets"
cat > "${PROJECT_DIR}/assets/fairfield_county_logo.txt" << 'EOL'
Place your Fairfield County logo image here.
Recommended size: 200x200 pixels.
Rename this file to fairfield_county_logo.png after adding the image.
EOL
print_success "Created asset placeholder"

# Generate synthetic data
print_header "Generating Synthetic Data"
if command_exists python || command_exists python3; then
    PYTHON_CMD="python"
    if ! command_exists python && command_exists python3; then
        PYTHON_CMD="python3"
    fi
    
    echo "Generating synthetic data for development..."
    "$PYTHON_CMD" "${PROJECT_DIR}/scripts/generate_synthetic_data.py"
    if [ $? -eq 0 ]; then
        print_success "Generated synthetic data"
    else
        print_error "Failed to generate synthetic data"
        print_info "You can generate synthetic data manually with: python scripts/generate_synthetic_data.py"
    fi
else
    print_info "Python not found, skipping synthetic data generation"
    print_info "You can generate synthetic data manually with: python scripts/generate_synthetic_data.py"
fi

# Final message
print_header "Setup Complete"
echo "The Fairfield County Housing Market Analysis project has been set up successfully."
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run the application with: streamlit run app.py"
echo "3. Generate synthetic data with: python scripts/generate_synthetic_data.py"
echo ""
echo "Refer to README.md for more information."
