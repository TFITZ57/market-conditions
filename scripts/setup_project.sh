#!/bin/bash

# Fairfield County Housing Market Analysis - Project Setup Script
# This script initializes the project structure and installs dependencies

set -e  # Exit immediately if a command exits with a non-zero status

echo "Setting up Fairfield County Housing Market Analysis project..."

# Ensure working directory is project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Create required directories
echo "Creating project directories..."
mkdir -p .streamlit
mkdir -p api_data/raw api_data/processed api_data/synthetic
mkdir -p assets
mkdir -p logs
mkdir -p src/data_collection
mkdir -p src/data_processing
mkdir -p src/visualization
mkdir -p src/ai_analysis
mkdir -p src/utils
mkdir -p src/ui/components src/ui/pages
mkdir -p tests/unit tests/integration tests/fixtures

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.py
chmod +x scripts/*.sh

# Create virtual environment if it doesn't exist
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
  python -m venv venv
  echo "Virtual environment created."
else
  echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install development dependencies if present
if [ -f "dev-requirements.txt" ]; then
  pip install -r dev-requirements.txt
fi

# Generate synthetic data
echo "Generating synthetic data..."
python scripts/generate_synthetic_data.py --dataset all

# Generate GeoJSON data for Fairfield County towns
echo "Generating GeoJSON data for Fairfield County towns..."
python scripts/generate_fairfield_geojson.py

echo "Project setup complete!"
echo "To start the application, run: streamlit run app.py" 