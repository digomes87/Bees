# Makefile for Bee Migration Analysis Project

.PHONY: help install install-dev test lint format clean run-analysis generate-charts setup-env docs

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  setup-env    - Setup development environment"
	@echo "  test         - Run tests"
	@echo "  test-cov     - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean temporary files"
	@echo "  run-analysis - Run complete bee migration analysis"
	@echo "  generate-charts - Generate all scientific charts"
	@echo "  docs         - Generate documentation"
	@echo "  security     - Run security checks"
	@echo "  all          - Run format, lint, test, and analysis"

# Installation
install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt

install-dev: install
	@echo "Installing development dependencies..."
	pip install -r requirements-dev.txt

# Environment setup
setup-env:
	@echo "Setting up development environment..."
	python -m venv .venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source .venv/bin/activate  # On Linux/Mac"
	@echo "  .venv\\Scripts\\activate     # On Windows"

setup-conda:
	@echo "Setting up conda environment..."
	conda env create -f environment.yml
	@echo "Conda environment created. Activate with:"
	@echo "  conda activate bee-migration-analysis"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	@echo "Running fast tests..."
	pytest tests/ -v -x --ff

# Code quality
lint:
	@echo "Running linting checks..."
	flake8 src/ tests/
	mypy src/

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

format-check:
	@echo "Checking code format..."
	black --check src/ tests/
	isort --check-only src/ tests/

# Security
security:
	@echo "Running security checks..."
	bandit -r src/
	safety check

# Analysis and charts
run-analysis:
	@echo "Running complete bee migration analysis..."
	cd src && python bee_analysis.py
	cd src && python bee_migration_analysis.py
	cd src && python comparative_analysis.py

generate-charts:
	@echo "Generating all scientific charts..."
	cd src && python generate_charts_english.py

generate-charts-pt:
	@echo "Generating charts in Portuguese..."
	cd src && python generate_charts.py

run-predictor:
	@echo "Running migration predictor..."
	cd src && python bee_migration_predictor.py

# Data processing
process-data:
	@echo "Processing raw data..."
	@echo "Note: Implement data processing scripts as needed"

download-data:
	@echo "Downloading external data..."
	@echo "Note: Implement data download scripts as needed"

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "Note: Add sphinx documentation generation here"

docs-serve:
	@echo "Serving documentation locally..."
	@echo "Note: Add local documentation server here"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -delete
	find . -type d -name ".mypy_cache" -delete
	find . -type f -name "*.log" -delete

clean-data:
	@echo "Cleaning processed data files..."
	rm -rf data/processed/*
	rm -rf results/*.csv
	rm -rf results/*.pkl

clean-images:
	@echo "Cleaning generated images..."
	rm -rf images/*.png
	rm -rf images/*.jpg
	rm -rf images/*.svg

# Development workflow
dev-setup: setup-env install-dev
	@echo "Development environment ready!"

dev-check: format lint test
	@echo "Development checks completed!"

# CI/CD simulation
ci: format-check lint test security
	@echo "CI checks completed!"

# Complete workflow
all: format lint test run-analysis generate-charts
	@echo "Complete workflow finished!"

# Git helpers
git-setup:
	@echo "Setting up git hooks..."
	pre-commit install

commit-check: format lint test
	@echo "Ready for commit!"

# Project information
info:
	@echo "Bee Migration Analysis Project"
	@echo "=============================="
	@echo "Python version: $(shell python --version)"
	@echo "Project structure:"
	@tree -L 2 -I '__pycache__|.git|.venv|*.pyc'

status:
	@echo "Project status:"
	@echo "- Source files: $(shell find src -name '*.py' | wc -l)"
	@echo "- Test files: $(shell find tests -name '*.py' 2>/dev/null | wc -l || echo 0)"
	@echo "- Data files: $(shell find data -name '*.csv' 2>/dev/null | wc -l || echo 0)"
	@echo "- Generated images: $(shell find images -name '*.png' 2>/dev/null | wc -l || echo 0)"
	@echo "- Model files: $(shell find results/models -name '*.pkl' 2>/dev/null | wc -l || echo 0)"

# Docker support (future)
docker-build:
	@echo "Building Docker image..."
	@echo "Note: Add Dockerfile and build command"

docker-run:
	@echo "Running Docker container..."
	@echo "Note: Add Docker run command"

# Jupyter notebook support
jupyter:
	@echo "Starting Jupyter Lab..."
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

notebook-clean:
	@echo "Cleaning notebook outputs..."
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb