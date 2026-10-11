.PHONY: help install install-dev hooks test test-site lint format format-check \
	analyze predict charts all site notebook docker-build docker-run clean check

help:
	@echo "Setup"
	@echo "  install        Install the package"
	@echo "  install-dev    Install the package with development tools"
	@echo "  hooks          Install the pre-commit hooks"
	@echo "Quality"
	@echo "  test           Run the Python tests with coverage"
	@echo "  test-site      Run the dashboard tests"
	@echo "  lint           Check the code with ruff"
	@echo "  format         Format the code with ruff"
	@echo "  check          Everything CI runs: format-check, lint, test, test-site"
	@echo "Run"
	@echo "  analyze        Trends, correlations and the summary report"
	@echo "  predict        Train and compare the models"
	@echo "  charts         Render the charts into images/"
	@echo "  all            analyze, predict and charts"
	@echo "  site           Build the package into the dashboard and serve it"
	@echo "  notebook       Open the tutorial in JupyterLab"
	@echo "Docker"
	@echo "  docker-build   Build the image"
	@echo "  docker-run     Run the whole pipeline in a container"
	@echo "Other"
	@echo "  clean          Remove caches and build leftovers"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,notebook]"

hooks:
	pre-commit install

test:
	pytest

test-site:
	node --test "tests/web/*.test.mjs"

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

format-check:
	ruff format --check src tests

check: format-check lint test test-site

analyze:
	bee-migration analyze

predict:
	bee-migration predict

charts:
	bee-migration charts

all:
	bee-migration all

site:
	pip wheel . --no-deps --wheel-dir docs/package
	python -m http.server 8000 --bind 127.0.0.1 --directory docs

notebook:
	jupyter lab notebooks/bee_migration_tutorial.ipynb

docker-build:
	docker compose build

docker-run:
	docker compose run --rm analysis all

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build docs/package src/*.egg-info
