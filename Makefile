# dextra - Data Extraction package for InvOCR
# Development and deployment automation

.PHONY: help install test lint format build clean publish

# Default target
help:
	@echo "dextra Development Commands"
	@echo "=========================="
	@echo "install     - Install dependencies"
	@echo "test        - Run tests"
	@echo "lint        - Run linting"
	@echo "format      - Format code"
	@echo "build       - Build package"
	@echo "clean       - Clean build artifacts"
	@echo "publish     - Publish package to PyPI"

install:
	poetry install
	poetry run pre-commit install

test:
	poetry run pytest --cov=dextra --cov-report=html

lint:
	poetry run flake8 dextra/
	poetry run mypy dextra/

format:
	poetry run black dextra/ tests/
	poetry run isort dextra/ tests/

build:
	poetry build

clean:
	rm -rf dist/ build/ *.egg-info/
	rm -rf .coverage htmlcov/
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

publish:
	poetry publish --build
