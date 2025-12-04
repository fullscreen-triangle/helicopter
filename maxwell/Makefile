# Makefile for maxwell package

.PHONY: help install install-dev test lint format clean demo benchmark docs

help:
	@echo "Maxwell Package - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install package and dependencies"
	@echo "  make install-dev   - Install with development dependencies"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make demo          - Run demo script"
	@echo "  make benchmark     - Run benchmarks"
	@echo "  make docs          - Generate documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-hardware:
	pip install -e ".[hardware]"

install-all:
	pip install -e ".[dev,hardware]"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

lint:
	flake8 src/ tests/ scripts/
	mypy src/

format:
	black src/ tests/ scripts/ --line-length 100

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

demo:
	python demo_hccc_vision.py

demo-visualize:
	python -m scripts.run_demo --visualize --validate --output results/

benchmark:
	python -m scripts.run_benchmark --n-images 10 --output benchmark_results.json

process-example:
	python -m scripts.process_image examples/test_image.jpg --visualize --save-results

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  - README_IMPLEMENTATION.md"
	@echo "  - QUICK_START.md"
	@echo "  - IMPLEMENTATION_COMPLETE.md"
	@echo "  - ALGORITHM_IMPLEMENTATION_PROPOSAL.md"

# Development helpers
check: lint test
	@echo "All checks passed!"

dev-setup: install-dev
	pre-commit install || echo "pre-commit not available"

# Package distribution
build:
	python -m build

upload-test:
	python -m twine upload --repository testpypi dist/*

upload:
	python -m twine upload dist/*
