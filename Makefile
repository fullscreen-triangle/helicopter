# Helicopter Makefile - Reverse Pakati Visual Knowledge Extraction

.PHONY: help install install-dev test test-coverage lint format type-check security-check clean build docker-build docker-run docs serve-docs pre-commit setup-dev train deploy

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := helicopter
IMAGE_NAME := helicopter
VERSION := $(shell grep -E '^version' pyproject.toml | cut -d'"' -f2)

# Default target
help: ## Show this help message
	@echo "Helicopter - Reverse Pakati Visual Knowledge Extraction"
	@echo "======================================================"
	@echo ""
	@echo "Available commands:"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } END {printf "\033[0m"}' $(MAKEFILE_LIST)

##@ Development

install: ## Install the package and dependencies
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

setup-dev: ## Complete development setup
	@echo "Setting up development environment..."
	$(PYTHON) -m venv env
	@echo "Activate environment with: source env/bin/activate"
	@echo "Then run: make install-dev"

##@ Code Quality

lint: ## Run linting checks
	@echo "Running linting checks..."
	flake8 $(PROJECT_NAME)/ tests/
	isort --check-only $(PROJECT_NAME)/ tests/
	black --check $(PROJECT_NAME)/ tests/

format: ## Format code with black and isort
	@echo "Formatting code..."
	black $(PROJECT_NAME)/ tests/
	isort $(PROJECT_NAME)/ tests/

type-check: ## Run type checking with mypy
	@echo "Running type checks..."
	mypy $(PROJECT_NAME)/

security-check: ## Run security checks with bandit
	@echo "Running security checks..."
	bandit -r $(PROJECT_NAME)/

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

##@ Testing

test: ## Run tests
	@echo "Running tests..."
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	pytest tests/ --cov=$(PROJECT_NAME) --cov-report=html --cov-report=term-missing -v

test-integration: ## Run integration tests
	@echo "Running integration tests..."
	pytest tests/integration/ -v --timeout=300

test-gpu: ## Run GPU-specific tests
	@echo "Running GPU tests..."
	pytest tests/ -m gpu -v

##@ Documentation

docs: ## Build documentation
	@echo "Building documentation..."
	cd docs && make html

serve-docs: ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8080"
	cd docs/_build/html && python -m http.server 8080

##@ Building and Packaging

build: ## Build the package
	@echo "Building package..."
	$(PYTHON) -m build

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

##@ Docker

docker-build: ## Build Docker image
	@echo "Building Docker image..."
	$(DOCKER) build -t $(IMAGE_NAME):$(VERSION) .
	$(DOCKER) tag $(IMAGE_NAME):$(VERSION) $(IMAGE_NAME):latest

docker-build-dev: ## Build development Docker image
	@echo "Building development Docker image..."
	$(DOCKER) build --target development -t $(IMAGE_NAME):dev .

docker-build-prod: ## Build production Docker image
	@echo "Building production Docker image..."
	$(DOCKER) build --target production -t $(IMAGE_NAME):prod .

docker-run: ## Run Docker container
	@echo "Running Docker container..."
	$(DOCKER) run -it --rm -p 8000:8000 --gpus all $(IMAGE_NAME):latest

docker-run-dev: ## Run development Docker container
	@echo "Running development Docker container..."
	$(DOCKER) run -it --rm -p 8000:8000 -p 8888:8888 --gpus all -v $(PWD):/app $(IMAGE_NAME):dev

##@ Docker Compose

up: ## Start development environment with docker-compose
	@echo "Starting development environment..."
	$(DOCKER_COMPOSE) up -d helicopter-dev redis postgres

up-prod: ## Start production environment
	@echo "Starting production environment..."
	$(DOCKER_COMPOSE) --profile production up -d

up-training: ## Start training environment
	@echo "Starting training environment..."
	$(DOCKER_COMPOSE) --profile training up -d

up-monitoring: ## Start monitoring stack
	@echo "Starting monitoring stack..."
	$(DOCKER_COMPOSE) --profile monitoring up -d

up-all: ## Start all services
	@echo "Starting all services..."
	$(DOCKER_COMPOSE) --profile production --profile monitoring --profile worker up -d

down: ## Stop all services
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

##@ Machine Learning

download-models: ## Download pre-trained models
	@echo "Downloading pre-trained models..."
	$(PYTHON) scripts/download_models.py

train-example: ## Train example model
	@echo "Training example model..."
	helicopter train \
		--dataset-path data/examples/ \
		--domain medical \
		--base-model gpt2 \
		--epochs 5 \
		--batch-size 4

train-medical: ## Train medical domain model
	@echo "Training medical domain model..."
	helicopter train \
		--dataset-path data/medical/ \
		--domain medical \
		--base-model microsoft/DialoGPT-medium \
		--epochs 20 \
		--batch-size 8

evaluate: ## Evaluate trained models
	@echo "Evaluating models..."
	$(PYTHON) scripts/evaluate_models.py

##@ Data Processing

process-dataset: ## Process raw dataset for training
	@echo "Processing dataset..."
	helicopter process \
		--data-dir data/raw/ \
		--output-dir data/processed/

create-sample-data: ## Create sample dataset for testing
	@echo "Creating sample dataset..."
	$(PYTHON) scripts/create_sample_data.py

##@ Deployment

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	$(DOCKER_COMPOSE) -f docker-compose.staging.yml up -d

deploy-prod: ## Deploy to production environment
	@echo "Deploying to production..."
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

k8s-deploy: ## Deploy to Kubernetes
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/

##@ Database

db-init: ## Initialize database
	@echo "Initializing database..."
	$(PYTHON) scripts/init_database.py

db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	alembic upgrade head

db-reset: ## Reset database (WARNING: destructive)
	@echo "Resetting database..."
	$(DOCKER_COMPOSE) stop postgres
	$(DOCKER_COMPOSE) rm -f postgres
	docker volume rm helicopter_postgres-data
	$(DOCKER_COMPOSE) up -d postgres
	sleep 5
	make db-init

##@ Monitoring

monitor: ## Open monitoring dashboard
	@echo "Opening monitoring dashboard..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Flower: http://localhost:5555"

metrics: ## Show system metrics
	@echo "System metrics:"
	@echo "==============="
	$(PYTHON) scripts/show_metrics.py

##@ Utilities

jupyter: ## Start Jupyter notebook
	@echo "Starting Jupyter notebook..."
	$(DOCKER_COMPOSE) --profile development up -d jupyter
	@echo "Jupyter available at: http://localhost:8888 (token: helicopter)"

shell: ## Open shell in development container
	@echo "Opening shell in development container..."
	$(DOCKER_COMPOSE) exec helicopter-dev bash

gpu-status: ## Check GPU status
	@echo "GPU Status:"
	@echo "==========="
	nvidia-smi

check-deps: ## Check for dependency updates
	@echo "Checking for dependency updates..."
	pip list --outdated

update-deps: ## Update dependencies
	@echo "Updating dependencies..."
	pip-upgrade

benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	$(PYTHON) scripts/benchmark.py

##@ CI/CD

ci-test: ## Run CI test suite
	@echo "Running CI test suite..."
	make lint
	make type-check
	make security-check
	make test-coverage

ci-build: ## Build for CI
	@echo "Building for CI..."
	make clean
	make build
	make docker-build

release: ## Create a release
	@echo "Creating release $(VERSION)..."
	git tag v$(VERSION)
	git push origin v$(VERSION)
	make build
	twine upload dist/*

##@ Quick Commands

dev: up ## Alias for 'up' - start development environment
prod: up-prod ## Alias for 'up-prod' - start production environment
start: up ## Alias for 'up' - start development environment
stop: down ## Alias for 'down' - stop all services
restart: down up ## Restart development environment

# Special targets
.PHONY: all
all: clean install-dev lint type-check test build ## Run all quality checks and build

.PHONY: quick-test
quick-test: ## Run quick tests (no coverage)
	pytest tests/ -x --tb=short

.PHONY: full-check
full-check: lint type-check security-check test-coverage ## Run all code quality checks

# Help formatting
.DEFAULT_GOAL := help 