# AI Assistant Makefile
# Common development tasks

.PHONY: help install install-dev test lint format clean build docker-build docker-run docs

# Default target
help:
	@echo "AI Assistant - Available commands:"
	@echo "  install      - Install the package"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting (flake8, mypy)"
	@echo "  dead-code    - Check for unused code and imports"
	@echo "  dead-code-report - Generate dead code report file"
	@echo "  dead-code-analysis - Generate comprehensive dead code analysis"
	@echo "  dead-code-analysis-html - Generate HTML dead code analysis"
	@echo "  format       - Format code (black, isort)"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"
	@echo "  docs         - Generate documentation"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,test,docs]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-smoke:
	pytest tests/smoke/ -v -m smoke

test-unit:
	pytest tests/unit/ -v -m unit

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

# Code quality
lint:
	flake8 src tests
	mypy src

dead-code:
	vulture src --config pyproject.toml

dead-code-report:
	vulture src --config pyproject.toml > dead_code_report.txt && echo "Dead code report saved to dead_code_report.txt" || (cat dead_code_report.txt && rm dead_code_report.txt)

dead-code-analysis:
	python tools/dead_code_analyzer.py --output dead_code_analysis.txt && echo "Comprehensive dead code analysis saved to dead_code_analysis.txt"

dead-code-analysis-html:
	python tools/dead_code_analyzer.py --report-format html --output dead_code_analysis.html && echo "HTML dead code analysis saved to dead_code_analysis.html"

format:
	black src tests
	isort src tests

format-check:
	black --check src tests
	isort --check-only src tests

# Build and packaging
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Docker
docker-build:
	docker build -f docker/Dockerfile -t ai-assistant:latest .

docker-build-dev:
	docker build -f docker/Dockerfile.dev -t ai-assistant:dev .

docker-run:
	docker run -it --rm -p 8000:8000 ai-assistant:latest

docker-run-dev:
	docker-compose -f docker/docker-compose.dev.yml up

docker-down:
	docker-compose -f docker/docker-compose.dev.yml down

# Development tools
setup:
	./scripts/setup.sh

migrate:
	./scripts/migrate.sh

deploy:
	./scripts/deploy.sh

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8080

# Database
db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-revision:
	alembic revision --autogenerate -m "$(MESSAGE)"

# Security
security-scan:
	bandit -r src/

# Performance
profile:
	python -m cProfile -o profile.stats src/main.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('tottime').print_stats(20)"

benchmark:
	pytest tests/performance/ -v --benchmark-only

# Utilities
check-deps:
	pip-audit

update-deps:
	pip-compile requirements/base.txt
	pip-compile requirements/development.txt
	pip-compile requirements/testing.txt
	pip-compile requirements/production.txt

pre-commit:
	pre-commit run --all-files

ci: format-check lint test

# Development server
dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

dev-cli:
	python -m src.cli

# Monitoring
logs:
	tail -f data/logs/application/app.log

metrics:
	curl http://localhost:8000/metrics

health:
	curl http://localhost:8000/health