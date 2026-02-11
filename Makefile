.PHONY: install install-dev test smoke-test test-unit test-integration test-fast test-coverage lint format typecheck check clean build benchmark docs-serve docs-build

# Install package
install:
	pip install -e .

# Install with all development dependencies
install-dev:
	pip install -e ".[dev,all]"

# Run all tests
test:
	pytest tests/ -v

# Run smoke tests (fast verification, <10s)
smoke-test:
	pytest tests/smoke_test.py -v

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run unit tests quickly (minimal output)
test-fast:
	pytest tests/unit/ -q --tb=line

# Run integration tests only
test-integration:
	pytest tests/integration/ -v

# Run tests with coverage
test-coverage:
	pytest tests/ --cov=src/forge --cov-report=term-missing --cov-report=html

# Run linter
lint:
	ruff check src/ tests/

# Format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Run type checker
typecheck:
	mypy src/forge/

# Run all checks
check: format lint typecheck test

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build package
build: clean
	python -m build

# Run examples
example-basic:
	python examples/basic_usage.py

example-sklearn:
	python examples/sklearn_pipeline.py

example-kaggle:
	python examples/kaggle_workflow.py

# Run benchmarks
benchmark:
	python benchmarks/run_benchmarks.py

# Serve documentation locally
docs-serve:
	mkdocs serve

# Build documentation
docs-build:
	mkdocs build
