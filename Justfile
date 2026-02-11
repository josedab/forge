# Default recipe
default:
    @just --list

# Install in development mode
install:
    pip install -e ".[dev,all]"

# Run smoke tests (fast verification, <10s)
smoke-test:
    pytest tests/smoke_test.py -v

# Run all tests
test:
    pytest tests/ -v

# Run tests with coverage
test-cov:
    pytest tests/ --cov=src/forge --cov-report=term-missing --cov-report=html

# Run unit tests only
test-unit:
    pytest tests/unit/ -v

# Run unit tests quickly (minimal output)
test-fast:
    pytest tests/unit/ -q --tb=line

# Run integration tests only
test-integration:
    pytest tests/integration/ -v

# Run linting
lint:
    ruff check src/ tests/

# Format code
format:
    ruff format src/ tests/
    ruff check --fix src/ tests/

# Type checking
typecheck:
    mypy src/forge/

# Run all checks
check: format lint typecheck test

# Clean build artifacts
clean:
    rm -rf build/ dist/ *.egg-info/ src/*.egg-info/
    rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ .coverage
    find . -type d -name __pycache__ -exec rm -rf {} +

# Build package
build: clean
    python -m build

# Serve documentation locally
docs-serve:
    mkdocs serve

# Build documentation
docs-build:
    mkdocs build

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

benchmark-generators:
    python benchmarks/benchmark_generators.py

benchmark-selectors:
    python benchmarks/benchmark_selectors.py

benchmark-pipeline:
    python benchmarks/benchmark_pipeline.py

# Run notebooks (requires jupyter)
notebook:
    jupyter notebook notebooks/
