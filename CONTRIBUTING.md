# Contributing to Forge

Thank you for your interest in contributing to Forge! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/forge.git
   cd forge
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/forge-features/forge.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev,all]"
   ```

3. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Running Tests

We use `pytest` for testing. The `Makefile` provides convenient commands:

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run tests with coverage report
make test-coverage

# Run linting
make lint

# Run type checking
make typecheck

# Run all checks (format, lint, typecheck, test)
make check
```

### Running Specific Tests

```bash
# Single test file
pytest tests/unit/test_analyzer.py

# Single test function
pytest tests/unit/test_analyzer.py::test_type_inference

# Tests matching a pattern
pytest -k "test_numeric"
```

## Coding Standards

### Type Hints

All public functions and methods must have type hints:

```python
from __future__ import annotations

def process_features(
    X: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    ...
```

### Docstrings

Use Google-style docstrings for all public APIs:

```python
def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
    """Fit the transformer to the data.

    Args:
        X: Input features DataFrame.
        y: Target variable (optional, used for supervised selection).

    Returns:
        The fitted transformer instance.

    Raises:
        ValueError: If X contains invalid data types.
    """
```

### scikit-learn Compatibility

All transformers must follow scikit-learn conventions:

- Inherit from `BaseEstimator` and `TransformerMixin`
- Implement `fit()`, `transform()`, and `get_feature_names_out()`
- Use `check_is_fitted()` before transform
- No mutable default arguments in `__init__`
- Store all constructor parameters as instance attributes with the same name

### Code Style

We use `ruff` for linting and formatting:

```bash
# Format code
make format

# Check linting
make lint
```

Key style points:
- Line length: 100 characters
- Use `from __future__ import annotations` for forward references
- Prefer explicit imports over star imports
- Use `Self` type hint for method chaining

## Pull Request Process

1. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Add tests** for any new functionality

4. **Run the full test suite**:
   ```bash
   make check
   ```

5. **Commit your changes** with a clear message:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request** against the `main` branch

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] Self-review of the code has been performed
- [ ] Tests have been added that prove the fix/feature works
- [ ] All tests pass locally
- [ ] Type hints are complete and mypy passes
- [ ] Documentation has been updated if needed

## Adding New Features

### Adding a New Feature Generator

1. Create a module in the appropriate `generators/` subdirectory
2. Inherit from `BaseFeatureGenerator`:

   ```python
   from forge.generators.base import BaseFeatureGenerator

   class MyGenerator(BaseFeatureGenerator):
       def __init__(self, param: str = "default"):
           self.param = param

       def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
           self._validate_input(X)
           # Learn from data
           self._is_fitted = True
           return self

       def transform(self, X: pd.DataFrame) -> pd.DataFrame:
           self._check_is_fitted()
           # Generate features
           return X_transformed

       def get_feature_names_out(self) -> list[str]:
           return self._feature_names
   ```

3. Register in the generator registry
4. Export in `__init__.py`
5. Add comprehensive tests

### Adding a New Feature Selector

1. Create a module in `selectors/`
2. Inherit from `BaseFeatureSelector`:

   ```python
   from forge.selectors.base import BaseFeatureSelector

   class MySelector(BaseFeatureSelector):
       def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
           # Compute feature scores
           self._support_mask = ...
           return self

       def get_support(self, indices: bool = False):
           if indices:
               return np.where(self._support_mask)[0]
           return self._support_mask
   ```

3. Export in `__init__.py`
4. Add comprehensive tests

## Questions?

If you have questions, feel free to:
- Open an issue for discussion
- Start a GitHub Discussion
- Reach out to the maintainers

Thank you for contributing!
