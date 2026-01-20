# CLAUDE.md - Forge Feature Engineering Platform

This file provides context for Claude Code when working with the Forge codebase.

## Project Overview

Forge is an automated feature engineering platform for machine learning. It provides automatic feature generation, intelligent selection, and scikit-learn compatibility to reduce the time data scientists spend on feature engineering.

## Build & Test Commands

```bash
# Install in development mode
pip install -e ".[dev,all]"

# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only
make test-integration

# Run with coverage
make test-coverage

# Lint code
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks (format, lint, typecheck, test)
make check
```

## Project Structure

```
forge/
├── src/forge/
│   ├── __init__.py           # Public API exports
│   ├── _version.py           # Version info
│   ├── exceptions.py         # Custom exceptions
│   ├── types.py              # Type definitions
│   ├── analyzer/             # Data analysis module
│   │   ├── base.py           # DataAnalyzer class
│   │   ├── type_inference.py # Column type detection
│   │   ├── statistics.py     # Statistical profiling
│   │   ├── quality.py        # Data quality assessment
│   │   └── report.py         # Analysis reports
│   ├── generators/           # Feature generation
│   │   ├── base.py           # BaseFeatureGenerator ABC
│   │   ├── registry.py       # Generator registry
│   │   ├── numeric/          # Numeric feature generators
│   │   ├── categorical/      # Categorical feature generators
│   │   ├── temporal/         # Temporal feature generators
│   │   └── text/             # Text feature generators
│   ├── selectors/            # Feature selection
│   │   ├── base.py           # BaseFeatureSelector ABC
│   │   ├── statistical.py    # Chi-square, ANOVA, MI
│   │   ├── importance.py     # Tree-based importance
│   │   ├── correlation.py    # Correlation removal
│   │   ├── variance.py       # Low variance removal
│   │   └── shap_selector.py  # SHAP-based selection
│   ├── missing/              # Missing value handling
│   │   ├── strategies.py     # Imputation strategies
│   │   └── indicators.py     # Missing indicators
│   ├── transformers/         # sklearn transformers
│   │   ├── base.py           # ForgeTransformerMixin
│   │   ├── auto_transformer.py # AutoFeatureTransformer
│   │   ├── feature_pipeline.py # ForgePipeline
│   │   └── serialization.py  # Save/load utilities
│   ├── visualization/        # Plotting
│   │   ├── importance.py     # Feature importance plots
│   │   ├── correlation.py    # Correlation heatmaps
│   │   ├── shap_plots.py     # SHAP visualizations
│   │   └── distribution.py   # Distribution plots
│   └── utils/                # Utilities
│       ├── validation.py     # Input validation
│       ├── parallel.py       # Parallel processing
│       └── logging.py        # Logging config
├── tests/
│   ├── conftest.py           # Pytest fixtures
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
└── examples/                 # Usage examples
```

## Key Classes

### AutoFeatureTransformer (Main Entry Point)
- Location: `src/forge/transformers/auto_transformer.py`
- sklearn-compatible transformer for automatic feature engineering
- Methods: `fit()`, `transform()`, `fit_transform()`, `get_feature_names_out()`, `get_feature_importance()`

### DataAnalyzer
- Location: `src/forge/analyzer/base.py`
- Analyzes input data, infers types, computes statistics
- Methods: `analyze()`, `infer_types()`, `profile_statistics()`, `assess_quality()`

### BaseFeatureGenerator (ABC)
- Location: `src/forge/generators/base.py`
- Abstract base for all feature generators
- Inherits from sklearn's BaseEstimator and TransformerMixin

### BaseFeatureSelector (ABC)
- Location: `src/forge/selectors/base.py`
- Abstract base for all feature selectors
- Methods: `fit()`, `transform()`, `get_support()`

## Coding Conventions

### Type Hints
- Use type hints for all public functions and methods
- Use `from __future__ import annotations` for forward references
- Common types are defined in `src/forge/types.py`

### sklearn Compatibility
- All transformers inherit from `BaseEstimator` and `TransformerMixin`
- Implement `fit()`, `transform()`, `get_feature_names_out()`
- Use `check_is_fitted()` before transform
- Clone parameters in `__init__` (no mutable defaults)

### Error Handling
- Use custom exceptions from `src/forge/exceptions.py`
- Validate inputs early in public methods
- Provide helpful error messages with context

### Testing
- Tests are in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Unit tests test single components
- Integration tests test full pipelines

## Common Patterns

### Creating a New Feature Generator
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

### Creating a New Feature Selector
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

## Dependencies

Core:
- numpy, pandas - Data structures
- scikit-learn - ML framework
- scipy - Statistical functions

Optional:
- matplotlib, seaborn - Visualization
- shap - SHAP-based selection
- xgboost, lightgbm - Gradient boosting

## Common Tasks

### Adding a New Generator Type
1. Create module in appropriate `generators/` subdirectory
2. Inherit from `BaseFeatureGenerator`
3. Register in the generator registry
4. Export in `__init__.py`
5. Add tests

### Adding a New Selector
1. Create module in `selectors/`
2. Inherit from `BaseFeatureSelector`
3. Implement `fit()` and `get_support()`
4. Export in `__init__.py`
5. Add tests

### Running Specific Tests
```bash
# Single test file
pytest tests/unit/test_analyzer.py

# Single test
pytest tests/unit/test_analyzer.py::test_type_inference

# With coverage for specific module
pytest --cov=src/forge/analyzer tests/unit/test_analyzer.py
```
