# AGENTS.md - AI Assistant Context for Forge

This file provides context for AI coding assistants (GitHub Copilot, Claude, ChatGPT, Cursor, etc.) when working with the Forge codebase.

## Project Summary

**Forge** is an automated feature engineering platform for machine learning, written in Python. It provides:

- Automatic feature generation from numeric, categorical, temporal, and text data
- Intelligent feature selection using statistical methods, tree-based importance, and SHAP
- Full scikit-learn compatibility (works with Pipeline, GridSearchCV, cross_val_score)
- Data analysis with automatic type inference and quality assessment

## Technology Stack

- **Language**: Python 3.9+
- **Core Dependencies**: numpy, pandas, scikit-learn, scipy
- **Optional**: matplotlib, seaborn, shap, xgboost, lightgbm
- **Build System**: Hatch (PEP 517/518)
- **Testing**: pytest with pytest-cov
- **Linting**: Ruff
- **Type Checking**: mypy (strict mode)
- **Documentation**: MkDocs with Material theme

## Key Directories

```
forge/
├── src/forge/           # Main package source code
│   ├── analyzer/        # Data analysis and type inference
│   ├── generators/      # Feature generation (numeric, categorical, temporal, text)
│   ├── selectors/       # Feature selection algorithms
│   ├── transformers/    # sklearn-compatible transformers
│   ├── missing/         # Missing value handling
│   ├── visualization/   # Plotting utilities
│   └── utils/           # Shared utilities
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # Documentation (MkDocs)
├── examples/           # Usage examples
├── notebooks/          # Jupyter notebooks
└── benchmarks/         # Performance benchmarks
```

## Quick Commands

```bash
# Development setup
pip install -e ".[dev,all]"

# Run tests
make test                    # All tests
pytest tests/unit/          # Unit tests only
pytest tests/ -v --tb=short # Verbose with short traceback

# Code quality
make lint                   # Ruff linting
make format                 # Ruff formatting
make typecheck              # mypy type checking
make check                  # All checks

# Documentation
mkdocs serve                # Local docs server at localhost:8000
mkdocs build                # Build static docs

# Using Justfile (alternative)
just test
just lint
just check
```

## Coding Patterns

### Creating a Feature Generator

All generators inherit from `BaseFeatureGenerator`:

```python
from forge.generators.base import BaseFeatureGenerator
from typing import Self

class MyGenerator(BaseFeatureGenerator):
    def __init__(self, param: str = "default"):
        self.param = param

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._validate_input(X)
        # Learn parameters from data
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        # Generate new features
        return X_with_new_features

    def get_feature_names_out(self) -> list[str]:
        return self._generated_feature_names
```

### Creating a Feature Selector

All selectors inherit from `BaseFeatureSelector`:

```python
from forge.selectors.base import BaseFeatureSelector

class MySelector(BaseFeatureSelector):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        # Compute which features to keep
        self._support_mask = np.array([True, False, True, ...])
        return self

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self._support_mask)[0]
        return self._support_mask
```

### sklearn Compatibility Rules

1. All transformers must inherit from `BaseEstimator` and `TransformerMixin`
2. Constructor parameters must be stored as instance attributes with same names
3. No mutable default arguments in `__init__`
4. Use `check_is_fitted()` before `transform()`
5. Implement `get_feature_names_out()` for pipeline compatibility
6. Support `clone()` semantics

## Type Hints

- Use `from __future__ import annotations` for forward references
- Common types are in `src/forge/types.py`
- Run `make typecheck` to verify types

## Testing Guidelines

- Tests are in `tests/unit/` and `tests/integration/`
- Use fixtures from `tests/conftest.py`
- Property-based tests use Hypothesis (`tests/unit/test_properties.py`)
- Aim for >80% coverage

Example test:

```python
import pytest
import pandas as pd
from forge.generators.numeric import InteractionGenerator

def test_interaction_generator_fit_transform():
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    gen = InteractionGenerator(columns=['a', 'b'], operations=['multiply'])

    result = gen.fit_transform(X)

    assert 'a_multiply_b' in result.columns
    assert result['a_multiply_b'].tolist() == [4, 10, 18]
```

## Architecture Decisions

Key design decisions are documented in `docs/architecture/`:

- **ADR-0001**: sklearn compatibility via inheritance from BaseEstimator/TransformerMixin
- **ADR-0002**: Plugin registry pattern for extensible generators
- **ADR-0003**: Rule-based type inference with override capabilities

## Common Tasks

### Add a new generator type
1. Create module in `src/forge/generators/<category>/`
2. Inherit from `BaseFeatureGenerator`
3. Register in generator registry
4. Export in `__init__.py`
5. Add tests in `tests/unit/`
6. Document in `docs/api/generators.md`

### Add a new selector
1. Create module in `src/forge/selectors/`
2. Inherit from `BaseFeatureSelector`
3. Export in `__init__.py`
4. Add tests
5. Document in `docs/api/selectors.md`

### Fix a bug
1. Write a failing test that reproduces the bug
2. Fix the code
3. Verify test passes
4. Run `make check` to ensure no regressions

## Code Quality Standards

- **Formatting**: Ruff format (line length 88)
- **Linting**: Ruff with extended rules (E, F, I, N, W, UP, B, C4, SIM, D, PT, RUF, S, TCH)
- **Type Safety**: mypy strict mode, no implicit Any
- **Docstrings**: Google style
- **Test Coverage**: >80% target

## Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Project configuration, dependencies |
| `src/forge/__init__.py` | Public API exports |
| `src/forge/types.py` | Type definitions |
| `src/forge/exceptions.py` | Custom exceptions |
| `tests/conftest.py` | Shared pytest fixtures |
| `CLAUDE.md` | Detailed context for Claude Code |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

When making changes:
1. Create a feature branch
2. Make changes with tests
3. Run `make check`
4. Submit PR with clear description
5. Ensure CI passes
