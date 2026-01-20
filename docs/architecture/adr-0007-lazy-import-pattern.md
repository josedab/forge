# ADR-0007: Lazy Import Pattern

## Status

Accepted

## Context

Forge has a modular architecture with many components:
- DataAnalyzer
- AutoFeatureTransformer
- Multiple generators (numeric, categorical, temporal, text)
- Multiple selectors (statistical, importance, correlation, SHAP)
- Visualization utilities
- Optional integrations (SHAP, XGBoost, LightGBM)

Importing all components eagerly at package load time causes several issues:

1. **Slow import time**: Loading all modules delays startup
2. **Circular imports**: Interdependent modules cause import failures
3. **Unnecessary dependencies**: Optional features load even when unused
4. **Memory overhead**: All classes loaded into memory

We needed a strategy to provide a clean top-level API while avoiding these issues.

## Decision

Use Python's `__getattr__()` mechanism for lazy imports in the main `__init__.py`:

```python
# forge/__init__.py
"""Forge: Automated Feature Engineering Platform."""

from forge._version import __version__, __version_info__

__all__ = [
    "__version__",
    "__version_info__",
    # Core classes - lazily imported
    "AutoFeatureTransformer",
    "DataAnalyzer",
    "ForgePipeline",
    # Generators
    "InteractionGenerator",
    "PolynomialGenerator",
    "TargetEncoder",
    # Selectors
    "ImportanceSelector",
    "CorrelationSelector",
    # ... more exports
]

def __getattr__(name: str) -> object:
    """Lazy import of main classes to avoid circular imports and speed up loading."""

    # Core transformers
    if name == "AutoFeatureTransformer":
        from forge.transformers.auto_transformer import AutoFeatureTransformer
        return AutoFeatureTransformer

    if name == "DataAnalyzer":
        from forge.analyzer.base import DataAnalyzer
        return DataAnalyzer

    if name == "ForgePipeline":
        from forge.transformers.feature_pipeline import ForgePipeline
        return ForgePipeline

    # Generators
    if name == "InteractionGenerator":
        from forge.generators.numeric.interactions import InteractionGenerator
        return InteractionGenerator

    if name == "PolynomialGenerator":
        from forge.generators.numeric.polynomials import PolynomialGenerator
        return PolynomialGenerator

    if name == "TargetEncoder":
        from forge.generators.categorical.encoders import TargetEncoder
        return TargetEncoder

    # Selectors
    if name == "ImportanceSelector":
        from forge.selectors.importance import ImportanceSelector
        return ImportanceSelector

    if name == "CorrelationSelector":
        from forge.selectors.correlation import CorrelationSelector
        return CorrelationSelector

    # Optional components (may raise MissingDependencyError)
    if name == "ShapSelector":
        from forge.selectors.shap_selector import ShapSelector
        return ShapSelector

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return list of public names for tab completion."""
    return __all__
```

### Implementation Rules

1. **All public classes listed in `__all__`**: Enables IDE completion and documentation
2. **`__getattr__()` handles lazy loading**: Import happens on first access
3. **`__dir__()` returns `__all__`**: Tab completion works correctly
4. **Clear error messages**: `AttributeError` for unknown names

## Consequences

### Positive

1. **Fast initial import**:
   ```python
   # This is fast - only loads version info
   import forge

   # This triggers the actual import
   transformer = forge.AutoFeatureTransformer()
   ```

2. **Circular import resolution**: Modules can reference each other without import-time issues:
   ```python
   # generators/base.py can import from selectors
   # selectors/base.py can import from generators
   # No circular import because neither is loaded at package init
   ```

3. **Optional dependency handling**: SHAP selector only loads when accessed:
   ```python
   # Works without shap installed
   from forge import AutoFeatureTransformer

   # Only fails if shap not installed AND ShapSelector accessed
   from forge import ShapSelector  # Raises MissingDependencyError if no shap
   ```

4. **Memory efficiency**: Only used classes are loaded:
   ```python
   # User only needs DataAnalyzer
   from forge import DataAnalyzer
   # Generators, selectors, etc. never loaded
   ```

5. **IDE support preserved**: `__all__` and `__dir__()` enable:
   - Tab completion in IPython/Jupyter
   - Autocomplete in VS Code, PyCharm
   - Documentation generation

### Negative

1. **Delayed import errors**: Import errors surface at use time, not import time:
   ```python
   import forge  # Succeeds even if submodule has syntax error

   forge.AutoFeatureTransformer()  # Error surfaces here
   ```

2. **Type checker limitations**: Static analyzers may not understand `__getattr__()`:
   ```python
   from forge import AutoFeatureTransformer  # pyright may warn "unknown import"
   ```
   **Mitigation**: Add type stubs or `TYPE_CHECKING` imports.

3. **Slight runtime overhead**: First access has import cost:
   ```python
   # First call includes import time
   t1 = forge.AutoFeatureTransformer()  # ~100ms (includes import)
   # Subsequent calls are fast
   t2 = forge.AutoFeatureTransformer()  # ~0.1ms
   ```

4. **Debugging complexity**: Stack traces include `__getattr__()`:
   ```
   File "forge/__init__.py", line 42, in __getattr__
       from forge.transformers.auto_transformer import AutoFeatureTransformer
   ```

### Mitigations

1. **TYPE_CHECKING imports for type checkers**:
   ```python
   from typing import TYPE_CHECKING

   if TYPE_CHECKING:
       from forge.transformers.auto_transformer import AutoFeatureTransformer
   ```

2. **Eager import option for testing**:
   ```python
   # forge/_eager.py - imports everything for testing
   from forge.transformers.auto_transformer import AutoFeatureTransformer
   from forge.analyzer.base import DataAnalyzer
   # ... all imports
   ```

3. **Caching**: Python caches module imports, so repeated access is fast.

## Alternatives Considered

### 1. Eager Imports

Import everything at package load time.

```python
# forge/__init__.py
from forge.transformers.auto_transformer import AutoFeatureTransformer
from forge.analyzer.base import DataAnalyzer
# ... 30+ imports
```

**Rejected because:**
- Slow import time (~500ms vs ~10ms)
- Circular import issues
- Loads unused code

### 2. Submodule-Only Access

No top-level exports, require full import paths.

```python
from forge.transformers.auto_transformer import AutoFeatureTransformer
```

**Rejected because:**
- Poor developer experience
- Long import statements
- Not idiomatic for Python packages

### 3. importlib Lazy Loading

Use `importlib.util.LazyLoader`.

```python
import importlib.util

spec = importlib.util.find_spec("forge.transformers.auto_transformer")
loader = importlib.util.LazyLoader(spec.loader)
```

**Rejected because:**
- More complex implementation
- Less control over lazy behavior
- `__getattr__()` is simpler and sufficient

## References

- [PEP 562 - Module `__getattr__` and `__dir__`](https://peps.python.org/pep-0562/)
- [Python import system documentation](https://docs.python.org/3/reference/import.html)
