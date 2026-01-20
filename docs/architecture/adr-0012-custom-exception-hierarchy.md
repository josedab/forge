# ADR-0012: Custom Exception Hierarchy

## Status

Accepted

## Context

Forge operations can fail in many ways:
- Invalid input data (wrong type, missing columns)
- Configuration errors (invalid parameter combinations)
- Runtime errors (feature generation failures)
- Missing optional dependencies
- State errors (using unfitted transformers)

Using generic Python exceptions (`ValueError`, `TypeError`, `RuntimeError`) makes it difficult to:
- Distinguish Forge errors from other errors
- Handle errors programmatically
- Provide helpful, context-specific messages
- Enable selective error catching

We needed an exception strategy that improves error handling and debugging.

## Decision

Implement a **custom exception hierarchy** rooted in `ForgeError`:

```python
# forge/exceptions.py

class ForgeError(Exception):
    """Base exception for all Forge errors.

    All Forge-specific exceptions inherit from this class,
    enabling users to catch all Forge errors with a single except clause.
    """
    pass


class NotFittedError(ForgeError):
    """Raised when a transformer is used before fitting.

    Example:
        >>> transformer = AutoFeatureTransformer()
        >>> transformer.transform(X)  # Not fitted!
        NotFittedError: AutoFeatureTransformer is not fitted. Call fit() first.
    """
    pass


class ValidationError(ForgeError):
    """Raised when input validation fails.

    Base class for all validation-related errors.
    """
    pass


class ColumnNotFoundError(ValidationError):
    """Raised when a required column is not found in the DataFrame.

    Example:
        >>> generator = InteractionGenerator(columns=['a', 'b'])
        >>> generator.fit(df)  # df has no column 'b'
        ColumnNotFoundError: Column 'b' not found in DataFrame.
                             Available columns: ['a', 'c', 'd']
    """
    pass


class InvalidColumnTypeError(ValidationError):
    """Raised when a column has an unexpected type.

    Example:
        >>> generator = PolynomialGenerator(columns=['name'])
        >>> generator.fit(df)  # 'name' is string, not numeric
        InvalidColumnTypeError: Column 'name' has type 'object',
                                expected numeric type.
    """
    pass


class ConfigurationError(ForgeError):
    """Raised when configuration is invalid.

    Example:
        >>> AutoFeatureTransformer(max_features=-1)
        ConfigurationError: max_features must be positive, got -1
    """
    pass


class FeatureGenerationError(ForgeError):
    """Raised when feature generation fails.

    Example:
        >>> generator.transform(X)
        FeatureGenerationError: Failed to generate interaction features:
                                Division by zero in column 'price'
    """
    pass


class FeatureSelectionError(ForgeError):
    """Raised when feature selection fails.

    Example:
        >>> selector.fit(X, y)
        FeatureSelectionError: Cannot compute feature importance:
                               All features have zero variance
    """
    pass


class MissingDependencyError(ForgeError, ImportError):
    """Raised when an optional dependency is not installed.

    Inherits from ImportError for compatibility with import error handling.

    Example:
        >>> from forge import ShapSelector
        MissingDependencyError: ShapSelector requires the shap package.
                                Install with: pip install forge-features[shap]
    """
    pass
```

### Exception Hierarchy

```
ForgeError (base)
│
├── NotFittedError
│   └── Transformer used before fit()
│
├── ValidationError
│   ├── ColumnNotFoundError
│   │   └── Required column missing
│   └── InvalidColumnTypeError
│       └── Column has wrong type
│
├── ConfigurationError
│   └── Invalid parameter values
│
├── FeatureGenerationError
│   └── Generation failed
│
├── FeatureSelectionError
│   └── Selection failed
│
└── MissingDependencyError (also inherits ImportError)
    └── Optional dependency not installed
```

### Usage Patterns

**Raising exceptions with context:**

```python
def _validate_columns(self, X: pd.DataFrame) -> None:
    """Validate that required columns exist."""
    missing = set(self.columns) - set(X.columns)
    if missing:
        raise ColumnNotFoundError(
            f"Columns not found: {sorted(missing)}. "
            f"Available columns: {sorted(X.columns)}"
        )


def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    """Transform input data."""
    if not self._is_fitted:
        raise NotFittedError(
            f"{type(self).__name__} is not fitted. "
            f"Call fit() or fit_transform() first."
        )
```

**Catching specific errors:**

```python
try:
    transformer.fit(X, y)
except ColumnNotFoundError as e:
    # Handle missing column - maybe suggest similar column names
    print(f"Column error: {e}")
except ValidationError as e:
    # Handle any validation error
    print(f"Validation failed: {e}")
except ForgeError as e:
    # Handle any Forge error
    print(f"Forge error: {e}")
```

**Catching all Forge errors:**

```python
try:
    result = pipeline.fit_transform(X, y)
except ForgeError as e:
    logger.error(f"Feature engineering failed: {e}")
    # Fallback to raw features
    result = X
```

## Consequences

### Positive

1. **Selective catching**: Handle specific error types:
   ```python
   try:
       transformer.fit(X, y)
   except NotFittedError:
       transformer.fit(X_train, y_train)
   except ColumnNotFoundError as e:
       logger.warning(f"Missing columns, using available: {e}")
       transformer.columns = [c for c in transformer.columns if c in X.columns]
   ```

2. **Clear error messages**: Domain-specific context:
   ```
   ColumnNotFoundError: Column 'user_id' not found in DataFrame.
   Available columns: ['id', 'user_name', 'email']
   Did you mean: 'id'?
   ```

3. **Error categorization**: Group related errors:
   ```python
   # All validation errors
   except ValidationError:
       handle_bad_input()

   # All Forge errors
   except ForgeError:
       handle_forge_failure()
   ```

4. **Debugging support**: Error type indicates problem area:
   ```
   FeatureGenerationError → Check generators, input data
   FeatureSelectionError → Check selectors, target variable
   ConfigurationError → Check transformer parameters
   ```

5. **Documentation**: Exception docstrings document failure modes:
   ```python
   class ColumnNotFoundError(ValidationError):
       """Raised when a required column is not found.

       This typically occurs when:
       - Column names are misspelled
       - DataFrame was modified after fitting
       - Column was dropped during preprocessing

       Example:
           >>> generator.fit(df_train)
           >>> generator.transform(df_test)  # df_test missing column
           ColumnNotFoundError: ...
       """
   ```

### Negative

1. **Learning curve**: Users must learn Forge exceptions:
   ```python
   # User might write:
   except ValueError:  # Doesn't catch ColumnNotFoundError

   # Should write:
   except ForgeError:  # Catches all Forge errors
   ```

2. **Exception proliferation**: Risk of too many exception types:
   ```python
   # Avoid going too granular
   class NumericColumnNotFoundError(ColumnNotFoundError): ...
   class CategoricalColumnNotFoundError(ColumnNotFoundError): ...
   # This is over-engineering
   ```

3. **Maintenance overhead**: Each exception needs:
   - Clear docstring
   - Consistent message format
   - Test coverage

### Mitigations

1. **Inherit from stdlib**: `MissingDependencyError` inherits `ImportError` for compatibility
2. **Shallow hierarchy**: Keep hierarchy depth ≤ 3
3. **Consistent messages**: Use message templates

## Alternatives Considered

### 1. Standard Library Exceptions Only

Use `ValueError`, `TypeError`, `RuntimeError`, etc.

```python
if column not in X.columns:
    raise ValueError(f"Column {column} not found")
```

**Rejected because:**
- Can't distinguish Forge errors from other libraries
- No error categorization
- Generic messages less helpful

### 2. Error Codes

Use error codes instead of exception types.

```python
class ForgeError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")

raise ForgeError("E001", "Column not found")
```

**Rejected because:**
- Less Pythonic
- Codes are opaque
- Can't use isinstance() for handling

### 3. Result Types

Return Result objects instead of raising exceptions.

```python
from dataclasses import dataclass

@dataclass
class Result:
    success: bool
    value: Any = None
    error: str | None = None

def fit(self, X, y) -> Result:
    if error_condition:
        return Result(success=False, error="Column not found")
    return Result(success=True, value=self)
```

**Rejected because:**
- Non-Pythonic pattern
- Doesn't integrate with sklearn
- Verbose usage

### 4. Warnings Instead of Errors

Use warnings for recoverable issues.

```python
import warnings

if missing_values > threshold:
    warnings.warn(f"High missing rate: {missing_values:.1%}")
```

**Rejected because:**
- Warnings can be ignored
- Not appropriate for fatal errors
- Hard to handle programmatically

## Exception Guidelines

### When to Use Each Exception

| Exception | Use When |
|-----------|----------|
| `NotFittedError` | Method requires fitted state but transformer unfitted |
| `ColumnNotFoundError` | Required column missing from DataFrame |
| `InvalidColumnTypeError` | Column has wrong dtype for operation |
| `ConfigurationError` | Invalid parameter value in `__init__` |
| `FeatureGenerationError` | Generator fails during `transform()` |
| `FeatureSelectionError` | Selector fails during `fit()` or `transform()` |
| `MissingDependencyError` | Optional package not installed |
| `ValidationError` | Other input validation failures |
| `ForgeError` | Other Forge-specific errors |

### Message Format

```python
# Good: Specific, actionable
raise ColumnNotFoundError(
    f"Column '{column}' not found in DataFrame. "
    f"Available columns: {list(X.columns)[:10]}..."
)

# Bad: Vague, unhelpful
raise ColumnNotFoundError("Column not found")
```

## Related Decisions

- [ADR-0011: Optional Dependencies Strategy](adr-0011-optional-dependencies-strategy.md) - `MissingDependencyError` usage
- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - `NotFittedError` mirrors sklearn's pattern
