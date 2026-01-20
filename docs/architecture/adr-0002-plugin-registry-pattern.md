# ADR-0002: Plugin Registry Pattern

## Status

Accepted

## Context

Forge supports multiple types of feature generators:
- Numeric generators (interactions, polynomials, transformations)
- Categorical generators (encodings, combinations)
- Temporal generators (date parts, lags, rolling windows)
- Text generators (lengths, TF-IDF)

We needed a way to:
1. Automatically discover available generators
2. Select appropriate generators based on column types
3. Allow users to add custom generators without modifying core code
4. Enable/disable generators at runtime

## Decision

Implement a registry pattern where generators register themselves with metadata about their capabilities:

```python
# forge/generators/registry.py
from typing import Callable, Type

_GENERATOR_REGISTRY: dict[str, dict] = {}

def register_generator(
    name: str,
    column_types: list[str],
    priority: int = 50,
) -> Callable[[Type], Type]:
    """Decorator to register a generator class."""
    def decorator(cls: Type) -> Type:
        _GENERATOR_REGISTRY[name] = {
            "class": cls,
            "column_types": column_types,
            "priority": priority,
        }
        return cls
    return decorator

def get_generators_for_type(column_type: str) -> list[Type]:
    """Get all generators that handle a specific column type."""
    return [
        info["class"]
        for info in sorted(
            _GENERATOR_REGISTRY.values(),
            key=lambda x: x["priority"],
            reverse=True,
        )
        if column_type in info["column_types"]
    ]
```

### Generator Registration

Generators register themselves using the decorator:

```python
# forge/generators/numeric/interactions.py
from forge.generators.base import BaseFeatureGenerator
from forge.generators.registry import register_generator

@register_generator(
    name="interactions",
    column_types=["numeric"],
    priority=80,  # Higher priority = applied earlier
)
class InteractionGenerator(BaseFeatureGenerator):
    """Generate interaction features between numeric columns."""
    ...
```

### Auto-Discovery

The `AutoFeatureTransformer` uses the registry to select generators:

```python
class AutoFeatureTransformer:
    def fit(self, X, y=None):
        # Infer column types
        column_types = self._analyzer.infer_types(X)

        # Get generators for each type
        for col, dtype in column_types.items():
            generators = get_generators_for_type(dtype)
            for gen_class in generators:
                self._generators.append(gen_class())

        return self
```

## Consequences

### Positive

1. **Extensibility**: Users can add custom generators:
   ```python
   from forge.generators.base import BaseFeatureGenerator
   from forge.generators.registry import register_generator

   @register_generator("my_custom", column_types=["numeric"])
   class MyCustomGenerator(BaseFeatureGenerator):
       def transform(self, X):
           # Custom feature generation logic
           return X_custom
   ```

2. **Discoverability**: Easy to list all available generators:
   ```python
   from forge.generators.registry import list_generators

   for name, info in list_generators():
       print(f"{name}: {info['column_types']}")
   ```

3. **Selective activation**: Enable/disable generators by name:
   ```python
   transformer = AutoFeatureTransformer(
       include_generators=["interactions", "polynomials"],
       exclude_generators=["tfidf"],
   )
   ```

4. **Priority-based ordering**: Control which generators run first:
   ```python
   @register_generator("fast_gen", column_types=["numeric"], priority=90)
   @register_generator("slow_gen", column_types=["numeric"], priority=10)
   ```

### Negative

1. **Import side effects**: Generators must be imported to be registered. We handle this with explicit imports in `__init__.py`:
   ```python
   # forge/generators/__init__.py
   from forge.generators.numeric import interactions, polynomials
   from forge.generators.categorical import encoders
   ```

2. **Global state**: The registry is a module-level dictionary. This could cause issues in testing if not properly reset.

3. **Discovery limitations**: Generators in external packages need explicit registration calls.

### Neutral

1. **Similar to pytest plugins**: This pattern is well-established in the Python ecosystem (pytest, Flask, etc.).

## Alternatives Considered

### 1. Configuration File

Define available generators in a YAML/JSON config file.

**Rejected because:**
- Separates code from configuration
- Harder to add custom generators
- Would require parsing and validation

### 2. Class Scanning

Automatically scan for subclasses of `BaseFeatureGenerator`.

**Rejected because:**
- Relies on import side effects anyway
- No way to specify metadata (column types, priority)
- More magical and harder to debug

### 3. Manual Registration in AutoFeatureTransformer

Hardcode the list of generators in the main class.

**Rejected because:**
- Not extensible
- Requires modifying core code to add generators
- Mixes configuration with logic

## Related Decisions

- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - Generators must follow sklearn conventions
- [ADR-0003: Type Inference Approach](adr-0003-type-inference-approach.md) - Type inference determines which generators apply
