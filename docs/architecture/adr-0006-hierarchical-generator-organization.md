# ADR-0006: Hierarchical Generator Organization

## Status

Accepted

## Context

Forge provides many feature generators for different data types:
- Numeric: interactions, polynomials, log transforms, binning
- Categorical: target encoding, frequency encoding, one-hot encoding
- Temporal: date parts, lags, rolling windows, time deltas
- Text: length, word count, TF-IDF

We needed to decide how to organize these generators in the codebase.

Options considered:
1. **Flat structure**: All generators in a single `generators/` directory
2. **Hierarchical by type**: Subdirectories for each data type
3. **Hierarchical by operation**: Subdirectories by operation type (encoding, aggregation, etc.)
4. **Plugin-based**: External packages for each generator category

## Decision

Organize generators hierarchically by data type:

```
src/forge/generators/
├── __init__.py           # Public exports
├── base.py               # BaseFeatureGenerator ABC
├── registry.py           # Generator registry
├── numeric/
│   ├── __init__.py
│   ├── interactions.py   # InteractionGenerator
│   ├── polynomials.py    # PolynomialGenerator
│   ├── transformations.py # LogTransformer, SqrtTransformer, etc.
│   └── binning.py        # BinningGenerator
├── categorical/
│   ├── __init__.py
│   ├── encoders.py       # TargetEncoder, FrequencyEncoder
│   ├── onehot.py         # OneHotEncoder wrapper
│   └── combinations.py   # CategoryCombiner
├── temporal/
│   ├── __init__.py
│   ├── dateparts.py      # DatePartExtractor
│   ├── lags.py           # LagGenerator
│   └── rolling.py        # RollingWindowGenerator
└── text/
    ├── __init__.py
    ├── basic.py          # LengthExtractor, WordCounter
    └── tfidf.py          # TfidfGenerator
```

### Module Organization Rules

1. **One primary class per file** (with related helpers):
   ```python
   # generators/numeric/interactions.py
   class InteractionGenerator(BaseFeatureGenerator):
       """Generate interaction features between numeric columns."""
       ...

   class PairwiseInteractionGenerator(InteractionGenerator):
       """Generate all pairwise interactions."""
       ...
   ```

2. **Type-specific `__init__.py` exports**:
   ```python
   # generators/numeric/__init__.py
   from forge.generators.numeric.interactions import InteractionGenerator
   from forge.generators.numeric.polynomials import PolynomialGenerator
   from forge.generators.numeric.transformations import LogTransformer
   from forge.generators.numeric.binning import BinningGenerator

   __all__ = [
       "InteractionGenerator",
       "PolynomialGenerator",
       "LogTransformer",
       "BinningGenerator",
   ]
   ```

3. **Top-level exports for convenience**:
   ```python
   # generators/__init__.py
   from forge.generators.base import BaseFeatureGenerator
   from forge.generators.numeric import *
   from forge.generators.categorical import *
   from forge.generators.temporal import *
   from forge.generators.text import *
   ```

## Consequences

### Positive

1. **Discoverability**: Easy to find generators for a specific data type:
   ```python
   # Looking for categorical encoding? Check generators/categorical/
   from forge.generators.categorical import TargetEncoder, FrequencyEncoder
   ```

2. **Logical grouping**: Related generators are co-located:
   ```
   generators/temporal/
   ├── dateparts.py    # Extract year, month, day
   ├── lags.py         # Lag features
   └── rolling.py      # Rolling statistics
   ```

3. **Incremental imports**: Import only what you need:
   ```python
   # Import just numeric generators (faster, smaller memory)
   from forge.generators.numeric import InteractionGenerator

   # vs. importing everything
   from forge.generators import *
   ```

4. **Clear ownership**: Each subdirectory has focused responsibility:
   - `numeric/`: Continuous value transformations
   - `categorical/`: Discrete value encoding
   - `temporal/`: Time-based features
   - `text/`: String processing

5. **Parallel development**: Teams can work on different type directories without conflicts.

6. **Testing organization**: Test structure mirrors source:
   ```
   tests/unit/test_generators/
   ├── test_numeric/
   │   ├── test_interactions.py
   │   └── test_polynomials.py
   ├── test_categorical/
   └── test_temporal/
   ```

### Negative

1. **Deeper import paths**:
   ```python
   # Longer imports
   from forge.generators.numeric.interactions import InteractionGenerator

   # Mitigated by __init__.py re-exports
   from forge.generators.numeric import InteractionGenerator
   from forge.generators import InteractionGenerator  # Also works
   ```

2. **Cross-type generators**: Some generators span types:
   ```python
   # Where does this go?
   class NumericCategoricalInteraction(BaseFeatureGenerator):
       """Interact numeric with categorical columns."""
   ```
   **Resolution**: Place in the "primary" type, document in both.

3. **More `__init__.py` files to maintain**: Each subdirectory needs exports.

### Neutral

1. **Similar to sklearn**: sklearn uses `sklearn.preprocessing`, `sklearn.feature_extraction.text`, etc.

## Alternatives Considered

### 1. Flat Structure

All generators in a single directory.

```
generators/
├── base.py
├── interactions.py
├── polynomials.py
├── target_encoder.py
├── frequency_encoder.py
├── dateparts.py
├── lags.py
├── tfidf.py
└── ...  (20+ files)
```

**Rejected because:**
- Difficult to navigate with many files
- No logical grouping
- Hard to find related generators

### 2. Organization by Operation

Group by what the generator does rather than data type.

```
generators/
├── encoding/       # All encoders
├── aggregation/    # All aggregators
├── extraction/     # All extractors
└── transformation/ # All transformers
```

**Rejected because:**
- Less intuitive for users thinking "I have datetime data"
- Cross-cutting concerns (target encoder is both encoding and aggregation)
- Doesn't match how users think about their data

### 3. Plugin Packages

Separate packages for each category.

```
forge-generators-numeric
forge-generators-categorical
forge-generators-temporal
```

**Rejected because:**
- Complex dependency management
- Overkill for current scale
- Fragmented user experience

## Implementation Notes

### Adding a New Generator

1. Identify the appropriate type directory
2. Create module in that directory
3. Inherit from `BaseFeatureGenerator`
4. Register with the generator registry
5. Export in `__init__.py`
6. Add tests in corresponding test directory

```python
# generators/numeric/my_generator.py
from forge.generators.base import BaseFeatureGenerator
from forge.generators.registry import register_generator

@register_generator("my_generator", column_types=["numeric"])
class MyGenerator(BaseFeatureGenerator):
    """Generate custom numeric features."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        ...

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...
```

## Related Decisions

- [ADR-0002: Plugin Registry Pattern](adr-0002-plugin-registry-pattern.md) - Registry enables discovery across hierarchy
- [ADR-0003: Type Inference Approach](adr-0003-type-inference-approach.md) - Type inference determines which generators apply
