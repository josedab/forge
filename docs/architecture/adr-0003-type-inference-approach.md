# ADR-0003: Type Inference Approach

## Status

Accepted

## Context

Forge needs to automatically determine column types to apply appropriate feature generators. The challenge is that pandas dtypes don't always reflect the semantic type of data:

- A column with dtype `int64` could be:
  - A true numeric value (age, price)
  - A categorical ID (user_id, product_code)
  - An encoded category (0/1/2 for low/medium/high)

- A column with dtype `object` could be:
  - Free-form text (description, comment)
  - A categorical value (country, status)
  - A datetime string ("2024-01-15")

We needed a robust approach to infer semantic column types.

## Decision

Implement a multi-signal type inference system that considers:

1. **pandas dtype** as a starting hint
2. **Cardinality** (unique value count relative to total rows)
3. **Value patterns** (regex matching for dates, emails, etc.)
4. **Statistical properties** (distribution shape, value ranges)

### Inference Logic

```python
def infer_column_type(column: pd.Series) -> str:
    """Infer semantic type of a column."""
    dtype = column.dtype
    n_unique = column.nunique()
    n_total = len(column)
    cardinality_ratio = n_unique / n_total

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if _looks_like_datetime(column):
        return "datetime"

    # Check for numeric
    if pd.api.types.is_numeric_dtype(dtype):
        # High cardinality numeric = true numeric
        if cardinality_ratio > 0.05 or n_unique > 20:
            return "numeric"
        # Low cardinality numeric = likely categorical
        return "categorical"

    # Check for text vs categorical in object columns
    if dtype == "object":
        avg_length = column.str.len().mean()
        # Long strings = text
        if avg_length > 50:
            return "text"
        # Many unique short strings = categorical
        return "categorical"

    return "unknown"
```

### Configurable Thresholds

Users can adjust inference behavior:

```python
analyzer = DataAnalyzer(
    categorical_threshold=0.05,  # Cardinality ratio below this = categorical
    text_min_length=50,          # Avg length above this = text
    datetime_patterns=[...],      # Custom datetime patterns
)
```

### Override Mechanism

Users can explicitly specify column types:

```python
transformer = AutoFeatureTransformer(
    column_types={
        "user_id": "categorical",  # Override: treat as categorical
        "description": "text",      # Override: treat as text
    }
)
```

## Consequences

### Positive

1. **Automatic handling**: Users don't need to manually specify types for most columns:
   ```python
   # Just works - types inferred automatically
   transformer = AutoFeatureTransformer()
   transformer.fit_transform(df, y)
   ```

2. **Semantic accuracy**: Distinguishes between numeric values and numeric-encoded categories:
   ```python
   # Column "rating" with values [1, 2, 3, 4, 5]
   # Inferred as categorical, not numeric
   # Gets appropriate encoding, not polynomial features
   ```

3. **Flexibility**: Users can override when needed:
   ```python
   transformer = AutoFeatureTransformer(
       column_types={"product_id": "categorical"}
   )
   ```

4. **Transparency**: Type inference results are accessible:
   ```python
   analyzer = DataAnalyzer()
   report = analyzer.analyze(df)
   print(report.column_types)
   # {"age": "numeric", "country": "categorical", ...}
   ```

### Negative

1. **Potential misclassification**: Heuristics can fail:
   - A zip code column (5 digits, high cardinality) might be classified as numeric
   - A short text column might be classified as categorical

2. **Non-deterministic edge cases**: Columns near thresholds might be classified differently depending on sample:
   - 4.9% unique values → categorical
   - 5.1% unique values → numeric

3. **Performance overhead**: Type inference requires a data scan, adding latency to `fit()`.

### Neutral

1. **Common pattern**: This approach is similar to pandas' `infer_objects()` and AutoML systems like auto-sklearn.

## Alternatives Considered

### 1. Rely on pandas dtypes Only

Use dtype directly without semantic inference.

**Rejected because:**
- Would mishandle numeric-encoded categories
- Would miss datetime strings in object columns
- Users would need to pre-process data

### 2. Require Explicit Type Specification

Force users to declare all column types.

**Rejected because:**
- Poor user experience for large datasets
- Violates "automatic" in "automatic feature engineering"
- Many columns have obvious types

### 3. Machine Learning-Based Inference

Train a model to classify column types.

**Rejected because:**
- Over-engineering for this use case
- Would require training data
- Heuristics work well for most cases

## Implementation Notes

### Type Hierarchy

```
Column Types:
├── numeric
│   ├── continuous (float, high cardinality int)
│   └── discrete (low cardinality int, treated as numeric)
├── categorical
│   ├── low_cardinality (< 20 unique)
│   └── high_cardinality (>= 20 unique)
├── datetime
│   ├── date
│   └── timestamp
├── text
│   ├── short_text (< 100 chars avg)
│   └── long_text (>= 100 chars avg)
└── unknown
```

### Configuration Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `categorical_threshold` | 0.05 | Max cardinality ratio for categorical |
| `categorical_max_unique` | 20 | Max unique values for low cardinality |
| `text_min_length` | 50 | Min avg length for text classification |
| `datetime_patterns` | ISO 8601 + common | Patterns for datetime detection |

## Related Decisions

- [ADR-0002: Plugin Registry Pattern](adr-0002-plugin-registry-pattern.md) - Type inference feeds into generator selection
