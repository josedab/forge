# Feature Generation

Forge provides comprehensive feature generation capabilities for different data types.

## Numeric Features

### Aggregations

Generate group-by statistics for numeric columns:

```python
from forge.generators.numeric import AggregationGenerator

gen = AggregationGenerator(
    group_columns=['category'],
    agg_columns=['amount'],
    agg_functions=['mean', 'sum', 'std']
)
X_agg = gen.fit_transform(X)
```

### Interactions

Create interaction features between numeric columns:

```python
from forge.generators.numeric import InteractionGenerator

gen = InteractionGenerator(
    columns=['price', 'quantity'],
    operations=['multiply', 'divide', 'add']
)
X_interact = gen.fit_transform(X)
```

### Transformations

Apply mathematical transformations:

```python
from forge.generators.numeric import TransformationGenerator

gen = TransformationGenerator(
    columns=['income', 'age'],
    transformations=['log', 'sqrt', 'square']
)
X_transformed = gen.fit_transform(X)
```

### Polynomials

Generate polynomial features:

```python
from forge.generators.numeric import PolynomialGenerator

gen = PolynomialGenerator(degree=2, include_bias=False)
X_poly = gen.fit_transform(X[['feature1', 'feature2']])
```

## Categorical Features

### Encoders

Multiple encoding strategies:

```python
from forge.generators.categorical import TargetEncoder, FrequencyEncoder

# Target encoding
te = TargetEncoder(columns=['category'])
X_encoded = te.fit_transform(X, y)

# Frequency encoding
fe = FrequencyEncoder(columns=['category'])
X_freq = fe.fit_transform(X)
```

### Combinations

Create category pair interactions:

```python
from forge.generators.categorical import CategoryCombiner

gen = CategoryCombiner(columns=['category', 'region'])
X_combined = gen.fit_transform(X)
```

## Temporal Features

### Component Extraction

Extract date/time components:

```python
from forge.generators.temporal import DatetimeComponentGenerator

gen = DatetimeComponentGenerator(
    columns=['signup_date'],
    components=['year', 'month', 'dayofweek', 'hour']
)
X_temporal = gen.fit_transform(X)
```

### Lags

Create lagged features for time series:

```python
from forge.generators.temporal import LagGenerator

gen = LagGenerator(columns=['value'], lags=[1, 7, 30])
X_lagged = gen.fit_transform(X)
```

### Rolling Windows

Calculate rolling statistics:

```python
from forge.generators.temporal import RollingGenerator

gen = RollingGenerator(
    columns=['value'],
    windows=[7, 30],
    functions=['mean', 'std', 'min', 'max']
)
X_rolling = gen.fit_transform(X)
```

## Text Features

### Basic Statistics

Extract text statistics:

```python
from forge.generators.text import TextStatsGenerator

gen = TextStatsGenerator(columns=['description'])
X_stats = gen.fit_transform(X)
# Creates: length, word_count, char_count, etc.
```

### TF-IDF

Vectorize text with TF-IDF:

```python
from forge.generators.text import TfidfGenerator

gen = TfidfGenerator(columns=['text'], max_features=100)
X_tfidf = gen.fit_transform(X)
```
