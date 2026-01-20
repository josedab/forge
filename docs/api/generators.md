# Feature Generators

All feature generators inherit from `BaseFeatureGenerator` and implement the scikit-learn transformer interface.

## Numeric Generators

### AggregationGenerator

Generate group-by aggregation features.

```python
from forge.generators.numeric import AggregationGenerator

gen = AggregationGenerator(
    group_columns=['category'],
    agg_columns=['amount', 'price'],
    agg_functions=['mean', 'sum', 'std', 'min', 'max']
)
```

### InteractionGenerator

Create interaction features between columns.

```python
from forge.generators.numeric import InteractionGenerator

gen = InteractionGenerator(
    columns=['price', 'quantity', 'discount'],
    operations=['multiply', 'divide', 'add', 'subtract']
)
```

### TransformationGenerator

Apply mathematical transformations.

```python
from forge.generators.numeric import TransformationGenerator

gen = TransformationGenerator(
    columns=['income', 'age'],
    transformations=['log', 'sqrt', 'square', 'reciprocal']
)
```

### PolynomialGenerator

Generate polynomial features.

```python
from forge.generators.numeric import PolynomialGenerator

gen = PolynomialGenerator(degree=2, include_bias=False, interaction_only=False)
```

## Categorical Generators

### TargetEncoder

Encode categories using target statistics.

```python
from forge.generators.categorical import TargetEncoder

encoder = TargetEncoder(
    columns=['category', 'region'],
    smoothing=1.0
)
```

### FrequencyEncoder

Encode categories by their frequency.

```python
from forge.generators.categorical import FrequencyEncoder

encoder = FrequencyEncoder(columns=['category'])
```

### CategoryCombiner

Create combined category features.

```python
from forge.generators.categorical import CategoryCombiner

combiner = CategoryCombiner(columns=['category', 'region'])
```

## Temporal Generators

### DatetimeComponentGenerator

Extract datetime components.

```python
from forge.generators.temporal import DatetimeComponentGenerator

gen = DatetimeComponentGenerator(
    columns=['date'],
    components=['year', 'month', 'day', 'dayofweek', 'hour', 'quarter']
)
```

### LagGenerator

Create lagged features.

```python
from forge.generators.temporal import LagGenerator

gen = LagGenerator(
    columns=['value'],
    lags=[1, 7, 14, 30]
)
```

### RollingGenerator

Calculate rolling window statistics.

```python
from forge.generators.temporal import RollingGenerator

gen = RollingGenerator(
    columns=['value'],
    windows=[7, 14, 30],
    functions=['mean', 'std', 'min', 'max']
)
```

### DifferenceGenerator

Calculate time differences.

```python
from forge.generators.temporal import DifferenceGenerator

gen = DifferenceGenerator(
    columns=['value'],
    periods=[1, 7]
)
```

## Text Generators

### TextStatsGenerator

Extract text statistics.

```python
from forge.generators.text import TextStatsGenerator

gen = TextStatsGenerator(columns=['description'])
# Creates: length, word_count, avg_word_length, etc.
```

### TfidfGenerator

Generate TF-IDF features.

```python
from forge.generators.text import TfidfGenerator

gen = TfidfGenerator(
    columns=['text'],
    max_features=100,
    ngram_range=(1, 2)
)
```
