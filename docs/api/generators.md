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

### NumericTransformer

Apply mathematical transformations to numeric columns.

```python
from forge.generators.numeric import NumericTransformer

# Combined transformer with multiple options
gen = NumericTransformer(
    columns=['income', 'age'],
    log=True,      # Apply log1p transformation
    sqrt=True,     # Apply square root transformation
    square=False,  # Apply square transformation
    binning=False, # Create binned features
    n_bins=5       # Number of bins if binning=True
)
```

### LogTransformer

Apply log transformation to numeric columns.

```python
from forge.generators.numeric import LogTransformer

gen = LogTransformer(columns=['income', 'price'])
# Creates: income_log1p, price_log1p
```

### PowerTransformer

Apply power transformations (sqrt, square) to numeric columns.

```python
from forge.generators.numeric import PowerTransformer

gen = PowerTransformer(
    columns=['age', 'score'],
    transforms=['sqrt', 'square']
)
# Creates: age_sqrt, age_square, score_sqrt, score_square
```

### BinningTransformer

Create binned features from numeric columns.

```python
from forge.generators.numeric import BinningTransformer

gen = BinningTransformer(columns=['age'], n_bins=5)
# Creates: age_binned (categorical)
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

### OneHotEncoder

One-hot encode categorical features.

```python
from forge.generators.categorical import OneHotEncoder

encoder = OneHotEncoder(
    columns=['color', 'size'],
    drop_first=False,  # Whether to drop first category to avoid multicollinearity
    handle_unknown='ignore'
)
```

### OrdinalEncoder

Ordinal integer encode categorical features.

```python
from forge.generators.categorical import OrdinalEncoder

encoder = OrdinalEncoder(
    columns=['education'],
    categories={'education': ['high_school', 'bachelor', 'master', 'phd']}
)
```

### WoEEncoder

Weight of Evidence encoding for binary classification targets.

```python
from forge.generators.categorical import WoEEncoder

encoder = WoEEncoder(
    columns=['category'],
    regularization=1.0  # Laplace smoothing factor
)
# Requires binary target during fit
```

### CatBoostEncoder

CatBoost-style ordered target encoding that prevents target leakage.

```python
from forge.generators.categorical import CatBoostEncoder

encoder = CatBoostEncoder(
    columns=['category'],
    a=1.0  # Smoothing parameter
)
```

### LeaveOneOutEncoder

Leave-one-out target encoding for reduced target leakage.

```python
from forge.generators.categorical import LeaveOneOutEncoder

encoder = LeaveOneOutEncoder(
    columns=['category'],
    sigma=0.05  # Noise level for regularization
)
```

### HashingEncoder

Feature hashing for high-cardinality categorical features.

```python
from forge.generators.categorical import HashingEncoder

encoder = HashingEncoder(
    columns=['user_id'],
    n_components=32  # Number of hash buckets
)
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
