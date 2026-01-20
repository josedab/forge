# Feature Selectors

All selectors inherit from `BaseFeatureSelector` and implement the scikit-learn selector interface.

## Statistical Selectors

### ChiSquareSelector

Select features using chi-square test.

```python
from forge.selectors import ChiSquareSelector

selector = ChiSquareSelector(k=20)
X_selected = selector.fit_transform(X, y)
```

**Parameters:**

- `k`: Number of features to select

### ANOVASelector

Select features using ANOVA F-test.

```python
from forge.selectors import ANOVASelector

selector = ANOVASelector(k=20)
X_selected = selector.fit_transform(X, y)
```

### MutualInfoSelector

Select features using mutual information.

```python
from forge.selectors import MutualInfoSelector

selector = MutualInfoSelector(k=20, discrete_features='auto')
```

## Importance-Based Selectors

### ImportanceSelector

Select features using model-based importance.

```python
from forge.selectors import ImportanceSelector

selector = ImportanceSelector(
    estimator='random_forest',  # or 'xgboost', 'lightgbm'
    k=50,                       # or threshold=0.01
)
```

**Parameters:**

- `estimator`: Model to use (`'random_forest'`, `'xgboost'`, `'lightgbm'`)
- `k`: Number of features to select (mutually exclusive with `threshold`)
- `threshold`: Minimum importance threshold

### SHAPSelector

Select features using SHAP values. Requires `shap` extra.

```python
from forge.selectors import SHAPSelector

selector = SHAPSelector(
    k=20,
    estimator='xgboost'
)
X_selected = selector.fit_transform(X, y)

# Access SHAP values
shap_values = selector.shap_values_
```

## Filter Selectors

### VarianceSelector

Remove low-variance features.

```python
from forge.selectors import VarianceSelector

selector = VarianceSelector(threshold=0.01)
X_selected = selector.fit_transform(X)
```

### CorrelationSelector

Remove highly correlated features.

```python
from forge.selectors import CorrelationSelector

selector = CorrelationSelector(threshold=0.95)
X_selected = selector.fit_transform(X)
```

## Common Methods

All selectors implement:

### get_support

```python
# Get boolean mask
mask = selector.get_support()

# Get indices
indices = selector.get_support(indices=True)
```

### get_feature_names_out

```python
names = selector.get_feature_names_out()
```

### transform

```python
X_selected = selector.transform(X)
```
