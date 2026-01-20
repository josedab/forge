# Feature Selection

Forge provides multiple strategies for selecting the most important features.

## Statistical Selection

### Chi-Square Test

For categorical features with classification targets:

```python
from forge.selectors import ChiSquareSelector

selector = ChiSquareSelector(k=20)
X_selected = selector.fit_transform(X, y)
```

### ANOVA F-Test

For numeric features with classification targets:

```python
from forge.selectors import ANOVASelector

selector = ANOVASelector(k=20)
X_selected = selector.fit_transform(X, y)
```

### Mutual Information

Works with any feature type:

```python
from forge.selectors import MutualInfoSelector

selector = MutualInfoSelector(k=20)
X_selected = selector.fit_transform(X, y)
```

## Importance-Based Selection

### Tree-Based Importance

Use Random Forest feature importance:

```python
from forge.selectors import ImportanceSelector

selector = ImportanceSelector(
    estimator='random_forest',
    threshold=0.01
)
X_selected = selector.fit_transform(X, y)
```

### SHAP-Based Selection

Use SHAP values for selection (requires `shap` extra):

```python
from forge.selectors import SHAPSelector

selector = SHAPSelector(k=20)
X_selected = selector.fit_transform(X, y)

# Get SHAP values for analysis
shap_values = selector.shap_values_
```

## Filter Methods

### Variance Threshold

Remove low-variance features:

```python
from forge.selectors import VarianceSelector

selector = VarianceSelector(threshold=0.01)
X_selected = selector.fit_transform(X)
```

### Correlation Filter

Remove highly correlated features:

```python
from forge.selectors import CorrelationSelector

selector = CorrelationSelector(threshold=0.95)
X_selected = selector.fit_transform(X)
```

## Combining Selection Methods

Chain multiple selectors:

```python
from forge.transformers import ForgePipeline
from forge.selectors import VarianceSelector, CorrelationSelector, ImportanceSelector

pipeline = ForgePipeline([
    ('variance', VarianceSelector(threshold=0.01)),
    ('correlation', CorrelationSelector(threshold=0.9)),
    ('importance', ImportanceSelector(k=50)),
])

X_selected = pipeline.fit_transform(X, y)
```

## Getting Selection Results

All selectors support introspection:

```python
# Get selected feature mask
mask = selector.get_support()

# Get selected feature indices
indices = selector.get_support(indices=True)

# Get selected feature names
names = selector.get_feature_names_out()
```
