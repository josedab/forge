# Feature Selectors

All selectors inherit from `BaseFeatureSelector` and implement the scikit-learn selector interface.

## Statistical Selectors

### StatisticalSelector

Select features using statistical tests (chi-square, ANOVA F-test, or mutual information).

```python
from forge.selectors import StatisticalSelector

# Classification with chi-square
selector = StatisticalSelector(
    n_features=20,
    method='chi2'  # 'f_classif' for ANOVA, 'mutual_info_classif' for MI
)
X_selected = selector.fit_transform(X, y)

# Regression with F-test
selector = StatisticalSelector(
    n_features=20,
    method='f_regression'  # or 'mutual_info_regression'
)
```

**Parameters:**

- `n_features`: Number of features to select
- `method`: Statistical test to use:
  - `'chi2'`: Chi-square test (classification, non-negative features)
  - `'f_classif'`: ANOVA F-test (classification)
  - `'mutual_info_classif'`: Mutual information (classification)
  - `'f_regression'`: F-test (regression)
  - `'mutual_info_regression'`: Mutual information (regression)

### KBestSelector

Select K best features based on scores.

```python
from forge.selectors import KBestSelector

selector = KBestSelector(k=20, score_func='f_classif')
X_selected = selector.fit_transform(X, y)
```

## Importance-Based Selectors

### ImportanceSelector

Select features using tree-based feature importance.

```python
from forge.selectors import ImportanceSelector

# Select fixed number of features
selector = ImportanceSelector(
    n_features=50,
    task='classification'  # or 'regression', 'auto'
)

# Select by threshold
selector = ImportanceSelector(
    threshold='mean',  # or 'median', or float like 0.01
    task='auto'
)
X_selected = selector.fit_transform(X, y)

# Get feature importances
importances = selector.get_importances()
```

**Parameters:**

- `n_features`: Number of features to select (int or fraction)
- `threshold`: Importance threshold (`'mean'`, `'median'`, or float)
- `task`: Task type (`'classification'`, `'regression'`, or `'auto'`)
- `model`: Custom model to use (must have `feature_importances_`)
- `n_estimators`: Number of trees for default Random Forest
- `random_state`: Random state for reproducibility

### PermutationImportanceSelector

Select features using permutation importance.

```python
from forge.selectors import PermutationImportanceSelector

selector = PermutationImportanceSelector(
    n_features=20,
    n_repeats=10,
    random_state=42
)
X_selected = selector.fit_transform(X, y)
```

### ShapSelector

Select features using SHAP values. Requires `shap` extra.

```python
from forge.selectors import ShapSelector

selector = ShapSelector(
    n_features=20,
    task='classification'
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

### QuasiConstantSelector

Remove quasi-constant features (features with single value in > threshold of samples).

```python
from forge.selectors import QuasiConstantSelector

selector = QuasiConstantSelector(threshold=0.99)  # Remove if 99% same value
X_selected = selector.fit_transform(X)
```

### CorrelationSelector

Remove highly correlated features.

```python
from forge.selectors import CorrelationSelector

selector = CorrelationSelector(threshold=0.95)
X_selected = selector.fit_transform(X)
```

### MulticollinearitySelector

Remove features with high multicollinearity using VIF (Variance Inflation Factor).

```python
from forge.selectors import MulticollinearitySelector

selector = MulticollinearitySelector(vif_threshold=10.0)
X_selected = selector.fit_transform(X)
```

## Ensemble Selectors

### EnsembleImportanceSelector

Select features using ensemble of multiple importance methods.

```python
from forge.selectors import EnsembleImportanceSelector

selector = EnsembleImportanceSelector(
    n_features=20,
    methods=['random_forest', 'permutation', 'shap'],
    aggregation='mean'  # or 'median', 'rank'
)
X_selected = selector.fit_transform(X, y)
```

### StabilitySelector

Select features using stability selection (bootstrap + selection).

```python
from forge.selectors import StabilitySelector

selector = StabilitySelector(
    n_features=20,
    n_bootstrap=100,
    threshold=0.6  # Selection frequency threshold
)
X_selected = selector.fit_transform(X, y)
```

## AutoML Selectors

### BayesianFeatureSelector

Select features using Bayesian optimization.

```python
from forge.selectors import BayesianFeatureSelector

selector = BayesianFeatureSelector(
    n_features=20,
    n_iterations=50,
    random_state=42
)
X_selected = selector.fit_transform(X, y)
```

### SequentialFeatureSelector

Forward/backward sequential feature selection.

```python
from forge.selectors import SequentialFeatureSelector

selector = SequentialFeatureSelector(
    n_features=20,
    direction='forward',  # or 'backward'
    scoring='accuracy'
)
X_selected = selector.fit_transform(X, y)
```

### GeneticFeatureSelector

Select features using genetic algorithm.

```python
from forge.selectors import GeneticFeatureSelector

selector = GeneticFeatureSelector(
    n_features=20,
    n_generations=50,
    population_size=100,
    random_state=42
)
X_selected = selector.fit_transform(X, y)
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
