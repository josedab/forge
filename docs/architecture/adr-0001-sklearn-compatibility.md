# ADR-0001: scikit-learn Compatibility

## Status

Accepted

## Context

Forge needs to integrate with existing machine learning pipelines and workflows. Data scientists already have established practices using scikit-learn's ecosystem of tools, including:

- `Pipeline` for chaining transformations
- `GridSearchCV` and `RandomizedSearchCV` for hyperparameter tuning
- `cross_val_score` for model evaluation
- `clone()` for creating fresh estimator copies

We needed to decide how Forge transformers would interact with this ecosystem.

## Decision

All Forge transformers will inherit from scikit-learn's `BaseEstimator` and `TransformerMixin`, implementing the full sklearn transformer interface:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class BaseFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        """Learn from training data."""
        return self

    def transform(self, X):
        """Transform input data."""
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        return self._feature_names
```

### Key Implementation Rules

1. **No mutable default arguments** in `__init__`:
   ```python
   # Bad
   def __init__(self, columns=["a", "b"]):
       self.columns = columns

   # Good
   def __init__(self, columns=None):
       self.columns = columns
   ```

2. **All parameters stored as attributes** with same names:
   ```python
   def __init__(self, max_features=100, strategy="auto"):
       self.max_features = max_features
       self.strategy = strategy
   ```

3. **Use `check_is_fitted()` before transform**:
   ```python
   from sklearn.utils.validation import check_is_fitted

   def transform(self, X):
       check_is_fitted(self)
       return X_transformed
   ```

4. **Implement `get_feature_names_out()`** for feature tracking:
   ```python
   def get_feature_names_out(self, input_features=None):
       check_is_fitted(self)
       return np.array(self._feature_names)
   ```

## Consequences

### Positive

1. **Seamless integration** with existing sklearn workflows:
   ```python
   from sklearn.pipeline import Pipeline
   from sklearn.ensemble import RandomForestClassifier
   from forge import AutoFeatureTransformer

   pipe = Pipeline([
       ("features", AutoFeatureTransformer(max_features=50)),
       ("clf", RandomForestClassifier()),
   ])

   # Works with cross-validation
   scores = cross_val_score(pipe, X, y, cv=5)

   # Works with grid search
   GridSearchCV(pipe, {"features__max_features": [25, 50, 100]})
   ```

2. **Feature tracking** through pipelines:
   ```python
   pipe.fit(X, y)
   feature_names = pipe["features"].get_feature_names_out()
   ```

3. **Cloning** works correctly for cross-validation:
   ```python
   from sklearn.base import clone
   new_transformer = clone(transformer)  # Fresh copy with same params
   ```

### Negative

1. **Must follow sklearn's clone semantics**: Cannot rely on `__init__` being called after `fit()`. All fitted state must be in separately named attributes (conventionally with trailing underscore).

2. **Parameter constraints**: All constructor parameters must be stored as-is without modification. Validation happens in `fit()`, not `__init__()`.

3. **No method chaining in transform**: The `transform()` method must return data, not `self`.

### Neutral

1. **Dual inheritance**: All transformers inherit from both `BaseEstimator` and `TransformerMixin`, which is standard sklearn practice.

## Alternatives Considered

### 1. Custom Pipeline System

Create our own pipeline abstraction independent of sklearn.

**Rejected because:**
- Would require users to learn a new API
- Would not work with existing sklearn tools
- Would duplicate significant functionality

### 2. Wrapper Classes

Keep Forge internals separate and provide sklearn wrappers.

**Rejected because:**
- Added complexity with two parallel hierarchies
- Wrapping has overhead and complicates debugging
- Direct inheritance is cleaner

## References

- [sklearn Developer Guide: Developing scikit-learn estimators](https://scikit-learn.org/stable/developers/develop.html)
- [sklearn Clone Function](https://scikit-learn.org/stable/modules/generated/sklearn.base.clone.html)
