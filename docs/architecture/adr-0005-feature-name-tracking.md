# ADR-0005: Feature Name Tracking

## Status

Accepted

## Context

Feature engineering pipelines can generate hundreds of features from a handful of input columns. Without proper tracking, it becomes impossible to:

- Understand which original columns contributed to which generated features
- Debug issues when a specific feature has unexpected values
- Explain model predictions in terms of original data
- Reproduce feature engineering steps

We needed a systematic approach to track feature names through all transformations.

## Decision

Implement comprehensive feature name tracking through a `ForgeTransformerMixin` that all transformers inherit:

```python
class ForgeTransformerMixin:
    """Mixin providing feature name tracking for all Forge transformers."""

    _feature_names_in: list[str]
    _feature_names_out: list[str]

    @property
    def n_features_in_(self) -> int:
        """Number of input features."""
        return len(self._feature_names_in)

    @property
    def n_features_out_(self) -> int:
        """Number of output features."""
        return len(self._feature_names_out)

    @property
    def feature_names_in_(self) -> list[str]:
        """Names of input features."""
        return self._feature_names_in.copy()

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        check_is_fitted(self)
        return self._feature_names_out.copy()
```

### Naming Conventions

Generated features follow predictable naming patterns:

```python
# Interaction features
f"{col1}_{col2}_interaction"      # e.g., "age_income_interaction"

# Polynomial features
f"{col}_squared"                   # e.g., "age_squared"
f"{col}_cubed"                     # e.g., "age_cubed"

# Transformation features
f"{col}_log"                       # e.g., "income_log"
f"{col}_sqrt"                      # e.g., "income_sqrt"
f"{col}_binned"                    # e.g., "age_binned"

# Encoded features
f"{col}_encoded"                   # e.g., "category_encoded"
f"{col}_{value}"                   # e.g., "color_red", "color_blue" (one-hot)

# Temporal features
f"{col}_year"                      # e.g., "date_year"
f"{col}_month"                     # e.g., "date_month"
f"{col}_dayofweek"                 # e.g., "date_dayofweek"

# Lag features
f"{col}_lag_{n}"                   # e.g., "sales_lag_1", "sales_lag_7"

# Rolling features
f"{col}_rolling_{window}_{stat}"   # e.g., "sales_rolling_7_mean"
```

### Implementation Pattern

Each generator implements feature name generation:

```python
class InteractionGenerator(BaseFeatureGenerator):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        self._feature_names_in = list(X.columns)

        # Generate names for all interaction pairs
        self._feature_names_out = []
        for col1, col2 in self._get_interaction_pairs(X):
            self._feature_names_out.append(f"{col1}_{col2}_interaction")

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        features = {}
        for col1, col2 in self._get_interaction_pairs(X):
            name = f"{col1}_{col2}_interaction"
            features[name] = X[col1] * X[col2]
        return pd.DataFrame(features, index=X.index)
```

## Consequences

### Positive

1. **End-to-end traceability**: Features can be traced from output back to input:
   ```python
   transformer = AutoFeatureTransformer()
   transformer.fit(X, y)

   # See all generated features
   print(transformer.get_feature_names_out())
   # ['age', 'age_squared', 'age_income_interaction', 'category_A', ...]

   # Understand feature importance in context
   importance = transformer.get_feature_importance()
   # DataFrame with feature names and importance scores
   ```

2. **Pipeline compatibility**: Works with sklearn's feature name propagation:
   ```python
   from sklearn.pipeline import Pipeline

   pipe = Pipeline([
       ('features', AutoFeatureTransformer()),
       ('selector', SelectKBest(k=10)),
       ('model', RandomForestClassifier()),
   ])

   pipe.fit(X, y)
   # Feature names flow through pipeline
   selected_names = pipe['selector'].get_feature_names_out(
       pipe['features'].get_feature_names_out()
   )
   ```

3. **Debugging support**: Named features simplify debugging:
   ```python
   X_transformed = transformer.transform(X)

   # Inspect specific feature
   print(X_transformed['age_income_interaction'].describe())

   # Find features with NaN
   nan_features = X_transformed.columns[X_transformed.isna().any()]
   ```

4. **Model interpretability**: Feature names enable explanations:
   ```python
   # SHAP values with meaningful names
   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_transformed)
   shap.summary_plot(shap_values, X_transformed)
   # Plot shows "age_income_interaction" not "feature_47"
   ```

5. **Reproducibility**: Feature names document transformations:
   ```python
   # Save feature configuration
   config = {
       'input_features': transformer.feature_names_in_,
       'output_features': transformer.get_feature_names_out(),
   }
   ```

### Negative

1. **Memory overhead**: Storing feature names for large feature sets:
   - 1000 features × 50 chars average = ~50KB
   - Negligible for most use cases

2. **Name collision potential**: Generated names could conflict:
   ```python
   # If input has "age_squared", polynomial generator creates conflict
   # Mitigation: Check for existing names and add suffix
   ```

3. **Long names for complex features**:
   ```python
   # Deeply nested transformations create long names
   "category_A_age_interaction_rolling_7_mean"
   # Mitigation: Truncation option with hash suffix
   ```

### Mitigations

1. **Collision detection**: Check for duplicate names and add numeric suffixes
2. **Name length limits**: Optional truncation with hash for uniqueness
3. **Lazy name generation**: Only compute names when requested

## Alternatives Considered

### 1. Numeric Indices Only

Track features by position, not name.

**Rejected because:**
- Difficult to debug ("what is feature 147?")
- No semantic meaning
- Breaks when feature order changes

### 2. Separate Metadata Object

Store names in a separate metadata container.

**Rejected because:**
- Easy to lose sync with data
- More complex API
- Doesn't integrate with sklearn

### 3. Column Name Prefixing

Add transformer name as prefix to all outputs.

**Rejected because:**
- Creates very long names quickly
- Less readable
- Harder to understand feature meaning

## Related Decisions

- [ADR-0004: DataFrame-First Design](adr-0004-dataframe-first-design.md) - DataFrames enable named columns
- [ADR-0001: sklearn Compatibility](adr-0001-sklearn-compatibility.md) - `get_feature_names_out()` is sklearn standard
