# ADR-0004: DataFrame-First Design

## Status

Accepted

## Context

Forge is a feature engineering platform that needs to handle tabular data throughout its pipeline. We needed to decide on the primary data structure for inputs and outputs across all components.

Options considered:
1. **NumPy arrays**: sklearn's native format, efficient for numerical operations
2. **pandas DataFrames**: Rich metadata, column names, mixed types
3. **Hybrid approach**: Accept both, convert internally
4. **Custom data structure**: Purpose-built for feature engineering

Key requirements:
- Preserve column names through transformations
- Support mixed data types (numeric, categorical, text, datetime)
- Enable feature traceability and debugging
- Maintain compatibility with sklearn pipelines

## Decision

All Forge transformers accept and return **pandas DataFrames** as their primary data structure:

```python
class BaseFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Fit the generator to training data."""
        self._validate_input(X)
        # Learn from DataFrame
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform input DataFrame to output DataFrame."""
        # Generate features, preserving DataFrame structure
        return X_transformed

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Return output feature names."""
        return self._feature_names_out
```

### Key Implementation Rules

1. **Input validation accepts DataFrames**:
   ```python
   def _validate_input(self, X: pd.DataFrame) -> None:
       if not isinstance(X, pd.DataFrame):
           raise ValidationError("Input must be a pandas DataFrame")
   ```

2. **Column names preserved and tracked**:
   ```python
   def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
       self._feature_names_in = list(X.columns)
       # ... fitting logic
       self._feature_names_out = self._generate_feature_names()
       return self
   ```

3. **Output is always a DataFrame with named columns**:
   ```python
   def transform(self, X: pd.DataFrame) -> pd.DataFrame:
       result = self._compute_features(X)
       return pd.DataFrame(result, columns=self._feature_names_out, index=X.index)
   ```

4. **Index preservation**:
   ```python
   # Input index is preserved in output
   X_out = transformer.transform(X_in)
   assert X_out.index.equals(X_in.index)
   ```

## Consequences

### Positive

1. **Feature traceability**: Column names make it easy to trace features through the pipeline:
   ```python
   transformer.fit_transform(X, y)
   print(transformer.get_feature_names_out())
   # ['age', 'age_squared', 'age_log', 'income', 'income_binned', ...]
   ```

2. **Debugging support**: Named columns simplify debugging:
   ```python
   X_transformed = pipeline.transform(X)
   # Easy to inspect specific features
   print(X_transformed[['age_income_interaction', 'category_encoded']])
   ```

3. **Mixed type support**: DataFrames naturally handle mixed types:
   ```python
   # Single DataFrame can contain numeric, categorical, datetime columns
   df = pd.DataFrame({
       'age': [25, 30, 35],           # numeric
       'category': ['A', 'B', 'A'],   # categorical
       'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
   })
   ```

4. **Rich metadata**: DataFrames preserve dtypes, allowing type-aware processing:
   ```python
   numeric_cols = X.select_dtypes(include=[np.number]).columns
   categorical_cols = X.select_dtypes(include=['object', 'category']).columns
   ```

5. **Index alignment**: Operations automatically align on index:
   ```python
   # Concatenating generated features maintains row alignment
   result = pd.concat([X, generated_features], axis=1)
   ```

### Negative

1. **Performance overhead**: DataFrames are slower than raw NumPy arrays:
   - DataFrame creation has overhead
   - Column name lookups add latency
   - Memory usage higher due to metadata

2. **sklearn compatibility friction**: Some sklearn estimators expect NumPy arrays:
   ```python
   # May need conversion at pipeline boundaries
   final_model.fit(X_transformed.values, y)
   ```

3. **Memory duplication**: Creating new DataFrames for each transformation step increases memory usage compared to in-place NumPy operations.

### Mitigations

1. **Lazy evaluation where possible**: Defer DataFrame creation until needed
2. **NumPy for internal computations**: Use NumPy arrays internally, wrap in DataFrame only for output
3. **Chunked processing for large datasets**: Process in batches to manage memory

## Alternatives Considered

### 1. NumPy Arrays Only

Use NumPy arrays like most sklearn transformers.

**Rejected because:**
- Loses column names (critical for feature engineering)
- No mixed type support without structured arrays
- Debugging becomes difficult ("what is column 47?")

### 2. Accept Both, Convert Internally

Accept DataFrames or arrays, convert to common internal format.

**Rejected because:**
- Ambiguity in output format
- Complex conversion logic
- Users wouldn't know what to expect

### 3. Custom Data Structure

Create a Forge-specific data container.

**Rejected because:**
- Learning curve for users
- No ecosystem compatibility
- Duplicates pandas functionality

## References

- [pandas DataFrame documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html)
- [sklearn set_output API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html) - sklearn's newer DataFrame output support validates this approach
